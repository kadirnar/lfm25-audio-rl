"""Held-out evaluation with a frozen independent ASR scorer and saved audio."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import Experiment
from .data import digest, file_hash, load_dataset, safe_audio_path
from .lora import inject_lora, load_adapter_state
from .rewards import WhisperScorer, score_reply


def paired_bootstrap(before: list[float], after: list[float], seed: int = 42) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("Matched nonempty evaluation samples required")
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite evaluation score")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(delta, size=len(delta), replace=True).mean()) for _ in range(2000)]
    return {
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_seed": seed,
    }


def compare_runs(before_path: Path, after_path: Path) -> dict:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    for key in ["dataset_hash", "split", "seed", "sampling", "asr_model", "reward_version"]:
        if before[key] != after[key]:
            raise ValueError(f"Evaluation protocols differ: {key}")
    if any(row["reward"] is None for row in before["results"] + after["results"]):
        raise ValueError("Use compare-metrics for open-ended evaluation reports")
    a = {row["example_id"]: row["reward"]["total"] for row in before["results"]}
    b = {row["example_id"]: row["reward"]["total"] for row in after["results"]}
    if set(a) != set(b) or len(a) != len(before["results"]) or len(b) != len(after["results"]):
        raise ValueError("Evaluation IDs differ or contain duplicates")
    return paired_bootstrap([a[k] for k in sorted(a)], [b[k] for k in sorted(a)])


def evaluate(
    config: Experiment,
    data: Path,
    output: Path,
    checkpoint: Path | None,
    split: str = "test",
    limit: int | None = None,
    metrics_config=None,
    warmup: int = 1,
):
    rows, metadata = load_dataset(data, require_audio=True)
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate on validation or test, never the training split")
    rows = [r for r in rows if r.split == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty evaluation split")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real-model evaluation currently requires a BF16-capable CUDA GPU")
    from liquid_audio import LFM2AudioProcessor

    from .lfm import decode_audio, generate, recording_model_class

    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    processor = LFM2AudioProcessor.from_pretrained(
        config.model_id, revision=config.model_revision, device="cuda"
    ).eval()
    model = (
        recording_model_class()
        .from_pretrained(config.model_id, revision=config.model_revision, device="cuda")
        .eval()
    )
    model.requires_grad_(False)
    if checkpoint:
        inject_lora(model.lfm, config.lora_targets, config.lora_rank, config.lora_alpha)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Check the config/data identity before evaluating adapters.
        identity = digest({"config": config.model_dump(), "dataset": metadata["manifest_sha256"]})
        if state["identity"] != identity:
            raise ValueError(
                "Checkpoint config/dataset mismatch; use the training config and dataset"
            )
        load_adapter_state(model, state["adapters"])
    scorer = WhisperScorer(config.asr_model, config.asr_device)
    generation_source = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "adapter_sha256": file_hash(checkpoint) if checkpoint else "base",
        "seed": str(config.seed),
        "temperature": str(config.temperature),
        "max_new_tokens": str(config.max_new_tokens),
    }
    for _ in range(warmup):
        warm = generate(model, processor, safe_audio_path(data, rows[0].input_audio), config)
        decode_audio(processor, warm)
    results, predictions = [], []
    for row in rows:
        # Per-example seeding ensures that the evaluation subset/order is irrelevant.
        torch.manual_seed(config.seed + int(row.id[:8], 16))
        torch.cuda.reset_peak_memory_stats()
        rollout = generate(
            model, processor, safe_audio_path(data, row.input_audio), config, measure=True
        )
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        waveform = decode_audio(processor, rollout)
        torch.cuda.synchronize()
        timing = dict(rollout.timing, decode_seconds=time.perf_counter() - decode_start)
        timing["audio_ready_seconds"] = timing["generation_seconds"] + timing["decode_seconds"]
        timing["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        timing["environment"] = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "platform": platform.platform(),
            "precision": "bf16",
            "warmup_runs": str(warmup),
            "scope": "input_preparation_to_full_decoded_audio; synchronized; excludes_ASR_and_file_write",
        }
        asr = ""
        audio_path = output / f"{row.id}.wav"
        if waveform is not None:
            sf.write(audio_path, waveform, 24000)
            asr = scorer.transcribe(str(audio_path))
        default_mode = (
            "open"
            if row.provenance.get("reward_protocol") == "open_ended" or row.task == "dialogue"
            else "closed"
        )
        is_open = row.provenance.get("evaluation", {}).get("answer_mode", default_mode) == "open"
        reward = (
            None
            if is_open
            else score_reply(
                row.answer, rollout.text, asr, waveform, truncated=not rollout.terminated
            )
        )
        predictions.append(
            {
                "example_id": row.id,
                "generation": generation_source,
                "text": rollout.text,
                "asr": asr,
                "asr_model": config.asr_model,
                "audio": audio_path.name if waveform is not None else None,
                "audio_sha256": file_hash(audio_path) if waveform is not None else None,
                "terminated": rollout.terminated,
                "timing": timing,
                "error": "no_audio_output" if waveform is None else None,
            }
        )
        results.append(
            {
                "example_id": row.id,
                "text": rollout.text,
                "asr": asr,
                "answer": row.answer,
                "reward": reward.to_dict() if reward is not None else None,
                "audio": audio_path.name if waveform is not None else None,
            }
        )
    rewards = [r["reward"]["total"] for r in results if r["reward"] is not None]
    report = {
        "dataset_hash": metadata["manifest_sha256"],
        "split": split,
        "seed": config.seed,
        "asr_model": config.asr_model,
        "sampling": {
            "temperature": config.temperature,
            "top_k": None,
            "max_new_tokens": config.max_new_tokens,
        },
        "reward_version": "spoken-exact-v1",
        "checkpoint": str(checkpoint) if checkpoint else "base",
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "reward_coverage": len(rewards) / len(results),
        "warmup_runs": warmup,
        "results": results,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    if metrics_config is not None:
        from .metrics.runner import score_predictions

        score_predictions(metrics_config, data, prediction_path, output / "metrics", split, limit)
    return {
        "mean_reward": report["mean_reward"],
        "n": len(results),
        "output": str(output),
        "predictions": str(prediction_path),
    }
