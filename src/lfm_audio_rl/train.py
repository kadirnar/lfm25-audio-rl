"""Single-GPU LoRA research runner. No model download occurs before preflight."""

import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
import torch

from .config import UPSTREAM_COMMIT, Experiment
from .data import digest, load_dataset, safe_audio_path
from .lora import adapter_state, inject_lora, load_adapter_state, reference_policy
from .objectives import advantages, policy_loss, reinforce_loss
from .optimization import enable_gradient_checkpointing, learning_rate, make_optimizer
from .rewards import WhisperScorer, score_reply
from .tracking import Tracker, check_tracking


def preflight(config: Experiment, data: Path):
    rows, metadata = load_dataset(data, require_audio=True)
    train_rows = [row for row in rows if row.split == "train"]
    if config.algorithm != "sft" and any(
        r.provenance.get("reward_protocol") == "open_ended" for r in train_rows
    ):
        raise ValueError(
            "Dialogue data needs an open-ended reward; use SFT or grounded_qa with this runner"
        )
    if not train_rows:
        raise ValueError("No training examples; generate a larger dataset")
    if (config.algorithm == "sft" or config.anchor_weight) and any(
        not r.answer_audio for r in train_rows
    ):
        raise ValueError("SFT/anchoring requires reference answer speech for every training row")
    check_tracking(config.wandb)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The real LFM runner requires CUDA (upstream detokenizer uses .cuda()). Run `lfm-rl smoke` on CPU."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The initial GPU runner requires BF16 support")
    if importlib.metadata.version("liquid-audio") != "1.3.0":
        raise RuntimeError("This adapter is audited against liquid-audio==1.3.0")
    return train_rows, metadata


def save_checkpoint(path: Path, model, optimizer, step: int, identity: str):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 1,
            "identity": identity,
            "step": step,
            "adapters": adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, model, optimizer, identity: str) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity or state["schema_version"] != 1:
        raise ValueError("Checkpoint configuration or dataset identity mismatch")
    load_adapter_state(model, state["adapters"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state["step"]


def train_example(config, data, output, row, model, processor, scorer, step, micro_step=0):
    """Accumulate one prompt's loss. The caller owns zero_grad and optimizer.step."""
    from .lfm import (
        check_parity,
        decode_audio,
        generate,
        sampled_logps,
        score_rollout,
        supervised_batch,
    )

    scale = 1 / config.train_opt.gradient_accumulation_steps
    log = {}
    # Preserve the original ordering of reference preparation and policy sampling.
    anchor = (
        supervised_batch(model, processor, row, data)
        if config.algorithm == "sft" or config.anchor_weight
        else None
    )
    if config.algorithm != "sft":
        rollouts, refs, rewards, old_logps, errors = [], [], [], [], []
        for index in range(config.group_size):
            rollout = generate(model, processor, safe_audio_path(data, row.input_audio), config)
            with torch.no_grad():
                replay = score_rollout(
                    model, rollout, temperature=config.temperature, scope=config.scope
                )
                old = sampled_logps(rollout, config.scope)
                errors.append(check_parity(replay, old, config.parity_atol))
            with reference_policy(model):
                ref = score_rollout(
                    model, rollout, temperature=config.temperature, scope=config.scope
                )
            waveform = decode_audio(processor, rollout)
            asr = ""
            if waveform is not None:
                if config.train_opt.save_rollout_audio:
                    audio_path = (
                        output / f"step-{step:05d}-prompt-{micro_step}-candidate-{index}.wav"
                    )
                    sf.write(audio_path, waveform, 24000)
                    asr = scorer.transcribe(str(audio_path))
                else:
                    # ASR still needs a WAV file; remove it immediately after scoring.
                    with TemporaryDirectory(prefix="asr-", dir=output) as temporary:
                        audio_path = Path(temporary) / "reply.wav"
                        sf.write(audio_path, waveform, 24000)
                        asr = scorer.transcribe(str(audio_path))
            reward = score_reply(
                row.answer, rollout.text, asr, waveform, truncated=not rollout.terminated
            )
            with (output / "rollouts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "micro_step": micro_step,
                            "candidate": index,
                            "example_id": row.id,
                            "text": rollout.text,
                            "asr": asr,
                            "reward": reward.to_dict(),
                            "actions": old.numel(),
                            "parity_max_error": errors[-1],
                        }
                    )
                    + "\n"
                )
            rollouts.append(rollout)
            refs.append(ref.detach())
            old_logps.append(old.detach())
            rewards.append(reward.total)
        reward_tensor = torch.tensor([rewards], device=next(model.parameters()).device)
        adv = advantages(reward_tensor, config.algorithm)[0]
        total_loss = 0.0
        for index, rollout in enumerate(rollouts):
            new = score_rollout(model, rollout, temperature=config.temperature, scope=config.scope)[
                None
            ]
            mask = torch.ones_like(new, dtype=torch.bool)
            if config.algorithm in {"rloo", "reinforce"}:
                loss = reinforce_loss(new, mask, adv[index : index + 1])
                if config.kl_beta:
                    loss = loss + policy_loss(
                        new,
                        old_logps[index][None],
                        refs[index][None],
                        mask,
                        torch.zeros_like(adv[index : index + 1]),
                        beta=config.kl_beta,
                        reduction="sum",
                    )
            else:
                loss = policy_loss(
                    new,
                    old_logps[index][None],
                    refs[index][None],
                    mask,
                    adv[index : index + 1],
                    epsilon=config.clip_epsilon,
                    beta=config.kl_beta,
                    reduction="fixed" if config.algorithm == "dr_grpo" else "sequence",
                    max_actions=config.max_new_tokens
                    * (model.codebooks if config.scope == "joint" else 1),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy loss; update aborted")
            (loss * scale / config.group_size).backward()
            total_loss += float(loss.detach()) / config.group_size
        log.update(
            policy_loss=total_loss,
            mean_reward=sum(rewards) / len(rewards),
            reward_std=float(reward_tensor.std(correction=0)),
            zero_variance_group=float(len(set(rewards)) == 1),
            truncation_rate=sum(not r.terminated for r in rollouts) / len(rollouts),
            parity_max_error=max(errors),
            actions=sum(p.numel() for p in old_logps),
        )
    if anchor is not None:
        anchor_loss = model(anchor).loss
        if not torch.isfinite(anchor_loss):
            raise FloatingPointError("Non-finite SFT loss; update aborted")
        weight = 1.0 if config.algorithm == "sft" else config.anchor_weight
        (weight * scale * anchor_loss).backward()
        log["anchor_loss"] = float(anchor_loss.detach())
    return log


def train(config: Experiment, data: Path, output: Path, resume: bool = False):
    rows, dataset = preflight(config, data)
    from liquid_audio import LFM2AudioProcessor

    from .lfm import recording_model_class

    identity = digest({"config": config.training_dict(), "dataset": dataset["manifest_sha256"]})
    if resume:
        if not (output / "checkpoint.pt").exists():
            raise ValueError("Resume requires an existing checkpoint.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    processor = LFM2AudioProcessor.from_pretrained(
        config.model_id, revision=config.model_revision, device="cuda"
    ).eval()
    model = (
        recording_model_class()
        .from_pretrained(config.model_id, revision=config.model_revision, device="cuda")
        .eval()
    )
    model.requires_grad_(False)
    matches = inject_lora(model.lfm, config.lora_targets, config.lora_rank, config.lora_alpha)
    if config.train_opt.gradient_checkpointing:
        enable_gradient_checkpointing(model.lfm)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, config)
    start = load_checkpoint(output / "checkpoint.pt", model, optimizer, identity) if resume else 0
    scorer = (
        None if config.algorithm == "sft" else WhisperScorer(config.asr_model, config.asr_device)
    )
    if not resume:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        provenance = {
            "identity": identity,
            "config": config.model_dump(),
            "dataset": dataset,
            "code_commit": git.stdout.strip() if git.returncode == 0 else None,
            "upstream_audited_commit": UPSTREAM_COMMIT,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(),
            "packages": sorted(
                f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
            ),
            "lora_modules": matches,
            "trainable_parameters": sum(p.numel() for p in params),
            "status": "experimental_gpu_run",
            "reward_version": "spoken-exact-v1",
        }
        (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # An initialization or first-update failure can resume without losing
        # the exact initialized adapters and random state.
        save_checkpoint(output / "checkpoint.pt", model, optimizer, 0, identity)
    tracking_metadata = {
        "experiment": config.training_dict(),
        "dataset_sha256": dataset["manifest_sha256"],
        "train_examples": len(rows),
        "trainable_parameters": sum(p.numel() for p in params),
        "upstream_commit": UPSTREAM_COMMIT,
    }
    # Eval mode disables dropout without disabling gradients. Each optimizer
    # update averages sequential prompt groups sampled from the same policy.
    accumulation = config.train_opt.gradient_accumulation_steps
    checkpoint_step = start
    with Tracker(config.wandb, output, identity, tracking_metadata, start) as tracker:
        for step in range(start, config.steps):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(config, step + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            parts, example_ids = [], []
            for micro_step in range(accumulation):
                row = rows[(step * accumulation + micro_step) % len(rows)]
                example_ids.append(row.id)
                parts.append(
                    train_example(
                        config, data, output, row, model, processor, scorer, step + 1, micro_step
                    )
                )
            norm = torch.nn.utils.clip_grad_norm_(
                params, config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.monotonic() - begin
            log = {key: sum(p[key] for p in parts) / len(parts) for key in parts[0]}
            if "parity_max_error" in log:
                log["parity_max_error"] = max(p["parity_max_error"] for p in parts)
            if "actions" in log:
                log["actions"] = sum(p["actions"] for p in parts)
                log["actions_per_second"] = log["actions"] / seconds
            log.update(
                step=step + 1,
                example_id=example_ids[0],
                example_ids=example_ids,
                lr=lr,
                gradient_norm=float(norm),
                seconds=seconds,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                prompts_per_update=accumulation,
                prompts_per_second=accumulation / seconds,
            )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(log) + "\n")
            final = step + 1 == config.steps
            if (step + 1) % config.train_opt.checkpoint_every == 0 or final:
                save_checkpoint(output / "checkpoint.pt", model, optimizer, step + 1, identity)
                checkpoint_step = step + 1
            tracker.log(log, checkpoint_step, final=final)
            print(json.dumps(log), flush=True)
    return {"output": str(output), "steps": config.steps, "identity": identity}
