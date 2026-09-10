import importlib.metadata
import json
import math
from pathlib import Path

from ..data import digest, file_hash, load_dataset, safe_audio_path
from ..synthetic.storage import atomic_json, run_directory
from . import METRIC_VERSION
from .audio import audio_scores, loudness, pitch_comparison, prosody, read_audio
from .backends import Scorers, scorer_assets
from .config import EvaluationPolicy, Prediction
from .statistics import frechet_distance, slices, summarize
from .text import constraint_scores, corpus_text, diversity, normalize, text_scores
from .timing import timing_scores


def implementation_hash():
    root = Path(__file__).parent.parent
    files = list((root / "metrics").glob("*.py")) + [
        root / "data.py",
        root / "config.py",
        root / "rewards.py",
        root / "synthetic" / "config.py",
        root / "synthetic" / "text.py",
        root / "synthetic" / "storage.py",
    ]
    return digest({str(p.relative_to(root)): file_hash(p) for p in files})


def package_versions():
    result = {}
    for name in [
        "torch",
        "pydantic",
        "soundfile",
        "httpx",
        "torchaudio",
        "numpy",
        "scipy",
        "sacrebleu",
        "rouge-score",
        "pyloudnorm",
        "pystoi",
        "pesq",
        "pysptk",
        "librosa",
        "torchmetrics",
        "onnxruntime",
        "speechbrain",
        "bert-score",
        "transformers",
        "tokenizers",
        "timm",
        "torchvision",
        "huggingface-hub",
        "ctranslate2",
        "utmosv2",
        "faster-whisper",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
            direct = importlib.metadata.distribution(name).read_text("direct_url.json")
            if direct:
                commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
                if commit:
                    result[name + "_commit"] = commit
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def policy_for(row):
    value = {
        "answer_mode": "open"
        if row.provenance.get("reward_protocol") == "open_ended" or row.task == "dialogue"
        else "closed"
    }
    value.update(row.provenance.get("evaluation", {}))
    return EvaluationPolicy.model_validate(value)


def load_predictions(path):
    records = [
        Prediction.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len({p.example_id for p in records}) != len(records):
        raise ValueError("Duplicate prediction IDs")
    return {p.example_id: p for p in records}


def score_case(
    row,
    prediction,
    data_root,
    prediction_root,
    config,
    scorers,
    asr,
    judge_cache,
    index,
    judge_client=None,
):
    policy = policy_for(row)
    metrics, missing, counts, details = {}, {}, {}, {}
    waveform, sr = None, None
    valid = False
    if prediction.audio:
        path = safe_audio_path(prediction_root, prediction.audio)
        if not path.is_file():
            missing["audio"] = "Prediction audio file is missing"
        else:
            actual_hash = file_hash(path)
            if prediction.audio_sha256 and actual_hash != prediction.audio_sha256:
                raise ValueError(f"Prediction audio hash mismatch: {row.id}")
            details["audio_sha256"] = actual_hash
            try:
                wave, sr = read_audio(path)
                raw_scores = audio_scores(wave, sr, config)
                valid = bool(raw_scores["valid_audio"])
                waveform = wave.mean(axis=1)
                if "audio" in config.groups:
                    metrics.update(raw_scores)
            except (ValueError, RuntimeError) as exc:
                valid, waveform = False, None
                missing["audio"] = str(exc)
    else:
        missing["audio"] = "No prediction audio"
    if prediction.error:
        valid = False
        missing["generation"] = prediction.error
    metrics["valid_audio"] = float(valid)
    metrics["generation_failure"] = float(prediction.error is not None or not valid)
    metrics["truncated"] = (
        float(not prediction.terminated) if prediction.terminated is not None else None
    )
    details["termination_known"] = prediction.terminated is not None
    transcript = prediction.asr
    if valid and asr:
        transcript = asr.transcribe(str(path))
        details["asr_source"] = config.asr_model
    else:
        details["asr_source"] = prediction.asr_model or "provided_transcript_unverified_source"
    # Missing/invalid audio must not obtain spoken-answer credit from supplied text.
    spoken = transcript if valid else ""
    details.update(
        text=prediction.text,
        asr=transcript,
        reference=row.answer,
        answer_mode=policy.answer_mode,
        generation=prediction.generation,
    )

    def update(values):
        for key, value in values.items():
            if value is None:
                metrics[key] = None
                missing.setdefault(key, "Undefined for this input")
            elif not math.isfinite(float(value)):
                metrics[key] = None
                missing[key] = "Scorer returned a non-finite value"
                if config.fail_on_metric_error:
                    raise ValueError(f"Non-finite metric: {key}")
            else:
                metrics[key] = float(value)

    def optional(name, operation):
        try:
            update(operation())
        except Exception as exc:
            if config.fail_on_metric_error:
                raise RuntimeError(f"{name} failed for {row.id}: {exc}") from exc
            missing[name] = f"{type(exc).__name__}: {exc}"

    def compare_text(prefix, reference, hypothesis, alternatives=()):
        values, edits = text_scores(reference, hypothesis, config.normalization, alternatives)
        update({f"{prefix}_{k}": v for k, v in values.items()})
        counts.update({f"{prefix}_{k}": v for k, v in edits.items()})

    if waveform is not None and "audio" in config.groups:
        optional(
            "integrated_lufs",
            lambda: {"integrated_lufs": loudness(wave if wave.shape[1] == 2 else waveform, sr)},
        )

    if "text" in config.groups:
        if policy.answer_mode == "closed":
            compare_text("text_answer", row.answer, prediction.text or "", policy.accepted_answers)
            if spoken is not None:
                compare_text("spoken_answer", row.answer, spoken, policy.accepted_answers)
                metrics["task_success"] = metrics["spoken_answer_exact_match"] * float(
                    valid and prediction.terminated is not False
                )
            else:
                missing["spoken_answer"] = "Output audio has no ASR transcript"
                metrics["task_success"] = None
        else:
            missing["answer_exact_match"] = "Open-ended answer; use a semantic or human judge"
        if prediction.text is not None and spoken is not None:
            compare_text("tts", prediction.text, spoken)
        else:
            missing["tts"] = "Need generated text and its output-audio transcript"
        if prediction.input_asr is not None:
            compare_text("input_asr", row.prompt, prediction.input_asr)
    if "instructions" in config.groups:
        if policy.constraints and spoken is not None:
            update(constraint_scores(spoken, policy.constraints, config.normalization))
            # An absent spoken response cannot pass a purely negative constraint.
            if not valid:
                metrics.update(instruction_pass_rate=0.0, instruction_all_pass=0.0)
        else:
            missing["instructions"] = "No constraints or no spoken transcript"
    if "timing" in config.groups:
        update(
            timing_scores(
                prediction.timing,
                len(waveform) / sr if waveform is not None else None,
                len(normalize(transcript, config.normalization).split())
                if transcript is not None
                else None,
            )
        )
        if prediction.timing is None:
            missing["timing"] = "No measured timing supplied"
    reference, reference_sr = None, None
    if row.answer_audio and any(g in config.groups for g in ["reference", "prosody", "fad"]):
        ref, reference_sr = read_audio(safe_audio_path(data_root, row.answer_audio))
        reference = ref.mean(axis=1)
    if valid:
        if "prosody" in config.groups:

            def measure_prosody():
                result, pitch = prosody(waveform, sr, config)
                if policy.reference_relation == "aligned" and reference is not None:
                    _, ref_pitch = prosody(reference, reference_sr, config)
                    if ref_pitch.shape == pitch.shape:
                        result.update(pitch_comparison(ref_pitch, pitch))
                    else:
                        missing["pitch_comparison"] = "Reference pitch frames are not aligned"
                return result

            optional("prosody", measure_prosody)
        if "reference" in config.groups:
            if reference is None:
                missing["reference"] = "No reference answer waveform"
            else:

                def measure_reference():
                    values, skipped = scorers.intrusive(
                        reference, reference_sr, waveform, sr, policy.reference_relation
                    )
                    missing.update(skipped)
                    return values

                optional("reference", measure_reference)
        if "speaker" in config.groups:
            if policy.speaker_reference_audio:
                speaker_path = safe_audio_path(data_root, policy.speaker_reference_audio)
                if file_hash(speaker_path) != policy.speaker_reference_sha256:
                    raise ValueError("Speaker reference hash mismatch")
                spk, spk_sr = read_audio(speaker_path)
                optional(
                    "speaker",
                    lambda: scorers.speaker_similarity(spk.mean(axis=1), spk_sr, waveform, sr),
                )
            else:
                missing["speaker_cosine"] = "No explicitly requested target-speaker reference"
        for name in ["dnsmos", "nisqa", "utmos"]:
            if name in config.groups:
                optional(name, lambda name=name: getattr(scorers, name)(waveform, sr))
        if "bertscore" in config.groups:
            if transcript is not None:
                optional("bertscore", lambda: scorers.bertscore(row.answer, transcript))
            else:
                missing["bertscore"] = "No output-audio transcript"
        if "fad" in config.groups:
            if reference is None:
                missing["fad"] = "No reference audio distribution"
            else:
                try:
                    a = scorers.clap_embedding(reference, reference_sr).tolist()
                    b = scorers.clap_embedding(waveform, sr).tolist()
                    if not a or len(a) != len(b) or not all(math.isfinite(v) for v in a + b):
                        raise ValueError("Invalid or mismatched CLAP embeddings")
                    details["fad_reference_embedding"] = a
                    details["fad_generated_embedding"] = b
                except Exception as exc:
                    if config.fail_on_metric_error:
                        raise
                    missing["fad"] = str(exc)
        if "judge" in config.groups:
            if transcript is None:
                missing["judge"] = "No output-audio transcript"
            else:
                from .judge import judge_answer

                values, provenance = judge_answer(
                    config.judge,
                    judge_cache,
                    index,
                    row.prompt,
                    row.answer,
                    transcript,
                    policy.constraints,
                    judge_client,
                )
                update(values)
                details["judge"] = provenance
                if policy.expected_refusal is not None:
                    metrics["judge_refusal_accuracy"] = float(
                        bool(values["judge_refusal"]) == policy.expected_refusal
                    )
                    if policy.expected_refusal:
                        metrics["judge_unsafe_assistance_rate"] = values["judge_unsafe"]
                    else:
                        metrics["judge_overrefusal_rate"] = values["judge_refusal"]
    else:
        for name in config.groups:
            if name in {
                "prosody",
                "reference",
                "speaker",
                "bertscore",
                "dnsmos",
                "nisqa",
                "utmos",
                "fad",
                "judge",
            }:
                missing[name] = "Invalid or missing generated audio"
    # Emit explicit nulls for unavailable individual metrics; groups retain a
    # group-level reason when the model could not be run at all.
    for key in missing:
        metrics.setdefault(key, None)
    update(dict(metrics))
    return {
        "example_id": row.id,
        "semantic_group": row.semantic_group,
        "task": row.task,
        "split": row.split,
        "language": row.language,
        "tts_profile": row.provenance.get("tts", {}).get("name", "unknown")
        if isinstance(row.provenance.get("tts"), dict)
        else str(row.provenance.get("tts", "unknown")),
        "metrics": metrics,
        "counts": counts,
        "missing": missing,
        "details": details,
    }


def score_predictions(
    config,
    data_root,
    predictions_path,
    output,
    split="test",
    limit=None,
    scorer_factory=Scorers,
    asr=None,
    judge_client=None,
):
    rows, dataset = load_dataset(data_root)
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError("Invalid evaluation split")
    predictions = load_predictions(predictions_path)
    unknown = set(predictions) - {r.id for r in rows}
    if unknown:
        raise ValueError("Predictions contain IDs outside the dataset")
    rows = [r for r in rows if split == "all" or r.split == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty evaluation selection")
    for row in rows:
        policy = policy_for(row)
        if policy.speaker_reference_audio:
            if (
                file_hash(safe_audio_path(data_root, policy.speaker_reference_audio))
                != policy.speaker_reference_sha256
            ):
                raise ValueError("Speaker reference hash mismatch")
    assets = scorer_assets(config)
    versions = package_versions()
    protocol = {
        "config": config.model_dump(),
        "assets": assets,
        "packages": versions,
        "implementation_sha256": implementation_hash(),
        "asr_sources": [config.asr_model]
        if config.asr_model
        else sorted({p.asr_model or "provided_unknown" for p in predictions.values()}),
        "timing_environments": sorted(
            {
                json.dumps(p.timing.environment, sort_keys=True)
                for p in predictions.values()
                if p.timing
            }
        )
        if "timing" in config.groups
        else [],
    }
    # Hash every prediction waveform, including legacy files without declared hashes,
    # so resume cannot silently reuse a score after the waveform has changed.
    audio_hashes = {}
    for row in rows:
        pred = predictions.get(row.id)
        if pred and pred.audio:
            path = safe_audio_path(predictions_path.parent, pred.audio)
            audio_hashes[row.id] = file_hash(path) if path.is_file() else None
    identity = {
        "stage": METRIC_VERSION,
        "protocol": protocol,
        "dataset_hash": dataset["manifest_sha256"],
        "predictions_hash": file_hash(predictions_path),
        "audio_hashes": audio_hashes,
        "split": split,
        "ids": [r.id for r in rows],
    }
    with run_directory(output, identity):
        complete = output / "complete.json"
        if complete.exists():
            if file_hash(output / "metrics.json") != json.loads(complete.read_text())["sha256"]:
                raise ValueError("Completed metrics report hash mismatch")
            return json.loads((output / "metrics.json").read_text())
        cache = output / "cases"
        cache.mkdir(exist_ok=True)
        judge_cache = output / "judge"
        judge_cache.mkdir(exist_ok=True)
        scorers = scorer_factory(config, output / "models")
        if config.asr_model and asr is None:
            from ..rewards import WhisperScorer

            asr = WhisperScorer(config.asr_model, config.asr_device)
        results = []
        for index, row in enumerate(rows):
            receipt = cache / f"{digest(row.id)}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["sha256"] != digest(saved["row"]):
                    raise ValueError("Cached metric row hash mismatch")
                result = saved["row"]
            else:
                pred = predictions.get(
                    row.id,
                    Prediction(
                        example_id=row.id,
                        text="",
                        asr="",
                        terminated=False,
                        error="missing_prediction",
                    ),
                )
                result = score_case(
                    row,
                    pred,
                    data_root,
                    predictions_path.parent,
                    config,
                    scorers,
                    asr,
                    judge_cache,
                    index,
                    judge_client,
                )
                atomic_json(receipt, {"sha256": digest(result), "row": result})
            results.append(result)
        report = {
            "metric_version": METRIC_VERSION,
            "dataset_hash": dataset["manifest_sha256"],
            "predictions_hash": identity["predictions_hash"],
            "protocol_hash": digest(protocol),
            "protocol": protocol,
            "split": split,
            "n_expected": len(rows),
            "n_predictions": sum(r.id in predictions for r in rows),
            "bootstrap_unit": "semantic_group",
            "summary": summarize(results, config.bootstrap_samples, config.seed),
            "slices": slices(results, config.bootstrap_samples, config.seed),
            "corpus": {},
            "rows": results,
        }
        if "text" in config.groups:
            for channel, field in [("text", "text"), ("spoken", "asr")]:
                eligible = [
                    r
                    for r in results
                    if r["details"].get(field) is not None or r["metrics"]["generation_failure"]
                ]
                if eligible:
                    hypotheses = [r["details"].get(field) or "" for r in eligible]
                    if channel == "spoken":
                        hypotheses = [
                            h if r["metrics"]["valid_audio"] else ""
                            for h, r in zip(hypotheses, eligible, strict=True)
                        ]
                    report["corpus"][channel] = {
                        "reference_overlap": corpus_text(
                            [r["details"]["reference"] for r in eligible], hypotheses
                        ),
                        "diversity": diversity(hypotheses, config.normalization),
                        "n": len(eligible),
                        "coverage": len(eligible) / len(rows),
                    }
        if "fad" in config.groups:
            eligible = [
                r
                for r in results
                if "fad_generated_embedding" in r["details"]
                and "fad_reference_embedding" in r["details"]
            ]
            report["corpus"]["fad_clap"] = {
                "value": frechet_distance(
                    [r["details"]["fad_reference_embedding"] for r in eligible],
                    [r["details"]["fad_generated_embedding"] for r in eligible],
                )
                if len(eligible) >= 2
                else None,
                "n": len(eligible),
                "coverage": len(eligible) / len(rows),
                "embedding": "CLAP, mean of fixed ten-second chunks",
                "warning": "Finite-sample biased distribution diagnostic; not MOS or a test of answer correctness",
            }
        atomic_json(output / "metrics.json", report)
        atomic_json(complete, {"sha256": file_hash(output / "metrics.json")})
        return report
