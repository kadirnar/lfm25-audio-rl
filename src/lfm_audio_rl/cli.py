import argparse
import json
from itertools import product
from pathlib import Path

import yaml

from .config import DatasetRecipe, Experiment, read_config
from .data import build_dataset, digest, file_hash, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Synthetic speech and LFM2.5-Audio RL research")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser(
        "data", help="Generate a versioned symbolic/synthetic speech dataset"
    )
    data.add_argument("--config", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--require-audio", action="store_true")
    smoke = commands.add_parser(
        "smoke", help="CPU objective checks; does not train the audio model"
    )
    smoke.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Experimental single-CUDA-GPU LoRA runner")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate base or adapted model on held-out speech"
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--metrics-config", type=Path)
    evaluate.add_argument("--warmup", type=int, default=1)
    compare = commands.add_parser(
        "compare", help="Paired bootstrap comparison of matching evaluations"
    )
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    matrix = commands.add_parser("matrix", help="Write 81 experiment configs; does not launch jobs")
    matrix.add_argument("--output", type=Path, required=True)
    synth_text = commands.add_parser(
        "synth-text", help="Generate text through OpenRouter; resumes cached batches"
    )
    synth_text.add_argument("--config", type=Path, required=True)
    synth_text.add_argument("--output", type=Path, required=True)
    synth_audio = commands.add_parser(
        "synth-audio", help="Render a text dataset with isolated neural TTS workers"
    )
    synth_audio.add_argument("--config", type=Path, required=True)
    synth_audio.add_argument("--text", type=Path, required=True)
    synth_audio.add_argument("--output", type=Path, required=True)
    synth_plan = commands.add_parser(
        "synth-plan", help="Show dataset size and request limits without API calls"
    )
    synth_plan.add_argument("--text-config", type=Path, required=True)
    synth_plan.add_argument("--audio-config", type=Path, required=True)
    score = commands.add_parser("score", help="Score saved speech-to-speech predictions")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    score.add_argument("--limit", type=int)
    compare_metrics = commands.add_parser(
        "compare-metrics", help="Paired metric differences with grouped confidence intervals"
    )
    compare_metrics.add_argument("before", type=Path)
    compare_metrics.add_argument("after", type=Path)
    compare_metrics.add_argument("--allow-partial", action="store_true")
    human = commands.add_parser("score-human", help="Summarize real listening ratings")
    human.add_argument("--ratings", type=Path, required=True)
    human.add_argument("--kind", choices=["mos", "preference"], default="mos")
    human.add_argument("--metrics-report", type=Path)
    human.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "score":
        from .metrics.config import MetricConfig
        from .metrics.runner import score_predictions

        report = score_predictions(
            read_config(args.config, MetricConfig),
            args.data,
            args.predictions,
            args.output,
            args.split,
            args.limit,
        )
        result = {
            "n_expected": report["n_expected"],
            "n_predictions": report["n_predictions"],
            "report": str(args.output / "metrics.json"),
        }
    elif args.command == "compare-metrics":
        from .metrics.statistics import compare_reports

        result = compare_reports(
            json.loads(args.before.read_text()),
            json.loads(args.after.read_text()),
            allow_partial=args.allow_partial,
        )
    elif args.command == "score-human":
        from .metrics.human import summarize_ratings

        groups = None
        if args.metrics_report:
            report = json.loads(args.metrics_report.read_text())
            groups = {r["example_id"]: r["semantic_group"] for r in report["rows"]}
        result = summarize_ratings(
            [json.loads(line) for line in args.ratings.read_text().splitlines() if line.strip()],
            args.kind,
            groups,
        )
        result["ratings_sha256"] = file_hash(args.ratings)
        if args.metrics_report:
            result["metrics_report_sha256"] = file_hash(args.metrics_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "synth-text":
        from .synthetic.config import TextConfig
        from .synthetic.text import generate_text

        result = generate_text(read_config(args.config, TextConfig), args.output)
    elif args.command == "synth-audio":
        from .synthetic.audio import render_speech
        from .synthetic.config import SpeechConfig

        result = render_speech(read_config(args.config, SpeechConfig), args.text, args.output)
    elif args.command == "synth-plan":
        from .synthetic.config import SpeechConfig, TextConfig

        text_config = read_config(args.text_config, TextConfig)
        audio_config = read_config(args.audio_config, SpeechConfig)
        result = {
            "text_examples": text_config.count,
            "speech_examples_before_qa": text_config.count * len(audio_config.engines),
            "audio_clips_before_qa": 2 * text_config.count * len(audio_config.engines),
            "openrouter_model": text_config.model,
            "request_limit_including_retries": text_config.max_requests,
            "maximum_completion_tokens_per_request": text_config.max_tokens,
            "profiles": [e.name for e in audio_config.engines],
            "asr_checks": audio_config.qa.asr_model,
            "network_calls": 0,
        }
    elif args.command == "data":
        result = build_dataset(read_config(args.config, DatasetRecipe), args.output)
    elif args.command == "validate-data":
        rows, metadata = load_dataset(args.dataset, require_audio=args.require_audio)
        result = {"validated": len(rows), "manifest_sha256": metadata["manifest_sha256"]}
    elif args.command == "smoke":
        from .smoke import run_smoke

        result = run_smoke(args.output)
    elif args.command == "train":
        from .train import train as train_model

        result = train_model(
            read_config(args.config, Experiment), args.data, args.output, args.resume
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate as evaluate_model
        from .metrics.config import MetricConfig

        result = evaluate_model(
            read_config(args.config, Experiment),
            args.data,
            args.output,
            args.checkpoint,
            args.split,
            args.limit,
            read_config(args.metrics_config, MetricConfig) if args.metrics_config else None,
            args.warmup,
        )
    elif args.command == "compare":
        from .evaluate import compare_runs

        result = compare_runs(args.before, args.after)
    elif args.command == "matrix":
        args.output.mkdir(parents=True, exist_ok=False)
        jobs = []
        variants = [
            (method, scope)
            for method in ["grpo", "dr_grpo", "rloo", "reinforce"]
            for scope in ["text", "joint"]
        ] + [("sft", "joint")]
        for version, (method, scope), seed in product(
            ["v1_clean", "v2_noisy", "v3_compositional"], variants, [42, 43, 44]
        ):
            config = Experiment(algorithm=method, scope=scope, seed=seed)
            name = f"{version}-{method}-{scope}-{seed}"
            (args.output / f"{name}.yaml").write_text(
                yaml.safe_dump(config.model_dump(), sort_keys=True)
            )
            jobs.append(
                {
                    "name": name,
                    "config": f"{name}.yaml",
                    "dataset": f"data/{version}",
                    "config_hash": digest(config.model_dump()),
                    "dataset_hash": None,
                    "status": "planned_not_launched",
                }
            )
        (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        result = {"planned_jobs": len(jobs), "output": str(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
