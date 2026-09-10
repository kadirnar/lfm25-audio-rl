import argparse
import json
from itertools import product
from pathlib import Path

import yaml

from .config import DatasetRecipe, Experiment, read_config
from .data import build_dataset, digest, load_dataset


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
    compare = commands.add_parser(
        "compare", help="Paired bootstrap comparison of matching evaluations"
    )
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    matrix = commands.add_parser("matrix", help="Write 81 experiment configs; does not launch jobs")
    matrix.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "data":
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

        result = evaluate_model(
            read_config(args.config, Experiment),
            args.data,
            args.output,
            args.checkpoint,
            args.split,
            args.limit,
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
