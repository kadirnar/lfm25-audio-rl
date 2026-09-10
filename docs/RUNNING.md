# Running experiments

Run these commands from the repository folder. See the [README](../README.md) for installation and the [dataset guide](DATASETS.md) for the files you need.

## Test the code on CPU

```bash
uv sync --extra dev
uv run pytest -q
uv run lfm-rl smoke --output runs/cpu-smoke
```

The smoke command checks eight training objectives using a tiny test model. Its scores do not measure speech quality.

To also test small versions of the upstream audio model's layers:

```bash
uv sync --extra dev --extra model
uv run pytest -q
```

These tests do not download pretrained model weights.

## Create the three dataset versions

Install `espeak-ng` first on Linux, then run:

```bash
uv run lfm-rl data --config configs/datasets/v1_clean.yaml --output data/v1_clean
uv run lfm-rl data --config configs/datasets/v2_noisy.yaml --output data/v2_noisy
uv run lfm-rl data --config configs/datasets/v3_compositional.yaml --output data/v3_compositional
```

For a small preview on macOS:

```bash
uv run lfm-rl data --config configs/datasets/macos_preview.yaml --output data/macos-preview
```

This uses the installed Samantha voice. You can change the voice in the YAML file. The preview has only eight examples and may have no validation or test examples.

## Evaluate the model before training

Real-model commands need a BF16-capable NVIDIA GPU, a compatible CUDA setup, and the model and ASR packages:

```bash
uv sync --extra model --extra asr --extra dev
uv run lfm-rl evaluate --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/base-validation --split validation
```

ASR means speech recognition. It checks the words in the model's spoken reply. These commands download the configured model weights. For repeatable comparisons, use the same fixed ASR model files for every run.

## Train

```bash
uv run lfm-rl train --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-v1
```

The default run takes ten steps. It updates small LoRA adapters while keeping the original model weights fixed. It also learns from reference answers through `anchor_weight: 0.1`.

For an SFT run, use `configs/experiments/sft.yaml` and a new output folder. SFT means learning directly from the provided question-and-answer examples.

GPU training with the pretrained checkpoint still needs validation. Memory use, speed, and speech improvements have not been measured for this repository.

## Evaluate after training

```bash
uv run lfm-rl evaluate --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-validation --split validation \
  --checkpoint runs/grpo-v1/checkpoint.pt
uv run lfm-rl compare runs/base-validation/evaluation.json runs/grpo-validation/evaluation.json
```

Use the same training config and dataset when loading its checkpoint. Compare results from the same questions and evaluation settings. Listen to saved audio as well as reading scores.

Use validation data to choose settings. Use the test split for the final comparison.

## Resume an interrupted run

```bash
uv run lfm-rl train --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-v1 --resume
```

Keep the config, dataset, and output folder the same. Resume restores the saved training state. Starting a new RL run from a separate SFT run is not supported yet.

## Prepare a larger comparison

```bash
uv run lfm-rl matrix --output runs/experiment-plan
```

This writes 81 configurations covering three datasets, three seeds, and nine training choices. It prepares files only; it does not start training.

## Output files

| File | Contents |
|---|---|
| `run.json` | Training settings, dataset checksum, model version, and package versions |
| `metrics.jsonl` | Scores, losses, and timing for each training step |
| `rollouts.jsonl` | Generated reply text, recognized speech, and rewards for RL runs |
| `checkpoint.pt` | Saved adapters and training state |
| `evaluation.json` | Results for each evaluation question |
| `*.wav` | Generated speech to listen to |

Keep generated datasets and training outputs outside Git. More detailed controls and planned features are listed in the [experiment plan](EXPERIMENTS.md).
