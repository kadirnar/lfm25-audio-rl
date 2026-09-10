# LFM2.5-Audio RL

Train [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) on synthetic speech and compare different training methods.

The model takes a spoken question and produces a spoken answer.

## Dataset format

Use **JSONL for the list of examples** and **WAV for the audio**. JSONL means one JSON object per line.

```text
data/v1_clean/
  dataset.json
  manifest.jsonl
  audio/
    question-001.wav
    answer-001.wav
```

Each example contains the question text, question audio, and correct answer text. Include answer audio for supervised training and for the default RL settings.

Use mono, 24 kHz WAV files. See [dataset examples and field names](docs/DATASETS.md).

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/kadirnar/lfm25-audio-rl.git
cd lfm25-audio-rl
uv sync --extra dev --extra synthetic --extra metrics
uv run pytest -q
```

## Create speech data

On Linux, install `espeak-ng` first. It turns the generated questions and answers into speech.

```bash
uv run lfm-rl data --config configs/datasets/v1_clean.yaml --output data/v1_clean
uv run lfm-rl validate-data data/v1_clean --require-audio
```

Other recipes add noise or harder questions. For a small macOS example, use `configs/datasets/macos_preview.yaml`.

## Use OpenRouter and neural TTS

Generate text with OpenRouter, then create speech with MOSS-TTS, Echo-TTS, Qwen3-TTS, or Zonos. Reuse the same text to compare different models and voices.

See the [synthetic data guide](docs/SYNTHETIC_PIPELINE.md) and [TTS setup](docs/TTS_SETUP.md). The adapters have CPU mock tests; real GPU generation still needs validation.

## Train

Training needs an NVIDIA GPU with BF16 support and a compatible CUDA setup.

```bash
uv sync --extra model --extra asr --extra dev
uv run lfm-rl train --config configs/experiments/grpo_text.yaml --data data/v1_clean --output runs/grpo-v1
```

Start with the default ten steps. See [how to test, evaluate, and resume a run](docs/RUNNING.md).

## Evaluate answers

Score answer correctness, speech quality, speaker similarity, prosody, and speed. The default setup works on saved audio and transcripts without model downloads or API calls.

```bash
uv run lfm-rl score --config configs/metrics/default.yaml \
  --data data/v1_clean --predictions runs/base/predictions.jsonl \
  --output runs/base-scores
```

See the [evaluation guide](docs/EVALUATION.md), [metric research](docs/METRIC_RESEARCH.md), and [optional scorer setup](docs/METRIC_SETUP.md). Learned scorers have API mock tests; pretrained quality results still need a real evaluation run.

If you changed environments for training, install the `metrics` extra again before scoring.

## Available methods

The training code supports SFT, GRPO, Dr. GRPO-style training, RLOO, and REINFORCE.

DPO, IPO, and SimPO have tested loss functions. Their complete audio training pipelines are not ready yet.

Tests pass on CPU. Training with the pretrained audio model still needs GPU validation.

## More information

- [Dataset format](docs/DATASETS.md)
- [Running experiments](docs/RUNNING.md)
- [Research report](docs/RESEARCH.md)
- [Experiment plan](docs/EXPERIMENTS.md)
- [What works and what is unfinished](docs/STATUS.md)

This repository's code uses the MIT license. The model and other packages have [their own licenses](docs/THIRD_PARTY.md).
