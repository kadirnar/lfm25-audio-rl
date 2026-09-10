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
uv sync --extra dev
uv run pytest -q
```

## Create speech data

On Linux, install `espeak-ng` first. It turns the generated questions and answers into speech.

```bash
uv run lfm-rl data --config configs/datasets/v1_clean.yaml --output data/v1_clean
uv run lfm-rl validate-data data/v1_clean --require-audio
```

Other recipes add noise or harder questions. For a small macOS example, use `configs/datasets/macos_preview.yaml`.

## Train

Training needs an NVIDIA GPU with BF16 support and a compatible CUDA setup.

```bash
uv sync --extra model --extra asr --extra dev
uv run lfm-rl train --config configs/experiments/grpo_text.yaml --data data/v1_clean --output runs/grpo-v1
```

Start with the default ten steps. See [how to test, evaluate, and resume a run](docs/RUNNING.md).

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
