# LFM2.5-Audio RL

Research and initial training infrastructure for synthetic-data post-training of [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B).

Start with the [research report](docs/RESEARCH.md), [implementation status](docs/STATUS.md), and [experiment protocol](docs/EXPERIMENTS.md).

**This is an initial research implementation.** CPU objective tests and tests using a tiny randomly initialized upstream audio architecture have been run. The pretrained 1.5B model has **not** been trained or benchmarked here. The real runner requires a BF16-capable NVIDIA GPU and must pass its sampler/replay check on that hardware.

## What is available

| Component | Available now |
|---|---|
| Synthetic data | Symbolically verified English questions, reference answers, local TTS, noise variant, compositional variant, stable semantic splits, manifest/audio hashes |
| Online optimization | GRPO, Dr. GRPO-style normalization, RLOO, REINFORCE; experimental LFM LoRA runner |
| Supervised baseline | Joint text/audio SFT; optional fixed-weight SFT anchor during online RL |
| Offline preferences | DPO, IPO, SimPO losses and executable CPU optimization checks; full audio preference-data runner is a next step |
| Model integration | Interleaved speech input/output, differentiable replay, all eight codebooks, stop-event handling, text/joint action scopes |
| Evaluation | Independent ASR of generated speech, exact task correctness, audio validity, text/speech consistency, paired bootstrap comparisons |
| Experiment management | Strict YAML configs, 81-run matrix generator, local metrics, adapter/optimizer/RNG checkpoints, resume |

PPO with a critic, complete DAPO, KTO, ALPO, adaptive WavAlign reproduction, distributed training, neural TTS providers, and online preference-pair generation are researched but not implemented. An SFT anchor in this repository is a fixed-weight experiment, not a reproduction of WavAlign's adaptive controller.

## Install and test

Python 3.12 is required. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```bash
git clone https://github.com/kadirnar/lfm25-audio-rl.git
cd lfm25-audio-rl
uv sync --extra dev
uv run pytest -q
uv run lfm-rl smoke --output runs/cpu-smoke
```

The smoke command optimizes tiny categorical policies with eight objectives. Its numbers verify implementation behavior; they measure no speech capability and cannot rank methods for LFM2.5-Audio.

To test the actual upstream layers at small random sizes, without downloading model weights:

```bash
uv sync --extra dev --extra model
uv run pytest -q
```

## Generate synthetic datasets

First verify manifests without installing a speech engine:

```bash
uv run lfm-rl data --config configs/datasets/text_only.yaml --output data/text-only
uv run lfm-rl validate-data data/text-only
```

For speech, install `espeak-ng` using your Linux package manager. The bundled providers are development tools; the research report specifies a path to diverse neural TTS datasets.

```bash
uv run lfm-rl data --config configs/datasets/v1_clean.yaml --output data/v1_clean
uv run lfm-rl data --config configs/datasets/v2_noisy.yaml --output data/v2_noisy
uv run lfm-rl data --config configs/datasets/v3_compositional.yaml --output data/v3_compositional
uv run lfm-rl validate-data data/v1_clean --require-audio
```

On macOS, `configs/datasets/macos_preview.yaml` uses the installed `Samantha` voice through `say` and generates eight examples. You can change the voice to another locally installed voice. Both input speech and reference answer speech are synthesized; no speaker recordings are collected.

Output directories must be new. Each dataset contains `dataset.json`, `manifest.jsonl`, and, when enabled, `audio/`. Keep generated datasets and checkpoints outside Git. A text-only manifest is deliberately rejected by the real training command.

## First GPU experiment

Install a matching CUDA-compatible PyTorch/torchaudio build for your platform. The lock resolves dependencies, but a local CUDA driver/runtime still has to support the chosen wheels.

```bash
uv sync --extra model --extra asr --extra dev
uv run lfm-rl evaluate --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/base-validation --split validation
uv run lfm-rl train --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-text-v1
uv run lfm-rl evaluate --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-validation --split validation \
  --checkpoint runs/grpo-text-v1/checkpoint.pt
uv run lfm-rl compare runs/base-validation/evaluation.json runs/grpo-validation/evaluation.json
```

These commands download the pinned base model and the configured ASR model when run. They do not provision a cloud GPU or call paid APIs. Start with the ten-step configuration and review saved audio before scaling. GPU fit, speed, quality, and long-run stability are unmeasured.

The default freezes the base model and trains LoRA projections in its language backbone. The reference policy disables the adapters, avoiding a second model copy. The encoder, codec, and detokenizer stay frozen. Text-only action losses can still change generated audio through the shared backbone.

Resume an interrupted run with the same config, dataset, and output directory:

```bash
uv run lfm-rl train --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-text-v1 --resume
```

Do not use `--resume` to change algorithms or initialize from SFT: checkpoints currently require an identical run identity. Initializing a new RL experiment from an SFT adapter needs a separate warm-start feature.

## Compare methods and data versions

```bash
uv run lfm-rl matrix --output runs/experiment-plan
```

This writes **81 planned configurations**: three datasets × three seeds × nine method/scope combinations. It does not launch any training. Four online methods each have text and joint scopes; SFT supplies the ninth combination. Data hashes are bound when a job actually starts.

Run metadata includes the resolved config, dataset manifest hash, model revision, installed packages, adapter targets, and training metrics. Audio scores use `spoken-exact-v1`; its coefficients and checks are currently code-defined. Use a pinned local ASR checkpoint for controlled studies: the convenient `base.en` alias alone is not an immutable reward-model revision.

## Reading the code

- `data.py`: semantic generation, synthesis, validation and provenance.
- `objectives.py`: masked objectives and explicit reduction choices.
- `lfm.py`: upstream model callbacks, action recording and differentiable replay.
- `lora.py`: trainable adapters and frozen reference policy.
- `train.py`: serial rollout groups, optimization, anchoring and checkpoints.
- `evaluate.py`: held-out speech scoring and comparison.
- `tests/test_lfm_contract.py`: upstream cached generation versus teacher-forced probabilities, gradient flow and terminal events.

## Licensing

Original repository code is MIT-licensed. Model weights and installed dependencies retain their own terms; the MIT license does not relicense them. See [third-party notes](docs/THIRD_PARTY.md). No pretrained weights, external corpora, credentials, or generated voice assets are committed.
