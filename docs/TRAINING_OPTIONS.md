# W&B and training options

The trainer supports Weights & Biases logging and configurable training options. W&B is disabled by default. The original optimizer, learning rate, and one-prompt update remain the defaults.

## Install

```bash
uv sync --locked --extra model --extra asr --extra wandb --extra dev
```

Training still needs a BF16-capable NVIDIA GPU. Installing the W&B extra does not start a run or upload data.

## Start with offline logging

Add this section to an experiment YAML file:

```yaml
wandb:
  mode: offline
  project: lfm25-audio-rl
  name: grpo-v1-seed42
  group: grpo-v1
  tags: [grpo, synthetic]
  log_every: 1
```

Run training normally:

```bash
uv run lfm-rl train --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/grpo-v1 \
  --wandb-mode offline --wandb-project lfm25-audio-rl
```

CLI values override the YAML file. The available W&B overrides are `--wandb-mode`, `--wandb-project`, `--wandb-entity`, and `--wandb-name`.

Offline mode saves W&B files under the training output folder. It needs no API key. To upload one of these runs later, log in and run `wandb sync` with its `offline-run-...` directory.

## Online logging

Run `wandb login` in your environment, or provide `WANDB_API_KEY` through your environment or secret manager. Then set `wandb.mode: online` or use `--wandb-mode online`.

Use `entity` for your W&B username or team. The trainer uses the existing account; it does not create an account for you. Keep credentials out of YAML files and Git.

Online mode sends training settings, the dataset checksum, numeric training metrics, and W&B's run/system metadata to your chosen project. This integration does not upload audio, transcripts, datasets, checkpoints, or source code. It disables console capture and automatic code/Git capture. Existing local files are still written normally.

Selected online logging must initialize successfully. A logging error is reported instead of silently changing the run to offline mode. To train without W&B, select `disabled`.

## Logged metrics

The W&B charts use `trainer/step` as the optimizer-update axis. Metrics have a `train/` prefix.

| Metric | Meaning |
|---|---|
| `policy_loss` | Mean RL loss across prompt groups |
| `anchor_loss` | Mean SFT or anchor loss before its weighting |
| `mean_reward` | Mean candidate reward |
| `reward_std` | Mean within-prompt reward standard deviation |
| `zero_variance_group` | Fraction of prompt groups with identical rewards |
| `truncation_rate` | Fraction of unfinished candidate replies |
| `parity_max_error` | Largest sampler-versus-replay log-probability error |
| `lr` | Learning rate used for this update |
| `gradient_norm` | Gradient norm before clipping |
| `seconds` | Update computation, synthesis, ASR, and rollout writing time |
| `peak_gpu_bytes` | Peak PyTorch allocated CUDA memory during this update |
| `prompts_per_update` | Number of accumulated prompts |
| `prompts_per_second` | Accumulated prompts divided by measured update time |
| `actions_per_second` | Selected RL actions divided by measured update time |
| `checkpoint_step` | Most recent saved optimizer update |

SFT runs omit RL-only metrics. The measured update time excludes final metrics writing, checkpoint saving, and W&B logging. It is not total job runtime. Peak allocated memory is not total device memory use.

`log_every` controls how often W&B receives an update. The last update is always logged. Local JSONL metrics are written every update. Full speech evaluation reports remain in the [evaluation pipeline](EVALUATION.md); they are not automatically sent to W&B.

## Training controls

Add a `train_opt` section to an experiment config:

```yaml
train_opt:
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
  optimizer: adamw
  weight_decay: 0.0
  betas: [0.9, 0.999]
  eps: 0.00000001
  scheduler: cosine
  warmup_steps: 10
  min_lr_ratio: 0.1
  checkpoint_every: 10
  save_rollout_audio: false
```

Use at least ten total `steps` with this warmup example. The ready-to-edit [example config](../configs/experiments/grpo_optimized.yaml) uses 100 optimizer updates and offline W&B.

| Option | Default | Effect |
|---|---|---|
| `gradient_accumulation_steps` | `1` | Average gradients across this many sequential prompts before one optimizer update |
| `gradient_checkpointing` | `false` | Recompute backbone activations during backward to reduce saved activations |
| `optimizer` | `adamw` | Use `adamw_fused` to request PyTorch's fused CUDA implementation |
| `weight_decay` | `0.0` | AdamW decay applied to the trainable LoRA parameters |
| `betas` | `[0.9, 0.999]` | AdamW moment coefficients |
| `eps` | `1e-8` | AdamW numerical stability constant |
| `scheduler` | `constant` | Also supports `linear` and `cosine` |
| `warmup_steps` | `0` | Increase LR from `lr / warmup_steps` to the configured LR |
| `min_lr_ratio` | `0.1` | Final decay target as a fraction of the configured LR |
| `checkpoint_every` | `1` | Save after this many optimizer updates; also save initialization and the final update |
| `save_rollout_audio` | `true` | Keep generated training WAV files; `false` removes temporary ASR files after scoring |

Two options also have CLI overrides: `--gradient-accumulation-steps 2` and `--gradient-checkpointing`. Use `--no-gradient-checkpointing` to disable it explicitly.

## What accumulation changes

`steps` counts optimizer updates. Each update processes `gradient_accumulation_steps` prompts. Each RL prompt still produces its own `group_size` candidates. Advantages are calculated within each prompt group, then the group losses are averaged. Gradients are clipped once before the optimizer update.

For example, 100 updates × 2 accumulated prompts × 4 candidates means 800 RL rollouts. The model is not updated between those two prompt groups. Only one prompt group is held at a time. SFT averages the per-example losses across accumulated examples; it is not a global token-weighted packed batch.

Accumulation increases the effective batch size. It does not make an update cheaper. Compare runs using a matched rollout/token budget as well as optimizer-step counts.

## What activation checkpointing changes

The trainer keeps the model in eval mode to disable dropout while still computing gradients. The usual training-mode checkpoint switch would not be enough for this runner.

Our wrapper applies PyTorch non-reentrant checkpointing to LFM backbone layers during uncached gradient passes. Cached generation and frozen-reference scoring bypass it. It does not checkpoint the Conformer, depthformer, codec, or ASR model.

Recomputation trades extra backward work for fewer saved backbone activations. It does not shrink model weights, ASR memory, or retained rollout inputs. GPU memory and speed changes have not been measured yet. The original sampler/replay parity check stays enabled.

`adamw_fused` is also optional. It requires CUDA parameters and uses PyTorch's fused implementation. Its speed and numerical behavior still need validation on the target GPU. The default `adamw` keeps PyTorch's standard optimizer selection.

## Learning-rate schedules

Warmup is measured in optimizer updates, not candidates or prompts. Constant scheduling holds the configured LR after warmup. Linear and cosine schedules decay across the remaining updates to `lr * min_lr_ratio` when at least two decay updates exist. With one remaining update, that update uses the base LR.

The trainer computes each LR from the update number and config. A checkpoint's step therefore determines the next LR without a separate scheduler-state file. Resume tests cover this behavior.

## Resume

```bash
uv run lfm-rl train --config configs/experiments/grpo_optimized.yaml \
  --data data/v1_clean --output runs/grpo-optimized --resume
```

Keep the training settings and data unchanged, including any CLI training overrides. Checkpoints restore adapters, optimizer state, Torch/CUDA random state, and Python random state. Changing W&B settings does not change checkpoint identity. Old checkpoints using the original training defaults keep the same identity.

Online sessions reuse the stored W&B run ID when mode, project, and entity match. Offline W&B cannot append to an earlier offline session, so each restart gets a new run ID in the same group. `wandb-state.json` records these sessions. Changing mode, project, or entity also starts a new session.

W&B is a log, not a checkpoint store. If a crash occurs after an unsaved update, training restarts from the last checkpoint. Local metrics and W&B history can contain repeated optimizer-step values from the failed attempt. The `checkpoint_step` metric shows what was saved. This is why the integration uses a custom step axis instead of forcing W&B's internal history index backward.

An initial step-0 checkpoint is saved before W&B initialization and training. This lets a failed startup or first update resume from the same adapters and random state.

A larger `checkpoint_every` reduces checkpoint writes but loses more progress after a crash. Setting `save_rollout_audio: false` saves disk space; ASR still runs and briefly writes a temporary WAV file. Keep audio when you need to investigate reward errors or listen to training examples.

## Validation and sources

Tests compare checkpointed and ordinary policy/SFT losses and LoRA gradients on the actual upstream architecture at tiny random sizes. They also cover accumulation, learning-rate boundaries, exact CPU adapter-state resume, final checkpoint saving, temporary audio cleanup, and W&B run handling.

The real W&B SDK was exercised in offline mode without credentials. Online requests are mocked. No pretrained GPU training, speed benchmark, memory-saving benchmark, or online W&B upload was performed.

API references: [W&B init](https://docs.wandb.ai/models/ref/python/functions/init), [W&B resume](https://docs.wandb.ai/models/runs/resuming), [custom metric axes](https://docs.wandb.ai/models/track/log/customize-logging-axes), [PyTorch checkpointing](https://docs.pytorch.org/docs/2.8/checkpoint.html), [PyTorch AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).
