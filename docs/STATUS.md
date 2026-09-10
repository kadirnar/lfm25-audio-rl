# Implementation and validation status

## Completed in the initial revision

- A 4,000+ word research report with 27 numbered primary-source references.
- Three versioned synthetic-data recipes, text-only development mode, and a macOS voice preview recipe.
- Content-addressed manifests/audio, deterministic semantic grouping and split validation.
- GRPO, Dr. GRPO-style, RLOO, REINFORCE, DPO, IPO and SimPO objectives, plus an SFT baseline.
- An experimental single-GPU LFM runner for the four online methods and SFT, with backbone LoRA and optional fixed SFT anchoring.
- Sampling-event recording, differentiable text/joint replay, terminal handling and a probability parity gate.
- A frozen-base reference using disabled adapters, adapter checkpoints, optimizer/RNG restoration and strict resume identity.
- Independent ASR-based spoken-answer scoring, saved audio and held-out evaluation with paired bootstrap comparisons.
- A generator for 81 planned configurations across data versions, methods, action scopes and seeds.
- A dependency lock and GitHub Actions workflow.

## Local validation

Validation used Python 3.12.12 on macOS arm64, PyTorch 2.8.0, liquid-audio 1.3.0 and Transformers 4.57.6. No pretrained model weights were downloaded for these checks.

**27 tests passed.** They cover objective gradients and hand-calculated cases, masking and reduction, synthetic split stability and tamper detection, waveform/reward failures, LoRA reference isolation, checkpoint/RNG restoration, paired-comparison checks, and the actual upstream architecture at tiny random dimensions.

The upstream architecture tests exercise an attention-plus-convolution language backbone, a Conformer input encoder and all eight audio codebooks. Cached generation agrees with teacher-forced selected-action log probabilities within `2e-4` on this FP32 CPU fixture, for both text and joint scopes. Gradients reach LoRA parameters. Terminal text decisions are recorded; terminal audio events retain only their first-codebook probability.

The eight-objective CPU smoke run improved each toy policy's target probability from its initial 0.25. See [machine-readable smoke evidence](validation/smoke.json). The fixture is categorical, not speech: those numbers are neither LFM performance results nor a ranking of RL methods.

Local speech generation produced eight input/reference pairs using the macOS Samantha voice. The sixteen mono 24 kHz WAV files passed manifest/audio hash validation. The preview remains a local deliverable outside Git, and all eight examples happened to fall in its training split. It is a synthesis preview, not an evaluation dataset.

The matrix command produced 81 valid experiment configurations. The real training preflight rejected this machine because CUDA is unavailable, before a model download or training output directory was created. Lint and formatting checks passed.

## Unverified or incomplete

| Area | Boundary |
|---|---|
| Pretrained LFM training | No GPU training, pretrained inference, quality measurement, throughput or VRAM-fit claim yet |
| GPU numerical parity | FP32 tiny CPU parity passed; BF16 checkpoint parity on target CUDA hardware still required |
| CUDA complete pipeline | Scorer loading, codec/detokenizer interaction, full-model SFT/backward and real checkpoint evaluation remain unverified |
| Offline preferences | DPO/IPO/SimPO losses are tested; persisted audio preference data and a full trainer are not implemented |
| SFT initialization | SFT can run independently; a new RL run cannot yet warm-start from an SFT adapter with a distinct reference policy |
| WavAlign/ALPO | Fixed anchoring and action masks are available; adaptive mixing and reward-specific routing are research plans |
| PPO/DAPO/KTO | Research only; no trainer under these names |
| Data scale/diversity | Symbolic development tasks and local TTS; no neural teacher/voice factory, broad conversation corpus or multilingual adaptation |
| Distributed/production use | Single process and serial rollouts; no concurrent sampling on one model instance, distributed engine or serving integration |
| Scorer immutability | Default ASR alias is not a pinned checkpoint; controlled studies should use a pinned local model |
| Resume logging | Checkpoint restores completed update state; append-only logs can retain attempts from an interrupted, uncheckpointed step |
| Real-world evaluation | No external benchmark scores, human preferences or demonstrated synthetic-to-real improvement |

The [experiment protocol](EXPERIMENTS.md) defines the next validation sequence and acceptance criteria. The initial code is a starting point for experiments, not evidence that the target model has improved.
