# Experiment protocol

## First sequence

1. Run CPU objectives and tiny upstream architecture tests.
2. Generate clean synthetic speech and validate all content hashes.
3. On a CUDA GPU, evaluate the unchanged checkpoint on validation speech.
4. Run ten steps of SFT; verify finite losses, a changed adapter and valid spoken replies.
5. Run text-action GRPO with the default fixed SFT anchor from the base checkpoint. Check reward variance and sampler/replay errors before increasing the step budget.
6. Compare base and adapted validation replies with the same scorer, decoding controls and prompt IDs. Listen to a random subset, including failures.
7. Test RLOO, Dr. GRPO and REINFORCE; then compare text/joint action scopes.
8. Repeat promising methods on noisy and compositional data, keeping the evaluation protocol fixed.
9. Select settings with validation data and evaluate the reserved test set once for the final comparison.

Each command is available in the README. The macOS preview is deliberately tiny and can contain no held-out examples; use the 128-example recipes for a first split and a larger separately designed corpus for meaningful experiments.

## Matrix

The matrix command produces 81 configurations: 3 data versions × 3 seeds × (4 online methods × 2 action scopes + 1 SFT baseline). Every config begins with ten steps and is a pilot, not a completed scientific experiment.

| Axis | Initial levels | Extension |
|---|---|---|
| Semantic data | Direct arithmetic/comparison; two-step arithmetic | Grounded conversational tasks, tool use, acoustic intent |
| Acoustic data | Clean; seeded 15 dB SNR white noise | Neural voices, unseen engines, room response, rate, accent |
| Objective | SFT, GRPO, Dr. GRPO-style, RLOO, REINFORCE | Offline DPO/IPO/SimPO runner, complete PPO/DAPO, routed advantages |
| Action mask | Text; joint text plus stochastic codebook decisions | Separate acoustic/semantic weights with precise objective definitions |
| Initialization | Pinned base model | SFT warm start after implementing a separate initialization API |
| Reference | Frozen base via disabled LoRA | Frozen SFT reference after warm-start support |
| Seeds | 42, 43, 44 | More seeds if variation obscures effects |
| Group size | 4 | 2, 8 after profiling |
| Anchor | Fixed weight 0.1 | 0, 0.03, 0.3; adaptive controller later |
| KL coefficient | 0.02 | 0, 0.01, 0.05 with separate speech checks |

Do not simultaneously change data, reward and optimizer when attributing an improvement. One especially useful comparison holds the semantic rows fixed between v1 and v2. v3 changes task difficulty, so its gains or losses cannot be attributed to acoustic augmentation.

## Run contract

For each completed run retain:

- Dataset recipe, semantic split, exact manifest and waveform hashes.
- Weight revision, package lock, training code commit, config and any initialization adapter.
- TTS/checkpoint/voice provenance and transformations, including seeds.
- Reward definition and a pinned local ASR checkpoint; the default `base.en` alias is convenient for development but can move.
- Sampling policy, action mask, reduction, group size, gradient clipping and reference settings.
- Completed optimizer updates, token/audio/scoring workload, GPU time and peak memory.
- Validation/test per-example results and saved audio, with unsuccessful generations included.

The initial run metadata covers config, manifest, model revision, packages, targets and metrics. Full compute accounting, reward checkpoint content hashes, neural provider metadata, and tracking across external services are next steps. A version-controlled scorer implementation and saved transcripts make current scoring decisions inspectable.

## Interpreting the objectives

`grpo` centers/scales rewards per prompt and averages the selected action surrogate per response. `dr_grpo` removes reward scale normalization and uses a fixed action bound. `rloo` uses a leave-one-out reward baseline and summed action log probabilities; `reinforce` uses raw rewards and summed log probabilities. RLOO/REINFORCE regularization also sums sampled reference penalties. The same numerical KL coefficient therefore does not imply identical regularization strength across reductions; tune it transparently.

The initial loop makes one update per group. At initial ratio one, GRPO clipping usually has no effect; multiple optimization epochs would make it more active but are not implemented. Sampling/replay mismatch can also move the ratio, so it is checked before every group is used.

Text-action training still conditions on previous generated audio and updates a backbone shared with speech generation. It does not isolate every downstream acoustic effect. Joint scope uses individual codebook actions with a terminal-audio exception: only codebook zero represents the probability of the canonical end frame.

DPO uses summed sequence scores. IPO and SimPO use length-normalized scores in the included functions. They have mathematical and toy tests but no full audio pair loader/trainer yet. PPO and DAPO names are rejected by the CLI until their complete training contracts exist.

## Criteria for moving beyond pilots

Before scaling, require a completed model reply with valid audio, finite gradients, probability parity within a documented tolerance, a changed adapter that survives save/reload, and repeatable evaluation. Inspect a sample of successes and every systematic failure category. Report zero-variance groups rather than dropping them without accounting.

For reward design, inspect cases where the transcript and audio disagree, numeral normalization changes the verdict, or ASR gives an answer a listener does not hear. For generalization, introduce an unseen TTS voice/engine and richer semantic templates before increasing training hours.

Use paired bootstrap intervals for matched evaluation examples and report results from all three training seeds separately. The comparison command handles per-example score differences; it does not automatically estimate variation across training seeds or account for repeated hyperparameter selection.

## Next implementation milestones

| Priority | Deliverable | Acceptance condition |
|---|---|---|
| 1 | Pretrained CUDA integration validation | Complete spoken rollout, parity, gradient update, checkpoint reload and base/adapted evaluation |
| 2 | Neural synthetic data provider | Pinned engine/checkpoint/voices, independently checked content, source/augmented waveform hashes |
| 3 | Explicit SFT warm start | Separate initialization and reference identities; resume remains strict |
| 4 | Persisted candidate/preference data | Exact token traces, audio, scoring provenance, confidence/tie handling and pair validation |
| 5 | Offline DPO/IPO/SimPO audio runner | Same evaluation protocol as online methods; summed/normalized objectives verified |
| 6 | Broader task and reward suite | Grounded answers plus listening audit; no reliance on arithmetic exact match for open dialogue |
| 7 | Complete PPO/DAPO/routed-advantage variants | Tests for critic/GAE, dynamic sampling or modality routing as appropriate |
| 8 | Multi-GPU/serving integration | Verified likelihood parity across inference and training engines, deterministic sharding and complete checkpoints |
