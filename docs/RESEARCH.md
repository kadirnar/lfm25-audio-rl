# Reinforcement learning for LFM2.5-Audio with synthetic speech

## Research conclusion

The strongest starting experiment is a synthetic English speech-to-speech task with verifiable answers, a frozen audio encoder and detokenizer, LoRA updates to the language backbone, and a comparison of SFT, text-action GRPO with a supervised anchor, and a leave-one-out policy-gradient baseline. Full text-and-audio optimization should follow only after validating action probabilities and measuring spoken output quality. This is an experimental recommendation, not an established best recipe for this checkpoint.

The literature supports investigating both modality-separated and joint optimization. ParaS2S provides direct speech-to-speech GRPO evidence; WavAlign motivates semantic preference updates with acoustic supervision; more recent ALPO work argues for separately routing semantic and acoustic advantages. These results come from other model/task settings and do not establish which method will win on LFM2.5-Audio. A controlled comparison should therefore retain all three design directions in the roadmap.[^6][^7][^8]

Synthetic-only training is a reasonable research constraint. However, a successful score on speech made by the training synthesizer establishes only performance within that synthetic distribution. A separate evaluation using unseen synthesis engines, speakers, acoustic conditions, and optionally licensed human recordings is necessary to assess transfer. The initial code deliberately starts with small, verifiable tasks and exposes its limitations instead of treating a synthetic demonstration as evidence of conversational improvement.

## Scope and evidence

The implementation audit was performed on September 10, 2026. The target is `LiquidAI/LFM2.5-Audio-1.5B`, with model revision `c362a0625dfe45aa588dce5f0ada28a7e5707628`. The inspected `liquid-audio` source revision is `19e65845923a7f136442c95137884ec61eb386aa`; its package version is 1.3.0. The code pins the package version and weight revision. Package locking and small upstream architecture tests accompany the integration.

Primary sources include the model card, official implementation and trainer, original RL papers, speech-specific post-training papers, benchmark repositories, and synthesis/scoring model documentation. The report distinguishes documented interfaces, experimental findings reported by paper authors, and proposed experiments. It does not infer published GPU throughput, training cost, or quality improvements for this repository; none have been measured.

The initial task is English spoken instruction following with synthetic input speech and generated spoken replies. The target model card lists English support. Adapting it to Turkish or other languages is a separate capability-building question and should first establish a supervised speech baseline. The presence of a language in a text backbone does not demonstrate reliable speech recognition or generation in that language.[^1]

## Model architecture and training interfaces

The model couples a roughly 1.2B language backbone with a FastConformer speech encoder and an RQ-transformer audio generator. Output uses eight codebooks and a Mimi-compatible token representation, followed by an audio detokenizer. The documented context length is 32,768 tokens. Interleaved generation is the conversational path, while sequential generation supports other tasks such as ASR and TTS.[^1]

The concrete training boundary is more important than the high-level architecture. The current `LFM2AudioModel` exposes `logits(batch)`, returning text logits, audio logits, and corresponding labels. Audio logits flatten frames in frame-major, codebook-minor order. The backbone predicts a state for each audio frame, and the depth transformer conditions later codebooks on earlier codes within that frame. The standard forward method computes weighted supervised losses; its codebook weights are training weights, not factors in a normalized joint probability.[^2]

Consequently, the RL adapter uses selected action log probabilities from logits rather than treating the existing scalar supervised loss as a policy log probability. A joint frame probability requires the sum of conditional log probabilities over its stochastic codebook decisions. Reusing the weighted SFT loss would silently change the RL objective. This distinction also matters for DPO: a true sequence likelihood is a sum, while a length-normalized score defines a different preference objective.

The official preprocessing pipeline includes conversation segments, an interleaved mapper, and a trainer. The mapper supplies a supervision mask and matches text/audio blocks to the model's interleaving schedule. This substantially reduces the amount of custom SFT infrastructure required. It does not itself provide the on-policy sampling, reference-policy scoring, reward computation, or experiment bookkeeping needed for RL.[^3][^4]

### Action accounting

The audited model configuration uses six text tokens followed by twelve audio frames in its interleaved schedule. Input speech is represented by continuous encoder features and is not an action to optimize. Prompt text, audio input positions, assistant headers, and padding must remain outside the response action mask. Special output tokens that are actually sampled belong to the trajectory.

Two stop conditions require special attention. The interleaved generator samples a text end-of-turn token and then stops before yielding that token. Separately, an audio frame with end-of-audio in its first codebook is emitted with every codebook set to the same sentinel. The probability of that emitted terminal audio event is determined by its first codebook; the discarded higher-codebook samples must be marginalized rather than scored as if seven sentinel decisions had been independently emitted. These behaviors were found by inspecting the generation implementation.[^2]

The initial adapter records sampling callbacks, reconstructs the full response, and checks cached sampling against teacher-forced replay. Sampling uses float32 probability computation at a positive temperature and full vocabulary support. Top-k sampling is disabled in real RL configs because it changes the behavior distribution and creates support mismatches if replay uses unrestricted logits. This conservative restriction can be relaxed later with an explicitly matched truncated-policy implementation.

The processor decodes ordinary audio codes into 24 kHz waveforms. The audited detokenizer construction includes a hard-coded CUDA transfer, so the real runner currently requires CUDA. Codebook sentinel frames must be removed before decoding, but the last ordinary audio frame must be preserved if a trajectory was truncated.[^5]

## Relevant speech reinforcement-learning evidence

| Work | Directly relevant evidence | Consequence for this project |
|---|---|---|
| ParaS2S | Studies content/style appropriateness at the waveform level and applies GRPO to speech-to-speech models. | Establish a joint-output baseline and evaluate what listeners hear, not only generated text. |
| WavAlign | Identifies difficulties with sparse preferences over dense speech sequences; proposes text-token preference updates with an adaptive supervised anchor. | Make text-action RL plus supervised audio anchoring an early candidate. |
| ALPO / ParaIntent | Separates textual and acoustic advantages and routes them to corresponding token subsets. Reported evaluation uses a Chinese, single-turn setting and one interleaved backbone. | Test routed advantages later, with independent reward calibration; do not assume cross-model transfer. |
| GRPO for LLM TTS | Uses ASR-based character error and confidence signals for generated speech. | Treat independent transcription as a practical reward component, while distinguishing TTS from conversational answer quality. |
| Synthetic-speech GRPO for ASR | Reports improved synthetic-to-real domain adaptation compared with SFT in a specific ASR study. | Include a synthetic-only SFT versus RL comparison; do not interpret ASR results as evidence of spoken-response naturalness. |
| SpeechGym | Investigates audio-native voice-agent RL in interactive task environments. | Reserve a later multi-turn track for stateful simulated tasks with verifiable outcomes. |

These findings come from the original publications.[^6][^7][^8][^9][^10][^11] They motivate experiments rather than establishing a universal ranking. Architectural differences include parallel speech streams, interleaved streams, and separated reasoning/speaking modules. Reward models and evaluation languages also differ. A method that improves acoustic emotion on one backbone may degrade intelligibility or task success on another.

One quantitative result deserves careful interpretation. The synthetic-banking ASR study reports 36.71% WER after synthetic-data SFT and 22.09% after GRPO on its setup. Its real-speech conditions also show that the relative benefit depends on the data regime. The relevant hypothesis is that RL can extract useful behavioral corrections from synthetic data; the reported percentage improvement is not a forecast for this project.[^10]

The newer speech papers also disagree in emphasis. WavAlign favors restricting preference-driven updates to the semantic channel, whereas ALPO reports gains from routing separate rewards to both channels. This is a useful experimental fork. Reward quality, model initialization, gradient balance, and acoustic token density could explain different outcomes. Reproducing one paper's objective name without its reward model, data protocol, and gradient routing would not resolve that question.

## Algorithm portfolio

### Baselines and first online methods

SFT establishes whether the examples, audio encoding, interleaving, and trainable parameters support the task. It is also the control for determining whether RL offers value beyond additional exposure to synthetic data. Rejection sampling followed by SFT is another useful later control: generate several responses, retain verified successes, and train on them with a documented acceptance rule. Neither should be mislabeled as an online RL algorithm.

REINFORCE provides a transparent policy-gradient baseline: increase the summed log probability of actions in high-reward trajectories. A baseline may reduce variance, but its construction must be stated. The initial implementation's plain REINFORCE variant has no fitted or group-centered baseline. RLOO subtracts the mean reward of the other samples generated for the same prompt, avoiding inclusion of the current sample in its own baseline.[^12]

GRPO uses the group-relative reward signal and a clipped policy surrogate. It avoids a learned critic, which is attractive when each rollout already requires speech generation, decoding, and scoring. The code's GRPO variant uses population-standard-deviation normalization within each group and averages selected action losses per sequence. Zero-variance groups yield zero preference advantage, which is logged. The independent SFT anchor can still provide an update.[^13]

Dr. GRPO motivates removing reward standard-deviation scaling and response-length normalization to address biases. The initial `dr_grpo` option uses centered, unscaled advantages and a fixed maximum-action denominator. This is an explicitly defined adaptation to speech actions, not a claim to reproduce an entire published training recipe. In joint scope, the fixed bound counts up to eight codebook decisions per generation step.[^14]

For GRPO variants, the sampled nonnegative reference penalty is `exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1`. It is a sample-based regularizer, not a full-distribution KL measurement, particularly when samples come from an older policy. The code retains an explicit coefficient and records the sampling distribution. Current TRL documentation is useful for understanding how normalization and clipping variants differ, but its generic examples are not a ready-made eight-codebook adapter for this model.[^15]

### Offline preference methods

DPO trains from chosen/rejected completions without an online RL loop or a separately learned reward model. It uses a reference-corrected sequence log-ratio difference. For synthetic speech, both candidates should answer the same input audio, and preferences should be based on a reproducible scoring protocol or human judgments. Candidate provenance and rejection reasons are as important as the pairwise label.[^16]

IPO is worth comparing when preferences are noisy or close to deterministic. Its quadratic target differs materially from DPO's logistic loss. The implementation uses length-normalized scores for this variant and tests it separately. SimPO uses a reference-free, length-normalized score and a margin, reducing reference computation at the cost of changing the objective.[^17][^18]

The repository includes these three losses and working toy optimization checks. It does not yet implement a complete LFM audio preference-dataset runner. That requires persisted candidates, exact interleaving or token traces, pair validity checks, and pairwise evaluation. A tested loss function is only one part of that pipeline.

KTO is a later option when the data naturally supplies desirable/undesirable labels instead of comparable pairs. It should be added with the required reference and distribution terms rather than approximated by a binary classifier and given the same name.[^19]

### Higher-complexity online methods

PPO with a learned value model remains a valuable comparator when investigating temporal credit assignment or multi-turn tasks. It adds value predictions, advantage estimation, critic optimization, and more memory/implementation complexity. The current runner intentionally has no critic and must not be described as a complete PPO implementation merely because its GRPO loss uses clipping.[^20]

DAPO combines several changes, including asymmetric clipping, dynamic sampling and token-level loss aggregation. Implementing only one of them does not reproduce DAPO. In speech, dynamic sampling must account for both valid audio and useful reward variance; silently dropping failed generations can change the effective training distribution. A faithful experiment should report rejection rates, accepted prompts, generated audio seconds and total scoring work.[^21]

Modality-routed GRPO, adaptive SFT/RL mixing and multi-objective constrained optimization are longer-term directions. Begin with separate semantic and acoustic reward logs before introducing separate advantages. A weighted sum can hide compensation between a wrong answer and attractive speech. A constrained formulation could instead require intelligibility and content correctness while optimizing naturalness, but it needs meaningful thresholds and robust reward measurements.

## Synthetic data strategy

### Stage 1: verified development data

The implemented data generator creates arithmetic and comparison questions with exact programmatic answers. It can synthesize both the user question and a reference assistant reply. This gives a low-cost way to verify preprocessing, SFT, rollout scoring, and reward behavior without dependence on an external text-generation API or a subjective judge. It is a development corpus, not a broad conversational dataset.

The first three recipes deliberately separate two kinds of change. `v1_clean` uses direct tasks; `v2_noisy` preserves their semantic identities and splits while adding seeded waveform noise; `v3_compositional` introduces two-step questions. Dataset names communicate the intended intervention, while content hashes establish the exact artifact used. Changes in voice or waveform do not move a semantic problem between train and test.

The implemented split prevents exact semantic duplicates across splits. It does not provide unseen-template, unseen-task-family, or unseen-speaker generalization. Those are separate evaluation dimensions. The initial TTS engines also have limited expressiveness; results should not be generalized to natural conversation from this corpus alone.

### Stage 2: diverse synthesis

The next data factory should combine a deterministic task schema with several speech renderers. Candidate engines include Kokoro for compact local generation, Piper for an established local speech ecosystem, and F5-TTS for reference-conditioned experiments. Their model and voice terms differ. F5-TTS, for example, distinguishes its MIT code from pretrained weights under CC-BY-NC; model choice must consider the specific checkpoint rather than only the repository license.[^22][^23][^24]

Recommended diversity axes are speaker/voice, speaking rate, accent where supported, pause structure, microphone response, reverberation, background noise and room acoustics. Each augmentation needs parameters and an RNG seed. Keep a clean source waveform alongside the transformed input. Validate content after synthesis with independent ASR, and distinguish a bad synthesis from a model failure.

Start with 1,000–5,000 verified semantic examples and a few renderings per example as a planning target, then scale only after inspecting reward variance and evaluation transfer. These are proposed experiment sizes, not an empirically optimized sample budget. More synthetic hours from one voice can amplify a narrow acoustic distribution without adding semantic coverage.

### Stage 3: conversational and paralinguistic tasks

For conversational content, create structured scenarios with grounded facts and a hidden answer specification: schedules, inventory, directions, object descriptions, or fictional service interactions. The teacher may propose a surface form, but factual fields should remain programmatically checkable. Open-ended responses require a different correctness evaluator than the exact-match arithmetic reward.

For paralinguistic tasks, hold the transcript constant while varying the intended vocal cue. If the label is recoverable from the text alone, the task does not demonstrate acoustic understanding. Include transcript-only controls, neutralized-prosody inputs, contradictory acoustic/text cues and unseen speakers. The ParaIntent study's neutralization experiments illustrate why those controls matter.[^8]

For multi-turn work, use a synthetic environment with explicit state and terminal outcomes. A dialogue can be fluent while leaving the simulated task incomplete. Track both spoken correctness and environment-state success. SpeechGym is relevant evidence for this direction, but adding its full interaction loop is a separate project phase.[^11]

### Version and provenance contract

Every production example should carry a stable semantic ID, task family, prompt version, answer specification, text-teacher identity if used, TTS engine/checkpoint, voice ID, language, synthesis parameters, augmentation parameters, seed, waveform hashes, licensing/provenance fields, QA results, and split. The current schema covers the development subset and permits provenance extensions; richer neural-voice manifests remain future work.

Version the semantic corpus, acoustic rendering, candidate-rollout pool, preference labels and reward calibration separately. Otherwise a run that appears to compare two algorithms may actually compare different questions, voices or judges. Keep the base model immutable and record any SFT initialization separately from RL hyperparameters.

## Rewards and failure analysis

The initial `spoken-exact-v1` reward verifies the answer in an independently recognized spoken reply. Correct internal text alone cannot earn a positive reward. Invalid, silent, severely clipped, overlong or truncated output is rejected. On valid speech, spoken correctness gates smaller contributions from text correctness and text/speech consistency. This reward is suitable for the exact-answer development tasks and deliberately unsuitable for evaluating open-ended conversational quality.

Independent ASR is essential because the model's generated text and audible words can diverge. For TTS, compare ASR against the intended transcript. For speech-to-speech conversation, first assess whether the audible answer is correct for the question, then measure its consistency with generated text. Comparing the reply to the user's question transcript would incorrectly reward repetition. Faster Whisper provides a practical local scoring implementation, but its model should be pinned and kept fixed across comparisons.[^25]

WER and CER are useful content checks, not comprehensive naturalness metrics. ASR can misrecognize unusual names, accents and numbers, and can prefer an unnatural pronunciation that its own model finds easy. Keep raw transcripts and waveform examples. Audit false penalties and disagreements with a second recognizer before interpreting a small score change as a model improvement.

Learned MOS or emotion models should initially be auxiliary metrics. The 2026 UTMOS robustness study documents adversarial vulnerabilities, strengthening the case for checking learned quality scores against independent listening judgments. A rise in one predictor under direct optimization does not by itself demonstrate a more pleasant or appropriate voice.[^26]

Reward hacking tests should include silence, noise, clipped speech, repeated answers, correct text with wrong spoken content, wrong text with correct speech, prematurely stopped audio, extra numeric answers, evaluator-prompt injection in generated text, and failures to terminate. For future LLM judges, treat the response as untrusted content and use a fixed rubric. Calibrate judge reliability on examples outside the training set.

Record individual components as well as the total. Useful diagnostics include reward variance by prompt, all-zero groups, truncation frequency, invalid audio frequency, spoken/text answer disagreement, response duration, semantic accuracy, policy drift and gradient norm by trainable module. A scalar mean alone can hide a collapse that improves a small subset of easy questions.

## Evaluation design

Use three evaluation layers. First, implementation checks verify masks, label alignment, terminal handling, likelihood parity and optimizer behavior. Second, a frozen synthetic evaluation suite measures task success under matched and shifted conditions. Third, external audio benchmarks and listening tests assess broader transfer. Passing the first layer does not imply success on the second or third.

VoiceBench includes both synthetic and human-recorded input subsets and covers several voice-assistant task types. It is a useful external input-side benchmark. Its standard response scoring should be supplemented with actual output-audio evaluation for a speech-to-speech claim. The framework should never silently substitute model-emitted text for what was spoken.[^27]

Select hyperparameters on validation data. Reserve test data for final comparisons. Keep the same prompt identities, reward scorer, sampling controls and evaluation subset across checkpoints. Compare per-prompt differences using paired bootstrap intervals and repeat training over multiple seeds. Report both mean effects and variation between runs; a within-test-set bootstrap does not capture training-seed variability.

Blind listening comparisons should separately rate intelligibility, naturalness, correctness, style appropriateness and preference. Present randomized A/B pairs at matched playback conditions. Do not normalize away meaningful clipping or silence failures without reporting them. Speaker similarity matters only when the task explicitly requires a target voice; it is not automatically a quality objective for a conversational assistant.

The initial evaluator saves generated audio, ASR, reference answers and per-example rewards. Its comparison command requires matching dataset hashes and evaluator settings. It supplies a paired interval over those examples. External benchmarks, human evaluation, multi-rater agreement and rich domain-shift suites are specified research work, not completed integrations.

## Compute and experiment budgeting

Memory depends on audio length, total mixed-sequence length, model precision, adapter placement, scoring models and rollout scheduling. Counting only 1.5B parameter weights is inadequate. A single BF16 copy of 1.5B parameters is about 3 GB in decimal units; training activations, optimizer states, codec components and generation caches are additional. A conventional full-fine-tuning accounting with FP32 master weights and moments can approach 16 bytes per parameter before activations. These are illustrative calculations, not measured allocations for this implementation.

The initial engineering assumption is one BF16-capable NVIDIA GPU, with 24–48 GB considered a planning range and 48 GB offering more experimentation headroom. This is not a fit guarantee or a purchasing recommendation. Start with four serial candidates, short responses, backbone LoRA and CPU ASR. Profile actual peak memory before increasing group size or sequence length. Run the CPU tests on ordinary hardware.

Equal optimizer-step counts do not establish equal compute. GRPO generates several candidates per prompt; offline methods pay an earlier candidate-generation cost; SFT consumes reference audio. Report model updates, unique semantic examples, generated tokens, audio seconds, scorer invocations, GPU-hours and data-generation work. Compare methods at both matched data exposure and matched total compute.

The 81-config matrix is a planning aid, not permission to launch a costly full sweep. Begin with a small pilot, eliminate broken reward/data configurations, then use a limited number of well-controlled comparisons. Larger group sizes can improve relative ranking information but also increase decoding and scorer cost. An all-zero group distribution is a reason to revisit task difficulty or reward density before buying more samples.

## Implementation decisions and remaining gaps

The initial repository uses direct PyTorch objectives around the official audio model rather than assuming a generic text trainer can handle the model's codebook distribution. LoRA adapters preserve an inexpensive frozen-base reference; sampled trajectories are replayed with gradients; online updates are serial to limit simultaneous graphs. All real-model entry points are marked experimental.

Implemented online methods make one update per rollout group. There is no configurable multi-epoch PPO loop, learned value model, distributed engine or dynamic resampling controller. Checkpoint resume restores adapters, optimizer and RNG state within an identical config/data identity. It is not a warm-start API for moving an SFT adapter into a differently configured RL run. Interrupted logs can contain pre-checkpoint attempts, so checkpoint step is the authoritative completed-update count.

The first dataset providers are local development synthesizers. Neural TTS, large-scale text generation, external preference pools, broad conversational judges, cloud launchers and experiment-tracker integrations remain future work. The code's strict validation rejects unimplemented algorithm names and unsupported sampling controls to avoid producing mislabeled runs.

The most important next milestone is a pretrained-checkpoint CUDA integration run: load the pinned artifacts, generate a complete spoken reply, confirm sampled/replayed likelihood agreement, take a finite-gradient LoRA step, save/reload the adapter and compare held-out audio against the base model. Until that happens, CPU results demonstrate infrastructure correctness only.

## Sources

[^1]: Liquid AI. [LFM2.5-Audio-1.5B model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B). Accessed September 10, 2026.
[^2]: Liquid AI. [LFM2AudioModel source](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/src/liquid_audio/model/lfm2_audio.py). Audited pinned source, September 10, 2026.
[^3]: Liquid AI. [Conversation training mapper](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/src/liquid_audio/data/mapper.py). Audited pinned source.
[^4]: Liquid AI. [Trainer implementation](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/src/liquid_audio/trainer.py) and [training example](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/examples/train.py). Audited pinned source.
[^5]: Liquid AI. [Audio processor and ChatState](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/src/liquid_audio/processor.py). Audited pinned source.
[^6]: Shu-wen Yang et al. [ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-aware Speech-to-Speech Interaction](https://arxiv.org/abs/2511.08723). November 2025; ICLR 2026 version also available through OpenReview.
[^7]: Yifu Chen et al. [WavAlign: Enhancing Intelligence and Expressiveness in Spoken Dialogue Models via Adaptive Hybrid Post-Training](https://arxiv.org/html/2604.14932v1). April 2026.
[^8]: Xiang Lin et al. [Towards More Expressive Spoken LLMs: Fine-Grained Intent Benchmarking and Acoustic-Lexical Decoupled Policy Optimization](https://arxiv.org/html/2608.03054v1). August 2026. Primary paper; study limitations discussed in its conclusion.
[^9]: Chang Liu et al. [Group Relative Policy Optimization for Text-to-Speech with Large Language Models](https://arxiv.org/html/2509.18798v1). September 2025.
[^10]: Shashi Kumar et al. [When Synthetic Speech Is All You Have: Better Call GRPO](https://arxiv.org/html/2607.08409v1). July 2026.
[^11]: Jiajun Fan et al. [SpeechGym: An Audio-Native Gym for Training Voice Agents via Reinforcement Learning](https://arxiv.org/html/2608.26432v1). August 2026. Primary paper.
[^12]: Arash Ahmadian et al. [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740). February 2024.
[^13]: Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). February 2024.
[^14]: Zichen Liu et al. [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783). March 2025.
[^15]: Hugging Face. [TRL GRPO trainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer). Main documentation accessed September 10, 2026; evolving implementation, not a pinned dependency of this project.
[^16]: Rafael Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). May 2023; revised July 2024.
[^17]: Mohammad Gheshlaghi Azar et al. [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036). October 2023.
[^18]: Yu Meng et al. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734). May 2024.
[^19]: Kawin Ethayarajh et al. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306). February 2024.
[^20]: John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). July 2017.
[^21]: Qiying Yu et al. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476). March 2025.
[^22]: hexgrad. [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M). Accessed September 10, 2026.
[^23]: Open Home Foundation. [Piper repository](https://github.com/OHF-Voice/piper1-gpl). Accessed September 10, 2026.
[^24]: F5-TTS authors. [Official F5-TTS implementation and license notes](https://github.com/SWivid/F5-TTS). Accessed September 10, 2026.
[^25]: SYSTRAN and contributors. [Faster Whisper](https://github.com/SYSTRAN/faster-whisper). Accessed September 10, 2026.
[^26]: Wen-Chin Huang and Tomoki Toda. [Attacking UTMOS: Probing the Robustness of a Speech Quality Assessment Model](https://arxiv.org/abs/2606.31105). June 2026. Primary paper.
[^27]: VoiceBench authors. [VoiceBench official repository](https://github.com/MatthewCYM/VoiceBench). Accessed September 10, 2026.
