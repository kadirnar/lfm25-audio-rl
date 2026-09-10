# Evaluation metric research

Research date: 10 September 2026.

This report defines the evaluation suite for LFM2.5-Audio RL experiments in this repository. It covers answer quality, speech intelligibility, acoustic quality, speaker identity, prosody, speed, and human assessment. It also explains where a score would be misleading.

The implementation is in `src/lfm_audio_rl/metrics/`. See [how to run it](EVALUATION.md) and [optional scorer setup](METRIC_SETUP.md). The suite implements the metrics listed here, not every possible audio metric or every published benchmark protocol.

## 1. What the model needs to be evaluated on

LFM2.5-Audio accepts speech and generates text and audio. Its speech-to-speech path interleaves text and audio tokens. The released model is oriented toward English. Its published evaluation includes VoiceBench, but those benchmark results do not establish the naturalness of this repository's generated audio or the quality of an RL checkpoint. [Model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B), [official inference code](https://github.com/Liquid4All/liquid-audio).

The central distinction is between **answer correctness** and **speech delivery**. A fluent answer can be wrong; a correct text answer can be omitted or changed in speech. Our design therefore scores generated text against the expected answer, recognized output speech against the expected answer, and recognized output speech against the model's own text. We also retain the audio for listening.

For the first RL study, use spoken task success and output-speech WER as primary task measures, then inspect failures, naturalness, and speed separately. For open conversation, use an answer judge and human review instead of exact-match accuracy. This is a proposed study design, not a claim that one metric has been validated as a universal reward for LFM.

## 2. Answer correctness and intelligibility

WER is `(substitutions + deletions + insertions) / reference words`. CER uses characters instead. Minimum-edit alignment finds the error counts. We implement the counts locally, with explicit normalization and empty-reference behavior. Corpus WER divides summed errors by summed reference words; it does not average percentages from short and long utterances equally. Insertions can make WER exceed 100%. JiWER documents the general edit-distance approach and why empty references need an explicit convention. [JiWER documentation](https://jitsi.github.io/jiwer/).

Our empty-reference convention is `null` for a nonempty hypothesis, while retaining its insertion count. Both empty has rate zero. This differs from JiWER 4's per-example numerical convention and is deliberately recorded. Missing audio produces an empty spoken hypothesis, so a failed answer is counted as deletions when the reference is nonempty.

Exact match and token F1 are useful for synthetic arithmetic, recall, and short factual answers. They allow explicit accepted alternatives. They do not solve open-ended factual evaluation. Token F1 measures bag-of-word overlap and can overlook meaning-changing word order or negation. WER against a reference paraphrase is likewise not a reliable correctness score for open conversation.

An independent ASR system estimates the words in generated speech. Its own mistakes affect the scores. Keep the same pinned ASR model and decoding settings for every run, listen to disagreements, and avoid using only the training reward's recognizer as the final judge. Our wrapper is English-only, which matches the initial dataset scope. Input-ASR WER checks the synthetic input recording, not the assistant's internal comprehension.

## 3. Text overlap and semantic similarity

The suite uses SacreBLEU for corpus BLEU, chrF, and TER and saves their signatures. It uses the Google ROUGE implementation for mean ROUGE-L F1 without stemming. These are descriptive reference-overlap scores. BLEU and chrF are on a 0–100 scale; ROUGE-L F1 is 0–1. TER is a percentage and can exceed 100. Higher BLEU/chrF/ROUGE-L and lower TER indicate closer reference overlap. A different correct answer can still score poorly. [SacreBLEU](https://github.com/mjpost/sacrebleu), [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).

BERTScore compares contextual token representations and returns precision, recall, and F1. Our adapter requires a local encoder and an explicit layer count; the setup example uses RoBERTa-large with 17 layers, no IDF, and no baseline rescaling. Record these choices when comparing results. Semantic similarity is not factual verification, and a high score does not guarantee that a spoken answer followed the request. [BERTScore](https://github.com/Tiiiger/bert_score).

Distinct unigram/bigram/trigram ratios and duplicate-response fraction help detect repeated templates or collapsed replies. We compute these over corpus text. They are not quality objectives: random or unnecessarily long answers can appear diverse. Compare them only on matched sample sizes and tasks.

## 4. Instruction following and open-ended judging

IFEval evaluates instructions with verifiable constraints. We implement seven small checks: required phrase, forbidden phrase, prefix, suffix, exact normalized text, minimum word count, and maximum word count. These are useful for synthetic tasks with clear labels. They are **not** the full IFEval instruction inventory or its official strict/loose aggregation. [IFEval paper](https://arxiv.org/abs/2311.07911).

Our OpenRouter rubric rates transcribed answer correctness, relevance, helpfulness, and instruction following from 1 to 5. It also records refusal and unsafe assistance. The prompt treats the example as data, asks for brief reasoning, and permits valid paraphrases. Structured output and local schema validation reject malformed scores. The adapter requires compatible provider parameters and saves the returned provider/model metadata. [OpenRouter structured output documentation](https://openrouter.ai/docs/guides/features/structured-outputs).

LLM judging remains fallible. A reference answer may be wrong, a transcript may lose tone or words, and judges may prefer particular styles. Review a stratified sample of disagreements and compare judge results with human ratings. Keep judge responses cached; remote model names and deterministic-looking settings do not guarantee permanent reproducibility. Use a separate judge from the synthetic-text generator when practical, then measure agreement instead of assuming independence.

Refusal accuracy needs an explicit expected-refusal label. On harmful requests, unsafe-assistance rate describes the judge's classification of the reply. On benign requests, overrefusal rate measures unnecessary refusal. These are generic evaluation fields, not a reproduced safety benchmark or proof of safety.

## 5. Audio validity and basic diagnostics

The default validity check rejects missing, unreadable, non-finite, very quiet, excessively clipped, or out-of-duration output. Defaults require 0.1–120 seconds, mono-averaged RMS at least `1e-4`, peak amplitude at most 1, and at most 1% of samples at or above absolute amplitude 0.999. These are configurable engineering thresholds, not perceptual standards.

We report duration, channels, sample rate, RMS and peak dBFS, DC offset, clipping fraction, and energy-silence statistics. Pauses use 20 ms frames and a default −40 dBFS threshold. This detects low energy; it does not recognize speech. Breathing, background noise, music, and quiet speech can affect the results. A waveform passing these checks can still contain no useful speech.

Integrated LUFS uses pyloudnorm's BS.1770 implementation. Signals shorter than 0.4 seconds or with undefined gated loudness receive `null`. We measure the original signal rather than loudness-normalizing it before evaluation. A loudness difference may matter to listeners, but “louder” is not a general improvement. [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm).

## 6. Predicted naturalness and speech quality

UTMOSv2 targets naturalness assessment of synthetic speech. Our adapter uses `fusion_stage3`, fold 0, seed-42 checkpoint architecture, one prediction repetition, no silence removal, and a fixed local inference seed. It reports `utmos_v2`, not measured human MOS. Its model can be useful as one independent held-out quality predictor, but should be calibrated against listeners for this dataset. [Official UTMOSv2 repository and paper link](https://github.com/sarulab-speech/UTMOSv2).

NISQA v2 predicts overall speech quality and four impairment dimensions. The adapter uses TorchMetrics 1.8.2, whose returned order is overall, noisiness, discontinuity, coloration, and loudness. We checked the source to avoid mislabeling those dimensions. This is the general NISQA model, **not NISQA-TTS**. The latter is a different predictor discussed upstream. [NISQA](https://github.com/gabrielmittag/NISQA), [pinned adapter source](https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/audio/nisqa.py).

DNSMOS P.835 was developed to evaluate noise suppressors. Our non-personalized adapter reports P.808, signal, background, and overall predictions in the library's output order. Its domain is a reason to treat it mainly as an acoustic diagnostic for clean/noisy experiments. A high DNSMOS score says nothing about whether an answer is true. The upstream implementation repeats short clips to its analysis window, which further limits interpretation of very short replies. [DNSMOS P.835 paper](https://arxiv.org/abs/2110.01763), [pinned adapter source](https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/audio/dnsmos.py).

All three predictors have training-domain limits. Do not average them into an alleged universal MOS, clamp their predictions into a more attractive range, or reward-hack them without an independent listening test. Model-file setup and terms are documented separately. The code leaves inapplicable scores empty and stops on unexpected scorer failures by default.

## 7. Reference-based audio metrics

These metrics compare a generated recording with another waveform. They are appropriate for codec reconstruction, controlled degradation, or matched speech targets. They are usually inappropriate for two different valid answers.

SI-SDR measures reconstruction after removing mean and fitting a scale factor. It is insensitive to a global gain change. Our implementation uses a relative numerical floor, giving a maximum of 120 dB for a perfect reconstruction. Ordinary SNR retains gain differences. Both require aligned equal-length signals; silent targets have undefined scores. These metrics are not naturalness measures. [SI-SDR analysis](https://arxiv.org/abs/1811.02508).

STOI and ESTOI estimate intelligibility using a reference. The implementation uses `pystoi`, resamples to 16 kHz, and requires at least one second under this protocol. PESQ uses wideband 16 kHz mode through `pesq`. Higher scores indicate better agreement under the respective algorithms, not better answer reasoning. Short or unsuitable signals can cause native scorer errors. [STOI implementation](https://github.com/mpariente/pystoi), [PESQ wrapper](https://github.com/ludlows/PESQ).

ViSQOL compares perceptual similarity and returns MOS-LQO. We use its speech mode at 16 kHz and an explicitly supplied speech SVR model. It needs a separately built native package. Although ViSQOL has its own alignment machinery, our initial wrapper conservatively requires a declared aligned reference and equal resampled lengths, keeping the comparison contract consistent. [ViSQOL](https://github.com/google/visqol).

MCD compares mel-cepstral coefficients for recordings of the same content. This implementation resamples to 16 kHz, uses 1024-sample Hamming frames every 80 samples, drops frames below RMS `1e-4`, computes 24 coefficients with alpha 0.42, excludes coefficient zero, and applies DTW. The reported value is `10 * sqrt(2) / ln(10)` times mean Euclidean coefficient distance along the path. A `1e-8` periodogram floor handles zero bins. DTW is limited to 16 million cells. Lower is closer. [PySPTK mcep API](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.mcep.html).

MCD results depend strongly on framing, silence policy, coefficient definition, and alignment. Our number is comparable only with the same protocol. Time warping can hide duration errors, so report duration and pause diagnostics separately. A TTS teacher's timbre is not automatically the desired LFM output timbre.

## 8. Speaker identity and prosody

Speaker similarity uses the SpeechBrain ECAPA VoxCeleb encoder, mono audio at 16 kHz, and cosine similarity of embeddings. It requires an explicit target-speaker reference and its checksum. LFM is not assumed to clone the input user's voice. A reference from a TTS engine is also not automatically a speaker target. Higher cosine similarity means closer embeddings, not guaranteed identity or naturalness. [SpeechBrain model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

Prosody uses librosa pYIN at 16 kHz with 1024-sample frames, 160-sample hops, and configurable 50–600 Hz pitch bounds. We report voiced-frame fraction, median F0, interquartile range, and pitch spread in semitones. Aligned references additionally allow pitch RMSE in cents, correlation, and voiced/unvoiced mismatch. Correlation is empty for constant or insufficient jointly voiced pitch. [pYIN API](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html).

Pitch extractors can fail on creaky, breathy, noisy, or otherwise unusual speech. Pitch spread alone is not expressiveness, and flatness is not always undesirable. A different speaker can legitimately have a different pitch range. Use task-specific targets and listening tests for prosody judgments.

## 9. Distribution distance

Fréchet Audio Distance compares distributions of audio embeddings. The original FAD work evaluated music enhancement with its own embedding protocol. We implement the Gaussian mean/covariance distance with **CLAP** embeddings and name the result `fad_clap` to make that choice explicit. It is not numerically interchangeable with original VGGish FAD. [Original FAD paper](https://arxiv.org/abs/1812.08466), [selected CLAP model](https://huggingface.co/laion/clap-htsat-unfused).

Our adapter uses fixed ten-second chunks at 48 kHz, averages chunk embeddings per utterance, then compares full covariance estimates for the reference and generated sets. It needs at least two valid paired recordings; that is a computational minimum, not a recommended scientific sample size. Covariance estimates are unstable with small samples, and FAD is biased by sample count.

Use the same reference distribution and equal dataset selections across runs. Report the included count and coverage. A low distance to a synthetic teacher distribution does not establish correct answers, naturalness, or generalization to human speech. There is no per-utterance FAD and no paired FAD confidence interval in the current command.

## 10. Speed and resource use

Useful latency endpoints are first text, first audio token, first playable audio, and complete decoded audio. They must not be substituted for each other. LFM's present evaluation runner measures the first two and complete audio; it decodes the full sequence afterward. It cannot establish streaming time to first playable audio from a codec-token timestamp.

The runner includes prompt preparation and synchronizes CUDA around measured stages. It excludes model loading, ASR, and file writing, and records warmup count and environment. Real-time factor is complete-audio latency divided by output duration. Peak allocated CUDA bytes is a process allocation diagnostic, not total device memory consumption. These choices define our instrumentation, not a published LFM speed result.

Externally supplied playable-chunk traces support stall counts, stall duration, first playable chunk, and arrival-gap statistics. The calculation assumes immediate playback from the first chunk without extra buffering. Full-duplex response timing, interruption recovery, concurrency, network delay, and server queueing need additional serving instrumentation and are not measured by this offline runner.

## 11. Human evaluation and experimental design

Human listening remains necessary for naturalness, expressiveness, intelligibility, and overall preference. P.800 addresses subjective transmission-quality assessment, while P.808 addresses crowdsourced speech-quality assessment. Consult the actual protocols when designing a formal study. Our code only validates and summarizes rating records; it does not implement all requirements of either recommendation. [ITU-T P.800](https://www.itu.int/rec/T-REC-P.800), [ITU-T P.808](https://www.itu.int/rec/T-REC-P.808).

We support separate 1–5 dimensions and A/B/tie preferences. Each item gets equal weight after averaging its listener ratings. Use randomized blinded presentation, a consistent task description, and suitable listener checks. For speaker similarity, supply the intended voice reference to listeners. For helpfulness, include the user's request. Keep human ratings distinct from predicted MOS in tables and filenames.

Confidence intervals in this repository resample semantic groups, keeping TTS renditions of one question together. This addresses correlated variants; it does not remove judge bias or account for every listener effect. Paired comparisons use the same examples. Missing-score coverage is checked, failure rates stay visible, and intervals over a single semantic group remain empty.

Separate model selection from final testing. Choose metrics and thresholds before examining the final test split. Compare multiple training seeds and include a real-speech holdout before claiming transfer beyond synthetic data. The current synthetic split groups question meaning, not necessarily speaker identity; a speaker-disjoint study needs a separately constructed split.

## 12. RL diagnostics and benchmark boundaries

The helper module computes teacher-forced target NLL, bits per token, and perplexity with a target mask. Text and audio tokenizations need separate reporting. Self-sampled action likelihood is not held-out target perplexity. The pass@k helper uses the estimator `1 - C(n-c, k) / C(n, k)` for independent attempts at an identical prompt. This formula originated in code-model evaluation; its use here requires a fixed speech-task correctness test. [Pass@k source](https://arxiv.org/abs/2107.03374).

Within-prompt reward standard deviation and zero-variance-group fraction help diagnose whether RL candidate groups supply a learning signal. They do not rank perceptual speech quality and are not added to the reward automatically.

VoiceBench covers spoken instruction understanding under content, speaker, and environment variation, using task-specific evaluators. It is a useful external evaluation target. This repository's generic answer judge, simple constraints, and acoustic metrics do not reproduce its complete official protocol. Run the upstream data, prompts, evaluator, and scorer settings to report a VoiceBench result. The same restriction applies to IFEval, AudioBench, and safety benchmarks. [VoiceBench paper](https://arxiv.org/abs/2410.17196), [official evaluation code](https://github.com/MatthewCYM/VoiceBench).

No pretrained LFM quality result, learned-scorer score, paid judge result, or human listening score was produced while implementing this suite. Numerical signal tests, mocked scorer contracts, small random upstream-model tests, and report/CLI tests establish software behavior only. GPU pretrained validation and a listening study remain the next experimental steps.
