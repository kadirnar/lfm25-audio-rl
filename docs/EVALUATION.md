# Evaluate speech-to-speech answers

Use the same held-out questions to compare the base model and each RL checkpoint. Keep the ASR model, metric settings, and hardware fixed.

The evaluation code checks what the model said, how its speech sounds, and how long it took. It saves each score separately. There is no combined score that hides a bad result in one area.

For the reasoning and sources, read [metric research](METRIC_RESEARCH.md). For optional model downloads, read [scorer setup](METRIC_SETUP.md).

## Start with the default metrics

Run commands from the repository folder:

```bash
uv sync --locked --extra metrics --extra dev
uv run lfm-rl score --config configs/metrics/default.yaml \
  --data data/v1_clean \
  --predictions runs/base/predictions.jsonl \
  --output runs/base-scores --split test
```

This command scores saved predictions. The default config makes no API calls and loads no pretrained scorer. It uses the ASR transcripts saved in the predictions file. It checks text, audio, simple instructions, and supplied timing.

Set `asr_model` to a local Faster Whisper model directory to transcribe the output audio again. Install the `asr` extra first. A name such as `large-v3` can download weights; a pinned local directory gives a more controlled comparison. The current ASR wrapper uses English, beam size 1, and no VAD or previous-text conditioning.

## Generate predictions with LFM

This step needs a BF16-capable CUDA GPU. It loads the real LFM model and ASR scorer.

```bash
uv sync --locked --extra model --extra asr --extra metrics
uv run lfm-rl evaluate --config configs/experiments/grpo_text.yaml \
  --data data/v1_clean --output runs/base --split test \
  --metrics-config configs/metrics/default.yaml --warmup 1
```

For an adapted model, add `--checkpoint runs/grpo-v1/checkpoint.pt` and use a new output folder. Use the training config and dataset that belong to that checkpoint.

The command writes generated WAV files, `predictions.jsonl`, the older reward report, and `metrics/metrics.json`. Open-ended answers can be evaluated, but the older exact-answer reward is left empty for them. The online RL trainer still needs an appropriate training reward; the evaluation judge is not connected to training.

## Prediction format

Use JSONL: one object per line. This is one example shown with line breaks for readability:

```json
{
  "example_id": "the-id-from-the-dataset",
  "generation": {
    "model_id": "LiquidAI/LFM2.5-Audio-1.5B",
    "model_revision": "the-pinned-revision",
    "adapter_sha256": "base"
  },
  "text": "The cup is red.",
  "asr": "the cup is red",
  "asr_model": "my-fixed-ASR-version",
  "audio": "audio/reply.wav",
  "terminated": true,
  "timing": {
    "generation_seconds": 1.6,
    "decode_seconds": 0.2,
    "audio_ready_seconds": 1.8,
    "first_audio_token_seconds": 0.4,
    "environment": {"gpu": "record the GPU name", "precision": "bf16"}
  }
}
```

`example_id` is required. Audio paths are relative to the predictions file and must stay inside that folder. Add `audio_sha256` when exporting your own predictions; the scorer also hashes every audio file itself. `generation` is optional metadata with string values. LFM evaluation fills in the model revision, adapter hash, seed, and sampling settings.

`text` means the model's generated text. `asr` means words recognized from its generated audio. These can differ. Use `null` when a transcript is unavailable, and `""` when recognition actually returned no words. A supplied transcript is trusted as data; use a fixed independent ASR model for real comparisons.

Optional `input_asr` contains an independently recognized input question. It is compared with the dataset's prompt. It is not a measure of the model's internal understanding.

Use `error` for a generation failure, such as `"out_of_memory"`. Missing predictions count as failed examples. A broken or silent waveform cannot receive spoken-answer credit through a supplied transcript. A valid waveform without ASR has unknown content scores, with reduced coverage. Omitted `terminated` means termination is unknown; it does not prove a complete reply.

## Tell the scorer what references mean

The existing dataset format stays the same. Add evaluation settings inside an example's `provenance`:

```json
{
  "evaluation": {
    "answer_mode": "closed",
    "accepted_answers": ["crimson"],
    "reference_relation": "none",
    "constraints": [
      {"kind": "contains", "value": "red"},
      {"kind": "max_words", "value": 10}
    ]
  }
}
```

This object belongs inside `provenance`, alongside the existing generator metadata. Changing the manifest also requires updating its checksum in `dataset.json`; publish a new dataset version instead of editing a recorded experiment's data.

Use `closed` for a question with a known short answer. Accepted alternatives affect exact match and token F1. WER and CER use the main answer only. Use `open` for a conversation with many valid answers. Dialogue and synthetic examples marked `reward_protocol: open_ended` default to this mode. Exact-answer accuracy is skipped for open answers.

Reference relations have three meanings:

| Value | Meaning |
|---|---|
| `none` | Default. The answer recording is not a waveform target. |
| `same_content` | Both recordings speak the same words, but timing can differ. Enables MCD with time alignment. |
| `aligned` | Same speech content with matched timing. Enables waveform and aligned pitch comparisons. |

Another TTS engine reading an answer is usually a `same_content` reference at most. A correct paraphrase needs `none`. PESQ, STOI, SI-SDR, SNR, and ViSQOL require declared alignment and equal lengths after resampling in this implementation. The code does not silently crop mismatched signals.

For speaker similarity, add `speaker_reference_audio` and `speaker_reference_sha256` to the evaluation object. This recording must be the requested **output speaker** and live inside the dataset folder. The scorer never assumes that the user's input voice is the desired output voice.

For refusal studies, add `expected_refusal: true` for requests that should be refused, or `false` for benign requests. The judge then reports refusal accuracy and the relevant unsafe-assistance or overrefusal rate. Have people review these labels and judgments.

## Metric groups

Choose groups in a YAML config. Copy `configs/metrics/full.yaml` to start an optional setup and remove groups you do not need.

| Group | Scores |
|---|---|
| `text` | Exact match, token F1, WER, CER; corpus BLEU, chrF, TER, ROUGE-L; text diversity |
| `audio` | Validity, duration, sample rate, channels, loudness, clipping, energy silence and pauses |
| `instructions` | Contains, excludes, starts with, ends with, exact text, minimum and maximum word counts |
| `timing` | Measured latency, real-time factor, memory, speech rate, and optional stream stalls |
| `prosody` | Voiced fraction, pitch median, pitch spread, and eligible reference pitch errors |
| `reference` | SI-SDR, SNR, STOI, ESTOI, PESQ, ViSQOL, MCD |
| `speaker` | ECAPA speaker-embedding cosine similarity |
| `bertscore` | BERTScore precision, recall, and F1 on spoken transcripts |
| `dnsmos` | P.808, speech signal, background, and overall quality predictions |
| `nisqa` | Overall quality, noisiness, discontinuity, coloration, and loudness predictions |
| `utmos` | UTMOSv2 naturalness prediction |
| `fad` | Corpus Fréchet distance using CLAP audio embeddings |
| `judge` | OpenRouter correctness, relevance, helpfulness, instruction following, refusal, and unsafe assistance |

`valid_audio`, `generation_failure`, and `truncated` are always reported. Learned quality scores need valid generated audio. Missing references or transcripts produce `null` with a reason. Missing model files fail before scoring. By default, dependency and scorer errors stop the run. Set `fail_on_metric_error: false` to record optional scorer failures instead; inspect coverage before comparing scores. Judge request, budget, and malformed-response errors always stop the run to avoid silently changing the judged sample.

## OpenRouter judge

Use a model and provider that support structured JSON output. Put the API key in `OPENROUTER_API_KEY`, not in a YAML file. A judge config can look like this:

```yaml
groups: [text, audio, judge]
judge:
  model: qwen/qwen3.5-35b-a3b
  max_requests: 100
  max_attempts: 3
  max_tokens: 1024
```

The judge receives the question, reference answer, spoken transcript, and constraints. It does not receive audio and cannot score voice naturalness. Each of the four answer ratings ranges from 1 to 5. Refusal and unsafe assistance are separate boolean judgments.

Requests and responses are cached. `max_requests` counts attempts, including retries and interrupted requests. It is a request ceiling, not a dollar budget. Returned model, provider, usage, rubric version, and explanation are recorded. You can set `judge.providers` to fix an ordered provider list and disable fallback. A seed and temperature zero do not guarantee identical remote responses, so preserve the cached responses.

## Timing

The LFM runner measures input preparation through generation and final audio decoding, after the configured warmup. It synchronizes CUDA around measured stages. ASR and file writing are excluded. Model loading is also excluded. It records GPU name, precision, package versions, and peak allocated CUDA memory.

The current runner decodes the full response after token generation. It reports `first_audio_token_seconds` and `audio_ready_seconds`. It does **not** report time to first playable audio. The first field measures codec-token availability, not sound that can be played.

If another serving system measures actual playable output, supply `first_audio_seconds`. For streaming, supply `chunks` as pairs of `[arrival_seconds, playable_duration_seconds]`. The scorer computes gaps and stalls assuming playback starts with the first chunk and runs at normal speed. It does not model a network or player by itself. Record the clock origin and environment consistently.

Real-time factor is `audio_ready_seconds / audio_seconds`; lower is faster. A value below 1 means complete generation took less time than playback. This alone does not prove smooth streaming. Summaries include p50, p95, and p99 for durations and memory. Tail estimates from a few examples are unstable.

## Read and compare reports

`metrics.json` contains settings and hashes, per-example results, overall summaries, task/TTS-profile/language/split slices, and corpus metrics. Each summary gives the value, sample count, semantic-group count, coverage, missing count, and a 95% bootstrap interval when at least two groups exist.

For example, `spoken_answer_corpus_wer` divides all word errors by all reference words. `spoken_answer_wer` averages example-level error rates. Prefer the corpus rate when describing WER. WER may exceed 1 because of insertions. Empty references retain insertion counts; the per-example rate is `null` if the hypothesis is nonempty. Both empty gives zero.

Text channels have different purposes:

| Prefix | Comparison |
|---|---|
| `text_answer` | Reference answer versus generated text |
| `spoken_answer` | Reference answer versus recognized output speech |
| `tts` | Generated text versus recognized output speech |
| `input_asr` | Prompt text versus recognized input speech |

The default normalization uses Unicode NFKC, lowercase, punctuation removal, and whitespace cleanup. `verbatim` preserves case and punctuation. `dataset` uses the synthetic generator's normalization, including its small-number spelling. Corpus overlap tools use their own recorded tokenization signatures, independently of this option.

```bash
uv run lfm-rl compare-metrics \
  runs/base-scores/metrics.json runs/rl-scores/metrics.json
```

Differences are **after minus before**. A negative WER difference is an improvement; a positive exact-match difference is an improvement. Pitch, loudness, and duration are diagnostics with no universal best direction.

Comparisons require the same dataset hash, example IDs, and scoring protocol. Changed model metadata is allowed. Changed scoring assets, ASR sources, or measured hardware are not. Unequal score coverage is rejected by default. `--allow-partial` uses only shared scored pairs and reports omissions. It can hide difficult failures, so report failures separately.

Bootstrap sampling keeps all versions of a semantic example together. It does not treat four TTS renditions as four independent questions. Corpus WER/CER comparisons use paired count totals, including different generated-text lengths for TTS WER. BLEU, chrF, TER, ROUGE-L, diversity, and FAD are corpus outputs and do not receive paired intervals from this command.

Repeat a scoring command to resume it. Dataset, prediction, waveform, local scorer, code, and configuration changes require a new output folder. A crash may leave `.lock`; remove it only after checking that its recorded process has stopped. Per-example receipts and the final report are checksummed. Do not run multiple workers against one output folder.

## Human ratings

Collect real listening ratings in JSONL. This is a format example, not measured evidence:

```json
{"example_id":"example-1","rater_id":"listener-1","dimension":"naturalness","score":4}
```

Dimensions are `naturalness`, `intelligibility`, `speaker_similarity`, `helpfulness`, and `prosody`. Scores are integers from 1 to 5. Use separate files for each model and a clear listening rubric. Randomize presentation order and hide model names during collection.

```bash
uv run lfm-rl score-human --ratings ratings/base.jsonl \
  --metrics-report runs/base-scores/metrics.json --output ratings/base-summary.json
```

For paired listening, use `winner: "A"`, `"B"`, or `"tie"` and pass `--kind preference`. Keep a record of which model A and B mean. Ties contribute half a vote to B. The summary averages ratings within each example before averaging examples, so extra listeners on one example do not give it extra weight. Duplicate ratings from the same rater for the same item and dimension are rejected.

The intervals resample examples or their semantic groups, conditional on the collected listeners. They do not estimate uncertainty over a new population of listeners. This code summarizes ratings; it does not provide a listening website or certify an ITU-compliant experiment.

## RL study helpers

`lfm_audio_rl.metrics.diagnostics` provides `target_likelihood`, `pass_at_k`, and `reward_groups` for research scripts. These are separate from the speech report.

Feed `target_likelihood` teacher-forced held-out target-token log probabilities at temperature 1, with a boolean target mask. Call it separately for text and audio. It returns NLL, bits per token, and perplexity. Comparing perplexity across different tokenizers, action scopes, or codebook reductions is not valid.

`pass_at_k(n, correct, k)` needs independent policy attempts at the same question and a fixed correctness rule. `reward_groups` summarizes a prompt-by-candidate reward matrix and flags zero-variance groups. Neither function proves that a model produces natural speech.
