# Dataset format

Use **JSONL files for records** and **WAV files for speech**. Each line of a JSONL file is one example.

Keep each dataset version in its own folder:

```text
data/v1_clean/
  dataset.json
  manifest.jsonl
  audio/
    question-001.wav
    answer-001.wav
```

`manifest.jsonl` lists the examples. `dataset.json` describes the dataset and contains a checksum, which detects changes to the list. The audio files contain the actual speech.

## What each method needs

| Training method | What to save for each question |
|---|---|
| SFT: learn from example answers | Question text and audio, correct answer text and audio |
| GRPO, Dr. GRPO, RLOO, REINFORCE | Question text and audio, plus correct answer text for the current reward function |
| Online RL with the default settings | Also include correct answer audio; the default run mixes RL with supervised examples |
| DPO, IPO, SimPO: learn from preferred answers | Question text and audio, plus a preferred reply and a less preferred reply, each with text and audio |

The first three rows use the same record format below. The preference-pair format is a proposal; its loader and full trainer are not implemented yet.

## A record the current code accepts

This example is shown on several lines so it is easy to read. Save each record on **one line** in `manifest.jsonl`.

```json
{
  "schema_version": 1,
  "id": "example-001",
  "semantic_group": "addition-2-3",
  "split": "train",
  "task": "addition",
  "language": "en",
  "prompt": "What is two plus three? Reply with only the number.",
  "answer": "five",
  "input_audio": "audio/question-001.wav",
  "input_sha256": "REPLACE_WITH_INPUT_FILE_SHA256",
  "answer_audio": "audio/answer-001.wav",
  "answer_sha256": "REPLACE_WITH_ANSWER_FILE_SHA256",
  "provenance": {
    "synthetic": true,
    "tts": "espeak",
    "voice": "en-us",
    "seed": 42
  }
}
```

The two checksum values are placeholders. Replace them with the SHA-256 checksums of the actual WAV files. The dataset generator does this automatically.

| Field | Meaning |
|---|---|
| `schema_version` | Use `1` for this format. |
| `id` | A unique name for this example. |
| `semantic_group` | A shared name for versions of the same underlying question. |
| `split` | `train`, `validation`, or `test`. |
| `task` | The kind of question, such as `addition`. |
| `language` | Use `en`; the current record format supports English. |
| `prompt` | The text spoken in the question audio. |
| `answer` | The correct answer used for teaching or scoring. |
| `input_audio` | Path to the spoken question, relative to the dataset folder. |
| `input_sha256` | Checksum of the question audio file. |
| `answer_audio` | Path to the spoken reference answer, when needed. |
| `answer_sha256` | Checksum of the reference answer audio file. |
| `provenance` | How the example was created: speech engine, voice, seed, and other details. |

The model receives the question audio. `prompt` stores its transcript for inspection. The generated reply is scored against `answer`.

The current reward checks exact answers after basic text normalization. Start with tasks that have clear answers. Open-ended conversations need a different reward function.

### RL without reference answer audio

For online RL without supervised examples, set these dataset fields to `null`:

```json
{
  "answer_audio": null,
  "answer_sha256": null
}
```

This is a fragment of a record, not a complete example. Also set `anchor_weight: 0` in the experiment YAML. The default is `0.1`, so the default run requires reference answer audio.

Keep `answer` in the record: the current reward still needs the correct answer text. The model generates candidate replies during training; you do not prepare those replies in advance.

## Audio files

Use mono WAV files at 24,000 Hz with 16-bit PCM samples as the project default. Use the same format for questions and reference answers.

The current loader also accepts mono audio at other sample rates of at least 8,000 Hz. The model's processor handles the input resampling. You do not need to prepare codec tokens or interleaved text/audio tokens yourself.

Each file should contain one complete question or answer. Keep the spoken words consistent with its text field. Avoid empty, silent, or clipped recordings.

## Dataset metadata

The generator creates `dataset.json` alongside the records:

```json
{
  "schema_version": 1,
  "recipe": {
    "version": "v1_clean",
    "seed": 42,
    "count": 1,
    "tts": "espeak",
    "voice": "en-us",
    "sample_rate": 24000,
    "snr_db": null,
    "synthesize_answers": true
  },
  "manifest_sha256": "REPLACE_WITH_MANIFEST_FILE_SHA256",
  "count": 1,
  "splits": {"train": 1, "validation": 0, "test": 0},
  "has_speech": true
}
```

This describes the one-record example above. For a real dataset, the counts must match its records. Calculate `manifest_sha256` from the final `manifest.jsonl` file, including its line breaks. Recalculate it whenever the file changes.

## Splits and versions

Use `train` to update the model, `validation` to choose settings, and `test` to measure the final result. The generator assigns approximately 80%, 10%, and 10% to these splits. Very small datasets can have empty splits.

Keep different voices, noisy copies, and paraphrases of the same question in the same split. Give them distinct `id` values but the same `semantic_group`. The loader checks this within a dataset; keep the grouping consistent across versions too.

The current recipes are:

| Version | Contents |
|---|---|
| `v1_clean` | Arithmetic and comparison questions with clean speech |
| `v2_noisy` | The same questions and splits, with noise added to question audio |
| `v3_compositional` | Harder questions with two addition steps |

Save changes in a new dataset folder so old experiment results still point to the original files. Put extra details inside `provenance`; the current loader rejects unknown fields at the top level of a record.

## Preference pairs: proposed format

A preference example contains two replies to the **same** spoken question. `chosen` is the preferred reply; `rejected` is the less preferred reply.

```json
{
  "id": "pair-001",
  "semantic_group": "addition-2-3",
  "split": "train",
  "prompt": "What is two plus three? Reply with only the number.",
  "input_audio": "audio/question-001.wav",
  "chosen": {"text": "five", "audio": "audio/chosen-001.wav"},
  "rejected": {"text": "six", "audio": "audio/rejected-001.wav"},
  "preference_reason": "The chosen spoken answer is correct."
}
```

This is a planning example, not a format accepted by the current training command. A full preference dataset will also need file checksums, the model and settings that generated each reply, and how the preference was decided. Retain exact generated token sequences when available for later training support.

## Create and check a dataset

With `espeak-ng` installed, run:

```bash
uv run lfm-rl data --config configs/datasets/v1_clean.yaml --output data/v1_clean
uv run lfm-rl validate-data data/v1_clean --require-audio
```

Use a new output folder. The command creates the audio, records, metadata, and checksums together.

`configs/datasets/text_only.yaml` creates text records for development. It cannot be used to train the speech model until input audio is added.
