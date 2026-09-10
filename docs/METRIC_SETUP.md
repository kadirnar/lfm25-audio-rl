# Set up optional scorers

The default evaluation config needs no scorer weights. Add optional models only for the measurements you plan to use. These commands are setup instructions; the repository does not include their weights.

Use Python 3.12. Run commands from the repository folder. Keep assets in `work/scorers/`, which is ignored by Git.

## Packages

```bash
uv sync --locked --extra metrics --extra metrics-models --extra dev
```

This installs the Python adapters for BERTScore, SpeechBrain, CLAP, DNSMOS, and NISQA. It does not download their pretrained weights. TorchMetrics is pinned to 1.8.2 because its scorer APIs and output order were checked against that version.

For native PESQ and MCD support:

```bash
uv sync --locked --extra metrics --extra metrics-models --extra metrics-native --extra dev
```

PESQ and PySPTK may need a C/C++ compiler. The native extra pins setuptools below 81 because PySPTK 1.0.1 imports `pkg_resources`. On Linux, install your distribution's compiler tools first. On macOS, use the Xcode command-line tools. These native metrics were exercised on macOS with local test signals.

For LFM generation, also include `--extra model --extra asr`. Check that PyTorch and Torchaudio match the intended CUDA environment before a pretrained run.

## BERTScore, speaker similarity, and CLAP

The following revisions were checked on 10 September 2026. Download only the scorers you want. This script downloads files; it does not run inference.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "speechbrain/spkrec-ecapa-voxceleb",
    revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
    local_dir="work/scorers/ecapa",
)
snapshot_download(
    "FacebookAI/roberta-large",
    revision="722cf37b1afa9454edce342e7895e588b6ff1d59",
    local_dir="work/scorers/roberta-large",
)
snapshot_download(
    "laion/clap-htsat-unfused",
    revision="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
    local_dir="work/scorers/clap",
)
```

The config fields are:

```yaml
speaker_model:
  path: work/scorers/ecapa
bert_model:
  path: work/scorers/roberta-large
  layers: 17
clap_model:
  path: work/scorers/clap
```

Select `speaker`, `bertscore`, or `fad` in `groups` to use them. A speaker model alone is not enough: each scored example also needs an explicitly requested target-speaker reference. See the [evaluation guide](EVALUATION.md).

Sources: [ECAPA model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large), [CLAP model](https://huggingface.co/laion/clap-htsat-unfused).

## DNSMOS

Download the two non-personalized ONNX models from the pinned [Microsoft DNS-Challenge source](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS).

Put them in this exact layout:

```text
work/scorers/dnsmos/
  DNSMOS/
    model_v8.onnx
    sig_bak_ovr.onnx
```

Use `dnsmos_weights_dir: work/scorers/dnsmos` and add `dnsmos` to `groups`. The adapter uses CPU ONNX inference with one thread. Its output order is P.808, signal, background, overall.

The wrapper checks that both files exist before calling TorchMetrics. Upstream code has download fallbacks for unusable assets; validate the ONNX files and block network access at the environment level when an offline run is required.

## NISQA

Obtain `weights/nisqa.tar` from the pinned [NISQA repository](https://github.com/gabrielmittag/NISQA/tree/fe84f0f252abec382b24367d5b22498a7ce34dbb/weights). If cloning through Git LFS, materialize the binary instead of leaving a pointer file.

Store it as `work/scorers/nisqa/nisqa.tar`. Set `nisqa_weights_dir: work/scorers/nisqa` and add `nisqa` to `groups`.

This adapter uses the general NISQA model through TorchMetrics. It does not load the separate NISQA-TTS model. Check NISQA's model terms before using or distributing the weights; its pretrained models are listed under CC-BY-NC-SA 4.0 upstream.

## UTMOSv2

UTMOSv2 is installed separately because its source and model dependencies need a dedicated compatibility check. Start with the metrics extras above, then install the reviewed source revision:

```bash
uv pip install --python .venv/bin/python \
  'git+https://github.com/sarulab-speech/UTMOSv2.git@cc2700db57bb83ee13dc31ebe1b868c254e15d09'
```

Download the fold-0 checkpoint:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    "sarulab-speech/UTMOSv2",
    "fold0_s42_best_model.pth",
    revision="506474f2b33dc77c234d668cc419be1861899cad",
    local_dir="work/scorers/utmos",
)
```

Use `utmos_checkpoint: work/scorers/utmos/fold0_s42_best_model.pth` and add `utmos` to `groups`. The adapter fixes the architecture to `fusion_stage3`, fold 0, checkpoint seed 42. Inference uses one repetition and keeps silence. It fixes the inference random seed without replacing the caller's random state.

The upstream constructor also initializes an EfficientNet image backbone and `facebook/wav2vec2-base`. It can download auxiliary files even when the final checkpoint is local. Prepare these caches before an offline run. The report hashes the configured final checkpoint and records installed package/source versions; it does not inventory every auxiliary cache file. Preserve that environment for a controlled study.

Run scoring with `.venv/bin/lfm-rl` after the manual install. A later `uv sync` can remove manually installed packages or change the environment. Preserve the resolved environment with `uv pip freeze --python .venv/bin/python` in the run records. The complete pretrained UTMOS environment has not been validated here.

Sources: [reviewed UTMOSv2 source](https://github.com/sarulab-speech/UTMOSv2/tree/cc2700db57bb83ee13dc31ebe1b868c254e15d09), [weights](https://huggingface.co/sarulab-speech/UTMOSv2).

## ViSQOL

Build the Python package using the instructions in the [reviewed ViSQOL repository](https://github.com/google/visqol/tree/38d0b0163e441047d4429bf07ad09e5b9031d02c). This is a native Bazel build and is not part of the uv extras. Install the resulting package into the scoring environment.

Copy that revision's `model/libsvm_nu_svr_model.txt` to `work/scorers/visqol/libsvm_nu_svr_model.txt`. Set `visqol_model` to this path. Select the `reference` group and include `visqol` in `reference_metrics`.

The adapter uses speech mode, 16 kHz, and the explicit SVR file. Its Python API contract is tested with a fake native object; a real local ViSQOL build has not been run. Keep the build revision alongside the report. The report hashes the SVR file, but does not prove the native extension came from that revision.

## Check the selected setup

Copy `configs/metrics/full.yaml`, remove unwanted groups, and change the asset paths. Paths are resolved from the working directory, not from the YAML file's directory. Keep the scorer files unchanged between base and RL evaluations.

Start with a small validation subset:

```bash
.venv/bin/lfm-rl score --config configs/metrics/my-study.yaml \
  --data data/v1_clean --predictions runs/base/predictions.jsonl \
  --output runs/scorer-check --split validation --limit 8
```

Use predictions for that selection. Missing selected predictions are recorded as failures; they are not generated by `score`. Eight cases are a setup check, not a reliable quality study.

Selected local assets are hashed before scoring. No model score is replaced with a dummy value. By default a load or inference error stops the run with its cause. For a larger run, inspect `coverage`, `missing`, generation failures, and a sample of saved audio before interpreting mean scores.

## Validation boundary

The repository tests use real text-metric libraries, waveform calculations, pYIN, STOI, ESTOI, PESQ, and MCD on test signals. They use mocked API contracts for learned scorers, ViSQOL, and OpenRouter. No learned metric weights or LFM pretrained weights were downloaded for those tests, and no paid judge requests were sent.

The [status page](STATUS.md) records the current test results. Installation of all optional pretrained scorers together, GPU runtime behavior, prediction quality, and human agreement still need validation on the intended machine.
