# Set up the TTS workers

Use separate Python environments for the four models. The commands below run from the repository root on Linux. They install upstream projects at the code revisions used to write the adapters.

These setup instructions have been checked against the upstream code and dependency files. They still need a complete install and synthesis test on a CUDA machine. Driver and system-library compatibility depend on your machine.

Install `git`, `uv`, `ffmpeg`, `espeak-ng`, and `sox` first. Use Python 3.12. Store the environments outside the source checkouts so the checkout stays clean.

```bash
mkdir -p work/tts-sources work/tts-envs
```

## MOSS-TTS

This adapter uses **MOSS-TTS v1.5** and its audio tokenizer. It supports direct speech and a reference voice. It does not select MOSS-TTSD, Local, Nano, or Realtime. The [official example](https://github.com/OpenMOSS/MOSS-TTS) uses a processor conversation, model generation, and processor audio decoding.

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git work/tts-sources/MOSS-TTS
git -C work/tts-sources/MOSS-TTS checkout 934d6826b084c46a0d033402174d5f8ac4ed2519
uv venv --python 3.12 work/tts-envs/moss
uv pip install --python work/tts-envs/moss/bin/python \
  torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python work/tts-envs/moss/bin/python \
  -e 'work/tts-sources/MOSS-TTS[torch-runtime]' soundfile
```

The worker loads pinned local snapshots for both the model and codec. MOSS uses Hugging Face remote model code from those snapshots.

## Echo-TTS

This is **jordandare/echo-tts**, using `jordand/echo-tts-base` and `jordand/fish-s1-dac-min`. It supports direct speech and a reference voice. The [official API](https://github.com/jordandare/echo-tts) uses `sample_pipeline`.

```bash
git clone https://github.com/jordandare/echo-tts.git work/tts-sources/echo-tts
git -C work/tts-sources/echo-tts checkout 2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e
uv venv --python 3.12 work/tts-envs/echo
uv pip install --python work/tts-envs/echo/bin/python \
  torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python work/tts-envs/echo/bin/python \
  -r work/tts-sources/echo-tts/requirements.txt soundfile
```

The worker pins the model, PCA state, and Fish codec downloads. It keeps Echo's 640-latent setting. Reducing that setting can truncate the text; it is not a simple speed control. Use short turns. `echo_steps`, `echo_cfg_text`, and `echo_cfg_speaker` control its sampler.

Echo's stated audio-output license is CC-BY-NC-SA-4.0. The pipeline records it in the dataset.

## Qwen3-TTS

The default is **Qwen3-TTS 12Hz 1.7B CustomVoice**, with the English preset `Ryan`. The adapter also supports the **1.7B Base** model for voice cloning. See the [official Python examples](https://github.com/QwenLM/Qwen3-TTS).

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git work/tts-sources/Qwen3-TTS
git -C work/tts-sources/Qwen3-TTS checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e
uv venv --python 3.12 work/tts-envs/qwen
uv pip install --python work/tts-envs/qwen/bin/python \
  torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python work/tts-envs/qwen/bin/python \
  -e work/tts-sources/Qwen3-TTS
```

Use `speaker` to change the preset and `instruct` for style instructions. The VoiceDesign and 0.6B variants are not connected in this first adapter. For cloning, add these fields to the Qwen engine profile:

```yaml
qwen_mode: voice_clone
reference_audio: /absolute/path/to/your-reference.wav
reference_text: The exact words spoken in that recording.
voice_id: speaker-a
voice_terms: Describe the permission or license for this recording.
```

This selects the pinned Base checkpoint automatically and caches the reference prompt in memory. Style instructions apply to CustomVoice, not cloning.

## Zonos

This adapter uses **Zonos v0.1 Transformer**, with optional reference voice conditioning. It does not load the Hybrid model or ZONOS2. The [official API](https://github.com/Zyphra/Zonos) builds conditioning, generates codes, and decodes them with DAC.

```bash
git clone https://github.com/Zyphra/Zonos.git work/tts-sources/Zonos
git -C work/tts-sources/Zonos checkout bc40d98e1e1ab54fc65c483be127a90e3c7c0645
uv venv --python 3.12 work/tts-envs/zonos
uv pip install --python work/tts-envs/zonos/bin/python \
  torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python work/tts-envs/zonos/bin/python \
  -e work/tts-sources/Zonos 'transformers==4.48.1'
```

The worker pins the main weights, DAC, and speaker encoder. `zonos_cfg_scale` controls guidance. The first adapter disables compilation for a simpler initial run.

## Add voices

For MOSS, Echo, or Zonos, add `reference_audio`, `voice_id`, and `voice_terms` to the engine profile. Use a clear recording whose terms permit your intended use. Qwen cloning also needs `reference_text` as shown above.

Without a reference, MOSS, Echo, and Zonos choose an unconditioned voice. The same profile name does not guarantee the same speaker across clips. Use a reference or a Qwen preset when speaker consistency matters.

To try several voices, repeat the engine profile with a new `name` and reference. Each profile produces its own dataset rows. The current pipeline uses the profile's voice settings for both the question and answer.

`max_new_tokens` applies to MOSS, Qwen, and Zonos. Echo uses its own sampler settings. Unsupported controls cause a config error instead of being silently ignored.

## Check a small run

Use the [pipeline guide](SYNTHETIC_PIPELINE.md) to generate two text examples, then render them with one engine. Check the worker log, listen to both question and answer files, and inspect ASR results and rejection reasons. Repeat for each engine before a larger run.

The worker downloads weights on its first run. Source commits and model revisions are listed in `src/lfm_audio_rl/synthetic/config.py`. Runtime package versions are saved with every speech dataset. Preserve those versions for later comparisons; these setup commands do not lock every transitive dependency.
