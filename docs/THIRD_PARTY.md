# Third-party components

This repository calls separately installed packages and does not redistribute their source or weights.

| Component | Terms to consult | Scope |
|---|---|---|
| LFM2.5-Audio and liquid-audio | [LFM Open License v1.0](https://github.com/Liquid4All/liquid-audio/blob/19e65845923a7f136442c95137884ec61eb386aa/LICENSE) | Model/package terms are separate from this repository's MIT license. Section 5 addresses commercial use and a revenue threshold. |
| Canary/NeMo and Mimi components | [Upstream model card attribution](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B#license) | Upstream distinguishes code licenses from checkpoint licenses, including CC-BY-4.0 weights. Preserve applicable notices in redistributed artifacts. |
| eSpeak NG | [Project license](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING) | Optional locally installed development synthesizer. |
| macOS say | Installed operating-system and voice terms | Local preview provider; generated previews are not included in the repository. |
| Faster Whisper | [Project license](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) | Optional ASR scoring implementation; consult checkpoint terms independently. |

For each future neural TTS engine, record the engine license, exact checkpoint, voice provenance, output terms, and whether synthetic outputs may be redistributed. Do this when selecting the actual provider; an engine's code license does not establish the rights of every available voice or weight file. Model downloads are performed by upstream libraries only when a real-model command is run.

## Synthetic speech workers

Adapters follow the examples in [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), [Echo-TTS](https://github.com/jordandare/echo-tts), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), and [Zonos](https://github.com/Zyphra/Zonos). Model and source revisions are recorded in `src/lfm_audio_rl/synthetic/config.py`. The repositories and weights are downloaded separately, not included here.

Echo's repository is primarily MIT code, with some Apache-2.0 files. Its weights and generated audio are CC-BY-NC-SA-4.0 according to the upstream license statement, including through the Fish codec. The pipeline preserves that declaration. The other selected main checkpoints declare Apache-2.0; this does not establish blanket licensing for generated datasets, reference recordings, or LLM outputs. Preserve the applicable terms for every source.

## Evaluation scorers

The metric code calls these projects through optional adapters. It does not bundle their source, binaries, or weights. Package, checkpoint, dataset, and API terms can differ.

| Component | Upstream source or terms | Use here |
|---|---|---|
| SacreBLEU | [Repository](https://github.com/mjpost/sacrebleu) | BLEU, chrF, TER |
| ROUGE | [Google Research](https://github.com/google-research/google-research/tree/master/rouge) | ROUGE-L |
| BERTScore and RoBERTa | [BERTScore](https://github.com/Tiiiger/bert_score), [model card](https://huggingface.co/FacebookAI/roberta-large) | Semantic text similarity |
| SpeechBrain ECAPA | [Model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Requested target-speaker similarity |
| CLAP | [Model card](https://huggingface.co/laion/clap-htsat-unfused) | Audio distribution embeddings |
| UTMOSv2 | [Code](https://github.com/sarulab-speech/UTMOSv2), [weights](https://huggingface.co/sarulab-speech/UTMOSv2) | Predicted naturalness; auxiliary backbones have their own terms |
| NISQA | [Repository license section](https://github.com/gabrielmittag/NISQA#license) | General speech-quality model; upstream lists MIT code and CC-BY-NC-SA 4.0 pretrained models |
| DNSMOS | [DNS-Challenge](https://github.com/microsoft/DNS-Challenge) | Non-personalized acoustic quality predictions |
| TorchMetrics | [Repository](https://github.com/Lightning-AI/torchmetrics) | Pinned DNSMOS/NISQA Python adapters; scorer-model terms remain separate |
| PESQ | [Wrapper](https://github.com/ludlows/PESQ) | Native reference scorer; inspect its bundled implementation and applicable PESQ terms |
| STOI/ESTOI | [pystoi](https://github.com/mpariente/pystoi) | Reference intelligibility |
| ViSQOL | [Repository](https://github.com/google/visqol) | Separately built reference-quality scorer |
| librosa, PySPTK, pyloudnorm | [librosa](https://github.com/librosa/librosa), [PySPTK](https://github.com/r9y9/pysptk), [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | Pitch, mel cepstra, and loudness |
| OpenRouter and selected provider | [OpenRouter documentation](https://openrouter.ai/docs) | Optional paid text judging under the selected service/model terms |

See [scorer setup](METRIC_SETUP.md) for reviewed source and checkpoint revisions. No trained scorer, benchmark dataset, native binary, or human rating data is redistributed in this change.

## Training tracking

Optional W&B logging uses the separately installed [Weights & Biases SDK](https://github.com/wandb/wandb). Online logging uses the selected W&B account and service terms. The code does not upload model checkpoints or speech files. Offline SDK tests do not contact a W&B project.
