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
