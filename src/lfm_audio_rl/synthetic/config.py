from typing import Literal

from pydantic import Field, model_validator

from ..config import StrictModel

# Audited on 2026-09-10. These identify the supported API and weight versions.
ENGINES = {
    "moss-tts": {
        "source": "https://github.com/OpenMOSS/MOSS-TTS",
        "source_revision": "934d6826b084c46a0d033402174d5f8ac4ed2519",
        "model_id": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "revision": "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        "auxiliary": {
            "OpenMOSS-Team/MOSS-Audio-Tokenizer": "3cd226ba2947efa357ef453bcad111b6eafba782"
        },
        "model_license": "Apache-2.0",
        "output_license": "Not specified separately by upstream; retain source and voice terms",
    },
    "echo-tts": {
        "source": "https://github.com/jordandare/echo-tts",
        "source_revision": "2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e",
        "model_id": "jordand/echo-tts-base",
        "revision": "06731f67000241d55cff6b0a961f22f5f151f24f",
        "auxiliary": {"jordand/fish-s1-dac-min": "18b770e5b62b1a58ebd7423401a937cc063d8729"},
        "model_license": "CC-BY-NC-SA-4.0",
        "output_license": "CC-BY-NC-SA-4.0",
    },
    "qwen3-tts": {
        "source": "https://github.com/QwenLM/Qwen3-TTS",
        "source_revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "auxiliary": {},
        "model_license": "Apache-2.0",
        "output_license": "Not specified separately by upstream; retain source and voice terms",
    },
    "zonos": {
        "source": "https://github.com/Zyphra/Zonos",
        "source_revision": "bc40d98e1e1ab54fc65c483be127a90e3c7c0645",
        "model_id": "Zyphra/Zonos-v0.1-transformer",
        "revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
        "auxiliary": {
            "descript/dac_44khz": "c1bc521685adf9cfe247bc39a5ca58917eda1ac4",
            "Zyphra/Zonos-v0.1-speaker-embedding": "9fe3dbdc3c703a8cf7141b2d0f550f6d55f67260",
        },
        "model_license": "Apache-2.0",
        "output_license": "Not specified separately by upstream; retain source and voice terms",
    },
}


class TextConfig(StrictModel):
    version: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=3)
    mode: Literal["grounded_qa", "dialogue"] = "grounded_qa"
    topics: list[str] = Field(default_factory=lambda: ["everyday objects"], min_length=1)
    count: int = Field(default=100, ge=1, le=100000)
    batch_size: int = Field(default=10, ge=1, le=50)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    temperature: float = Field(default=0.8, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=128, le=32768)
    max_requests: int = Field(default=30, ge=1, le=100000)
    max_attempts: int = Field(default=3, ge=1, le=8)
    timeout_seconds: float = Field(default=90, gt=0, le=600)
    providers: list[str] = Field(default_factory=list)


class EngineConfig(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    engine: Literal["moss-tts", "echo-tts", "qwen3-tts", "zonos"]
    python: str
    source_dir: str
    reference_audio: str | None = None
    reference_text: str | None = None
    voice_id: str = "unconditioned"
    voice_terms: str = "No reference recording"
    speaker: str = "Ryan"
    instruct: str = ""
    qwen_mode: Literal["custom_voice", "voice_clone"] = "custom_voice"
    max_new_tokens: int = Field(default=2048, ge=64, le=8192)
    echo_steps: int = Field(default=40, ge=1, le=100)
    echo_cfg_text: float = Field(default=3, gt=0, le=20)
    echo_cfg_speaker: float = Field(default=8, ge=0, le=30)
    zonos_cfg_scale: float = Field(default=2, gt=1, le=10)

    @model_validator(mode="after")
    def controls(self):
        controls = {
            "max_new_tokens": {"moss-tts", "qwen3-tts", "zonos"},
            "echo_steps": {"echo-tts"},
            "echo_cfg_text": {"echo-tts"},
            "echo_cfg_speaker": {"echo-tts"},
            "zonos_cfg_scale": {"zonos"},
            "reference_text": {"qwen3-tts"},
            "qwen_mode": {"qwen3-tts"},
            "speaker": {"qwen3-tts"},
            "instruct": {"qwen3-tts"},
        }
        for key, engines in controls.items():
            if key in self.model_fields_set and self.engine not in engines:
                raise ValueError(f"{key} does not apply to {self.engine}")
        if self.engine == "qwen3-tts":
            if self.qwen_mode == "voice_clone" and not (
                self.reference_audio and self.reference_text
            ):
                raise ValueError("Qwen voice_clone requires reference_audio and reference_text")
            if self.qwen_mode == "custom_voice" and self.reference_audio:
                raise ValueError("Use qwen_mode: voice_clone with reference_audio")
            if self.qwen_mode == "voice_clone" and self.instruct:
                raise ValueError("Qwen voice_clone does not accept instruct")
        elif self.qwen_mode != "custom_voice" or self.instruct or self.speaker != "Ryan":
            raise ValueError("qwen_mode, speaker and instruct apply only to Qwen")
        if self.reference_audio and (
            self.voice_id == "unconditioned" or self.voice_terms == "No reference recording"
        ):
            raise ValueError("Reference audio requires a voice_id and its voice_terms")
        if self.reference_text and not self.reference_audio:
            raise ValueError("reference_text requires reference_audio")
        return self

    def resolved(self):
        result = dict(ENGINES[self.engine], **self.model_dump())
        if self.engine == "qwen3-tts" and self.qwen_mode == "voice_clone":
            result.update(
                model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                revision="fd4b254389122332181a7c3db7f27e918eec64e3",
            )
        return result


class AudioQA(StrictModel):
    min_seconds: float = Field(default=0.15, gt=0)
    max_seconds: float = Field(default=30, gt=0, le=120)
    min_rms: float = Field(default=0.0001, gt=0, lt=1)
    max_clipped_fraction: float = Field(default=0.01, ge=0, le=1)
    asr_model: str | None = None
    max_wer: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def duration(self):
        if self.min_seconds >= self.max_seconds:
            raise ValueError("min_seconds must be less than max_seconds")
        return self


class SpeechConfig(StrictModel):
    version: str = Field(min_length=1, max_length=80)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    engines: list[EngineConfig] = Field(min_length=1)
    qa: AudioQA = Field(default_factory=AudioQA)
    worker_timeout_seconds: float = Field(default=1800, gt=0, le=7200)

    @model_validator(mode="after")
    def names(self):
        if len({e.name for e in self.engines}) != len(self.engines):
            raise ValueError("Engine profile names must be unique")
        return self
