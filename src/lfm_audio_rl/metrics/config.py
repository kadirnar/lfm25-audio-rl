from typing import Literal

from pydantic import Field, model_validator

from ..config import StrictModel


class ModelSpec(StrictModel):
    # A local directory makes the exact scorer files explicit and hashable.
    path: str
    layers: int | None = Field(default=None, ge=1)


class JudgeConfig(StrictModel):
    model: str
    max_requests: int = Field(default=100, ge=1)
    max_attempts: int = Field(default=3, ge=1, le=8)
    max_tokens: int = Field(default=1024, ge=128, le=4096)
    timeout_seconds: float = Field(default=90, gt=0, le=600)
    seed: int = Field(default=42, ge=0)
    providers: list[str] = Field(default_factory=list)


class MetricConfig(StrictModel):
    normalization: Literal["basic", "verbatim", "dataset"] = "basic"
    groups: list[
        Literal[
            "text",
            "audio",
            "timing",
            "instructions",
            "prosody",
            "reference",
            "speaker",
            "bertscore",
            "dnsmos",
            "nisqa",
            "utmos",
            "fad",
            "judge",
        ]
    ] = ["text", "audio", "timing", "instructions"]
    reference_metrics: list[Literal["si_sdr", "snr", "stoi", "estoi", "pesq", "visqol", "mcd"]] = [
        "si_sdr",
        "snr",
        "stoi",
        "estoi",
    ]
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    seed: int = Field(default=42, ge=0)
    device: str = "cpu"
    # If absent, use supplied ASR transcripts. Otherwise transcribe output audio.
    asr_model: str | None = None
    asr_device: Literal["cpu", "cuda"] = "cpu"
    min_audio_seconds: float = Field(default=0.1, gt=0)
    max_audio_seconds: float = Field(default=120, gt=0, le=600)
    silence_dbfs: float = Field(default=-40, ge=-100, le=-10)
    clipping_threshold: float = Field(default=0.999, gt=0, le=1)
    max_clipped_fraction: float = Field(default=0.01, ge=0, le=1)
    pitch_min_hz: float = Field(default=50, ge=20)
    pitch_max_hz: float = Field(default=600, le=2000)
    speaker_model: ModelSpec | None = None
    bert_model: ModelSpec | None = None
    clap_model: ModelSpec | None = None
    utmos_checkpoint: str | None = None
    nisqa_weights_dir: str | None = None
    dnsmos_weights_dir: str | None = None
    visqol_model: str | None = None
    judge: JudgeConfig | None = None
    # Fail selected model/dependency errors by default. Data inapplicability is
    # reported separately and never replaced with a synthetic quality score.
    fail_on_metric_error: bool = True

    @model_validator(mode="after")
    def valid(self):
        if len(set(self.groups)) != len(self.groups) or not self.groups:
            raise ValueError("Select unique, nonempty metric groups")
        if len(set(self.reference_metrics)) != len(self.reference_metrics):
            raise ValueError("Reference metrics must be unique")
        if (
            self.min_audio_seconds >= self.max_audio_seconds
            or self.pitch_min_hz >= self.pitch_max_hz
        ):
            raise ValueError("Invalid audio or pitch bounds")
        required = {
            "speaker": "speaker_model",
            "bertscore": "bert_model",
            "fad": "clap_model",
            "utmos": "utmos_checkpoint",
            "nisqa": "nisqa_weights_dir",
            "dnsmos": "dnsmos_weights_dir",
            "judge": "judge",
        }
        for group, field in required.items():
            if group in self.groups and getattr(self, field) is None:
                raise ValueError(f"{group} requires {field}")
        if "bertscore" in self.groups and self.bert_model.layers is None:
            raise ValueError("BERTScore requires the selected encoder layer count")
        if (
            "reference" in self.groups
            and "visqol" in self.reference_metrics
            and not self.visqol_model
        ):
            raise ValueError("ViSQOL requires the speech-mode model file")
        return self


class Timing(StrictModel):
    environment: dict[str, str] = Field(default_factory=dict)
    # All elapsed values are seconds relative to this request's start.
    first_text_seconds: float | None = Field(default=None, ge=0)
    first_audio_token_seconds: float | None = Field(default=None, ge=0)
    first_audio_seconds: float | None = Field(default=None, ge=0)
    generation_seconds: float | None = Field(default=None, ge=0)
    decode_seconds: float | None = Field(default=None, ge=0)
    audio_ready_seconds: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    # Actual playable chunks, not codec tokens. Arrival times are elapsed seconds.
    chunks: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def order(self):
        if self.audio_ready_seconds is not None:
            for value in [
                self.first_text_seconds,
                self.first_audio_token_seconds,
                self.first_audio_seconds,
            ]:
                if value is not None and value > self.audio_ready_seconds:
                    raise ValueError("First output cannot arrive after the completed audio")
        previous = -1.0
        for arrival, duration in self.chunks:
            if arrival < previous or arrival < 0 or duration <= 0:
                raise ValueError("Chunk arrivals must be ordered and durations positive")
            previous = arrival
        return self


class Prediction(StrictModel):
    example_id: str
    generation: dict[str, str] = Field(default_factory=dict)
    text: str | None = None
    asr: str | None = None
    asr_model: str | None = None
    input_asr: str | None = None
    audio: str | None = None
    audio_sha256: str | None = None
    terminated: bool | None = None
    timing: Timing | None = None
    error: str | None = None


class Constraint(StrictModel):
    kind: Literal[
        "contains", "not_contains", "starts_with", "ends_with", "max_words", "min_words", "exact"
    ]
    value: str | int

    @model_validator(mode="after")
    def valid(self):
        if self.kind in {"max_words", "min_words"}:
            if type(self.value) is not int or self.value < 0:
                raise ValueError("Word limits require nonnegative integers")
        elif not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Text constraints require nonempty strings")
        return self


class EvaluationPolicy(StrictModel):
    answer_mode: Literal["closed", "open"] = "closed"
    accepted_answers: list[str] = Field(default_factory=list)
    # Reference metrics never assume another TTS engine's answer is aligned.
    reference_relation: Literal["none", "same_content", "aligned"] = "none"
    speaker_reference_audio: str | None = None
    speaker_reference_sha256: str | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    expected_refusal: bool | None = None

    @model_validator(mode="after")
    def speaker(self):
        if bool(self.speaker_reference_audio) != bool(self.speaker_reference_sha256):
            raise ValueError("Speaker reference needs both path and SHA256")
        if any(not a.strip() for a in self.accepted_answers):
            raise ValueError("Accepted answers cannot be empty")
        return self
