"""Verifiable spoken-answer rewards. Audio QA is not a naturalness metric."""

from dataclasses import asdict, dataclass

import numpy as np

from .data import normalize


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / len(ref)  # May legitimately exceed one due to insertions.


@dataclass(frozen=True)
class Reward:
    total: float
    spoken_correctness: float
    text_correctness: float
    text_speech_consistency: float
    valid_audio: bool
    failure: str | None

    def to_dict(self):
        return asdict(self)


def score_reply(
    answer: str,
    text: str,
    asr: str,
    waveform: np.ndarray | None,
    *,
    sample_rate: int = 24000,
    truncated: bool = False,
) -> Reward:
    spoken = float(normalize(answer) == normalize(asr))
    textual = float(normalize(answer) == normalize(text))
    consistency = max(0.0, 1.0 - word_error_rate(text, asr)) if normalize(text) else 0.0
    failure = None
    if truncated:
        failure = "truncated"
    elif waveform is None or waveform.size == 0:
        failure = "missing_audio"
    elif waveform.ndim != 1 or sample_rate <= 0 or not np.isfinite(waveform).all():
        failure = "invalid_waveform"
    elif waveform.size / sample_rate > 30:
        failure = "duration_exceeded"
    elif np.sqrt(np.mean(waveform.astype(np.float64) ** 2)) < 1e-4:
        failure = "silence"
    elif np.mean(np.abs(waveform) >= 0.999) > 0.01:
        failure = "clipping"
    # Reward actual spoken correctness first. Consistency alone must not reward a
    # wrong but perfectly transcribed answer. This is a verifiable-task reward.
    total = spoken * (0.8 + 0.1 * textual + 0.1 * consistency) if failure is None else 0.0
    return Reward(total, spoken, textual, consistency, failure is None, failure)


class WhisperScorer:
    def __init__(self, model: str, device: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model, device=device, compute_type="int8" if device == "cpu" else "float16"
        )

    def transcribe(self, path: str) -> str:
        segments, _ = self.model.transcribe(
            path, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=False
        )
        return " ".join(segment.text.strip() for segment in segments)
