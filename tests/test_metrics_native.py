"""Numerical signal checks, not model-quality evaluations."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from lfm_audio_rl.metrics.audio import mel_cepstral_distortion, prosody
from lfm_audio_rl.metrics.backends import Scorers
from lfm_audio_rl.metrics.config import MetricConfig


def signal():
    t = np.arange(32000) / 16000
    x = 0.2 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += 0.005 * np.random.default_rng(1).normal(size=t.shape)
    x[:2000] = 0
    return x


def test_pyin_finds_known_pitch():
    values, _ = prosody(signal(), 16000, MetricConfig())
    assert values["f0_median_hz"] == pytest.approx(150, abs=2)
    assert values["voiced_frame_fraction"] > 0.8


def test_stoi_estoi_identity():
    x = signal()
    cfg = MetricConfig(groups=["reference"], reference_metrics=["stoi", "estoi"])
    result, skipped = Scorers(cfg, ".").intrusive(x, 16000, x, 16000, "aligned")
    assert not skipped
    assert result["stoi"] == pytest.approx(1) and result["estoi"] == pytest.approx(1)


def test_native_pesq_identity_and_noise():
    pytest.importorskip("pesq")
    x = signal()
    cfg = MetricConfig(groups=["reference"], reference_metrics=["pesq"])
    scorer = Scorers(cfg, ".")
    clean, _ = scorer.intrusive(x, 16000, x, 16000, "aligned")
    noisy, _ = scorer.intrusive(
        x, 16000, x + 0.1 * np.random.default_rng(1).normal(size=x.shape), 16000, "aligned"
    )
    assert clean["pesq_wb"] > 4.5 and noisy["pesq_wb"] < clean["pesq_wb"]


def test_native_mcd_silence_floor_and_identity():
    pytest.importorskip("pysptk")
    x = signal()
    assert mel_cepstral_distortion(x, x, 16000) == pytest.approx(0, abs=1e-10)
    with pytest.raises(ValueError, match="non-silent"):
        mel_cepstral_distortion(np.zeros_like(x), x, 16000)
