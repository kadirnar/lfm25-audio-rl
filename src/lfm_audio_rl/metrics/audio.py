import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_audio(path: Path):
    wave, sr = sf.read(path, dtype="float64", always_2d=True)
    if (
        not wave.size
        or not np.isfinite(wave).all()
        or not 8000 <= sr <= 192000
        or wave.shape[1] not in (1, 2)
    ):
        raise ValueError("Invalid audio samples, channels, or sample rate")
    return wave, sr


def resample(wave, sr, target):
    if sr == target:
        return np.asarray(wave, dtype=np.float64)
    gcd = math.gcd(sr, target)
    return resample_poly(wave, target // gcd, sr // gcd)


def _longest_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def audio_scores(wave, sr, config):
    if wave.ndim == 1:
        wave = wave[:, None]
    mono = wave.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(wave)))
    clipped = float(np.mean(np.abs(wave) >= config.clipping_threshold))
    seconds = len(mono) / sr
    frame = max(1, round(sr * 0.02))
    frames = [mono[i : i + frame] for i in range(0, len(mono), frame)]
    silence = np.array([np.sqrt(np.mean(x**2)) < 10 ** (config.silence_dbfs / 20) for x in frames])
    lengths = np.array([len(x) for x in frames]) / sr
    active = np.flatnonzero(~silence)
    leading = float(lengths[: active[0]].sum()) if len(active) else seconds
    trailing = float(lengths[active[-1] + 1 :].sum()) if len(active) else seconds
    valid = (
        config.min_audio_seconds <= seconds <= config.max_audio_seconds
        and rms >= 1e-4
        and clipped <= config.max_clipped_fraction
        and peak <= 1.0
    )
    return {
        "valid_audio": float(valid),
        "audio_seconds": seconds,
        "channels": wave.shape[1],
        "sample_rate": sr,
        "rms_dbfs": 20 * math.log10(rms) if rms > 0 else None,
        "peak_dbfs": 20 * math.log10(peak) if peak > 0 else None,
        "dc_offset": float(np.mean(mono)),
        "clipped_fraction": clipped,
        "energy_silence_fraction": float(lengths[silence].sum() / seconds),
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "longest_energy_pause_seconds": min(seconds, _longest_run(silence) * frame / sr),
    }


def loudness(wave, sr):
    import pyloudnorm as pyln

    if len(wave) / sr < 0.4:
        return None
    value = float(pyln.Meter(sr).integrated_loudness(wave))
    return value if np.isfinite(value) else None


def aligned_scores(reference, degraded):
    if reference.shape != degraded.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("Aligned metrics require equal-length mono signals; no silent cropping")
    target, prediction = reference - reference.mean(), degraded - degraded.mean()
    energy = float(target @ target)
    if energy <= 1e-15:
        return {"si_sdr_db": None, "snr_db": None}
    projection = target * float(prediction @ target) / energy
    if float(prediction @ prediction) <= 1e-15:
        return {"si_sdr_db": None, "snr_db": 0.0}
    noise = prediction - projection
    # A documented 1e-12 relative floor bounds perfect reconstructions at 120 dB.
    projected_energy = float(projection @ projection)
    si_sdr = (
        10 * np.log10(projected_energy / max(float(noise @ noise), projected_energy * 1e-12))
        if projected_energy > energy * 1e-15
        else -120.0
    )
    mse = float((degraded - reference) @ (degraded - reference))
    raw_energy = float(reference @ reference)
    snr = 10 * np.log10(raw_energy / max(mse, raw_energy * 1e-12))
    return {"si_sdr_db": float(si_sdr), "snr_db": float(snr)}


def prosody(wave, sr, config):
    import librosa

    mono = resample(wave, sr, 16000)
    f0, voiced, _ = librosa.pyin(
        mono,
        sr=16000,
        fmin=config.pitch_min_hz,
        fmax=config.pitch_max_hz,
        frame_length=1024,
        hop_length=160,
    )
    pitch = f0[np.isfinite(f0) & voiced]
    result = {
        "voiced_frame_fraction": float(np.mean(voiced)),
        "f0_median_hz": float(np.median(pitch)) if pitch.size else None,
        "f0_iqr_hz": float(np.diff(np.quantile(pitch, [0.25, 0.75]))[0]) if pitch.size else None,
        "f0_std_semitones": float(np.std(12 * np.log2(pitch))) if pitch.size else None,
    }
    return result, f0


def pitch_comparison(reference_f0, degraded_f0):
    if reference_f0.shape != degraded_f0.shape:
        raise ValueError("Pitch comparison requires aligned frames")
    a, b = np.isfinite(reference_f0), np.isfinite(degraded_f0)
    both = a & b
    return {
        "voicing_error_rate": float(np.mean(a != b)),
        "f0_rmse_cents": float(
            np.sqrt(np.mean((1200 * np.log2(degraded_f0[both] / reference_f0[both])) ** 2))
        )
        if both.any()
        else None,
        "f0_correlation": float(np.corrcoef(reference_f0[both], degraded_f0[both])[0, 1])
        if both.sum() > 2 and np.std(reference_f0[both]) > 0 and np.std(degraded_f0[both]) > 0
        else None,
    }


def mel_cepstral_distortion(reference, degraded, sr):
    import librosa
    import pysptk

    def coefficients(wave):
        x = resample(wave, sr, 16000)
        if len(x) < 1024:
            raise ValueError("MCD requires at least one complete analysis frame")
        frames = librosa.util.frame(x, frame_length=1024, hop_length=80).T.copy()
        # Remove energy-silent frames before DTW. This is a declared analysis
        # policy, not a speech detector; pause duration is scored separately.
        frames = frames[np.sqrt(np.mean(frames**2, axis=1)) >= 1e-4]
        if not len(frames):
            raise ValueError("MCD is undefined without non-silent analysis frames")
        frames *= np.hamming(1024)
        return pysptk.mcep(frames, order=24, alpha=0.42, etype=1, eps=1e-8)[:, 1:]

    a, b = coefficients(reference), coefficients(degraded)
    if len(a) * len(b) > 16_000_000:
        raise ValueError("MCD DTW exceeds 16 million cells; evaluate shorter utterances")
    _, path = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
    distances = np.linalg.norm(a[path[:, 0]] - b[path[:, 1]], axis=1)
    return float(10 * np.sqrt(2) / np.log(10) * np.mean(distances))
