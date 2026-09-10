import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("scipy")

from lfm_audio_rl.config import DatasetRecipe, Experiment
from lfm_audio_rl.data import build_dataset, load_dataset
from lfm_audio_rl.synthetic import audio
from lfm_audio_rl.synthetic.config import AudioQA, EngineConfig, SpeechConfig
from lfm_audio_rl.synthetic.storage import publish
from lfm_audio_rl.train import preflight


def tone(path, sr=44100):
    wave = 0.2 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)
    sf.write(path, np.stack([wave, wave], axis=1), sr, subtype="FLOAT")


def test_audio_conversion_and_transcript_gate(tmp_path):
    raw, result = tmp_path / "native.wav", tmp_path / "audio.wav"
    tone(raw)
    report = audio.normalize_audio(raw, result, AudioQA(), "hello there", lambda _: "Hello there.")
    wave, sr = sf.read(result)
    assert sr == 24000 and wave.shape == (24000,)
    assert sf.info(result).subtype == "PCM_16"
    assert report["wer"] == 0 and report["native_channels"] == 2
    with pytest.raises(audio.AudioRejected, match="WER"):
        audio.normalize_audio(raw, result, AudioQA(), "hello there", lambda _: "wrong words")
    assert not result.exists()


@pytest.mark.parametrize("kind", ["silent", "nan", "clipped", "short"])
def test_bad_audio_rejected(tmp_path, kind):
    wave = (
        np.zeros(24000)
        if kind == "silent"
        else np.ones(24000) * (float("nan") if kind == "nan" else 1.0)
    )
    if kind == "short":
        wave = np.ones(10) * 0.1
    path = tmp_path / "bad.wav"
    sf.write(path, wave, 24000, subtype="FLOAT")
    with pytest.raises(audio.AudioRejected):
        audio.normalize_audio(path, tmp_path / "out.wav", AudioQA(), "hi")


def test_render_resume_hashes_and_cross_engine_splits(tmp_path, monkeypatch):
    text_path, output = tmp_path / "text", tmp_path / "speech"
    build_dataset(DatasetRecipe(version="v1_clean", count=2), text_path)
    config = SpeechConfig(
        version="v1",
        engines=[
            EngineConfig(name="echo", engine="echo-tts", python="unused", source_dir="unused"),
            EngineConfig(name="zonos", engine="zonos", python="unused", source_dir="unused"),
        ],
    )
    monkeypatch.setattr(audio, "resolve_profiles", lambda _: [p.resolved() for p in config.engines])
    calls = []

    class FakeWorker:
        runtime = {"test_fixture": True}

        def __init__(self, profile, log, timeout):
            self.profile = profile

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def synthesize(self, text, seed, output):
            calls.append((text, seed))
            if len(calls) == 3:
                raise RuntimeError("simulated interruption")
            tone(output)

    with pytest.raises(RuntimeError, match="interruption"):
        audio.render_speech(config, text_path, output, FakeWorker)
    assert not (output / "dataset.json").exists()
    result = audio.render_speech(config, text_path, output, FakeWorker)
    assert result["count"] == 4 and len(calls) == 9
    rows, _ = load_dataset(output, require_audio=True)
    assert rows[0].semantic_group == rows[2].semantic_group
    assert rows[0].split == rows[2].split
    assert rows[0].id != rows[2].id
    assert "CC-BY-NC-SA-4.0" in result["output_licenses"]
    assert audio.render_speech(config, text_path, output, FakeWorker) == result
    # Tampering also fails when resuming a partially published run.
    (output / "dataset.json").unlink()
    (output / rows[0].input_audio).write_bytes(b"invalid")
    with pytest.raises(ValueError, match="hash"):
        audio.render_speech(config, text_path, output, FakeWorker)


def test_dialogue_not_used_with_exact_rl_reward(tmp_path):
    tmp_path = tmp_path / "data"
    build_dataset(DatasetRecipe(version="v1_clean", count=10), tmp_path)
    rows, _ = load_dataset(tmp_path)
    tone(tmp_path / "dummy.wav", 24000)
    from lfm_audio_rl.data import file_hash

    rows = [
        r.model_copy(
            update={
                "input_audio": "dummy.wav",
                "answer_audio": "dummy.wav",
                "input_sha256": file_hash(tmp_path / "dummy.wav"),
                "answer_sha256": file_hash(tmp_path / "dummy.wav"),
                "provenance": {"reward_protocol": "open_ended"},
            }
        )
        for r in rows
    ]
    # tone() makes stereo; export fixture as mono to satisfy existing loader.
    wave, sr = sf.read(tmp_path / "dummy.wav")
    sf.write(tmp_path / "dummy.wav", wave.mean(axis=1), sr)
    sha = file_hash(tmp_path / "dummy.wav")
    rows = [r.model_copy(update={"input_sha256": sha, "answer_sha256": sha}) for r in rows]
    publish(tmp_path, rows, {})
    with pytest.raises(ValueError, match="open-ended reward"):
        preflight(Experiment(), tmp_path)
