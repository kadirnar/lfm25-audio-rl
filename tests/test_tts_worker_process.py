import sys

import pytest

pytest.importorskip("scipy")

from lfm_audio_rl.synthetic import audio


def test_worker_process_protocol_and_key_isolation(tmp_path, monkeypatch):
    # Exercise the real worker entry point and descriptor redirection with a
    # fake backend. No model, network request, or CUDA allocation is needed.
    original = audio.WORKER_SCRIPT
    script = tmp_path / "fixture.py"
    script.write_text(f"""
import importlib.util
import os
import numpy as np
import torch
spec = importlib.util.spec_from_file_location("worker", {str(original)!r})
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
def load(profile):
    assert "OPENROUTER_API_KEY" not in os.environ
    print("library Python stdout")
    os.write(1, b"library native stdout\\n")
    def generate(text, seed):
        return np.ones((24000, 1), dtype=np.float32) * 0.1, 24000
    return generate
worker.load_backend = load
torch.cuda.get_device_name = lambda i: "test fixture"
worker.main()
""")
    monkeypatch.setattr(audio, "WORKER_SCRIPT", script)
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-forward-this-test-key")
    profile = {"python": sys.executable, "source_dir": str(tmp_path)}
    log = tmp_path / "worker.log"
    with audio.Worker(profile, log, timeout=30) as worker:
        worker.synthesize("hello", 42, tmp_path / "a.wav")
        worker.synthesize("goodbye", 43, tmp_path / "b.wav")
        assert worker.runtime["gpu"] == "test fixture"
    assert worker.process.poll() is not None
    assert (tmp_path / "a.wav").exists() and (tmp_path / "b.wav").exists()
    logs = log.read_text()
    assert "library Python stdout" in logs and "library native stdout" in logs
    assert "never-forward-this-test-key" not in logs


def test_unresponsive_worker_is_terminated(tmp_path, monkeypatch):
    script = tmp_path / "hang.py"
    script.write_text("import threading\nthreading.Event().wait()\n")
    monkeypatch.setattr(audio, "WORKER_SCRIPT", script)
    worker = audio.Worker(
        {"python": sys.executable, "source_dir": str(tmp_path)}, tmp_path / "log", timeout=0.2
    )
    with pytest.raises(RuntimeError, match="timed out"), worker:
        pass
    assert worker.process.poll() is not None
