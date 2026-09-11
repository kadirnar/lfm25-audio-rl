import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lfm_audio_rl.config import WandbConfig
from lfm_audio_rl.tracking import Tracker


def sdk(monkeypatch):
    runs = []

    def create(**kwargs):
        # A logger must not perturb subsequent candidate sampling.
        random.random()
        np.random.random()
        torch.rand(1)
        run = SimpleNamespace(
            summary={},
            url="https://wandb.ai/test/run",
            log=Mock(),
            finish=Mock(),
            define_metric=Mock(),
        )
        runs.append((kwargs, run))
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=create))
    monkeypatch.setattr("lfm_audio_rl.tracking.check_tracking", lambda _: None)
    return runs


def test_disabled_tracker_has_no_sdk_calls_or_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lfm_audio_rl.tracking.check_tracking",
        lambda _: pytest.fail("disabled tracker imported W&B"),
    )
    with Tracker(WandbConfig(), tmp_path, "identity", {}) as tracker:
        tracker.log({"step": 1, "loss": 2}, 1)
    assert list(tmp_path.iterdir()) == []


def test_online_resume_and_scalar_logging_preserve_rng(tmp_path, monkeypatch):
    runs = sdk(monkeypatch)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cfg = WandbConfig(mode="online", project="test", entity="test", log_every=2)
    with Tracker(cfg, tmp_path, "identity", {"seed": 42}) as tracker:
        tracker.log({"step": 1, "loss": 3.0}, 1)
        tracker.log({"step": 2, "loss": 2.0, "text": "do not upload", "bad": float("nan")}, 2)
        tracker.log({"step": 3, "loss": 1.0}, 3, final=True)
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    kwargs, run = runs[0]
    assert kwargs["resume"] is None and kwargs["save_code"] is False
    assert kwargs["settings"]["console"] == "off"
    assert run.log.call_count == 2
    payload = run.log.call_args_list[0].args[0]
    assert payload == {"trainer/step": 2, "train/checkpoint_step": 2, "train/loss": 2.0}
    with Tracker(cfg, tmp_path, "identity", {}, start_step=3):
        pass
    assert runs[1][0]["id"] == kwargs["id"] and runs[1][0]["resume"] == "allow"
    assert runs[1][1].summary["resumed_from_step"] == 3
    with pytest.raises(ValueError, match="another training run"):
        with Tracker(cfg, tmp_path, "different", {}):
            pass


def test_offline_resume_creates_grouped_segments_and_marks_failure(tmp_path, monkeypatch):
    runs = sdk(monkeypatch)
    cfg = WandbConfig(mode="offline")
    with Tracker(cfg, tmp_path, "identity", {}):
        pass
    with pytest.raises(RuntimeError, match="training failed"):
        with Tracker(cfg, tmp_path, "identity", {}, start_step=2):
            raise RuntimeError("training failed")
    assert runs[0][0]["id"] != runs[1][0]["id"]
    assert runs[0][0]["group"] == runs[1][0]["group"]
    assert runs[1][0]["resume"] is None
    runs[1][1].finish.assert_called_once_with(exit_code=1)
    state = json.loads((tmp_path / "wandb-state.json").read_text())
    assert state["segments"][-1]["status"] == "failed"


def test_real_offline_sdk_writes_local_run(tmp_path):
    pytest.importorskip("wandb")
    with Tracker(
        WandbConfig(mode="offline", project="lfm-rl-test"),
        tmp_path,
        "test-fixture",
        {"fixture": True},
    ) as tracker:
        tracker.log({"step": 1, "policy_loss": 0.5, "mean_reward": 0.75}, 1)
        assert tracker.run.offline
    assert list((tmp_path / "wandb").glob("offline-run-*/run-*.wandb"))


def test_cleanup_failure_preserves_training_exception(tmp_path, monkeypatch):
    runs = sdk(monkeypatch)
    with pytest.warns(RuntimeWarning, match="preserving"):
        with pytest.raises(ValueError, match="original training error"):
            with Tracker(WandbConfig(mode="offline"), tmp_path, "identity", {}):
                runs[0][1].finish.side_effect = RuntimeError("cleanup failed")
                raise ValueError("original training error")
    assert (
        json.loads((tmp_path / "wandb-state.json").read_text())["segments"][-1]["status"]
        == "failed"
    )
