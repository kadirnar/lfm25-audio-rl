import json

import pytest

from lfm_audio_rl.evaluate import compare_runs, paired_bootstrap


def test_paired_interval_and_protocol_mismatch(tmp_path):
    interval = paired_bootstrap([0, 0, 0], [1, 1, 1])
    assert interval["mean_delta"] == 1 and interval["ci95"] == [1, 1]
    protocol = {
        "dataset_hash": "abc",
        "split": "test",
        "seed": 42,
        "sampling": {},
        "asr_model": "pinned-test-model",
        "reward_version": "v1",
        "results": [{"example_id": "x", "reward": {"total": 0}}],
    }
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(protocol))
    protocol["asr_model"] = "different-model"
    b.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="asr_model"):
        compare_runs(a, b)
