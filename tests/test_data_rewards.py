import json

import numpy as np
import pytest
from pydantic import ValidationError

from lfm_audio_rl.config import DatasetRecipe, Experiment
from lfm_audio_rl.data import build_dataset, load_dataset, safe_audio_path, semantic_examples
from lfm_audio_rl.rewards import score_reply, word_error_rate


def test_semantics_and_splits_are_stable_across_acoustic_versions():
    clean = semantic_examples(DatasetRecipe(version="v1_clean", count=1000))
    noisy = semantic_examples(DatasetRecipe(version="v2_noisy", snr_db=15, count=1000))
    assert [(r.semantic_group, r.split, r.prompt, r.answer) for r in clean] == [
        (r.semantic_group, r.split, r.prompt, r.answer) for r in noisy
    ]
    assert len({r.id for r in clean}) == 1000
    assert {r.split for r in clean} == {"train", "validation", "test"}


def test_deterministic_build_tampering_and_immutable_output(tmp_path):
    recipe = DatasetRecipe(version="v1_clean", count=32)
    a, b = tmp_path / "a", tmp_path / "b"
    assert build_dataset(recipe, a) == build_dataset(recipe, b)
    rows, metadata = load_dataset(a)
    assert len(rows) == metadata["count"] == 32
    with pytest.raises(FileExistsError):
        build_dataset(recipe, a)
    with pytest.raises(ValueError, match="text-only"):
        load_dataset(a, require_audio=True)
    with (a / "manifest.jsonl").open("a") as stream:
        stream.write(json.dumps({}) + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_dataset(a)
    with pytest.raises(ValueError, match="inside"):
        safe_audio_path(a, "../secret.wav")


def test_audio_reward_cannot_be_won_with_text_or_silence():
    wave = np.full(2400, 0.1)
    assert score_reply("twenty five", "25", "25", wave).total == 1
    assert score_reply("twenty five", "25", "twenty six", wave).total == 0
    assert score_reply("25", "25", "25", np.zeros(2400)).total == 0
    assert score_reply("25", "25", "25", wave, truncated=True).total == 0
    assert score_reply("25", "25", "25", np.full(2400, np.nan)).total == 0
    assert score_reply("25", "25", "25", None).total == 0
    assert word_error_rate("a", "a b c") == 2


def test_config_rejects_unknown_and_unsupported_controls():
    with pytest.raises(ValidationError):
        Experiment(top_k=4)
    with pytest.raises(ValidationError):
        Experiment(algorithm="ppo")
    with pytest.raises(ValidationError):
        Experiment(unknown=True)
    with pytest.raises(ValidationError):
        DatasetRecipe(version="v2_noisy")
    with pytest.raises(ValidationError):
        Experiment(lr=float("nan"))
