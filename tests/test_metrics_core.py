import numpy as np
import pytest

pytest.importorskip("sacrebleu")

from lfm_audio_rl.metrics.audio import aligned_scores, audio_scores, pitch_comparison
from lfm_audio_rl.metrics.config import Constraint, MetricConfig, Timing
from lfm_audio_rl.metrics.human import summarize_ratings
from lfm_audio_rl.metrics.statistics import clustered_interval, frechet_distance, summarize
from lfm_audio_rl.metrics.text import (
    constraint_scores,
    corpus_text,
    diversity,
    edit_counts,
    text_scores,
)
from lfm_audio_rl.metrics.timing import timing_scores


def test_error_counts_and_normalization_have_defined_empty_semantics():
    values, counts = text_scores("A red cup", "a blue cup and spoon")
    assert values["wer"] == 1
    assert counts["words"] == {
        "errors": 3,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 2,
        "reference_units": 3,
        "rate": 1,
    }
    assert text_scores("", "one two")[0]["wer"] is None
    assert edit_counts([], ["one", "two"])["insertions"] == 2
    assert text_scores("red blue", "")[0]["wer"] == 1
    assert text_scores("red", "red blue green")[0]["wer"] == 2
    assert text_scores("25", "twenty five", "dataset")[0]["exact_match"] == 1
    assert text_scores("25", "twenty five", "basic")[0]["exact_match"] == 0
    assert text_scores("red", "crimson", alternatives=["crimson"])[0]["token_f1"] == 1


def test_corpus_metrics_use_real_libraries_and_signatures():
    reference = ["The red cup is on the table.", "Please open the small blue book."]
    scores = corpus_text(reference, reference)
    assert scores["bleu"]["value"] == pytest.approx(100)
    assert scores["chrf"]["value"] == pytest.approx(100)
    assert scores["ter"]["value"] == 0
    assert scores["rouge_l_f1"]["value"] == 1
    assert "version:" in scores["bleu"]["signature"]
    assert corpus_text(reference, ["", ""])["bleu"]["value"] == 0
    assert diversity(["hello hello", "hello"])["distinct_1"] == 1 / 3


def test_constraints_do_not_match_substrings_and_require_spoken_content():
    constraints = [Constraint(kind="contains", value="red"), Constraint(kind="max_words", value=2)]
    assert constraint_scores("hundred", constraints)["instruction_pass_rate"] == 0.5
    assert constraint_scores("red cup", constraints)["instruction_all_pass"] == 1
    with pytest.raises(ValueError):
        Constraint(kind="max_words", value="two")


def test_audio_diagnostics_and_scale_invariant_sdr():
    sr = 24000
    t = np.arange(sr) / sr
    reference = 0.1 * np.sin(2 * np.pi * 250 * t)
    a = aligned_scores(reference, reference)
    b = aligned_scores(reference, 2 * reference)
    assert a["si_sdr_db"] == pytest.approx(120)
    assert b["si_sdr_db"] == pytest.approx(a["si_sdr_db"])
    assert b["snr_db"] == pytest.approx(0, abs=1e-10)
    assert aligned_scores(reference, np.zeros_like(reference))["si_sdr_db"] is None
    with pytest.raises(ValueError, match="equal-length"):
        aligned_scores(reference, reference[:-1])
    wave = np.concatenate([np.zeros(4800), reference, np.zeros(2400)])
    scores = audio_scores(wave, sr, MetricConfig())
    assert scores["leading_silence_seconds"] == pytest.approx(0.2)
    assert scores["trailing_silence_seconds"] == pytest.approx(0.1)
    assert scores["energy_silence_fraction"] == pytest.approx(0.3 / 1.3)
    assert scores["valid_audio"] == 1
    assert audio_scores(np.ones(sr), sr, MetricConfig())["valid_audio"] == 0


def test_pitch_and_streaming_timing_are_not_token_latency():
    values = pitch_comparison(np.array([100.0, 200.0, np.nan]), np.array([200.0, 400.0, 100.0]))
    assert values["f0_rmse_cents"] == pytest.approx(1200)
    assert values["voicing_error_rate"] == pytest.approx(1 / 3)
    timing = Timing(
        first_audio_token_seconds=0.1,
        audio_ready_seconds=3.0,
        chunks=[(0.5, 1.0), (1.0, 0.5), (2.5, 1.0)],
    )
    result = timing_scores(timing, 2.5, 5)
    assert "first_audio_seconds" not in result
    assert result["first_audio_token_seconds"] == 0.1
    assert result["first_playable_chunk_seconds"] == 0.5
    assert result["stream_stall_seconds"] == 0.5
    assert result["real_time_factor"] == 1.2
    assert result["speech_words_per_minute"] == 120
    with pytest.raises(ValueError):
        Timing(first_audio_seconds=2, audio_ready_seconds=1)
    with pytest.raises(ValueError):
        Timing(chunks=[(1, 1), (0, 1)])


def test_grouped_intervals_and_micro_wer():
    result = clustered_interval([0, 0, 1, 1], ["a", "a", "b", "b"], samples=100)
    assert result["n_groups"] == 2 and result["n"] == 4
    assert result["value"] == 0.5
    rows = [
        {
            "semantic_group": "a",
            "metrics": {"spoken_answer_wer": 1.0},
            "counts": {"spoken_answer_words": {"errors": 1, "reference_units": 1}},
        },
        {
            "semantic_group": "b",
            "metrics": {"spoken_answer_wer": 0.0},
            "counts": {"spoken_answer_words": {"errors": 0, "reference_units": 9}},
        },
    ]
    result = summarize(rows, samples=100)
    assert result["spoken_answer_wer"]["value"] == 0.5
    assert result["spoken_answer_corpus_wer"]["value"] == 0.1
    assert clustered_interval([1], ["a"])["ci95"] is None


def test_frechet_gaussian_distance_and_human_item_weighting():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert frechet_distance(a, a) == pytest.approx(0, abs=1e-8)
    assert frechet_distance(a, a + 1) == pytest.approx(2, abs=1e-8)
    with pytest.raises(ValueError):
        frechet_distance(a[:1], a)
    ratings = [
        {"example_id": "a", "rater_id": "r1", "dimension": "naturalness", "score": 5},
        {"example_id": "a", "rater_id": "r2", "dimension": "naturalness", "score": 5},
        {"example_id": "b", "rater_id": "r1", "dimension": "naturalness", "score": 1},
    ]
    result = summarize_ratings(ratings)
    assert result["dimensions"]["naturalness"]["value"] == 3
    assert result["n_raters"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_ratings(ratings + ratings[:1])
    prefs = [
        {"example_id": "a", "rater_id": "r1", "winner": "B"},
        {"example_id": "b", "rater_id": "r1", "winner": "tie"},
    ]
    assert (
        summarize_ratings(prefs, "preference")["dimensions"]["B_preference_with_half_ties"]["value"]
        == 0.75
    )


def test_target_likelihood_pass_at_k_and_reward_collapse():
    from lfm_audio_rl.metrics.diagnostics import pass_at_k, reward_groups, target_likelihood

    likelihood = target_likelihood(np.log([0.5, 0.25, 0.01]), np.array([True, True, False]))
    assert likelihood["perplexity"] == pytest.approx(np.sqrt(8))
    assert likelihood["bits_per_token"] == pytest.approx(1.5)
    assert likelihood["n_tokens"] == 2
    with pytest.raises(ValueError):
        target_likelihood([0.1])
    assert pass_at_k(10, 2, 3) == pytest.approx(1 - 56 / 120)
    assert pass_at_k(10, 0, 3) == 0
    assert pass_at_k(10, 10, 3) == 1
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)
    assert reward_groups([[1, 1], [0, 1]])["zero_variance_group_fraction"] == 0.5


def test_pairing_corpus_rates_with_different_generated_text_lengths():
    from lfm_audio_rl.metrics.statistics import paired_ratios

    result = paired_ratios([1, 0], [1, 9], [1, 0], [2, 18], ["a", "b"], samples=100)
    assert result["value"] == pytest.approx(-0.05)
    assert result["ci95"][0] <= result["value"] <= result["ci95"][1]
    rows = [
        {"semantic_group": str(i), "metrics": {"audio_ready_seconds": v}, "counts": {}}
        for i, v in enumerate([1, 2, 3])
    ]
    assert summarize(rows)["audio_ready_seconds"]["quantiles"]["p50"] == 2
