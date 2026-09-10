import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("sacrebleu")

from lfm_audio_rl.data import Example, digest, file_hash
from lfm_audio_rl.metrics.config import MetricConfig
from lfm_audio_rl.metrics.runner import score_predictions
from lfm_audio_rl.metrics.statistics import compare_reports
from lfm_audio_rl.synthetic.storage import publish


def fixture(tmp_path, dialogue=False):
    data, predictions = tmp_path / "data", tmp_path / "predictions"
    data.mkdir()
    predictions.mkdir()
    rows = [
        Example(
            id=digest(i)[:24],
            semantic_group=str(i),
            split="test",
            task="dialogue" if dialogue else "color_recall",
            prompt="What color is the cup?",
            answer="red",
            provenance={"reward_protocol": "open_ended" if dialogue else "exact_answer"},
        )
        for i in range(3)
    ]
    publish(data, rows, {})
    t = np.arange(24000) / 24000
    sf.write(predictions / "a.wav", 0.1 * np.sin(2 * np.pi * 200 * t), 24000)
    records = [
        {
            "example_id": rows[0].id,
            "text": "red",
            "asr": "red",
            "asr_model": "test-asr",
            "audio": "a.wav",
            "audio_sha256": file_hash(predictions / "a.wav"),
            "terminated": True,
        },
        {
            "example_id": rows[1].id,
            "text": "blue",
            "asr": "blue",
            "asr_model": "test-asr",
            "audio": "a.wav",
            "terminated": False,
        },
    ]
    path = predictions / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return data, path, rows, records


def test_complete_report_counts_failures_and_resumes(tmp_path):
    data, path, rows, records = fixture(tmp_path)
    cfg = MetricConfig(bootstrap_samples=100)
    out = tmp_path / "report"
    report = score_predictions(cfg, data, path, out)
    assert report["n_expected"] == 3 and report["n_predictions"] == 2
    assert report["summary"]["task_success"]["value"] == pytest.approx(1 / 3)
    assert report["summary"]["spoken_answer_corpus_wer"]["value"] == pytest.approx(2 / 3)
    assert report["summary"]["generation_failure"]["value"] == pytest.approx(1 / 3)
    assert report["corpus"]["spoken"]["n"] == 3
    assert score_predictions(cfg, data, path, out) == report
    (out / "complete.json").unlink()
    assert score_predictions(cfg, data, path, out) == report
    assert "NaN" not in (out / "metrics.json").read_text()
    result = compare_reports(report, report, samples=100)
    assert result["metrics"]["task_success"]["value"] == 0
    with pytest.raises(ValueError, match="different configuration"):
        score_predictions(cfg.model_copy(update={"normalization": "verbatim"}), data, path, out)
    # Legacy predictions without hashes are still protected by run identity.
    sf.write(path.parent / "a.wav", np.zeros(24000), 24000)
    with pytest.raises(ValueError, match="different configuration"):
        score_predictions(cfg, data, path, out)


def test_unknown_ids_and_audio_escape_are_errors(tmp_path):
    data, path, rows, records = fixture(tmp_path)
    records[0]["example_id"] = "unknown"
    path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="outside"):
        score_predictions(MetricConfig(), data, path, tmp_path / "out")
    records[0]["example_id"] = rows[0].id
    records[0]["audio"] = "../outside.wav"
    path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="inside"):
        score_predictions(MetricConfig(), data, path, tmp_path / "out")


def test_open_answers_are_not_exact_match_scored(tmp_path):
    data, path, _, _ = fixture(tmp_path, dialogue=True)
    report = score_predictions(MetricConfig(bootstrap_samples=100), data, path, tmp_path / "report")
    assert "task_success" not in report["summary"]
    assert "spoken_answer_exact_match" not in report["summary"]
    assert "tts_wer" in report["summary"]
    assert report["rows"][0]["missing"]["answer_exact_match"].startswith("Open-ended")


def test_comparison_rejects_coverage_and_protocol_changes(tmp_path):
    data, path, _, _ = fixture(tmp_path)
    report = score_predictions(MetricConfig(bootstrap_samples=100), data, path, tmp_path / "report")
    other = json.loads(json.dumps(report))
    other["rows"][0]["metrics"]["task_success"] = None
    with pytest.raises(ValueError, match="Coverage differs"):
        compare_reports(report, other)
    assert (
        compare_reports(report, other, allow_partial=True)["metrics"]["task_success"][
            "omitted_pairs"
        ]
        == 1
    )
    other["protocol_hash"] = "different"
    with pytest.raises(ValueError, match="protocol_hash"):
        compare_reports(report, other)


def test_optional_scorer_failures_and_untranscribed_audio_are_visible(tmp_path):
    data, path, _, records = fixture(tmp_path)
    records[0]["asr"] = None
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture only")

    class FakeScorers:
        def __init__(self, *args):
            pass

        def utmos(self, *args):
            return {"utmos_v2": float("nan")}

    cfg = MetricConfig(
        groups=["text", "utmos"], utmos_checkpoint=str(weights), bootstrap_samples=100
    )
    with pytest.raises(RuntimeError, match="Non-finite"):
        score_predictions(cfg, data, path, tmp_path / "strict", scorer_factory=FakeScorers)
    report = score_predictions(
        cfg.model_copy(update={"fail_on_metric_error": False}),
        data,
        path,
        tmp_path / "lenient",
        scorer_factory=FakeScorers,
    )
    assert report["rows"][0]["metrics"]["task_success"] is None
    assert report["summary"]["utmos_v2"]["coverage"] == 0
    assert report["summary"]["task_success"]["coverage"] == pytest.approx(2 / 3)
    assert (
        "Non-finite" in report["rows"][0]["missing"].get("utmos", "")
        or "non-finite" in report["rows"][0]["missing"]["utmos_v2"]
    )


def test_tts_corpus_comparison_allows_different_answers(tmp_path):
    data, path, _, records = fixture(tmp_path)
    cfg = MetricConfig(bootstrap_samples=100)
    before = score_predictions(cfg, data, path, tmp_path / "before")
    records[0]["text"] = "the cup is red"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    after = score_predictions(cfg, data, path, tmp_path / "after")
    comparison = compare_reports(before, after, samples=100)
    assert comparison["metrics"]["tts_corpus_wer"]["value"] == pytest.approx(0.6)


def test_cli_score_compare_and_human(tmp_path, monkeypatch, capsys):
    from lfm_audio_rl.cli import main

    data, path, rows, _ = fixture(tmp_path)
    cfg = tmp_path / "metrics.yaml"
    cfg.write_text("bootstrap_samples: 100\n")
    out = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "lfm-rl",
            "score",
            "--config",
            str(cfg),
            "--data",
            str(data),
            "--predictions",
            str(path),
            "--output",
            str(out),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["n_expected"] == 3
    monkeypatch.setattr(
        "sys.argv",
        ["lfm-rl", "compare-metrics", str(out / "metrics.json"), str(out / "metrics.json")],
    )
    main()
    assert json.loads(capsys.readouterr().out)["metrics"]["task_success"]["value"] == 0
    ratings = tmp_path / "ratings.jsonl"
    ratings.write_text(
        json.dumps(
            {
                "example_id": rows[0].id,
                "rater_id": "fixture",
                "dimension": "naturalness",
                "score": 3,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "lfm-rl",
            "score-human",
            "--ratings",
            str(ratings),
            "--metrics-report",
            str(out / "metrics.json"),
            "--output",
            str(tmp_path / "human.json"),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["dimensions"]["naturalness"]["value"] == 3


def test_evaluate_exports_predictions_and_runs_metric_stage(tmp_path, monkeypatch):
    import importlib
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    from lfm_audio_rl import lfm
    from lfm_audio_rl.config import Experiment
    from lfm_audio_rl.metrics.config import Prediction

    evaluation = importlib.import_module("lfm_audio_rl.evaluate")
    data, _, rows, _ = fixture(tmp_path, dialogue=True)
    rows = [r.model_copy(update={"input_audio": "input.wav"}) for r in rows]
    monkeypatch.setattr(
        evaluation, "load_dataset", lambda *a, **kw: (rows, {"manifest_sha256": "fixture"})
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 128)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fixture GPU")
    model = SimpleNamespace(requires_grad_=lambda _: None)
    model.eval = lambda: model
    processor = SimpleNamespace()
    processor.eval = lambda: processor
    fake = ModuleType("liquid_audio")
    fake.LFM2AudioProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)
    monkeypatch.setitem(sys.modules, "liquid_audio", fake)
    monkeypatch.setattr(
        lfm,
        "recording_model_class",
        lambda: SimpleNamespace(from_pretrained=lambda *a, **kw: model),
    )
    calls = []

    def generate(*args, measure=False):
        calls.append(measure)
        return SimpleNamespace(
            timing={"generation_seconds": 1, "first_audio_token_seconds": 0.2},
            text="red",
            terminated=True,
        )

    monkeypatch.setattr(lfm, "generate", generate)
    monkeypatch.setattr(
        lfm, "decode_audio", lambda *a: 0.1 * np.sin(2 * np.pi * 200 * np.arange(24000) / 24000)
    )
    monkeypatch.setattr(
        evaluation, "WhisperScorer", lambda *a: SimpleNamespace(transcribe=lambda _: "red")
    )
    # The metric stage reads the real published dataset; only GPU generation is faked.
    result = evaluation.evaluate(
        Experiment(),
        data,
        tmp_path / "eval",
        None,
        metrics_config=MetricConfig(bootstrap_samples=100),
    )
    predictions = [
        Prediction.model_validate_json(line)
        for line in (tmp_path / "eval" / "predictions.jsonl").read_text().splitlines()
    ]
    assert calls == [False, True, True, True]
    assert len(predictions) == 3 and result["mean_reward"] is None
    assert predictions[0].generation["adapter_sha256"] == "base"
    assert predictions[0].timing.audio_ready_seconds >= 1
    assert predictions[0].timing.first_audio_seconds is None
    report = json.loads((tmp_path / "eval" / "metrics" / "metrics.json").read_text())
    assert report["summary"]["valid_audio"]["value"] == 1
