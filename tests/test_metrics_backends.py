"""Upstream API contracts with fake models; these tests never download weights."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

from lfm_audio_rl.metrics.backends import Scorers, scorer_assets
from lfm_audio_rl.metrics.config import MetricConfig, ModelSpec


def module(monkeypatch, name, **members):
    value = ModuleType(name)
    value.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, value)
    return value


def test_quality_model_dimension_order_and_local_assets(tmp_path, monkeypatch):
    dn = tmp_path / "dn"
    (dn / "DNSMOS").mkdir(parents=True)
    for name in ["model_v8.onnx", "sig_bak_ovr.onnx"]:
        (dn / "DNSMOS" / name).write_bytes(b"test fixture")
    nq = tmp_path / "nq"
    nq.mkdir()
    (nq / "nisqa.tar").write_bytes(b"test fixture")
    dns_call = Mock(return_value=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    nis_call = Mock(return_value=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    dnmod = module(
        monkeypatch,
        "torchmetrics.functional.audio.dnsmos",
        deep_noise_suppression_mean_opinion_score=dns_call,
    )
    nqmod = module(
        monkeypatch,
        "torchmetrics.functional.audio.nisqa",
        non_intrusive_speech_quality_assessment=nis_call,
        _load_nisqa_model=SimpleNamespace(cache_clear=Mock()),
    )
    cfg = MetricConfig(
        groups=["dnsmos", "nisqa"], dnsmos_weights_dir=str(dn), nisqa_weights_dir=str(nq)
    )
    scorer = Scorers(cfg, tmp_path)
    wave = np.ones(24000)
    assert scorer.dnsmos(wave, 24000) == {
        "dnsmos_p808": 1,
        "dnsmos_signal": 2,
        "dnsmos_background": 3,
        "dnsmos_overall": 4,
    }
    assert scorer.nisqa(wave, 24000) == {
        "nisqa_overall": 1,
        "nisqa_noisiness": 2,
        "nisqa_discontinuity": 3,
        "nisqa_coloration": 4,
        "nisqa_loudness": 5,
    }
    assert dnmod.DNSMOS_DIR == str(dn) and nqmod.NISQA_DIR == str(nq)
    assert dns_call.call_args.kwargs["personalized"] is False
    before = scorer_assets(cfg)
    (nq / "nisqa.tar").write_bytes(b"changed fixture")
    assert before["nisqa"]["sha256"] != scorer_assets(cfg)["nisqa"]["sha256"]


def test_speaker_resampling_and_bertscore_arguments(tmp_path, monkeypatch):
    encoder = SimpleNamespace(encode_batch=Mock(return_value=torch.tensor([[[3.0, 4.0]]])))
    from_hparams = Mock(return_value=encoder)
    module(
        monkeypatch,
        "speechbrain.inference.speaker",
        EncoderClassifier=SimpleNamespace(from_hparams=from_hparams),
    )
    bert = SimpleNamespace(
        score=Mock(return_value=(torch.tensor([0.8]), torch.tensor([0.6]), torch.tensor([0.7])))
    )
    constructor = Mock(return_value=bert)
    module(monkeypatch, "bert_score", BERTScorer=constructor)
    cfg = MetricConfig(
        groups=["speaker", "bertscore"],
        speaker_model=ModelSpec(path=str(tmp_path)),
        bert_model=ModelSpec(path=str(tmp_path), layers=17),
    )
    scorer = Scorers(cfg, tmp_path)
    assert scorer.speaker_similarity(np.ones(8000), 8000, np.ones(24000), 24000)[
        "speaker_cosine"
    ] == pytest.approx(1)
    assert encoder.encode_batch.call_args.args[0].shape == (1, 16000)
    assert from_hparams.call_count == 1
    assert from_hparams.call_args.kwargs["overrides"]["pretrained_path"] == str(tmp_path)
    assert scorer.bertscore("reference", "hypothesis")["bertscore_f1"] == pytest.approx(0.7)
    bert.score.assert_called_once_with(["hypothesis"], ["reference"])
    assert constructor.call_args.kwargs["num_layers"] == 17
    assert constructor.call_args.kwargs["idf"] is False


def test_utmos_and_clap_inference_contracts(tmp_path, monkeypatch):
    predict = Mock(return_value=np.array([3.5]))
    create = Mock(return_value=SimpleNamespace(predict=predict))
    module(monkeypatch, "utmosv2", create_model=create)
    cfg = MetricConfig(
        groups=["utmos", "fad"],
        utmos_checkpoint=str(tmp_path / "model.pth"),
        clap_model=ModelSpec(path=str(tmp_path)),
    )
    scorer = Scorers(cfg, tmp_path)
    assert scorer.utmos(np.ones(24000), 24000)["utmos_v2"] == 3.5
    assert create.call_args.kwargs["checkpoint_path"] == str(tmp_path / "model.pth")
    assert predict.call_args.kwargs["remove_silent_section"] is False
    assert predict.call_args.kwargs["num_repetitions"] == 1
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.get_audio_features.return_value = torch.tensor([[1.0, 2.0]])
    processor = Mock(return_value={"input_features": torch.ones(1, 1)})
    model_load, processor_load = Mock(return_value=model), Mock(return_value=processor)
    module(
        monkeypatch,
        "transformers",
        ClapModel=SimpleNamespace(from_pretrained=model_load),
        ClapProcessor=SimpleNamespace(from_pretrained=processor_load),
    )
    embedding = scorer.clap_embedding(np.ones(11 * 16000), 16000)
    assert embedding.tolist() == [1.0, 2.0]
    assert [len(c.kwargs["audio"]) for c in processor.call_args_list] == [480000, 48000]
    assert model_load.call_args.kwargs["local_files_only"] is True
    assert processor_load.call_args.kwargs["local_files_only"] is True


def test_intrusive_metrics_require_explicit_alignment_and_visqol_speech_mode(tmp_path, monkeypatch):
    api = SimpleNamespace(Create=Mock(), Measure=Mock(return_value=SimpleNamespace(moslqo=4.1)))

    def cfg_factory():
        return SimpleNamespace(audio=SimpleNamespace(), options=SimpleNamespace())

    module(monkeypatch, "visqol", visqol_lib_py=SimpleNamespace(VisqolApi=lambda: api))
    module(monkeypatch, "visqol.pb2", visqol_config_pb2=SimpleNamespace(VisqolConfig=cfg_factory))
    cfg = MetricConfig(
        groups=["reference"], reference_metrics=["visqol"], visqol_model=str(tmp_path / "svr.txt")
    )
    scorer = Scorers(cfg, tmp_path)
    a = np.ones(24000)
    values, missing = scorer.intrusive(a, 24000, a, 24000, "same_content")
    assert not values and "aligned" in missing["visqol_moslqo"]
    values, missing = scorer.intrusive(a, 24000, a[:-200], 24000, "aligned")
    assert not values and "different lengths" in missing["visqol_moslqo"]
    values, missing = scorer.intrusive(a, 24000, a, 24000, "aligned")
    assert values["visqol_moslqo"] == 4.1 and not missing
    config = api.Create.call_args.args[0]
    assert config.audio.sample_rate == 16000 and config.options.use_speech_scoring is True
    assert api.Measure.call_args.args[0].dtype == np.float64


def test_inference_seed_is_reproducible_and_preserves_caller_state():
    import random

    from lfm_audio_rl.metrics.backends import inference_seed

    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    def sample():
        with inference_seed(42, "cpu"):
            return [random.random(), np.random.random(), torch.rand(1).item()]

    assert sample() == sample()
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
