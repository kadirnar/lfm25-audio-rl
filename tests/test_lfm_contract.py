"""Actual upstream layers at tiny random sizes; no pretrained weights or downloads."""

# Optional dependency must be skipped before importing its modules.
# ruff: noqa: E402

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("liquid_audio")
pytestmark = pytest.mark.model

from liquid_audio import ChatState, LFMModality
from liquid_audio.model.conformer.encoder import ConformerEncoderConfig
from liquid_audio.model.lfm2_audio import DepthformerConfig, LFM2AudioConfig
from liquid_audio.utils import mel2emb_len
from transformers import Lfm2Config

from lfm_audio_rl.lfm import (
    Event,
    check_parity,
    pack_rollout,
    recording_model_class,
    sampled_logps,
    score_rollout,
)
from lfm_audio_rl.lora import inject_lora


@pytest.fixture
def tiny_model():
    torch.manual_seed(3)
    encoder = ConformerEncoderConfig(
        feat_in=128,
        feat_out=-1,
        n_layers=1,
        d_model=32,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=16,
        causal_downsampling=False,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=2,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=[-1, -1],
        xscaling=False,
        untie_biases=True,
        pos_emb_max_len=100,
        conv_kernel_size=9,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        dropout=0,
        dropout_pre_encoder=0,
        dropout_emb=0,
        dropout_att=0,
    )
    config = LFM2AudioConfig(
        architectures=["LFM2AudioModel"],
        codebooks=8,
        tie_audio_embeddings=True,
        semantic_codebook_factor=1,
        codebook_weight="linear",
        interleaved_n_text=2,
        interleaved_n_audio=2,
        preprocessor={},
        encoder=encoder,
        lfm=Lfm2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            layer_types=["full_attention", "conv"],
            block_multiple_of=32,
            block_auto_adjust_ff_dim=False,
        ),
        depthformer=DepthformerConfig(layers=1, dim=64, tie=True),
    )
    model = recording_model_class()(config).eval()
    model.requires_grad_(False)
    inject_lora(model.lfm, ["q_proj", "v_proj"], rank=2, alpha=4)
    model.events = []
    return model


class Tokenizer:
    def encode(self, *args, **kwargs):
        return torch.tensor([[1]])

    def decode(self, tokens, **kwargs):
        return "test"


def chat_fixture():
    processor = SimpleNamespace(text=Tokenizer(), device=torch.device("cpu"))
    chat = ChatState(processor, dtype=torch.float32)
    chat.text = torch.tensor([[1, 9]])
    chat.audio_in = torch.randn(128, 17)
    chat.audio_in_lens = torch.tensor([17])
    chat.modality_flag = torch.tensor(
        [
            [int(LFMModality.TEXT)]
            + [int(LFMModality.AUDIO_IN)] * int(mel2emb_len(17))
            + [int(LFMModality.TEXT)]
        ]
    )
    return chat, processor


@pytest.mark.parametrize("scope", ["text", "joint"])
def test_real_upstream_cached_sampling_matches_teacher_forcing(tiny_model, scope):
    chat, processor = chat_fixture()
    for _ in tiny_model.generate_interleaved(
        **chat, max_new_tokens=10, text_temperature=0.9, audio_temperature=0.9
    ):
        pass
    events = tiny_model.events
    assert any(e.kind == "audio" for e in events)
    rollout = pack_rollout(chat, events, processor)
    replayed = score_rollout(tiny_model, rollout, temperature=0.9, scope=scope)
    error = check_parity(replayed, sampled_logps(rollout, scope), tolerance=2e-4)
    assert error < 2e-4
    (-replayed.mean()).backward()
    gradients = [p.grad for p in tiny_model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_text_stop_is_recorded_and_audio_eos_is_marginalized(tiny_model):
    logits = torch.full((256,), -100.0)
    logits[7] = 100
    tiny_model._sample_text_token(logits, temperature=1.0)
    assert tiny_model.events[-1].tokens.item() == 7
    chat, processor = chat_fixture()
    events = [
        Event("audio", torch.arange(8), torch.full((8,), -1.0)),
        Event("audio", torch.full((8,), 2048), torch.tensor([-1.0])),
        Event("text", torch.tensor([7]), torch.tensor([-1.0])),
    ]
    rollout = pack_rollout(chat, events, processor)
    assert rollout.terminated
    assert rollout.audio_codes.shape == (8, 1)
    replay = score_rollout(tiny_model, rollout, temperature=1.0, scope="joint")
    assert replay.numel() == 10  # Eight codebooks, one EOS decision, one text stop.


def test_parity_failure_blocks_updates():
    with pytest.raises(RuntimeError, match="no update"):
        check_parity(torch.tensor([-2.0]), torch.tensor([-1.0]), tolerance=0.05)


def test_audio_sampler_records_only_terminal_first_codebook(tiny_model, monkeypatch):
    for index, layer in enumerate(tiny_model.depth_embeddings):

        def forced_logits(*args, _index=index, **kwargs):
            logits = torch.full((2049,), -100.0)
            logits[2048 if _index == 0 else _index] = 100.0
            return logits

        monkeypatch.setattr(layer, "get_logits", forced_logits)
    result = tiny_model._sample_audio_frame(torch.randn(64), temperature=1.0)
    event = tiny_model.events[-1]
    assert result[0] == 2048 and result[1] == 1
    assert (event.tokens == 2048).all()
    assert event.logps.shape == (1,)


def test_measured_generation_reports_tokens_without_inventing_playable_audio(
    tiny_model, monkeypatch
):
    from lfm_audio_rl import lfm
    from lfm_audio_rl.config import Experiment
    from lfm_audio_rl.metrics.config import Timing

    chat, processor = chat_fixture()
    monkeypatch.setattr(lfm, "prompt_chat", lambda *args: chat)
    rollout = lfm.generate(tiny_model, processor, None, Experiment(max_new_tokens=10), measure=True)
    timing = Timing.model_validate(rollout.timing)
    assert timing.generation_seconds > 0
    assert timing.first_text_seconds is not None or timing.first_audio_token_seconds is not None
    assert timing.first_audio_seconds is None
    assert timing.audio_ready_seconds is None
