"""Mock official SDK boundaries; these do not claim pretrained GPU validation."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from lfm_audio_rl.synthetic.config import EngineConfig
from lfm_audio_rl.synthetic.worker import load_backend


def module(monkeypatch, name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    return value


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    downloads = []

    def snapshot_download(repo_id, *, revision, allow_patterns):
        downloads.append((repo_id, revision))
        return "/snapshots/" + revision

    def hf_hub_download(repo_id, filename, *, revision):
        downloads.append((repo_id, revision))
        return "/snapshots/" + revision + "/" + filename

    module(
        monkeypatch,
        "huggingface_hub",
        snapshot_download=snapshot_download,
        hf_hub_download=hf_hub_download,
    )
    return downloads


def profile(engine, **kwargs):
    return EngineConfig(
        name="test", engine=engine, python=sys.executable, source_dir=".", **kwargs
    ).resolved()


def test_qwen_uses_pinned_snapshot_and_custom_voice(monkeypatch, environment):
    calls = []

    class Qwen:
        @classmethod
        def from_pretrained(cls, path, *, device_map, dtype, attn_implementation):
            assert path.startswith("/snapshots/")
            return cls()

        def generate_custom_voice(self, *, text, language, max_new_tokens, speaker, instruct):
            calls.append((text, language, speaker, instruct))
            return [np.ones(24000) * 0.1], 24000

    module(monkeypatch, "qwen_tts", Qwen3TTSModel=Qwen)
    p = profile("qwen3-tts", instruct="Speak calmly.")
    wave, sr = load_backend(p)("Hello.", 42)
    assert calls == [("Hello.", "English", "Ryan", "Speak calmly.")]
    assert environment == [(p["model_id"], p["revision"])]
    assert wave.shape == (24000, 1) and sr == 24000


def test_qwen_clone_uses_base_and_cached_reference(monkeypatch, environment):
    calls = []

    class Qwen:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
            calls.append((ref_audio, ref_text, x_vector_only_mode))
            return "cached-clone"

        def generate_voice_clone(self, *, text, language, max_new_tokens, voice_clone_prompt):
            assert voice_clone_prompt == "cached-clone"
            return [np.ones(24000) * 0.1], 24000

    module(monkeypatch, "qwen_tts", Qwen3TTSModel=Qwen)
    p = profile(
        "qwen3-tts",
        qwen_mode="voice_clone",
        reference_audio="reference.wav",
        reference_text="This is my voice.",
        voice_id="speaker-a",
        voice_terms="test fixture",
    )
    generate = load_backend(p)
    generate("Hello.", 42)
    generate("Goodbye.", 43)
    assert len(calls) == 1
    assert environment[0][0].endswith("-Base")


def test_echo_pins_both_models_and_passes_seed(monkeypatch, environment):
    calls = []
    inference = module(monkeypatch, "inference")

    def load_model_from_hf(*, repo_id, device, delete_blockwise_modules):
        assert delete_blockwise_modules
        inference.hf_hub_download(repo_id, "pytorch_model.safetensors")
        return "model"

    def load_fish_ae_from_hf(*, device):
        inference.hf_hub_download("jordand/fish-s1-dac-min", "pytorch_model.safetensors")
        return "fish"

    def load_pca_state_from_hf(*, repo_id, device):
        inference.hf_hub_download(repo_id, "pca_state.safetensors")
        return "pca"

    def sample_pipeline(
        *, model, fish_ae, pca_state, sample_fn, text_prompt, speaker_audio, rng_seed
    ):
        calls.append((text_prompt, rng_seed))
        assert sample_fn.keywords["sequence_length"] == 640
        assert sample_fn.keywords["num_steps"] == 40
        assert speaker_audio is None
        return torch.ones(1, 1, 44100) * 0.1, text_prompt

    inference.load_model_from_hf = load_model_from_hf
    inference.load_fish_ae_from_hf = load_fish_ae_from_hf
    inference.load_pca_state_from_hf = load_pca_state_from_hf
    inference.sample_pipeline = sample_pipeline
    inference.sample_euler_cfg_independent_guidances = lambda *args, **kwargs: None
    p = profile("echo-tts")
    wave, sr = load_backend(p)("Hello.", 123)
    assert calls == [("Hello.", 123)]
    assert environment[0] == (p["model_id"], p["revision"])
    assert environment[1] == next(iter(p["auxiliary"].items()))
    assert sr == 44100 and wave.shape == (44100, 1)


def test_moss_processor_decode_and_codec_pin(monkeypatch, environment):
    class Tensor:
        def to(self, device):
            return self

    class Processor:
        audio_tokenizer = Tensor()
        model_config = SimpleNamespace(sampling_rate=48000)

        @classmethod
        def from_pretrained(cls, path, *, codec_path, trust_remote_code):
            assert codec_path.startswith("/snapshots/") and trust_remote_code
            return cls()

        def build_user_message(self, *, text, reference):
            assert reference == ["ref.wav"]
            return text

        def __call__(self, conversations, *, mode):
            assert conversations == [["Hello."]] and mode == "generation"
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

        def decode(self, outputs):
            return [SimpleNamespace(audio_codes_list=[torch.ones(2, 48000) * 0.1])]

    class Model:
        @classmethod
        def from_pretrained(cls, path, *, trust_remote_code, attn_implementation, dtype):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, *, input_ids, attention_mask, max_new_tokens):
            return "codes"

    module(monkeypatch, "transformers", AutoModel=Model, AutoProcessor=Processor)
    p = profile(
        "moss-tts", reference_audio="ref.wav", voice_id="speaker-a", voice_terms="test fixture"
    )
    wave, sr = load_backend(p)("Hello.", 42)
    assert wave.shape == (48000, 2) and sr == 48000
    assert environment == [(p["model_id"], p["revision"]), *p["auxiliary"].items()]


def test_zonos_pins_dac_and_uses_conditioning(monkeypatch, environment):
    module(monkeypatch, "torchaudio")
    module(monkeypatch, "zonos")
    autoencoder = module(monkeypatch, "zonos.autoencoder")
    module(monkeypatch, "zonos.speaker_cloning")
    calls = []

    class DAC:
        @classmethod
        def from_pretrained(cls, repo_id, *, revision):
            calls.append((repo_id, revision))
            return cls()

    class Zonos:
        autoencoder = SimpleNamespace(
            sampling_rate=44100, decode=lambda codes: torch.ones(1, 1, 44100) * 0.1
        )

        @classmethod
        def from_pretrained(cls, repo_id, *, revision, device):
            calls.append((repo_id, revision))
            autoencoder.DacModel.from_pretrained("descript/dac_44khz")
            return cls()

        def eval(self):
            return self

        def prepare_conditioning(self, cond):
            assert cond == {"text": "Hello.", "speaker": None, "language": "en-us"}
            return cond

        def generate(self, cond, *, max_new_tokens, cfg_scale, progress_bar, disable_torch_compile):
            assert cfg_scale == 2 and not progress_bar and disable_torch_compile
            return "codes"

    module(monkeypatch, "transformers.models.dac", DacModel=DAC)
    module(monkeypatch, "zonos.conditioning", make_cond_dict=lambda **kwargs: kwargs)
    module(monkeypatch, "zonos.model", Zonos=Zonos)
    p = profile("zonos")
    wave, sr = load_backend(p)("Hello.", 42)
    assert calls == [
        (p["model_id"], p["revision"]),
        ("descript/dac_44khz", p["auxiliary"]["descript/dac_44khz"]),
    ]
    assert wave.shape == (44100, 1) and sr == 44100
