"""Run this file with an engine's Python, not as part of the LFM environment.

Protocol: one init JSON line, then synthesis JSON lines. Results use a dedicated
stdout descriptor; library output (including native writes) goes to stderr.
"""

import importlib.metadata
import json
import os
import random
import sys
import traceback
from functools import partial


def load_backend(profile):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("Neural TTS workers currently require CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neural TTS workers require BF16 support")
    device = "cuda:0"
    dtype = torch.bfloat16
    model_id, revision = profile["model_id"], profile["revision"]
    revisions = {model_id: revision, **profile["auxiliary"]}

    def download(repo_id, filename, **kwargs):
        # Echo and Zonos speaker loaders omit revision. Pin those calls explicitly.
        kwargs["revision"] = revisions[repo_id]
        return hf_hub_download(repo_id, filename, **kwargs)

    def snapshot(repo_id):
        return snapshot_download(
            repo_id,
            revision=revisions[repo_id],
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.npz",
            ],
        )

    engine = profile["engine"]
    reference = profile["reference_audio"]
    if engine == "moss-tts":
        from transformers import AutoModel, AutoProcessor

        torch.backends.cuda.enable_cudnn_sdp(False)
        path = snapshot(model_id)
        codec = snapshot("OpenMOSS-Team/MOSS-Audio-Tokenizer")
        processor = AutoProcessor.from_pretrained(path, codec_path=codec, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        model = (
            AutoModel.from_pretrained(
                path, trust_remote_code=True, attn_implementation="sdpa", dtype=dtype
            )
            .to(device)
            .eval()
        )

        def synthesize(text, seed):
            args = {"text": text}
            if reference:
                args["reference"] = [reference]
            batch = processor([[processor.build_user_message(**args)]], mode="generation")
            codes = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=profile["max_new_tokens"],
            )
            messages = processor.decode(codes)
            wave = messages[0].audio_codes_list[0]
            return wave, processor.model_config.sampling_rate

    elif engine == "echo-tts":
        import inference

        inference.hf_hub_download = download
        model = inference.load_model_from_hf(
            repo_id=model_id, device=device, delete_blockwise_modules=True
        )
        fish = inference.load_fish_ae_from_hf(device=device)
        pca = inference.load_pca_state_from_hf(repo_id=model_id, device=device)
        speaker = inference.load_audio(reference).to(device) if reference else None
        sampler = partial(
            inference.sample_euler_cfg_independent_guidances,
            num_steps=profile["echo_steps"],
            cfg_scale_text=profile["echo_cfg_text"],
            cfg_scale_speaker=profile["echo_cfg_speaker"],
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=640,
        )

        def synthesize(text, seed):
            wave, _ = inference.sample_pipeline(
                model=model,
                fish_ae=fish,
                pca_state=pca,
                sample_fn=sampler,
                text_prompt=text,
                speaker_audio=speaker,
                rng_seed=seed,
            )
            return wave[0], 44100

    elif engine == "qwen3-tts":
        from qwen_tts import Qwen3TTSModel

        # A local snapshot pins both model and processor; the upstream wrapper does
        # not forward revision to its separate processor load.
        model = Qwen3TTSModel.from_pretrained(
            snapshot(model_id), device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        clone = None
        if profile["qwen_mode"] == "voice_clone":
            clone = model.create_voice_clone_prompt(
                ref_audio=reference, ref_text=profile["reference_text"], x_vector_only_mode=False
            )

        def synthesize(text, seed):
            args = {
                "text": text,
                "language": "English",
                "max_new_tokens": profile["max_new_tokens"],
            }
            if clone is not None:
                waves, sr = model.generate_voice_clone(**args, voice_clone_prompt=clone)
            else:
                waves, sr = model.generate_custom_voice(
                    **args, speaker=profile["speaker"], instruct=profile["instruct"]
                )
            return waves[0], sr

    elif engine == "zonos":
        import torchaudio
        import zonos.autoencoder as autoencoder
        import zonos.speaker_cloning as speaker_cloning
        from transformers.models.dac import DacModel
        from zonos.conditioning import make_cond_dict
        from zonos.model import Zonos

        class PinnedDAC:
            @staticmethod
            def from_pretrained(repo_id):
                return DacModel.from_pretrained(repo_id, revision=revisions[repo_id])

        autoencoder.DacModel = PinnedDAC
        speaker_cloning.hf_hub_download = download
        model = Zonos.from_pretrained(model_id, revision=revision, device=device).eval()
        speaker = None
        if reference:
            wave, sr = torchaudio.load(reference)
            speaker = model.make_speaker_embedding(wave, sr)

        def synthesize(text, seed):
            cond = make_cond_dict(text=text, speaker=speaker, language="en-us")
            conditioning = model.prepare_conditioning(cond)
            codes = model.generate(
                conditioning,
                max_new_tokens=profile["max_new_tokens"],
                cfg_scale=profile["zonos_cfg_scale"],
                progress_bar=False,
                disable_torch_compile=True,
            )
            return model.autoencoder.decode(codes)[0], model.autoencoder.sampling_rate

    else:
        raise ValueError("Unknown TTS engine")

    def generate(text, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            wave, sr = synthesize(text, seed)
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().float().cpu().numpy()
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim == 1:
            wave = wave[:, None]
        elif wave.ndim == 2:
            # Torch audio APIs are channels x samples. Qwen returns mono arrays.
            wave = wave.T
        else:
            raise ValueError(f"Unexpected waveform rank: {wave.ndim}")
        if wave.shape[1] not in (1, 2):
            raise ValueError("Expected one or two audio channels")
        return wave, int(sr)

    return generate


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()

    try:
        profile = json.loads(sys.stdin.readline())
        sys.path.insert(0, profile["source_dir"])
        generate = load_backend(profile)
        import soundfile as sf
        import torch

        packages = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        }
        send(
            {
                "event": "ready",
                "runtime": {
                    "python": sys.version,
                    "packages": packages,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                },
            }
        )
        for line in sys.stdin:
            job = json.loads(line)
            wave, sr = generate(job["text"], job["seed"])
            # Floating point WAV preserves invalid/clipped samples for host QA.
            sf.write(job["output"], wave, sr, subtype="FLOAT")
            send({"event": "done", "id": job["id"], "sample_rate": sr})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"event": "error", "error_type": type(exc).__name__})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
