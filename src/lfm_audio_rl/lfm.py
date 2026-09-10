"""Experimental liquid-audio 1.3.0 integration; CUDA validation is still required.

This module calls upstream implementations without vendoring their code. Sampling
callbacks retain the terminal text decision hidden by generate_interleaved().
The adapter is intentionally single-worker; a model instance cannot serve
concurrent rollouts while its audio-logit callbacks are installed.
"""

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F


@dataclass
class Event:
    kind: str
    tokens: torch.Tensor
    logps: torch.Tensor


@dataclass
class Rollout:
    batch: object
    events: list[Event]
    text: str
    audio_codes: torch.Tensor
    terminated: bool


def selected_logps(logits: torch.Tensor, labels: torch.Tensor, temperature: float):
    return (
        F.log_softmax(logits.float() / temperature, dim=-1)
        .gather(-1, labels.long().unsqueeze(-1))
        .squeeze(-1)
    )


def recording_model_class():
    from liquid_audio import LFM2AudioModel

    class RecordingLFM(LFM2AudioModel):
        events: list[Event]

        def _sample_text_token(self, logits, *, temperature=None, top_k=None):
            if temperature is None or temperature <= 0 or top_k is not None:
                raise ValueError("RL recording requires positive temperature and top_k=None")
            raw = logits.detach().float().clone()
            result = super()._sample_text_token(
                logits.float(), temperature=temperature, top_k=top_k
            )
            self.events.append(
                Event(
                    "text",
                    result.detach().clone(),
                    selected_logps(raw, result.reshape(()), temperature).reshape(1),
                )
            )
            return result

        def _sample_audio_frame(self, embedding, *, temperature=None, top_k=None):
            if temperature is None or temperature <= 0 or top_k is not None:
                raise ValueError("RL recording requires positive temperature and top_k=None")
            captured = []
            originals = []
            try:
                for layer in self.depth_embeddings:
                    original = layer.get_logits
                    originals.append((layer, original, "get_logits" in vars(layer)))

                    def capture(*args, _original=original, **kwargs):
                        logits = _original(*args, **kwargs).float()
                        captured.append(logits.detach().clone())
                        return logits

                    layer.get_logits = capture
                result = super()._sample_audio_frame(
                    embedding, temperature=temperature, top_k=top_k
                )
            finally:
                for layer, original, had_instance_attribute in originals:
                    if had_instance_attribute:
                        layer.get_logits = original
                    else:
                        del layer.get_logits
            logps = selected_logps(torch.stack(captured), result, temperature)
            canonical = result.detach().clone()
            if int(canonical[0]) == 2048:
                # Upstream overwrites the other seven samples. The emitted event
                # probability is p(codebook0=EOS), marginalizing those samples.
                canonical.fill_(2048)
                logps = logps[:1]
            self.events.append(Event("audio", canonical, logps.detach()))
            return result

    return RecordingLFM


def prompt_chat(processor, path: Path, codebooks: int):
    from liquid_audio import ChatState

    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError("Expected mono input speech")
    chat = ChatState(processor, codebooks=codebooks)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(torch.from_numpy(waveform.T.copy()), sr)
    chat.end_turn()
    chat.new_turn("assistant")
    return chat


def pack_rollout(chat, events: list[Event], processor) -> Rollout:
    from liquid_audio import LFMModality
    from liquid_audio.data.types import LFM2AudioModelInput

    if not events:
        raise ValueError("No generated actions")
    prompt_length = chat.modality_flag.shape[1]
    texts = [event.tokens for event in events if event.kind == "text"]
    audios = [event.tokens for event in events if event.kind == "audio"]
    text = torch.cat(texts).unsqueeze(0) if texts else chat.text.new_empty((1, 0))
    audio = torch.stack(audios, dim=1) if audios else chat.audio_out.new_empty((chat.codebooks, 0))
    flags = chat.modality_flag.new_tensor(
        [[int(LFMModality.TEXT if e.kind == "text" else LFMModality.AUDIO_OUT) for e in events]]
    )
    chat.append(text=text, audio_out=audio, modality_flag=flags)
    mask = torch.zeros_like(chat.modality_flag, dtype=torch.bool)
    mask[:, prompt_length:] = True
    batch = LFM2AudioModelInput(**dict(chat), supervision_mask=mask)
    transcript = processor.text.decode(text[0].tolist(), skip_special_tokens=True).strip()
    # Strip only actual EOS frames; never drop the last ordinary frame blindly.
    codes = audio[:, audio[0] != 2048]
    terminated = events[-1].kind == "text" and int(events[-1].tokens[0]) == 7
    return Rollout(batch, events, transcript, codes, terminated)


@torch.no_grad()
def generate(model, processor, input_path: Path, config) -> Rollout:
    chat = prompt_chat(processor, input_path, model.codebooks)
    model.events = []
    for _ in model.generate_interleaved(
        **chat,
        max_new_tokens=config.max_new_tokens,
        text_temperature=config.temperature,
        text_top_k=None,
        audio_temperature=config.temperature,
        audio_top_k=None,
    ):
        pass
    return pack_rollout(chat, list(model.events), processor)


def score_rollout(model, rollout: Rollout, *, temperature: float, scope: str):
    text_logits, audio_logits, text_labels, audio_labels = model.logits(rollout.batch)
    text_lp = selected_logps(text_logits, text_labels, temperature)
    audio_lp = selected_logps(audio_logits, audio_labels, temperature).reshape(-1, model.codebooks)
    result, text_index, audio_index = [], 0, 0
    for event in rollout.events:
        if event.kind == "text":
            result.append(text_lp[text_index : text_index + 1])
            text_index += 1
        else:
            if scope == "joint":
                result.append(audio_lp[audio_index, : event.logps.numel()])
            audio_index += 1
    if text_index != text_lp.numel() or audio_index != audio_lp.shape[0]:
        raise ValueError("Replay labels do not align with sampled events")
    if not result:
        raise ValueError("No selected actions")
    return torch.cat(result)


def sampled_logps(rollout: Rollout, scope: str):
    return torch.cat([e.logps for e in rollout.events if e.kind == "text" or scope == "joint"])


def check_parity(replayed: torch.Tensor, sampled: torch.Tensor, tolerance: float) -> float:
    if replayed.shape != sampled.shape or not torch.isfinite(replayed).all():
        raise RuntimeError("Sampler/replay shape or finite-value failure")
    maximum = float((replayed.detach() - sampled.detach()).abs().max())
    if maximum > tolerance:
        raise RuntimeError(
            f"Sampler/replay discrepancy {maximum:.6f} > {tolerance}; no update allowed"
        )
    return maximum


@torch.no_grad()
def decode_audio(processor, rollout: Rollout):
    codes = rollout.audio_codes
    if codes.shape[1] == 0 or (codes < 0).any() or (codes >= 2048).any():
        return None
    return processor.decode(codes.unsqueeze(0)).float().cpu().reshape(-1).numpy()


@torch.no_grad()
def supervised_batch(model, processor, row, root: Path):
    from liquid_audio.data.mapper import LFM2AudioChatMapper
    from liquid_audio.data.types import (
        AudioSegment,
        ChatMessage,
        InterleavedSegment,
        LFM2AudioModelInput,
        TextSegment,
    )

    from .data import safe_audio_path

    if not row.answer_audio:
        raise ValueError("SFT/anchoring requires synthesized reference answer audio")
    mapper = LFM2AudioChatMapper(
        processor,
        codebooks=model.codebooks,
        interleaved_text_tokens=model.conf.interleaved_n_text,
        interleaved_audio_tokens=model.conf.interleaved_n_audio,
    )
    sample = mapper(
        [
            ChatMessage(
                role="system",
                content=[TextSegment(text="Respond with interleaved text and audio.")],
            ),
            ChatMessage(
                role="user",
                content=[AudioSegment(audio=safe_audio_path(root, row.input_audio).read_bytes())],
            ),
            ChatMessage(
                role="assistant",
                content=[
                    InterleavedSegment(
                        text=row.answer, audio=safe_audio_path(root, row.answer_audio).read_bytes()
                    )
                ],
            ),
        ]
    )
    return LFM2AudioModelInput(**{name: getattr(sample, name) for name in sample.__slots__}).to(
        next(model.parameters()).device
    )
