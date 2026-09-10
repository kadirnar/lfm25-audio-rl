"""Deterministic semantic examples, local speech synthesis, and immutable manifests."""

import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import Field

from .config import DatasetRecipe, StrictModel


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number_words(n: int) -> str:
    small = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if not 0 <= n < 1000:
        raise ValueError("number_words supports integers in [0, 999]")
    if n < 20:
        return small[n]
    if n < 100:
        return tens[n // 10] + (" " + small[n % 10] if n % 10 else "")
    return small[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")


def normalize(text: str) -> str:
    text = re.sub(r"<\|[^>]+\|>", "", text.lower())
    text = re.sub(r"\b\d{1,3}\b", lambda m: number_words(int(m[0])), text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


class Example(StrictModel):
    schema_version: int = 1
    id: str
    semantic_group: str
    split: Literal["train", "validation", "test"]
    task: str
    prompt: str
    answer: str
    language: Literal["en"] = "en"
    input_audio: str | None = None
    input_sha256: str | None = None
    answer_audio: str | None = None
    answer_sha256: str | None = None
    provenance: dict = Field(default_factory=dict)


def semantic_examples(recipe: DatasetRecipe) -> list[Example]:
    rng = random.Random(recipe.seed)
    rows, seen = [], set()
    while len(rows) < recipe.count:
        a, b = rng.randrange(100), rng.randrange(100)
        task = "addition" if len(rows) % 2 == 0 else "comparison"
        group = {"task": task, "a": min(a, b), "b": max(a, b)}
        if recipe.version == "v3_compositional":
            task = "two_step"
            c = rng.randrange(100)
            group = {"task": task, "a": min(a, b), "b": max(a, b), "c": c}
            prompt = f"Add {number_words(a)} and {number_words(b)}, then add {number_words(c)}. Reply with only the final number."
            answer = number_words(a + b + c)
        elif task == "addition":
            prompt = (
                f"What is {number_words(a)} plus {number_words(b)}? Reply with only the number."
            )
            answer = number_words(a + b)
        else:
            prompt = f"Which number is larger, {number_words(a)} or {number_words(b)}? Reply with only that number."
            answer = number_words(max(a, b))
        group_id = digest(group)
        if group_id in seen:
            continue
        seen.add(group_id)
        # Split identity excludes voice, waveform, data version and paraphrase.
        bucket = int(group_id[:8], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        rows.append(
            Example(
                id=group_id[:20],
                semantic_group=group_id,
                split=split,
                task=task,
                prompt=prompt,
                answer=answer,
                provenance={
                    "generator": "symbolic-v1",
                    "seed": recipe.seed,
                    "recipe": recipe.version,
                    "operands": group,
                },
            )
        )
    return rows


def synthesize(text: str, target: Path, recipe: DatasetRecipe) -> None:
    if recipe.tts == "say":
        executable = shutil.which("say")
        if not executable:
            raise RuntimeError("The say provider requires macOS; use espeak on Linux")
        argv = [
            executable,
            "-v",
            recipe.voice,
            "-o",
            str(target),
            "--file-format=WAVE",
            f"--data-format=LEI16@{recipe.sample_rate}",
            text,
        ]
    elif recipe.tts == "espeak":
        executable = shutil.which("espeak-ng")
        if not executable:
            raise RuntimeError("Install espeak-ng or choose tts: none for text-only manifests")
        argv = [executable, "-v", recipe.voice, "-w", str(target), text]
    else:
        raise ValueError("No speech provider selected")
    subprocess.run(argv, check=True, capture_output=True, timeout=120)
    wave, sr = sf.read(target, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if sr != recipe.sample_rate:
        # Linear interpolation is sufficient for this development TTS provider.
        # Neural TTS production datasets should use a bandlimited resampler.
        length = round(len(wave) * recipe.sample_rate / sr)
        wave = np.interp(np.arange(length) * sr / recipe.sample_rate, np.arange(len(wave)), wave)
    if wave.size == 0 or not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5:
        raise ValueError("TTS produced empty, invalid, or silent audio")
    sf.write(target, wave, recipe.sample_rate, subtype="PCM_16")


def build_dataset(recipe: DatasetRecipe, output: Path) -> dict:
    # No accidental overwrite of a dataset under comparison.
    output.mkdir(parents=True, exist_ok=False)
    rows = semantic_examples(recipe)
    if recipe.tts != "none":
        (output / "audio").mkdir()
        for index, row in enumerate(rows):
            changes = {}
            for role, text in [("input", row.prompt), ("answer", row.answer)]:
                if role == "answer" and not recipe.synthesize_answers:
                    continue
                path = output / "audio" / f"{row.id}-{role}.wav"
                synthesize(text, path, recipe)
                if role == "input" and recipe.snr_db is not None:
                    wave, sr = sf.read(path, dtype="float32")
                    noise = np.random.default_rng(recipe.seed + index).normal(size=wave.shape)
                    rms = math.sqrt(float(np.mean(wave**2)))
                    noise *= rms / (10 ** (recipe.snr_db / 20)) / np.sqrt(np.mean(noise**2))
                    mixed = wave + noise
                    mixed /= max(1.0, float(np.max(np.abs(mixed))) / 0.95)
                    sf.write(path, mixed, sr, subtype="PCM_16")
                key = "input" if role == "input" else "answer"
                changes[f"{key}_audio"] = path.relative_to(output).as_posix()
                changes[f"{key}_sha256"] = file_hash(path)
            provenance = dict(
                row.provenance,
                tts=recipe.tts,
                voice=recipe.voice,
                platform=platform.platform(),
                synthetic=True,
            )
            rows[index] = row.model_copy(update={**changes, "provenance": provenance})
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    metadata = {
        "schema_version": 1,
        "recipe": recipe.model_dump(),
        "manifest_sha256": file_hash(manifest),
        "count": len(rows),
        "splits": {s: sum(r.split == s for r in rows) for s in ["train", "validation", "test"]},
        "has_speech": recipe.tts != "none",
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def safe_audio_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Audio paths must stay inside the dataset directory")
    return path


def load_dataset(root: Path, require_audio: bool = False) -> tuple[list[Example], dict]:
    metadata = json.loads((root / "dataset.json").read_text())
    if file_hash(root / "manifest.jsonl") != metadata["manifest_sha256"]:
        raise ValueError("Dataset manifest hash mismatch")
    rows = [
        Example.model_validate_json(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    if len(rows) != metadata["count"] or len({r.id for r in rows}) != len(rows):
        raise ValueError("Dataset has a count mismatch or duplicate IDs")
    groups = {}
    for row in rows:
        if groups.setdefault(row.semantic_group, row.split) != row.split:
            raise ValueError("Semantic group leakage across splits")
        if require_audio and not row.input_audio:
            raise ValueError(
                "Real model runs require synthetic input speech; this is a text-only manifest"
            )
        for role in ["input", "answer"]:
            relative, expected = getattr(row, f"{role}_audio"), getattr(row, f"{role}_sha256")
            if relative:
                path = safe_audio_path(root, relative)
                if not expected or file_hash(path) != expected:
                    raise ValueError(f"Audio hash mismatch: {relative}")
                info = sf.info(path)
                if info.frames <= 0 or info.channels != 1 or info.samplerate < 8000:
                    raise ValueError(f"Invalid audio metadata: {relative}")
    return rows, metadata
