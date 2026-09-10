import json
import math
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..data import digest, file_hash, load_dataset
from ..rewards import word_error_rate
from .config import AudioQA, SpeechConfig
from .storage import atomic_json, publish, run_directory

WORKER_SCRIPT = Path(__file__).with_name("worker.py")


class AudioRejected(ValueError):
    pass


def normalize_audio(source: Path, target: Path, qa: AudioQA, text: str, transcribe=None):
    wave, sr = sf.read(source, dtype="float32", always_2d=True)
    if (
        sr < 8000
        or sr > 192000
        or wave.shape[1] not in (1, 2)
        or not wave.size
        or not np.isfinite(wave).all()
    ):
        raise AudioRejected("Invalid waveform, channel count or sample rate")
    duration = len(wave) / sr
    if not qa.min_seconds <= duration <= qa.max_seconds:
        raise AudioRejected("Audio duration outside configured bounds")
    clipping = float(np.mean(np.abs(wave) >= 0.999))
    if clipping > qa.max_clipped_fraction or float(np.max(np.abs(wave))) > 1.0:
        raise AudioRejected("Audio is clipped or exceeds full scale")
    mono = wave.mean(axis=1)
    if float(np.sqrt(np.mean(mono**2))) < qa.min_rms:
        raise AudioRejected("Audio is silent or too quiet")
    factor = math.gcd(sr, 24000)
    mono = resample_poly(mono, 24000 // factor, sr // factor) if sr != 24000 else mono
    # Bandlimited conversion can overshoot slightly; retain native audio and record gain.
    gain = min(1.0, 0.98 / max(1e-12, float(np.max(np.abs(mono)))))
    mono *= gain
    sf.write(target, mono, 24000, subtype="PCM_16")
    report = {
        "native_sample_rate": sr,
        "native_channels": wave.shape[1],
        "seconds": duration,
        "clipped_fraction": clipping,
        "gain": gain,
        "asr_checked": transcribe is not None,
    }
    if transcribe:
        transcript = transcribe(target)
        wer = word_error_rate(text, transcript)
        report.update(transcript=transcript, wer=wer)
        if wer > qa.max_wer:
            target.unlink()
            raise AudioRejected(f"Transcript WER {wer:.3f} exceeds {qa.max_wer}")
    return report


class Worker:
    def __init__(self, profile: dict, log: Path, timeout: float):
        self.profile, self.log, self.timeout = profile, log, timeout
        self.process = None

    def __enter__(self):
        self.stderr = self.log.open("a")
        env = dict(os.environ)
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen(
                [self.profile["python"], "-u", str(WORKER_SCRIPT)],
                cwd=self.profile["source_dir"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                text=True,
                bufsize=1,
                env=env,
            )
            self.events = queue.Queue()

            def read():
                for line in self.process.stdout:
                    self.events.put(line)
                self.events.put(None)

            self.reader = threading.Thread(target=read, daemon=True)
            self.reader.start()
            self._send(self.profile)
            ready = self._receive("ready")
            self.runtime = ready["runtime"]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def _receive(self, event):
        try:
            line = self.events.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError(f"TTS worker timed out; inspect {self.log}") from None
        if line is None:
            raise RuntimeError(f"TTS worker exited; inspect {self.log}")
        response = json.loads(line)
        if response.get("event") != event:
            raise RuntimeError(
                f"TTS worker failed ({response.get('error_type', 'protocol error')}); inspect {self.log}"
            )
        return response

    def synthesize(self, text, seed, output):
        identity = digest({"text": text, "seed": seed, "output": str(output)})
        self._send({"id": identity, "text": text, "seed": seed, "output": str(output)})
        if self._receive("done")["id"] != identity:
            raise RuntimeError("TTS worker returned a different job ID")

    def __exit__(self, *args):
        if self.process is not None:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if hasattr(self, "reader"):
                self.reader.join(timeout=1)
            self.process.stdout.close()
        self.stderr.close()


def resolve_profiles(config: SpeechConfig):
    profiles = []
    for engine in config.engines:
        p = engine.resolved()
        for key in ["python", "source_dir", "reference_audio"]:
            if p.get(key):
                p[key] = str(
                    Path(p[key]).expanduser().absolute()
                    if key == "python"
                    else Path(p[key]).expanduser().resolve()
                )
        if not Path(p["python"]).is_file():
            raise ValueError(f"Worker Python not found for {engine.name}: {p['python']}")
        source = Path(p["source_dir"])
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True, capture_output=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if sha != p["source_revision"] or dirty:
            raise ValueError(f"{engine.name} needs a clean checkout at {p['source_revision']}")
        if p["reference_audio"]:
            p["reference_sha256"] = file_hash(Path(p["reference_audio"]))
        profiles.append(p)
    return profiles


def render_speech(
    config: SpeechConfig, text_data: Path, output: Path, worker_factory=Worker, transcribe=None
):
    text_rows, text_meta = load_dataset(text_data)
    profiles = resolve_profiles(config)
    identity = {
        "stage": "speech-v1",
        "config": config.model_dump(),
        "profiles": profiles,
        "text_manifest": text_meta["manifest_sha256"],
    }
    output = output.resolve()
    with run_directory(output, identity):
        if (output / "dataset.json").exists():
            return load_dataset(output, require_audio=True)[1]
        if config.qa.asr_model and transcribe is None:
            from faster_whisper import WhisperModel

            asr = WhisperModel(config.qa.asr_model, device="cpu", compute_type="int8")

            def transcribe(path):
                segments, _ = asr.transcribe(str(path), language="en", beam_size=5)
                return " ".join(segment.text for segment in segments)

        rows, rejected, runtimes = [], [], {}
        for folder in ["audio", "native", "clips", "logs"]:
            (output / folder).mkdir(exist_ok=True)
        for p in profiles:
            with worker_factory(
                p, output / "logs" / f"{p['name']}.log", config.worker_timeout_seconds
            ) as worker:
                runtime_file = output / "logs" / f"{p['name']}.runtime.json"
                if runtime_file.exists() and json.loads(runtime_file.read_text()) != worker.runtime:
                    raise ValueError(
                        "Worker runtime changed during resume; use a new output directory"
                    )
                atomic_json(runtime_file, worker.runtime)
                runtimes[p["name"]] = worker.runtime
                for row in text_rows:
                    example_id = digest({"text_id": row.id, "profile": p, "seed": config.seed})[:24]
                    changes, reports = {}, {}
                    try:
                        for role, text in [("input", row.prompt), ("answer", row.answer)]:
                            clip_id = f"{example_id}-{role}"
                            native = output / "native" / f"{clip_id}.wav"
                            target = output / "audio" / f"{clip_id}.wav"
                            receipt = output / "clips" / f"{clip_id}.json"
                            seed = int(
                                digest({"seed": config.seed, "row": row.id, "role": role})[:8], 16
                            ) % (2**31)
                            if receipt.exists():
                                report = json.loads(receipt.read_text())
                                if (
                                    file_hash(native) != report["native_sha256"]
                                    or file_hash(target) != report["sha256"]
                                ):
                                    raise ValueError("Cached audio hash mismatch")
                            else:
                                temporary = native.with_suffix(".tmp.wav")
                                worker.synthesize(text, seed, temporary)
                                temporary.replace(native)
                                report = normalize_audio(
                                    native, target, config.qa, text, transcribe
                                )
                                report.update(
                                    seed=seed,
                                    native_sha256=file_hash(native),
                                    sha256=file_hash(target),
                                )
                                atomic_json(receipt, report)
                            changes[f"{role}_audio"] = target.relative_to(output).as_posix()
                            changes[f"{role}_sha256"] = report["sha256"]
                            reports[role] = report
                    except AudioRejected as exc:
                        rejected.append(
                            {"text_id": row.id, "profile": p["name"], "reason": str(exc)}
                        )
                        (output / "rejected.jsonl").write_text(
                            "".join(json.dumps(r) + "\n" for r in rejected)
                        )
                        continue
                    provenance = dict(
                        row.provenance,
                        tts=p,
                        audio_qa=reports,
                        text_manifest_sha256=text_meta["manifest_sha256"],
                    )
                    rows.append(
                        row.model_copy(
                            update={"id": example_id, "provenance": provenance, **changes}
                        )
                    )
        (output / "rejected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
        return publish(
            output,
            rows,
            identity,
            rejected_count=len(rejected),
            requested_count=len(text_rows) * len(profiles),
            runtimes=runtimes,
            quality="waveform_and_asr" if transcribe else "waveform_only",
            output_licenses=sorted({p["output_license"] for p in profiles}),
        )
