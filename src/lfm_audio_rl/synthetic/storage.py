import importlib.metadata
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from ..data import digest, file_hash, load_dataset


def implementation_hash():
    folder = Path(__file__).parent
    files = list(folder.glob("*.py")) + [
        folder.parent / "data.py",
        folder.parent / "config.py",
        folder.parent / "rewards.py",
    ]
    return digest({str(p.relative_to(folder.parent)): file_hash(p) for p in files})


def runtime_identity():
    packages = {}
    for name in [
        "numpy",
        "scipy",
        "pydantic",
        "httpx",
        "soundfile",
        "faster-whisper",
        "ctranslate2",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages}


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@contextmanager
def run_directory(output: Path, identity: dict):
    identity = dict(
        identity, implementation_sha256=implementation_hash(), runtime=runtime_identity()
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        marker = output / "run.json"
        if marker.exists():
            if json.loads(marker.read_text())["identity"] != digest(identity):
                raise ValueError(
                    "Output belongs to a different configuration or input; use a new directory"
                )
        elif any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Output is nonempty and has no run.json")
        else:
            atomic_json(marker, {"identity": digest(identity), "config": identity})
        yield
    finally:
        os.close(fd)
        lock.unlink()


def publish(output: Path, rows, recipe: dict, **extra):
    if not rows:
        raise ValueError("No accepted examples; inspect rejected.jsonl")
    temp = output / "manifest.jsonl.tmp"
    temp.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    temp.replace(output / "manifest.jsonl")
    result = {
        "schema_version": 1,
        "recipe": recipe,
        "manifest_sha256": file_hash(output / "manifest.jsonl"),
        "count": len(rows),
        "splits": {s: sum(r.split == s for r in rows) for s in ["train", "validation", "test"]},
        "has_speech": all(r.input_audio for r in rows),
        **extra,
    }
    atomic_json(output / "dataset.json", result)
    load_dataset(output, require_audio=bool(result["has_speech"]))
    return result
