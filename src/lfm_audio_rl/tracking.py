"""Optional W&B scalar logging. Local checkpoints remain the source of truth."""

import importlib.util
import json
import math
import random
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def check_tracking(config):
    if config.mode != "disabled" and importlib.util.find_spec("wandb") is None:
        raise RuntimeError("W&B logging requires the wandb extra: uv sync --extra wandb")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Do not initialize CUDA just to log a CPU test or an offline run.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class Tracker:
    def __init__(self, config, output, identity, metadata, start_step=0):
        self.config, self.output, self.identity = config, Path(output), identity
        self.run = None
        self.state_path = self.output / "wandb-state.json"
        self.segment = None
        self.state = None
        self.metadata, self.start_step = metadata, start_step

    def __enter__(self):
        if self.config.mode == "disabled":
            return self
        check_tracking(self.config)
        with preserve_rng():
            import wandb

            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
                if self.state["identity"] != self.identity:
                    raise ValueError("W&B state belongs to another training run")
            else:
                self.state = {
                    "schema_version": 1,
                    "identity": self.identity,
                    "lineage": uuid.uuid4().hex[:12],
                    "segments": [],
                }
            previous = self.state["segments"][-1] if self.state["segments"] else None
            same_online_run = (
                previous is not None
                and all(
                    previous[k] == getattr(self.config, k) for k in ["mode", "entity", "project"]
                )
                and self.config.mode == "online"
            )
            run_id = previous["id"] if same_online_run else uuid.uuid4().hex[:12]
            self.segment = dict(
                id=run_id,
                mode=self.config.mode,
                project=self.config.project,
                entity=self.config.entity,
                start_step=self.start_step,
                status="starting",
            )
            self.state["segments"].append(self.segment)
            # Persist the ID before SDK initialization so a failed init is retryable.
            atomic_json(self.state_path, self.state)
            try:
                self.run = wandb.init(
                    id=run_id,
                    project=self.config.project,
                    entity=self.config.entity,
                    name=self.config.name or self.output.name,
                    group=self.config.group or self.state["lineage"],
                    tags=self.config.tags,
                    job_type="train",
                    mode=self.config.mode,
                    force=self.config.mode == "online",
                    resume="allow" if same_online_run else None,
                    reinit="create_new",
                    dir=str(self.output.resolve()),
                    config=dict(self.metadata, training_identity=self.identity),
                    save_code=False,
                    settings={"console": "off", "disable_git": True, "disable_code": True},
                )
                if self.run is None:
                    raise RuntimeError("W&B did not create a run")
                self.run.define_metric("trainer/step")
                self.run.define_metric("train/*", step_metric="trainer/step")
                self.run.summary["resumed_from_step"] = self.start_step
                self.run.summary["last_checkpoint_step"] = self.start_step
                self.segment.update(
                    status="running", url=self.run.url if self.config.mode == "online" else None
                )
                atomic_json(self.state_path, self.state)
            except BaseException:
                if self.run is not None:
                    try:
                        self.run.finish(exit_code=1)
                    except Exception:
                        warnings.warn(
                            "W&B cleanup failed after initialization error",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                self.segment["status"] = "failed"
                atomic_json(self.state_path, self.state)
                raise
        return self

    def log(self, metrics, checkpoint_step, final=False):
        if self.run is None or (metrics["step"] % self.config.log_every and not final):
            return
        values = {"trainer/step": metrics["step"], "train/checkpoint_step": checkpoint_step}
        for key, value in metrics.items():
            if key != "step" and isinstance(value, (float, int, bool)) and math.isfinite(value):
                values[f"train/{key}"] = value
        with preserve_rng():
            # Custom x-axis permits replay of an uncheckpointed update after a crash.
            # W&B's own history index remains monotonic; no explicit SDK step is set.
            self.run.log(values)
            self.run.summary["last_optimizer_step"] = metrics["step"]
            self.run.summary["last_checkpoint_step"] = checkpoint_step

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            with preserve_rng():
                status = "finished" if exc_type is None else "failed"
                try:
                    self.run.summary["training_status"] = status
                    self.run.finish(exit_code=0 if exc_type is None else 1)
                except Exception:
                    status = "failed"
                    if exc_type is None:
                        raise
                    warnings.warn(
                        "W&B cleanup failed; preserving the training error",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                finally:
                    self.segment["status"] = status
                    atomic_json(self.state_path, self.state)
        return False
