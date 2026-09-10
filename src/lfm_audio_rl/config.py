from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
UPSTREAM_COMMIT = "19e65845923a7f136442c95137884ec61eb386aa"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DatasetRecipe(StrictModel):
    version: Literal["v1_clean", "v2_noisy", "v3_compositional"]
    seed: int = Field(default=42, ge=0)
    count: int = Field(default=128, ge=1, le=10000)
    tts: Literal["none", "espeak", "say"] = "none"
    voice: str = "en-us"
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    snr_db: float | None = None
    synthesize_answers: bool = True

    @model_validator(mode="after")
    def check_noise(self):
        if self.version == "v2_noisy" and self.snr_db is None:
            raise ValueError("v2_noisy requires an explicit snr_db")
        if self.version != "v2_noisy" and self.snr_db is not None:
            raise ValueError("snr_db belongs to v2_noisy")
        return self


class Experiment(StrictModel):
    algorithm: Literal["grpo", "dr_grpo", "rloo", "reinforce", "sft"] = "grpo"
    scope: Literal["text", "joint"] = "text"
    seed: int = Field(default=42, ge=0)
    steps: int = Field(default=10, ge=1)
    group_size: int = Field(default=4, ge=2, le=32)
    lr: float = Field(default=1e-5, gt=0, le=0.1)
    kl_beta: float = Field(default=0.02, ge=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    max_new_tokens: int = Field(default=256, ge=8, le=4096)
    temperature: float = Field(default=1.0, gt=0)
    # Full-support sampling is deliberate: top-k changes the behavior distribution.
    top_k: None = None
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: float = Field(default=16, gt=0)
    lora_targets: list[str] = ["q_proj", "k_proj", "v_proj", "out_proj"]
    anchor_weight: float = Field(default=0.1, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    parity_atol: float = Field(default=0.05, gt=0, le=0.1)
    model_id: str = MODEL_ID
    model_revision: str = Field(default=MODEL_REVISION, pattern=r"^[0-9a-f]{40}$")
    asr_model: str = "base.en"
    asr_device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_run(self):
        if not self.lora_targets or any(not t for t in self.lora_targets):
            raise ValueError("lora_targets must be nonempty")
        if self.algorithm == "sft" and self.scope != "joint":
            raise ValueError("SFT baseline uses joint supervision")
        return self


def read_config(path: Path, cls):
    with path.open() as stream:
        return cls.model_validate(yaml.safe_load(stream))
