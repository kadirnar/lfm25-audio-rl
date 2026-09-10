"""Tiny categorical policies for verifying optimization, not an audio benchmark."""

import json
import math
from pathlib import Path

import torch

from .objectives import advantages, policy_loss, preference_loss, reinforce_loss

METHODS = ("sft", "grpo", "dr_grpo", "rloo", "reinforce", "dpo", "ipo", "simpo")


def smoke_method(method: str, *, steps: int = 80, seed: int = 42) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown smoke method: {method}")
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(torch.zeros(8, 4))
    optimizer = torch.optim.Adam([logits], lr=0.08)
    target = torch.arange(8) % 4
    mask = torch.ones((8, 1), dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad()
        lp = logits.log_softmax(-1)
        if method == "sft":
            loss = -lp.gather(1, target[:, None]).mean()
        elif method in {"dpo", "ipo", "simpo"}:
            negative = (target + torch.randint(1, 4, (8,))) % 4
            loss = preference_loss(
                lp.gather(1, target[:, None]),
                lp.gather(1, negative[:, None]),
                mask,
                mask,
                method=method,
                ref_chosen=torch.full((8, 1), -math.log(4)),
                ref_rejected=torch.full((8, 1), -math.log(4)),
                beta=0.5,
            )
        else:
            sampled = torch.multinomial(lp.detach().exp(), 8, replacement=True)
            reward = (sampled == target[:, None]).float()
            adv = advantages(reward, method).flatten()
            new = lp.gather(1, sampled).reshape(-1, 1)
            active = torch.ones_like(new, dtype=torch.bool)
            if method in {"rloo", "reinforce"}:
                loss = reinforce_loss(new, active, adv)
            else:
                loss = policy_loss(
                    new,
                    new.detach(),
                    torch.full_like(new, -math.log(4)),
                    active,
                    adv,
                    beta=0.01,
                    reduction="fixed" if method == "dr_grpo" else "sequence",
                    max_actions=1,
                )
        loss.backward()
        if not torch.isfinite(logits.grad).all():
            raise FloatingPointError("Non-finite smoke-test gradient")
        optimizer.step()
    probability = float(logits.detach().softmax(-1).gather(1, target[:, None]).mean())
    return {
        "backend": "toy_categorical",
        "is_audio_training": False,
        "method": method,
        "seed": seed,
        "steps": steps,
        "initial_target_probability": 0.25,
        "final_target_probability": probability,
        "final_loss": float(loss.detach()),
    }


def run_smoke(output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    results = [smoke_method(method) for method in METHODS]
    (output / "smoke.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["final_target_probability"] <= 0.4 for row in results):
        raise RuntimeError("An objective failed to improve the toy policy")
    return results
