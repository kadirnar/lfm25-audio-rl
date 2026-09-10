"""Small, explicit LoRA adapters with a frozen-base reference policy."""

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            update = F.linear(F.linear(x.to(self.a.dtype), self.a), self.b)
            out = out + update.to(out.dtype) * self.scale
        return out


def inject_lora(module: nn.Module, targets: list[str], rank: int, alpha: float) -> list[str]:
    module.requires_grad_(False)
    matches = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    if not matches:
        raise ValueError(f"No Linear modules matched LoRA targets {targets}")
    for name, child in matches:
        parent_path, _, key = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, key, LoRALinear(child, rank, alpha))
    return [name for name, _ in matches]


@contextmanager
def reference_policy(module: nn.Module):
    adapters = [m for m in module.modules() if isinstance(m, LoRALinear)]
    states = [m.enabled for m in adapters]
    try:
        for adapter in adapters:
            adapter.enabled = False
        with torch.no_grad():
            yield
    finally:
        for adapter, enabled in zip(adapters, states, strict=True):
            adapter.enabled = enabled


def adapter_state(module: nn.Module):
    return {name: p.detach().cpu() for name, p in module.named_parameters() if p.requires_grad}


def load_adapter_state(module: nn.Module, state: dict):
    params = {n: p for n, p in module.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError("Checkpoint adapter names do not match this model/config")
    with torch.no_grad():
        for name, param in params.items():
            if state[name].shape != param.shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            param.copy_(state[name])
