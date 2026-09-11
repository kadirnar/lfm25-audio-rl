"""Explicit optimizer settings and recomputation for teacher-forced replay."""

import functools
import math

import torch
from torch.utils.checkpoint import checkpoint


def make_optimizer(parameters, config):
    params = list(parameters)
    options = config.train_opt
    kwargs = dict(
        lr=config.lr, betas=options.betas, eps=options.eps, weight_decay=options.weight_decay
    )
    if options.optimizer == "adamw_fused":
        if not params or any(p.device.type != "cuda" for p in params):
            raise ValueError("adamw_fused requires CUDA parameters in this runner")
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def learning_rate(config, update):
    """LR for a one-based optimizer update; derivable from checkpoint step."""
    if not 1 <= update <= config.steps:
        raise ValueError("Update must be in [1, steps]")
    options = config.train_opt
    if options.warmup_steps and update <= options.warmup_steps:
        return config.lr * update / options.warmup_steps
    if options.scheduler == "constant":
        return config.lr
    remaining = config.steps - options.warmup_steps
    progress = (update - options.warmup_steps - 1) / max(1, remaining - 1)
    multiplier = (
        1 - progress if options.scheduler == "linear" else (1 + math.cos(math.pi * progress)) / 2
    )
    return config.lr * (options.min_lr_ratio + (1 - options.min_lr_ratio) * multiplier)


def enable_gradient_checkpointing(backbone):
    """Recompute LFM backbone layers only during uncached gradient passes.

    The runner deliberately keeps eval mode for dropout-free policy replay.
    Hugging Face's training-mode checkpoint switch would not apply here.
    Non-reentrant checkpointing supports frozen inputs with trainable LoRA.
    """
    layers = getattr(backbone, "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Expected the audited LFM backbone's layers")
    for layer in layers:
        if getattr(layer, "_lfm_rl_checkpointed", False):
            continue
        original = layer.forward

        @functools.wraps(original)
        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                if kwargs.get("past_key_values") is not None:
                    raise ValueError("Gradient checkpointing requires uncached replay")
                return checkpoint(
                    _original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
                )
            return _original(*args, **kwargs)

        layer.forward = forward
        layer._lfm_rl_checkpointed = True
    return len(layers)
