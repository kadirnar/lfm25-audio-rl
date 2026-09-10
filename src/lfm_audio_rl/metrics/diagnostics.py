"""Helpers for controlled RL studies; not automatic speech-quality rewards."""

import math

import numpy as np


def target_likelihood(log_probabilities, mask=None):
    """Teacher-forced held-out target tokens only, at temperature one.

    Call separately for text and audio tokens with an identical tokenization
    and scope. Generated-action log probabilities are not held-out perplexity.
    """
    values = np.asarray(log_probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional token log-probability array")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != values.shape or mask.dtype != bool:
            raise ValueError("Mask must be boolean with one entry per token")
        values = values[mask]
    if not values.size or not np.isfinite(values).all() or (values > 0).any():
        raise ValueError("Need finite nonpositive log probabilities for selected target tokens")
    nll = float(-values.mean())
    return {
        "n_tokens": len(values),
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "perplexity": math.exp(nll) if nll < math.log(np.finfo(float).max) else None,
    }


def pass_at_k(n, correct, k):
    """Unbiased estimator from n independent attempts at one identical prompt.

    The correctness rule must be fixed before sampling. TTS variants of the
    same dataset item are not independent policy attempts.
    """
    if any(type(v) is not int for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require integers with 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return float(-np.expm1(sum(np.log1p(-k / j) for j in range(n - correct + 1, n + 1))))


def reward_groups(rewards):
    """Diagnose exploration collapse from equal-size candidate groups."""
    x = np.asarray(rewards, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Need a finite prompt-by-candidate reward matrix")
    std = x.std(axis=1)
    return {
        "n_prompts": x.shape[0],
        "candidates_per_prompt": x.shape[1],
        "mean_reward": float(x.mean()),
        "mean_within_prompt_std": float(std.mean()),
        "zero_variance_group_fraction": float(np.mean(std == 0)),
    }
