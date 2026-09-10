"""Explicit objective variants. All log probabilities refer to actual sampled actions."""

import torch
from torch import Tensor
from torch.nn import functional as F


def advantages(rewards: Tensor, method: str) -> Tensor:
    """Shape [prompts, group]. RLOO excludes the current sample from its baseline."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Finite rewards with shape [prompts, group>=2] are required")
    centered = rewards - rewards.mean(-1, keepdim=True)
    if method == "grpo":
        return centered / rewards.std(-1, keepdim=True, correction=0).clamp_min(1e-6)
    if method == "dr_grpo":
        return centered
    if method == "rloo":
        return centered * rewards.shape[1] / (rewards.shape[1] - 1)
    if method == "reinforce":
        return rewards.clone()  # Deliberately no fitted or batch baseline.
    raise ValueError(f"Unknown advantage method: {method}")


def _validate(logps: Tensor, mask: Tensor):
    if logps.ndim != 2 or mask.shape != logps.shape or mask.dtype != torch.bool:
        raise ValueError("Expected log probabilities and boolean mask with shape [batch, actions]")
    if not mask.any(-1).all():
        raise ValueError("Each trajectory must have at least one selected action")
    if not torch.isfinite(logps[mask]).all():
        raise ValueError("Non-finite action log probability")


def reduce_actions(values: Tensor, mask: Tensor, reduction: str, max_actions: int | None = None):
    _validate(values, mask)
    values = values.masked_fill(~mask, 0)
    if reduction == "sequence":
        return (values.sum(-1) / mask.sum(-1)).mean()
    if reduction == "token":
        return values.sum() / mask.sum()
    if reduction == "fixed":
        if max_actions is None or max_actions < int(mask.sum(-1).max()):
            raise ValueError("fixed reduction requires a valid upper bound on action count")
        return values.sum() / (values.shape[0] * max_actions)
    if reduction == "sum":
        return values.sum(-1).mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def policy_loss(
    new: Tensor,
    old: Tensor,
    ref: Tensor,
    mask: Tensor,
    advantage: Tensor,
    *,
    epsilon: float = 0.2,
    beta: float = 0.02,
    reduction: str = "sequence",
    max_actions: int | None = None,
) -> Tensor:
    """Clipped token-action surrogate plus sampled k3 reference regularization.

    k3 is a nonnegative sample estimator; with old-policy samples after updates it is
    a surrogate regularizer, not an exact full-distribution KL measurement.
    """
    _validate(new, mask)
    if old.shape != new.shape or ref.shape != new.shape or advantage.shape != new.shape[:1]:
        raise ValueError("Mismatched policy loss shapes")
    _validate(old, mask)
    _validate(ref, mask)
    if not torch.isfinite(advantage).all():
        raise ValueError("Non-finite advantage")
    log_ratio = (new - old.detach()).masked_fill(~mask, 0)
    # Fail clearly instead of silently changing the objective by clamping log ratios.
    if (log_ratio[mask].abs() > 60).any():
        raise FloatingPointError("Importance ratio overflow; inspect sampler/replay parity")
    ratio = log_ratio.exp()
    adv = advantage.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    delta = (ref.detach() - new).masked_fill(~mask, 0)
    kl = delta.exp() - delta - 1 if beta else torch.zeros_like(new)
    return reduce_actions(-surrogate + beta * kl, mask, reduction, max_actions)


def reinforce_loss(logps: Tensor, mask: Tensor, advantage: Tensor) -> Tensor:
    _validate(logps, mask)
    return -(logps.masked_fill(~mask, 0).sum(-1) * advantage.detach()).mean()


def preference_loss(
    chosen: Tensor,
    rejected: Tensor,
    chosen_mask: Tensor,
    rejected_mask: Tensor,
    *,
    method: str = "dpo",
    ref_chosen: Tensor | None = None,
    ref_rejected: Tensor | None = None,
    beta: float = 0.1,
    margin: float = 0.5,
):
    """DPO uses sums; IPO and SimPO use length-normalized sequence scores."""
    _validate(chosen, chosen_mask)
    _validate(rejected, rejected_mask)
    if beta <= 0 or chosen.shape[0] != rejected.shape[0]:
        raise ValueError("Positive beta and matched pair batch sizes required")

    def score(x, mask):
        total = x.masked_fill(~mask, 0).sum(-1)
        return total if method == "dpo" else total / mask.sum(-1)

    gap = score(chosen, chosen_mask) - score(rejected, rejected_mask)
    if method == "simpo":
        return -F.logsigmoid(beta * gap - margin).mean()
    if ref_chosen is None or ref_rejected is None:
        raise ValueError("DPO/IPO require reference scores")
    _validate(ref_chosen, chosen_mask)
    _validate(ref_rejected, rejected_mask)
    gap = gap - (score(ref_chosen, chosen_mask) - score(ref_rejected, rejected_mask)).detach()
    if method == "dpo":
        return -F.logsigmoid(beta * gap).mean()
    if method == "ipo":
        return (gap - 1 / (2 * beta)).square().mean()
    raise ValueError(f"Unknown preference method: {method}")
