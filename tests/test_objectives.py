import math

import pytest
import torch

from lfm_audio_rl.objectives import advantages, policy_loss, preference_loss, reduce_actions
from lfm_audio_rl.smoke import METHODS, smoke_method


def test_advantages_match_hand_calculation():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]])
    torch.testing.assert_close(
        advantages(rewards, "rloo"), torch.tensor([[-1.5, 0, 1.5], [0, 0, 0.0]])
    )
    torch.testing.assert_close(
        advantages(rewards, "dr_grpo"), torch.tensor([[-1, 0, 1], [0, 0, 0.0]])
    )
    assert advantages(rewards, "grpo")[0].std(correction=0).item() == pytest.approx(1)
    assert advantages(rewards, "grpo")[1].count_nonzero() == 0
    with pytest.raises(ValueError):
        advantages(torch.tensor([[1.0]]), "rloo")


def test_clipping_and_masking_gradients():
    new = torch.tensor([[math.log(0.9), 100.0], [math.log(0.1), 100.0]], requires_grad=True)
    old = torch.tensor([[math.log(0.5), 0], [math.log(0.5), 0]])
    mask = torch.tensor([[True, False], [True, False]])
    # Both updates have already moved past the beneficial clipping boundary.
    loss = policy_loss(new, old, old, mask, torch.tensor([1.0, -1.0]), beta=0)
    assert loss.item() == pytest.approx((-1.2 + 0.8) / 2)
    loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros_like(new))


def test_prompt_padding_and_fixed_normalization():
    values = torch.tensor([[1.0, 2, float("nan")], [3, 999, 999]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert reduce_actions(values, mask, "sequence").item() == pytest.approx(2.25)
    assert reduce_actions(values, mask, "token").item() == pytest.approx(2)
    assert reduce_actions(values, mask, "fixed", 3).item() == pytest.approx(1)
    with pytest.raises(ValueError):
        reduce_actions(values, mask, "fixed", 1)


def test_reference_and_old_policy_receive_no_gradients():
    new = torch.tensor([[-0.8]], requires_grad=True)
    ref = torch.tensor([[-1.2]], requires_grad=True)
    old = torch.tensor([[-1.0]], requires_grad=True)
    loss = policy_loss(new, old, ref, torch.ones_like(new, dtype=torch.bool), torch.tensor([0.0]))
    assert loss > 0
    loss.backward()
    assert old.grad is None and ref.grad is None
    assert new.grad is not None


@pytest.mark.parametrize("method", ["dpo", "ipo", "simpo"])
def test_preferences_reward_chosen_and_penalize_rejected(method):
    chosen = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    rejected = torch.tensor([[-1.0]], requires_grad=True)
    cm, rm = torch.ones_like(chosen, dtype=torch.bool), torch.ones_like(rejected, dtype=torch.bool)
    loss = preference_loss(
        chosen,
        rejected,
        cm,
        rm,
        method=method,
        ref_chosen=chosen.detach(),
        ref_rejected=rejected.detach(),
    )
    loss.backward()
    assert (chosen.grad < 0).all() and (rejected.grad > 0).all()


@pytest.mark.parametrize("method", METHODS)
def test_objective_improves_independent_toy_policy(method):
    result = smoke_method(method)
    assert result["final_target_probability"] > 0.4
    assert result["is_audio_training"] is False
