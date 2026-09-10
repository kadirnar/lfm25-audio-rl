import pytest
import torch
from torch import nn

from lfm_audio_rl.lora import adapter_state, inject_lora, load_adapter_state, reference_policy


def test_lora_reference_freeze_and_checkpoint_roundtrip():
    model = nn.Sequential(nn.Linear(4, 3))
    x = torch.randn(2, 4)
    baseline = model(x).detach()
    inject_lora(model, ["0"], rank=2, alpha=4)
    torch.testing.assert_close(model(x), baseline)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(x).sum().backward()
    optimizer.step()
    assert not torch.allclose(model(x), baseline)
    assert model[0].base.weight.grad is None
    with reference_policy(model):
        torch.testing.assert_close(model(x), baseline)
    with pytest.raises(RuntimeError), reference_policy(model):
        raise RuntimeError("test state restoration")
    assert model[0].enabled
    state = {k: v.clone() for k, v in adapter_state(model).items()}
    expected = model(x).detach()
    with torch.no_grad():
        model[0].b.zero_()
    load_adapter_state(model, state)
    torch.testing.assert_close(model(x), expected)
