import random

import pytest
import torch
from torch import nn

from lfm_audio_rl.lora import inject_lora
from lfm_audio_rl.train import load_checkpoint, save_checkpoint


def test_checkpoint_restores_optimizer_adapter_and_rng(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = nn.Sequential(nn.Linear(4, 2))
    inject_lora(model, ["0"], rank=2, alpha=4)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params)
    x = torch.randn(3, 4)
    model(x).sum().backward()
    optimizer.step()
    expected = model(x).detach()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 1, "identity")
    torch_expected, python_expected = torch.rand(4), random.random()
    with torch.no_grad():
        for p in params:
            p.zero_()
    assert load_checkpoint(path, model, optimizer, "identity") == 1
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(torch.rand(4), torch_expected)
    assert random.random() == python_expected
    assert optimizer.state[params[0]]["step"] == 1
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, model, optimizer, "wrong")
