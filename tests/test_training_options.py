"""Run the optimizer loop on CPU with fake preprocessing, not pretrained speech."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from lfm_audio_rl.config import Experiment, TrainingOptions
from lfm_audio_rl.data import Example
from lfm_audio_rl.lora import adapter_state
from lfm_audio_rl.train import train, train_example


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lfm = nn.Sequential(nn.Linear(2, 1))

    def forward(self, value):
        return SimpleNamespace(loss=(self.lfm(value) - 1).square().mean())


def setup(monkeypatch):
    from lfm_audio_rl import lfm

    module = importlib.import_module("lfm_audio_rl.train")
    rows = [
        Example(
            id=str(i),
            semantic_group=str(i),
            split="train",
            task="color_recall",
            prompt="color?",
            answer="red",
            input_audio="input.wav",
            answer_audio="answer.wav",
        )
        for i in range(2)
    ]
    monkeypatch.setattr(module, "preflight", lambda *args: (rows, {"manifest_sha256": "fixture"}))
    for name in ["synchronize", "reset_peak_memory_stats", "manual_seed_all"]:
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test fixture")
    fake = ModuleType("liquid_audio")
    fake.LFM2AudioProcessor = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(eval=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "liquid_audio", fake)
    models = []

    def load(*args, **kwargs):
        model = Tiny()
        models.append(model)
        return model

    monkeypatch.setattr(lfm, "recording_model_class", lambda: SimpleNamespace(from_pretrained=load))

    def batch(model, processor, row, root):
        return torch.tensor([[1.0, float(int(row.id) + 1)]])

    monkeypatch.setattr(lfm, "supervised_batch", batch)
    return module, rows, models, batch


def config(**options):
    return Experiment(
        algorithm="sft",
        scope="joint",
        steps=4,
        lr=0.001,
        lora_targets=["0"],
        train_opt=TrainingOptions(**options),
    )


def test_accumulated_sft_gradient_equals_mean_of_prompt_losses(tmp_path, monkeypatch):
    from lfm_audio_rl.lora import inject_lora

    _, rows, _, batch = setup(monkeypatch)
    torch.manual_seed(7)
    model = Tiny()
    inject_lora(model.lfm, ["0"], 2, 4)
    expected = torch.stack([model(batch(model, None, row, None)).loss for row in rows]).mean()
    expected.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    cfg = config(gradient_accumulation_steps=2)
    for i, row in enumerate(rows):
        train_example(cfg, tmp_path, tmp_path, row, model, None, None, 1, i)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])


def test_resumed_optimization_matches_uninterrupted_run(tmp_path, monkeypatch):
    module, _, models, _ = setup(monkeypatch)
    cfg = config(gradient_accumulation_steps=2, scheduler="cosine", warmup_steps=1)
    train(cfg, tmp_path, tmp_path / "full")
    expected = {k: v.clone() for k, v in adapter_state(models[-1]).items()}
    original = module.save_checkpoint

    def crash(path, model, optimizer, step, identity):
        original(path, model, optimizer, step, identity)
        if step == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(module, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        train(cfg, tmp_path, tmp_path / "resume")
    monkeypatch.setattr(module, "save_checkpoint", original)
    train(cfg, tmp_path, tmp_path / "resume", resume=True)
    for key, value in adapter_state(models[-1]).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    logs = [
        json.loads(line)
        for line in (tmp_path / "resume" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in logs] == [1, 2, 3, 4]
    assert all(row["example_ids"] == ["0", "1"] for row in logs)
    assert logs[-1]["lr"] == pytest.approx(cfg.lr * 0.1)


def test_checkpoint_interval_always_saves_final_update(tmp_path, monkeypatch):
    module, _, _, _ = setup(monkeypatch)
    saved = []
    original = module.save_checkpoint

    def save(*args):
        saved.append(args[3])
        original(*args)

    monkeypatch.setattr(module, "save_checkpoint", save)
    train(config(checkpoint_every=3), tmp_path, tmp_path / "run")
    assert saved == [0, 3, 4]


@pytest.mark.parametrize("algorithm", ["grpo", "dr_grpo", "rloo", "reinforce"])
def test_rl_groups_accumulate_without_changing_sampling_policy(tmp_path, monkeypatch, algorithm):
    import numpy as np

    from lfm_audio_rl import lfm
    from lfm_audio_rl.lora import inject_lora

    _, rows, _, _ = setup(monkeypatch)
    model = Tiny()
    model.codebooks = 8
    inject_lora(model.lfm, ["0"], 2, 4)
    calls, files = [], []

    def score(model, rollout, **kwargs):
        z = model.lfm(torch.tensor([[1.0, float(rollout.prompt + 1)]])).reshape(1)
        return torch.log_softmax(torch.cat([z, -z]), 0)[rollout.action : rollout.action + 1]

    def generate(model, processor, path, config):
        index = len(calls)
        prompt, action = index // config.group_size, index % config.group_size
        rollout = SimpleNamespace(
            prompt=prompt, action=action, text="red" if action else "blue", terminated=True
        )
        with torch.no_grad():
            rollout.old = score(model, rollout)
        calls.append({k: v.clone() for k, v in adapter_state(model).items()})
        return rollout

    def transcribe(path):
        from pathlib import Path

        assert Path(path).is_file()
        files.append(Path(path))
        return "red" if len(files) % 2 == 0 else "blue"

    monkeypatch.setattr(lfm, "generate", generate)
    monkeypatch.setattr(lfm, "score_rollout", score)
    monkeypatch.setattr(lfm, "sampled_logps", lambda rollout, scope: rollout.old)
    monkeypatch.setattr(
        lfm, "decode_audio", lambda *args: 0.1 * np.sin(2 * np.pi * 200 * np.arange(2400) / 24000)
    )
    cfg = Experiment(
        algorithm=algorithm,
        group_size=2,
        anchor_weight=0,
        kl_beta=0,
        train_opt=TrainingOptions(gradient_accumulation_steps=2, save_rollout_audio=False),
    )
    stats = [
        train_example(
            cfg, tmp_path, tmp_path, row, model, None, SimpleNamespace(transcribe=transcribe), 1, i
        )
        for i, row in enumerate(rows)
    ]
    assert [s["mean_reward"] for s in stats] == [0.5, 0.5]
    assert all(s["parity_max_error"] == 0 for s in stats)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    for candidate in calls:
        for name in calls[0]:
            torch.testing.assert_close(candidate[name], calls[0][name])
    assert files and not any(path.exists() for path in files)
    assert not list(tmp_path.glob("asr-*"))
    records = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert [r["micro_step"] for r in records] == [0, 0, 1, 1]
