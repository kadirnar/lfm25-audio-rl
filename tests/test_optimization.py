import pytest
import torch

from lfm_audio_rl.config import Experiment, TrainingOptions, WandbConfig
from lfm_audio_rl.data import digest
from lfm_audio_rl.optimization import learning_rate, make_optimizer


def test_schedules_have_explicit_warmup_and_final_lr():
    for scheduler in ["linear", "cosine"]:
        config = Experiment(
            steps=6,
            lr=0.01,
            train_opt=TrainingOptions(scheduler=scheduler, warmup_steps=2, min_lr_ratio=0.1),
        )
        rates = [learning_rate(config, i) for i in range(1, 7)]
        assert rates[:3] == [0.005, 0.01, 0.01]
        assert rates[-1] == pytest.approx(0.001)
        assert rates[2:] == sorted(rates[2:], reverse=True)
        # Resumption derives exactly the same next LR from the committed step.
        assert [learning_rate(config, i) for i in range(4, 7)] == rates[3:]
    assert learning_rate(Experiment(steps=1), 1) == 1e-5
    with pytest.raises(ValueError):
        Experiment(steps=2, train_opt=TrainingOptions(warmup_steps=3))


def test_optimizer_parameters_and_cuda_requirement():
    parameter = torch.nn.Parameter(torch.ones(2))
    config = Experiment(
        lr=0.002, train_opt=TrainingOptions(weight_decay=0.1, betas=(0.8, 0.9), eps=1e-7)
    )
    optimizer = make_optimizer([parameter], config)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    parameter.sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    with pytest.raises(ValueError, match="CUDA"):
        make_optimizer([parameter], Experiment(train_opt=TrainingOptions(optimizer="adamw_fused")))
    with pytest.raises(ValueError):
        TrainingOptions(betas=(1, 0.99))


def test_tracking_changes_preserve_legacy_checkpoint_identity():
    config = Experiment()
    legacy = config.model_dump(exclude={"wandb", "train_opt"})
    assert config.training_dict() == legacy
    logged = Experiment(wandb=WandbConfig(mode="offline", name="my run"))
    assert digest(logged.training_dict()) == digest(legacy)
    changed = Experiment(train_opt=TrainingOptions(gradient_accumulation_steps=2))
    assert digest(changed.training_dict()) != digest(legacy)


def test_training_cli_overrides_are_validated(tmp_path, monkeypatch, capsys):
    from lfm_audio_rl.cli import main

    cfg = tmp_path / "config.yaml"
    cfg.write_text("algorithm: grpo\n")
    observed = []

    def fake(config, *args):
        observed.append(config)
        return {"ok": True}

    monkeypatch.setattr("lfm_audio_rl.train.train", fake)
    arguments = [
        "lfm-rl",
        "train",
        "--config",
        str(cfg),
        "--data",
        str(tmp_path),
        "--output",
        str(tmp_path / "run"),
        "--wandb-mode",
        "offline",
        "--wandb-project",
        "study",
        "--gradient-checkpointing",
        "--gradient-accumulation-steps",
        "2",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    assert observed[0].wandb.mode == "offline" and observed[0].wandb.project == "study"
    assert observed[0].train_opt.gradient_checkpointing
    assert observed[0].train_opt.gradient_accumulation_steps == 2
    monkeypatch.setattr("sys.argv", arguments[:-1] + ["0"])
    with pytest.raises(ValueError):
        main()
    assert len(observed) == 1
