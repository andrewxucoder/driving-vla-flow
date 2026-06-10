"""training.loop + training.optim — generic train loop + factories."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from driving_vla.training import (
    TrainConfig,
    TrainState,
    build_lr_scheduler,
    build_optimizer,
    train,
)


def _toy_model_and_loader():
    """Small linear regression for the loop smoke test."""
    model = torch.nn.Linear(4, 1)
    data = [
        {"x": torch.randn(8, 4), "y": torch.randn(8, 1)}
        for _ in range(3)
    ]
    return model, data


def test_train_runs_and_advances_step():
    model, data = _toy_model_and_loader()
    opt = build_optimizer(model.parameters(), lr=1e-3)

    def step_fn(batch):
        pred = model(batch["x"])
        return ((pred - batch["y"]) ** 2).mean()

    state = train(
        model=model,
        optimizer=opt,
        data_loader=data,
        step_fn=step_fn,
        config=TrainConfig(num_epochs=2, log_every=10),
    )
    assert state.step == 6  # 2 epochs × 3 batches
    assert state.epoch == 1
    assert torch.isfinite(torch.tensor(state.last_loss))


def test_train_rejects_non_scalar_loss():
    model, data = _toy_model_and_loader()
    opt = build_optimizer(model.parameters(), lr=1e-3)

    def step_fn(batch):
        return model(batch["x"])  # not scalar

    with pytest.raises(ValueError):
        train(model=model, optimizer=opt, data_loader=data, step_fn=step_fn)


def test_train_saves_checkpoint_at_epoch_end(tmp_path: Path):
    model, data = _toy_model_and_loader()
    opt = build_optimizer(model.parameters(), lr=1e-3)

    def step_fn(batch):
        return ((model(batch["x"]) - batch["y"]) ** 2).mean()

    save = tmp_path / "ckpt.pt"
    train(
        model=model,
        optimizer=opt,
        data_loader=data,
        step_fn=step_fn,
        config=TrainConfig(num_epochs=1, save_path=save),
    )
    assert save.exists()
    payload = torch.load(save, map_location="cpu", weights_only=False)
    assert "model_state" in payload and "optimizer_state" in payload


def test_train_state_dataclass_defaults():
    state = TrainState()
    assert state.step == 0 and state.epoch == 0


def test_build_optimizer_rejects_zero_lr():
    with pytest.raises(ValueError):
        build_optimizer([torch.nn.Parameter(torch.zeros(1))], lr=0)


def test_build_lr_scheduler_basic():
    opt = build_optimizer([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = build_lr_scheduler(opt, total_steps=10)
    assert sched is not None
    opt.step()
    sched.step()


def test_build_lr_scheduler_bad_args():
    opt = build_optimizer([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, total_steps=0)
    with pytest.raises(ValueError):
        build_lr_scheduler(opt, total_steps=10, min_lr_ratio=0)
