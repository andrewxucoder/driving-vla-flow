"""Generic training loop with a step_fn callback.

Decouples "how to compute loss on a batch" (script-level) from "how to run an
epoch / log / save" (this module). Used by every train_*.py script.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


StepFn = Callable[[dict[str, Any]], torch.Tensor]


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    last_loss: float = float("nan")
    losses: list[float] = field(default_factory=list)


@dataclass
class TrainConfig:
    num_epochs: int = 1
    log_every: int = 50
    grad_clip: float | None = 1.0
    save_path: str | Path | None = None
    save_every_epoch: bool = True


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    state: TrainState,
    cfg_metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_state": {
            "step": state.step,
            "epoch": state.epoch,
            "last_loss": state.last_loss,
        },
        "metadata": cfg_metadata,
    }
    torch.save(payload, path)


def train(
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    data_loader: Iterable[dict[str, Any]],
    step_fn: StepFn,
    config: TrainConfig | None = None,
    scheduler: LRScheduler | None = None,
    state: TrainState | None = None,
    log_fn: Callable[[TrainState], None] | None = None,
    cfg_metadata: dict[str, Any] | None = None,
) -> TrainState:
    """Run ``num_epochs`` over ``data_loader`` driven by ``step_fn``.

    ``step_fn(batch) -> scalar loss``. The training loop owns optimizer.step(),
    gradient clipping, scheduler stepping, logging, and checkpoint saves.
    """
    if config is None:
        config = TrainConfig()
    if state is None:
        state = TrainState()
    cfg_metadata = cfg_metadata or {}

    model.train()
    for epoch in range(config.num_epochs):
        state.epoch = epoch
        for batch in data_loader:
            loss = step_fn(batch)
            if loss.ndim != 0:
                raise ValueError(f"step_fn must return a scalar loss, got shape {tuple(loss.shape)}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            state.step += 1
            state.last_loss = float(loss.detach().cpu().item())
            state.losses.append(state.last_loss)

            if log_fn is not None and state.step % max(1, config.log_every) == 0:
                log_fn(state)

        if config.save_every_epoch and config.save_path is not None:
            _save_checkpoint(
                Path(config.save_path),
                model=model,
                optimizer=optimizer,
                state=state,
                cfg_metadata=cfg_metadata,
            )

    return state


__all__ = ["StepFn", "TrainConfig", "TrainState", "train"]
