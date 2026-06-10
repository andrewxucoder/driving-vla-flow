"""Optimizer / LR-scheduler factories.

Tiny helpers that keep training scripts uncluttered. All hyperparameters live
in YAML / dataclass configs at the script layer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """AdamW with sensible driving-policy defaults."""
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    return torch.optim.AdamW(
        list(params),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Cosine decay from initial lr down to ``min_lr_ratio * lr``."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}")
    base_lr = optimizer.param_groups[0]["lr"]
    return CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=base_lr * min_lr_ratio
    )


__all__ = ["build_lr_scheduler", "build_optimizer"]
