"""Trajectory-level metrics: ADE / FDE / collision proxy / progress.

Operate on tensors of shape ``[B, H, 2]`` (or compatible). Pure functions, no
hidden state — safe to call from any train/eval loop.
"""

from __future__ import annotations

import torch


def _check_2d_traj(name: str, t: torch.Tensor) -> None:
    if t.ndim != 3 or t.shape[-1] != 2:
        raise ValueError(f"{name} expected [B, H, 2], got {tuple(t.shape)}")


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error (mean over horizon of pointwise L2)."""
    _check_2d_traj("pred", pred)
    _check_2d_traj("target", target)
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    per_step = torch.linalg.norm(pred - target, dim=-1)  # [B, H]
    return per_step.mean(dim=-1)  # [B]


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error (L2 at the last horizon step)."""
    _check_2d_traj("pred", pred)
    _check_2d_traj("target", target)
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return torch.linalg.norm(pred[:, -1] - target[:, -1], dim=-1)  # [B]


def lateral_offset(pred: torch.Tensor) -> torch.Tensor:
    """Maximum absolute lateral offset along the horizon (y channel)."""
    _check_2d_traj("pred", pred)
    return pred[..., 1].abs().max(dim=-1).values  # [B]


def collision_proxy(
    pred: torch.Tensor,
    lane_half_width: float = 1.75,
) -> torch.Tensor:
    """Geometric collision proxy: 1.0 if max lateral offset exceeds half-lane width.

    A coarse stand-in for an actual collision check (which would need other
    agents). Useful as a regularising signal in BoN scoring and rule-based reward.
    """
    if lane_half_width <= 0:
        raise ValueError(f"lane_half_width must be > 0, got {lane_half_width}")
    return (lateral_offset(pred) > lane_half_width).float()  # [B]


def forward_progress(pred: torch.Tensor) -> torch.Tensor:
    """Forward distance from start to end (x channel only)."""
    _check_2d_traj("pred", pred)
    return pred[:, -1, 0] - pred[:, 0, 0]  # [B]


def jerk(pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute jerk = mean |Δ²p / Δt²| (Δt = 1 since indices are discrete)."""
    _check_2d_traj("pred", pred)
    if pred.shape[1] < 3:
        return torch.zeros(pred.shape[0], device=pred.device)
    accel = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # [B, H-2, 2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=-1)  # [B]


__all__ = [
    "ade",
    "collision_proxy",
    "fde",
    "forward_progress",
    "jerk",
    "lateral_offset",
]
