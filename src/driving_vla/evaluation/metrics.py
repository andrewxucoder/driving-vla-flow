from __future__ import annotations

import torch


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, dim=-1).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()


def jerk_proxy(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[1] < 4:
        return torch.tensor(0.0, device=traj.device)
    vel = traj[:, 1:] - traj[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    return torch.norm(jerk, dim=-1).mean()
