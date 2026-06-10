"""Direct waypoint regression head — MLP baseline."""

from __future__ import annotations

import torch
from torch import nn


class WaypointRegressionHead(nn.Module):
    """MLP that maps a single conditioning vector to a fixed-length trajectory chunk."""

    def __init__(
        self,
        horizon: int,
        traj_dim: int,
        cond_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if horizon <= 0 or traj_dim <= 0 or cond_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"all dims must be > 0; got horizon={horizon} traj_dim={traj_dim} "
                f"cond_dim={cond_dim} hidden_dim={hidden_dim}"
            )
        self.horizon = horizon
        self.traj_dim = traj_dim
        self.cond_dim = cond_dim

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * traj_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond expected [B, {self.cond_dim}], got {tuple(cond.shape)}"
            )
        batch = cond.shape[0]
        return self.net(cond).reshape(batch, self.horizon, self.traj_dim)


__all__ = ["WaypointRegressionHead"]
