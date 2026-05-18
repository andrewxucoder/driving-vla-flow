from __future__ import annotations

import torch
from torch import nn


class WaypointRegressionHead(nn.Module):
    """基于条件向量直接回归未来轨迹的 MLP 基线头。"""

    def __init__(self, horizon: int, traj_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.horizon = horizon
        self.traj_dim = traj_dim
        flat_dim = horizon * traj_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        return self.net(cond).reshape(b, self.horizon, self.traj_dim)
