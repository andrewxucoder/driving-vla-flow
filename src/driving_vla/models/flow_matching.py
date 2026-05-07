from __future__ import annotations

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.view(-1, 1))


class ConditionalFlowMatcher(nn.Module):
    """Conditional vector-field model for trajectory Flow Matching."""

    def __init__(self, horizon: int, traj_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.horizon = horizon
        self.traj_dim = traj_dim
        flat_dim = horizon * traj_dim
        self.time = TimeEmbedding(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(flat_dim + cond_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        h = self.time(t)
        inp = torch.cat([x_t.reshape(b, -1), cond, h], dim=-1)
        return self.net(inp).reshape(b, self.horizon, self.traj_dim)


def flow_matching_loss(model: ConditionalFlowMatcher, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Rectified-flow style objective with linear interpolation x_t=(1-t)x0+t*x1."""
    x0 = torch.randn_like(x1)
    b = x1.shape[0]
    t = torch.rand(b, device=x1.device)
    view_t = t.view(b, 1, 1)
    x_t = (1.0 - view_t) * x0 + view_t * x1
    target_v = x1 - x0
    pred_v = model(x_t, t, cond)
    return torch.mean((pred_v - target_v) ** 2)


@torch.no_grad()
def sample_flow(model: ConditionalFlowMatcher, cond: torch.Tensor, steps: int = 16) -> torch.Tensor:
    b = cond.shape[0]
    x = torch.randn(b, model.horizon, model.traj_dim, device=cond.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i / steps, device=cond.device)
        v = model(x, t, cond)
        x = x + dt * v
    return x
