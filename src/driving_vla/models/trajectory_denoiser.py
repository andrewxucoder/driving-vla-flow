"""Shared trajectory denoiser MLP backbone.

Flow Matching and Diffusion both consume ``(x_t, t, cond) → [B, H, D]`` vector
fields / noise predictions; only the training objective and sampler differ.
Sharing the backbone makes "same-architecture, two-objective" comparisons fair.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding + 2-layer MLP projection.

    Frequency schedule matches the standard transformer convention:

        freqs[i] = exp(-i * log(max_period) / (N - 1))   for i in [0, N-1]

    where ``N = num_frequencies = dim // 2``. ``forward`` concatenates sin/cos
    of the time scalar and projects through Linear → SiLU → Linear to ``dim``.
    """

    freqs: torch.Tensor

    def __init__(
        self,
        dim: int,
        num_frequencies: int | None = None,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if num_frequencies is None:
            num_frequencies = max(1, dim // 2)
        freqs = torch.exp(torch.linspace(0.0, -math.log(max_period), num_frequencies))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_frequencies, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1).to(dtype=self.freqs.dtype)
        scaled = t * self.freqs.unsqueeze(0)
        embed = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(embed)


class TrajectoryDenoiser(nn.Module):
    """Shared MLP backbone: (x_t, t, cond) → predicted trajectory tensor.

    Reshapes the trajectory chunk to a flat vector, concatenates with the time
    embedding and conditioning latent, runs a small MLP, then reshapes back.
    """

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
        self.hidden_dim = hidden_dim

        flat_dim = horizon * traj_dim
        self.time = TimeEmbedding(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(flat_dim + cond_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[1] != self.horizon or x_t.shape[2] != self.traj_dim:
            raise ValueError(
                f"x_t expected [B, {self.horizon}, {self.traj_dim}], got {tuple(x_t.shape)}"
            )
        if cond.ndim != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond expected [B, {self.cond_dim}], got {tuple(cond.shape)}"
            )
        batch = x_t.shape[0]
        time_emb = self.time(t)
        inp = torch.cat([x_t.reshape(batch, -1), cond, time_emb], dim=-1)
        return self.net(inp).reshape(batch, self.horizon, self.traj_dim)


__all__ = ["TimeEmbedding", "TrajectoryDenoiser"]
