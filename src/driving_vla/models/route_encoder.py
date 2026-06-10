"""Route polyline encoder — VectorNet-style.

Encodes an ego-frame look-ahead route (the long-range plan beyond the
prediction window) into a single ``[B, latent_dim]`` latent that plugs into
:class:`DrivingObservationEncoder`.

Architecture::

    [B, N, 2]  ego-frame waypoints
        → per-point MLP encoder           → [B, N, hidden_dim]
        → + learnable segment-index PE
        → 1-layer transformer encoder      → [B, N, hidden_dim]
        → masked mean pool over waypoints  → [B, hidden_dim]
        → 2-layer MLP projection           → [B, latent_dim]

Padding/mask handling: ``forward`` accepts an optional ``mask: [B, N]`` where
``1`` = valid waypoint, ``0`` = padding. When provided, padded positions are
excluded from the transformer attention and the pooling average.
"""

from __future__ import annotations

import torch
from torch import nn


class RouteEncoder(nn.Module):
    """VectorNet-style encoder for a single ego-frame route polyline."""

    def __init__(
        self,
        latent_dim: int,
        max_waypoints: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if max_waypoints <= 0:
            raise ValueError(f"max_waypoints must be > 0, got {max_waypoints}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.latent_dim = latent_dim
        self.max_waypoints = max_waypoints
        self.hidden_dim = hidden_dim

        self.point_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.segment_pe = nn.Parameter(
            torch.randn(1, max_waypoints, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.fuse = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        waypoints: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if waypoints.ndim != 3 or waypoints.shape[-1] != 2:
            raise ValueError(
                f"waypoints expected [B, N, 2], got {tuple(waypoints.shape)}"
            )
        batch, n, _ = waypoints.shape
        if n > self.max_waypoints:
            raise ValueError(
                f"got {n} waypoints, but max_waypoints={self.max_waypoints}"
            )
        if mask is not None:
            if mask.shape != (batch, n):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} does not match [B={batch}, N={n}]"
                )

        pts = self.point_proj(waypoints) + self.segment_pe[:, :n, :]

        if mask is None:
            fused = self.fuse(pts)
            pooled = fused.mean(dim=1)
        else:
            pad_mask = ~mask.bool()  # transformer expects True at padding positions
            fused = self.fuse(pts, src_key_padding_mask=pad_mask)
            weight = mask.unsqueeze(-1).to(fused.dtype)
            pooled = (fused * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        return self.projection(pooled)


__all__ = ["RouteEncoder"]
