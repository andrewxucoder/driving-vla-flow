"""Multi-view image encoder — minimal CNN baseline.

Inputs ``[B, V, C, H, W]``. Each view goes through a shared small CNN; the V
per-view embeddings are mean-pooled into a single ``[B, latent_dim]`` tensor.
This is intentionally lightweight (~few-hundred-K params) — heavier backbones
(CLIP / DINO / SigLIP frozen vision tower) can be swapped in via the same
``latent_dim`` contract that :class:`DrivingObservationEncoder` expects.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiViewImageEncoder(nn.Module):
    """Shared CNN over views, mean-pool across views."""

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {image_size}")
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"images expected [B, V, C, H, W], got {tuple(images.shape)}"
            )
        batch, views, c, h, w = images.shape
        flat = images.reshape(batch * views, c, h, w)
        per_view = self.cnn(flat).reshape(batch, views, self.latent_dim)
        return per_view.mean(dim=1)


__all__ = ["MultiViewImageEncoder"]
