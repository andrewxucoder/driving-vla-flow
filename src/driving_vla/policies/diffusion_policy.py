"""Diffusion policy — DDPM training + DDIM-style inference."""

from __future__ import annotations

import torch

from driving_vla.models.diffusion_head import (
    ConditionalDiffusionModel,
    diffusion_loss,
    sample_diffusion,
    sample_diffusion_n,
)
from driving_vla.policies.base_policy import BasePolicy


class DiffusionPolicy(BasePolicy):
    """Wraps :class:`ConditionalDiffusionModel` as a :class:`BasePolicy`."""

    head: ConditionalDiffusionModel

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: ConditionalDiffusionModel,
        sampling_steps: int = 50,
    ) -> None:
        super().__init__(
            encoder=encoder,
            head=head,
            action_dim=head.traj_dim,
            horizon=head.horizon,
        )
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be > 0, got {sampling_steps}")
        self.sampling_steps = sampling_steps

    def _compute_loss(
        self,
        target_chunk: torch.Tensor,
        cond_latent: torch.Tensor,
    ) -> torch.Tensor:
        return diffusion_loss(self.head, x0=target_chunk, cond=cond_latent)

    def _sample_chunk(self, cond_latent: torch.Tensor) -> torch.Tensor:
        return sample_diffusion(self.head, cond=cond_latent, steps=self.sampling_steps)

    def _sample_chunk_n(
        self,
        cond_latent: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        return sample_diffusion_n(
            self.head, cond=cond_latent, n=n, steps=self.sampling_steps
        )


__all__ = ["DiffusionPolicy"]
