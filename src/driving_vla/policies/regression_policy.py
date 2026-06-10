"""Regression policy — deterministic MLP head."""

from __future__ import annotations

import torch

from driving_vla.models.regression_head import WaypointRegressionHead
from driving_vla.policies.base_policy import BasePolicy


class RegressionPolicy(BasePolicy):
    """Direct waypoint regression — MSE loss, deterministic single-shot inference."""

    head: WaypointRegressionHead

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: WaypointRegressionHead,
    ) -> None:
        super().__init__(
            encoder=encoder,
            head=head,
            action_dim=head.traj_dim,
            horizon=head.horizon,
        )

    def _compute_loss(
        self,
        target_chunk: torch.Tensor,
        cond_latent: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.head(cond_latent)
        return torch.mean((pred - target_chunk) ** 2)

    def _sample_chunk(self, cond_latent: torch.Tensor) -> torch.Tensor:
        # Deterministic head: every "sample" is identical.
        return self.head(cond_latent)


__all__ = ["RegressionPolicy"]
