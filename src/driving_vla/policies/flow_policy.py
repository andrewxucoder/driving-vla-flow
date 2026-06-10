"""Flow-matching policy — rectified-flow training + ODE-integrated inference."""

from __future__ import annotations

import torch

from driving_vla.models.flow_matching import (
    ConditionalFlowMatcher,
    flow_matching_loss,
    sample_flow,
    sample_flow_n,
)
from driving_vla.policies.base_policy import BasePolicy


class FlowPolicy(BasePolicy):
    """Wraps :class:`ConditionalFlowMatcher` as a :class:`BasePolicy`."""

    head: ConditionalFlowMatcher

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: ConditionalFlowMatcher,
        sampling_steps: int = 32,
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
        return flow_matching_loss(self.head, x1=target_chunk, cond=cond_latent)

    def _sample_chunk(self, cond_latent: torch.Tensor) -> torch.Tensor:
        return sample_flow(self.head, cond=cond_latent, steps=self.sampling_steps)

    def _sample_chunk_n(
        self,
        cond_latent: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        # Independent N draws — different Gaussian seeds inside sample_flow_n.
        return sample_flow_n(
            self.head, cond=cond_latent, n=n, steps=self.sampling_steps
        )


__all__ = ["FlowPolicy"]
