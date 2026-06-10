"""BasePolicy contract — Encoder + head wrapped in a unified train/infer interface.

Structurally aligned with ``robot_vla.policies.BasePolicy`` (per V2_GREENFIELD_PLAN
R2): the surface area is identical, only the ``observation`` type differs
(driving uses :class:`DrivingTrajectorySample` instead of robot's ``Observation``).

Subclasses override two methods:

- ``_compute_loss(target_chunk, cond_latent) -> Tensor`` — training objective
- ``_sample_chunk(cond_latent) -> Tensor [B, H, A]`` — single-shot inference

The Best-of-N sampler ``_sample_chunk_n`` defaults to broadcasting the single
deterministic sample N times; stochastic heads (Flow / Diffusion) override it
to perform N independent draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from driving_vla.data.trajectory_sample import DrivingTrajectorySample


def observation_to_condition(
    observation: DrivingTrajectorySample,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Flatten a single :class:`DrivingTrajectorySample` into an encoder-ready dict.

    Adds a leading batch dim of size 1 to all tensors. Non-tensor fields
    (``instruction`` / ``metadata``) are passed through as length-1 lists where
    the encoder expects a sequence.
    """
    def _move(t: torch.Tensor) -> torch.Tensor:
        return t.to(device) if device is not None else t

    cond: dict[str, Any] = {}
    if observation.ego_state is not None:
        cond["ego_state"] = _move(observation.ego_state).unsqueeze(0)
    else:
        # ego_state is mandatory at the encoder; reuse history[-1] as fallback
        if observation.history is None or observation.history.shape[0] == 0:
            raise ValueError(
                "observation_to_condition needs either ego_state or non-empty history"
            )
        cond["ego_state"] = _move(observation.history[-1]).unsqueeze(0)

    cond["history"] = _move(observation.history).unsqueeze(0)
    if observation.command_id is not None:
        cond["command_id"] = torch.tensor(
            [int(observation.command_id)], dtype=torch.long, device=device
        )
    if observation.images is not None:
        cond["images"] = _move(observation.images).unsqueeze(0)
    if observation.instruction is not None:
        cond["instruction"] = [observation.instruction]
    if observation.route is not None:
        cond["route"] = _move(observation.route).unsqueeze(0)
    if observation.metadata:
        cond["metadata"] = [observation.metadata]
    return cond


class BasePolicy(nn.Module):
    """Encoder + head wrapper presenting a unified policy interface."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        action_dim: int,
        horizon: int,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or horizon <= 0:
            raise ValueError(
                f"action_dim/horizon must both be > 0; got {action_dim}/{horizon}"
            )
        self.encoder = encoder
        self.head = head
        self.action_dim = action_dim
        self.horizon = horizon

    def forward(
        self,
        condition: Mapping[str, Any],
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Training loss. ``target_chunk`` shape ``[B, H, action_dim]``."""
        if target_chunk.ndim != 3:
            raise ValueError(
                f"target_chunk expected [B, H, action_dim], got {tuple(target_chunk.shape)}"
            )
        _, h, a = target_chunk.shape
        if h != self.horizon or a != self.action_dim:
            raise ValueError(
                f"target_chunk shape [_, {h}, {a}] does not match policy "
                f"(horizon={self.horizon}, action_dim={self.action_dim})"
            )
        cond_latent = self.encoder(condition)
        return self._compute_loss(target_chunk, cond_latent)

    @torch.no_grad()
    def predict_chunk(self, observation: DrivingTrajectorySample) -> torch.Tensor:
        """Return a single ``[H, action_dim]`` chunk for the given observation."""
        device = self._device()
        cond = observation_to_condition(observation, device=device)
        cond_latent = self.encoder(cond)
        chunk = self._sample_chunk(cond_latent)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk.squeeze(0)
        return chunk.detach().cpu()

    @torch.no_grad()
    def predict_chunk_n(
        self,
        observation: DrivingTrajectorySample,
        n: int,
    ) -> torch.Tensor:
        """Best-of-N: return ``[N, H, action_dim]``."""
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        device = self._device()
        cond = observation_to_condition(observation, device=device)
        cond_latent = self.encoder(cond)
        chunks = self._sample_chunk_n(cond_latent, n=n)
        if chunks.ndim == 4 and chunks.shape[0] == 1:
            chunks = chunks.squeeze(0)
        return chunks.detach().cpu()

    def act_step(self, observation: DrivingTrajectorySample) -> torch.Tensor:
        """Single-step action ``[action_dim]``. Default: chunk[0]."""
        chunk = self.predict_chunk(observation)
        return chunk[0]

    def reset(self) -> None:
        """Hook for stateful policies (ACT-style temporal ensemble, etc.)."""
        return None

    # ── Subclass hooks ────────────────────────────────────────────────────

    def _compute_loss(
        self,
        target_chunk: torch.Tensor,
        cond_latent: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("BasePolicy._compute_loss must be overridden")

    def _sample_chunk(self, cond_latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BasePolicy._sample_chunk must be overridden")

    def _sample_chunk_n(
        self,
        cond_latent: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Default: broadcast a single deterministic sample N times.

        Stochastic heads override with N independent draws.
        """
        chunk = self._sample_chunk(cond_latent)  # [B, H, A]
        return chunk.unsqueeze(1).expand(-1, n, -1, -1).contiguous()

    def _device(self) -> torch.device:
        return next(self.encoder.parameters()).device


__all__ = ["BasePolicy", "observation_to_condition"]
