"""Learned reward model — small MLP that scores (condition, trajectory) pairs.

Trained from preference pairs (chosen / rejected) via Bradley-Terry loss.
Used as a swap-in alternative to the rule-based :func:`trajectory_reward` in
Best-of-N scoring at evaluation time.
"""

from __future__ import annotations

import torch
from torch import nn


class LearnedRewardModel(nn.Module):
    """MLP scorer over (cond_latent, flat_trajectory) → scalar reward.

    The condition encoder is left external — the model expects an already-
    encoded ``[B, cond_dim]`` latent.
    """

    def __init__(
        self,
        cond_dim: int,
        horizon: int,
        traj_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if cond_dim <= 0 or horizon <= 0 or traj_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"all dims must be > 0; got cond_dim={cond_dim} horizon={horizon} "
                f"traj_dim={traj_dim} hidden_dim={hidden_dim}"
            )
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.traj_dim = traj_dim

        flat_dim = horizon * traj_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim + flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cond: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond expected [B, {self.cond_dim}], got {tuple(cond.shape)}"
            )
        if traj.ndim != 3 or traj.shape[1] != self.horizon or traj.shape[2] != self.traj_dim:
            raise ValueError(
                f"traj expected [B, {self.horizon}, {self.traj_dim}], got {tuple(traj.shape)}"
            )
        batch = cond.shape[0]
        flat = traj.reshape(batch, -1)
        return self.net(torch.cat([cond, flat], dim=-1)).squeeze(-1)


def bradley_terry_loss(
    chosen_score: torch.Tensor,
    rejected_score: torch.Tensor,
) -> torch.Tensor:
    """Standard preference-pair reward training loss.

    L = -log σ(chosen - rejected), averaged over the batch.
    """
    if chosen_score.shape != rejected_score.shape:
        raise ValueError(
            f"score shapes must match; got {tuple(chosen_score.shape)} vs "
            f"{tuple(rejected_score.shape)}"
        )
    return -torch.nn.functional.logsigmoid(chosen_score - rejected_score).mean()


__all__ = ["LearnedRewardModel", "bradley_terry_loss"]
