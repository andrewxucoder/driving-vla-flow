"""Conditional DDPM trajectory head — x0 parameterization.

Predicts the *clean* trajectory ``x0`` directly (rather than the ε-noise added
to it) under a linear-β variance schedule. This keeps the head's output on
the same trajectory scale as the Flow head's velocity field, so the same
shared ``TrajectoryDenoiser`` backbone learns at comparable difficulty across
the two objectives.

Two distinct issues sank Round 0's diffusion head (ADE 6.4 m); both matter:

1. ε-parameterization on raw-metre trajectories is numerically awkward — the
   model must predict O(1) noise out of an O(10) input. x0-prediction fixes
   this (the head now outputs on the trajectory scale).
2. **Schedule length dominates.** ``β ∈ [1e-4, 0.02]`` only reaches
   ``ᾱ_T ≈ 0`` (signal fully destroyed) when ``T`` is large (~1000). At the
   Round-0 ``T=100`` the terminal ``ᾱ_T ≈ 0.36`` (√ᾱ ≈ 0.60), so the forward
   process still left ~60 % of the trajectory intact — yet the sampler starts
   from pure ``N(0, I)`` noise, querying the model far out-of-distribution at
   the first reverse step. x0-param alone (T=100) only recovered to ~4.7 m;
   ``diffusion_steps=1000`` brings ADE to ~0.47 m, competitive with Flow.
   Hence the default below is 1000, not 100.

Inference uses a DDIM-style deterministic sampler: at each step the model
emits an x0 estimate; we derive the ε estimate from it and re-noise to the
next step's noise level. The final step returns the x0 estimate directly.
Sampling cost is set by the DDIM step count (``sample_diffusion(steps=...)``),
not by ``diffusion_steps``, so a larger schedule adds no inference overhead.
"""

from __future__ import annotations

import torch

from driving_vla.models.trajectory_denoiser import TrajectoryDenoiser


class ConditionalDiffusionModel(TrajectoryDenoiser):
    """Conditional DDPM x0-predictor over trajectory chunks."""

    alpha_bar: torch.Tensor

    def __init__(
        self,
        horizon: int,
        traj_dim: int,
        cond_dim: int,
        hidden_dim: int,
        diffusion_steps: int = 1000,
    ) -> None:
        super().__init__(
            horizon=horizon,
            traj_dim=traj_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )
        if diffusion_steps <= 1:
            raise ValueError(f"diffusion_steps must be ≥ 2, got {diffusion_steps}")
        self.diffusion_steps = diffusion_steps
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))


def diffusion_loss(
    model: ConditionalDiffusionModel,
    x0: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """x0-prediction objective on a randomly-chosen diffusion step.

    ``L = MSE(model(x_t, t, cond), x0)``. The model's output lives on the
    trajectory scale (metres), matching the Flow head's velocity scale.
    """
    if x0.ndim != 3:
        raise ValueError(f"x0 expected [B, H, D], got {tuple(x0.shape)}")
    batch = x0.shape[0]
    t_idx = torch.randint(0, model.diffusion_steps, (batch,), device=x0.device)
    noise = torch.randn_like(x0)
    alpha = model.alpha_bar[t_idx].view(batch, 1, 1)
    x_t = torch.sqrt(alpha) * x0 + torch.sqrt(1.0 - alpha) * noise
    t = t_idx.to(dtype=torch.float32) / max(model.diffusion_steps - 1, 1)
    pred_x0 = model(x_t, t, cond)
    return torch.mean((pred_x0 - x0) ** 2)


@torch.no_grad()
def sample_diffusion(
    model: ConditionalDiffusionModel,
    cond: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """DDIM-style deterministic sampler over the x0-parameterised head.

    Each step:
      1. ``x0_pred = model(x_t, t, cond)``
      2. derive ``ε_pred = (x_t − √ᾱ_t · x0_pred) / √(1 − ᾱ_t)``
      3. ``x_{next} = √ᾱ_{next} · x0_pred + √(1 − ᾱ_{next}) · ε_pred``
    The final iteration returns ``x0_pred`` directly.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    batch = cond.shape[0]
    device = cond.device
    x = torch.randn(batch, model.horizon, model.traj_dim, device=device)
    alpha_bar = model.alpha_bar

    indices = (
        torch.linspace(model.diffusion_steps - 1, 0, steps, device=device)
        .round()
        .to(dtype=torch.long)
    )
    indices = torch.unique_consecutive(indices)

    for i, t_idx in enumerate(indices):
        t = torch.full(
            (batch,),
            float(t_idx.item()) / max(model.diffusion_steps - 1, 1),
            device=device,
        )
        x0_pred = model(x, t, cond)
        if i == len(indices) - 1:
            x = x0_pred
        else:
            alpha_t = alpha_bar[t_idx].view(1, 1, 1)
            alpha_next = alpha_bar[indices[i + 1]].view(1, 1, 1)
            eps_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
                (1.0 - alpha_t).clamp(min=1e-8)
            )
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1.0 - alpha_next) * eps_pred
    return x


@torch.no_grad()
def sample_diffusion_n(
    model: ConditionalDiffusionModel,
    cond: torch.Tensor,
    n: int,
    steps: int = 50,
) -> torch.Tensor:
    """Independent N-sample sampler for Best-of-N. Returns ``[B, N, H, D]``."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    batch = cond.shape[0]
    cond_exp = cond.unsqueeze(1).expand(-1, n, -1).reshape(batch * n, -1)
    flat = sample_diffusion(model, cond_exp, steps=steps)
    return flat.reshape(batch, n, model.horizon, model.traj_dim)


__all__ = [
    "ConditionalDiffusionModel",
    "diffusion_loss",
    "sample_diffusion",
    "sample_diffusion_n",
]
