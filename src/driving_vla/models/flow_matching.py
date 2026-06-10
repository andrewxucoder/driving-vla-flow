"""Conditional flow-matching trajectory head.

Rectified-flow style: train a vector field ``v_θ(x_t, t, cond)`` to predict
``x_1 - x_0`` where ``x_t = (1-t) x_0 + t x_1``, ``x_0 ~ N(0, I)``, and
``x_1`` is the expert trajectory chunk. At inference time, integrate the ODE
``dx/dt = v_θ`` from ``t=0`` (Gaussian noise) to ``t=1`` (sampled trajectory)
with Euler steps.
"""

from __future__ import annotations

import torch

from driving_vla.models.trajectory_denoiser import TrajectoryDenoiser


class ConditionalFlowMatcher(TrajectoryDenoiser):
    """Rectified-flow trajectory head.

    Inherits the shared (x_t, t, cond) → trajectory_tensor backbone; the class
    exists as a distinct symbol so ``isinstance(head, ConditionalFlowMatcher)``
    can dispatch on policy type.
    """


def flow_matching_loss(
    model: ConditionalFlowMatcher,
    x1: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Rectified-flow training objective.

    ``x1``: expert trajectory ``[B, H, D]``. ``cond``: encoded condition ``[B, L]``.
    """
    if x1.ndim != 3:
        raise ValueError(f"x1 expected [B, H, D], got {tuple(x1.shape)}")
    batch = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch, device=x1.device)
    t_view = t.view(batch, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_v = x1 - x0
    pred_v = model(x_t, t, cond)
    return torch.mean((pred_v - target_v) ** 2)


@torch.no_grad()
def sample_flow(
    model: ConditionalFlowMatcher,
    cond: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """Integrate the ODE ``dx/dt = v_θ(x, t, cond)`` from t=0 to t=1.

    Returns ``[B, H, D]`` sampled trajectories.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    batch = cond.shape[0]
    device = cond.device
    x = torch.randn(batch, model.horizon, model.traj_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((batch,), (i + 0.5) * dt, device=device)
        v = model(x, t, cond)
        x = x + dt * v
    return x


@torch.no_grad()
def sample_flow_n(
    model: ConditionalFlowMatcher,
    cond: torch.Tensor,
    n: int,
    steps: int = 32,
) -> torch.Tensor:
    """Independent N-sample sampler for Best-of-N.

    ``cond``: ``[B, L]``. Returns ``[B, N, H, D]`` with each (b, n) drawn from a
    separate Gaussian seed (no broadcasting tricks).
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    batch = cond.shape[0]
    cond_exp = cond.unsqueeze(1).expand(-1, n, -1).reshape(batch * n, -1)
    flat = sample_flow(model, cond_exp, steps=steps)  # [B*N, H, D]
    return flat.reshape(batch, n, model.horizon, model.traj_dim)


__all__ = [
    "ConditionalFlowMatcher",
    "flow_matching_loss",
    "sample_flow",
    "sample_flow_n",
]
