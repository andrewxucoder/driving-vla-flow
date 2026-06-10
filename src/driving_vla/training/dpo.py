"""Flow-DPO: rectified-flow logprob surrogate + DPO loss.

The flow vector field has no exact tractable density. We use the standard
surrogate based on the regression target:

    surrogate_logp(x | cond) ∝ -‖v_θ(x_t, t, cond) - (x - x_0)‖²

evaluated at a fixed grid of ``t`` values (or a single random ``t`` per pair).
For DPO we only need *differences* in logp, which makes the constant offsets
cancel — so the surrogate is exactly the negative regression residual.

Reference: Wallace et al., *Diffusion-DPO* (2023), adapted to rectified flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from driving_vla.models.flow_matching import ConditionalFlowMatcher


@dataclass
class DPOConfig:
    """DPO hyperparameters."""

    beta: float = 0.1
    """Inverse temperature — higher β = sharper preference enforcement."""

    timesteps_per_pair: int = 4
    """Number of random ``t`` samples averaged into the surrogate logp."""


def flow_logprob_surrogate(
    model: ConditionalFlowMatcher,
    x: torch.Tensor,
    cond: torch.Tensor,
    *,
    timesteps_per_pair: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Negative regression-residual surrogate for log p(x | cond).

    Returns ``[B]`` scores. Higher = closer to the flow's preferred trajectory.

    For each trajectory ``x``, samples ``timesteps_per_pair`` random ``t``
    values and averages ``-‖v_θ(x_t, t, cond) - (x - x_0)‖²`` across them.
    ``x_0`` is freshly drawn Gaussian noise each call — DPO only needs the
    *difference* of two such surrogates so the noise term cancels in expectation.
    """
    if x.ndim != 3:
        raise ValueError(f"x expected [B, H, D], got {tuple(x.shape)}")
    if timesteps_per_pair <= 0:
        raise ValueError(f"timesteps_per_pair must be > 0, got {timesteps_per_pair}")
    batch = x.shape[0]
    device = x.device

    total = torch.zeros(batch, device=device)
    for _ in range(timesteps_per_pair):
        if generator is None:
            x0 = torch.randn_like(x)
            t = torch.rand(batch, device=device)
        else:
            x0 = torch.empty_like(x).normal_(generator=generator)
            t = torch.rand(batch, device=device, generator=generator)
        t_view = t.view(batch, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * x
        target_v = x - x0
        pred_v = model(x_t, t, cond)
        residual = (pred_v - target_v).pow(2).reshape(batch, -1).mean(dim=-1)
        total = total - residual

    return total / timesteps_per_pair


def flow_dpo_loss(
    policy_head: ConditionalFlowMatcher,
    reference_head: ConditionalFlowMatcher | None,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    cond: torch.Tensor,
    *,
    config: DPOConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Flow-DPO scalar loss.

    L = -log σ( β · ( (logp_θ(chosen) - logp_θ(rejected))
                    - (logp_ref(chosen) - logp_ref(rejected)) ) )

    When ``reference_head is None`` the reference term is dropped (becomes
    plain preference learning rather than DPO; useful when you have no frozen
    initial-policy snapshot).
    """
    if config is None:
        config = DPOConfig()
    if chosen.shape != rejected.shape:
        raise ValueError(
            f"chosen / rejected shape mismatch: {tuple(chosen.shape)} vs "
            f"{tuple(rejected.shape)}"
        )

    pol_chosen = flow_logprob_surrogate(
        policy_head, chosen, cond,
        timesteps_per_pair=config.timesteps_per_pair, generator=generator,
    )
    pol_rejected = flow_logprob_surrogate(
        policy_head, rejected, cond,
        timesteps_per_pair=config.timesteps_per_pair, generator=generator,
    )
    pol_delta = pol_chosen - pol_rejected

    if reference_head is None:
        margin = config.beta * pol_delta
    else:
        with torch.no_grad():
            ref_chosen = flow_logprob_surrogate(
                reference_head, chosen, cond,
                timesteps_per_pair=config.timesteps_per_pair, generator=generator,
            )
            ref_rejected = flow_logprob_surrogate(
                reference_head, rejected, cond,
                timesteps_per_pair=config.timesteps_per_pair, generator=generator,
            )
            ref_delta = ref_chosen - ref_rejected
        margin = config.beta * (pol_delta - ref_delta)

    return -F.logsigmoid(margin).mean()


__all__ = ["DPOConfig", "flow_dpo_loss", "flow_logprob_surrogate"]
