"""training.dpo — flow logprob surrogate + DPO loss."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import ConditionalFlowMatcher
from driving_vla.training import DPOConfig, flow_dpo_loss, flow_logprob_surrogate


def _model(*, cond_dim: int = 16, horizon: int = 4, traj_dim: int = 2) -> ConditionalFlowMatcher:
    return ConditionalFlowMatcher(
        horizon=horizon, traj_dim=traj_dim, cond_dim=cond_dim, hidden_dim=32
    )


def test_surrogate_shape():
    m = _model()
    x = torch.zeros(3, 4, 2)
    cond = torch.zeros(3, 16)
    out = flow_logprob_surrogate(m, x, cond, timesteps_per_pair=2)
    assert out.shape == (3,)


def test_surrogate_invalid_timesteps():
    m = _model()
    with pytest.raises(ValueError):
        flow_logprob_surrogate(m, torch.zeros(1, 4, 2), torch.zeros(1, 16), timesteps_per_pair=0)


def test_surrogate_rejects_2d_x():
    m = _model()
    with pytest.raises(ValueError):
        flow_logprob_surrogate(m, torch.zeros(3, 8), torch.zeros(3, 16))


def test_dpo_loss_scalar():
    m = _model()
    chosen = torch.zeros(2, 4, 2)
    rejected = torch.ones(2, 4, 2) * 0.5
    cond = torch.zeros(2, 16)
    loss = flow_dpo_loss(m, None, chosen, rejected, cond, config=DPOConfig(timesteps_per_pair=2))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dpo_loss_with_reference():
    policy = _model()
    ref = _model()
    chosen = torch.randn(2, 4, 2)
    rejected = torch.randn(2, 4, 2)
    cond = torch.randn(2, 16)
    loss = flow_dpo_loss(
        policy, ref, chosen, rejected, cond, config=DPOConfig(timesteps_per_pair=2)
    )
    assert torch.isfinite(loss)


def test_dpo_loss_grad_flows_to_policy_only():
    policy = _model()
    ref = _model()
    for p in ref.parameters():
        p.requires_grad = False
    chosen = torch.randn(2, 4, 2)
    rejected = torch.randn(2, 4, 2)
    cond = torch.randn(2, 16)
    loss = flow_dpo_loss(policy, ref, chosen, rejected, cond, config=DPOConfig(timesteps_per_pair=2))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert all(p.grad is None for p in ref.parameters())


def test_dpo_shape_mismatch_raises():
    m = _model()
    with pytest.raises(ValueError):
        flow_dpo_loss(
            m, None,
            torch.zeros(2, 4, 2), torch.zeros(2, 5, 2),
            torch.zeros(2, 16),
        )
