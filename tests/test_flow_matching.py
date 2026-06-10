"""ConditionalFlowMatcher — loss + sample + sample_n."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import (
    ConditionalFlowMatcher,
    flow_matching_loss,
    sample_flow,
    sample_flow_n,
)


def _model(*, horizon: int = 4, traj_dim: int = 2, cond_dim: int = 32) -> ConditionalFlowMatcher:
    return ConditionalFlowMatcher(
        horizon=horizon, traj_dim=traj_dim, cond_dim=cond_dim, hidden_dim=64
    )


def test_loss_scalar():
    m = _model()
    x1 = torch.zeros(3, 4, 2)
    cond = torch.zeros(3, 32)
    loss = flow_matching_loss(m, x1=x1, cond=cond)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_grad_flows():
    m = _model()
    x1 = torch.randn(2, 4, 2)
    cond = torch.randn(2, 32)
    loss = flow_matching_loss(m, x1=x1, cond=cond)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()
    )
    assert has_grad


def test_loss_rejects_2d_x():
    m = _model()
    with pytest.raises(ValueError):
        flow_matching_loss(m, x1=torch.zeros(3, 8), cond=torch.zeros(3, 32))


def test_sample_shape():
    m = _model()
    m.eval()
    out = sample_flow(m, cond=torch.zeros(2, 32), steps=4)
    assert out.shape == (2, 4, 2)


def test_sample_n_shape():
    m = _model()
    m.eval()
    out = sample_flow_n(m, cond=torch.zeros(2, 32), n=5, steps=4)
    assert out.shape == (2, 5, 4, 2)


def test_sample_n_independent_draws():
    m = _model()
    m.eval()
    # Independent gaussian seeds → different samples per N
    torch.manual_seed(0)
    out = sample_flow_n(m, cond=torch.zeros(1, 32), n=4, steps=4)
    # At least one pair of N samples should differ
    diffs = [
        (out[0, i] - out[0, j]).abs().sum().item()
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    assert max(diffs) > 1e-6


def test_sample_rejects_zero_steps():
    m = _model()
    with pytest.raises(ValueError):
        sample_flow(m, cond=torch.zeros(1, 32), steps=0)


def test_sample_n_rejects_zero_n():
    m = _model()
    with pytest.raises(ValueError):
        sample_flow_n(m, cond=torch.zeros(1, 32), n=0)
