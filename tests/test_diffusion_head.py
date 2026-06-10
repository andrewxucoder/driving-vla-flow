"""ConditionalDiffusionModel — loss + sampler + sample_n."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import (
    ConditionalDiffusionModel,
    diffusion_loss,
    sample_diffusion,
    sample_diffusion_n,
)


def _model(
    *, horizon: int = 4, traj_dim: int = 2, cond_dim: int = 32, steps: int = 20
) -> ConditionalDiffusionModel:
    return ConditionalDiffusionModel(
        horizon=horizon,
        traj_dim=traj_dim,
        cond_dim=cond_dim,
        hidden_dim=64,
        diffusion_steps=steps,
    )


def test_alpha_bar_shape():
    m = _model(steps=20)
    assert m.alpha_bar.shape == (20,)
    # alpha_bar should be monotonically decreasing (cumprod of < 1)
    diffs = m.alpha_bar[1:] - m.alpha_bar[:-1]
    assert (diffs <= 0).all()


def test_diffusion_steps_min_2():
    with pytest.raises(ValueError):
        ConditionalDiffusionModel(
            horizon=4, traj_dim=2, cond_dim=16, hidden_dim=32, diffusion_steps=1
        )


def test_loss_scalar():
    m = _model()
    x0 = torch.zeros(3, 4, 2)
    cond = torch.zeros(3, 32)
    loss = diffusion_loss(m, x0=x0, cond=cond)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_grad_flows():
    m = _model()
    x0 = torch.randn(2, 4, 2)
    cond = torch.randn(2, 32)
    loss = diffusion_loss(m, x0=x0, cond=cond)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()
    )
    assert has_grad


def test_sample_shape():
    m = _model()
    m.eval()
    out = sample_diffusion(m, cond=torch.zeros(2, 32), steps=5)
    assert out.shape == (2, 4, 2)


def test_sample_n_shape():
    m = _model()
    m.eval()
    out = sample_diffusion_n(m, cond=torch.zeros(2, 32), n=3, steps=5)
    assert out.shape == (2, 3, 4, 2)


def test_sample_n_independent_draws():
    m = _model()
    m.eval()
    torch.manual_seed(0)
    out = sample_diffusion_n(m, cond=torch.zeros(1, 32), n=4, steps=5)
    diffs = [
        (out[0, i] - out[0, j]).abs().sum().item()
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    assert max(diffs) > 1e-6


def test_sample_rejects_zero_steps():
    m = _model()
    with pytest.raises(ValueError):
        sample_diffusion(m, cond=torch.zeros(1, 32), steps=0)


def test_x0_parameterization_learns_target():
    """Round-0 regression: ε-parameterization on raw-metre trajectories
    collapsed to ADE 6.4m. x0-parameterization should let a tiny model overfit
    a fixed target in a handful of steps — proxy for "actually learns the
    trajectory scale, not just the noise level"."""
    torch.manual_seed(0)
    m = _model(horizon=4, traj_dim=2, cond_dim=8, steps=20)
    target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])  # [1,4,2]
    cond = torch.zeros(1, 8)

    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    initial_loss = None
    for step in range(200):
        loss = diffusion_loss(m, x0=target, cond=cond)
        if step == 0:
            initial_loss = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = float(loss.item())
    assert final_loss < initial_loss * 0.1, (
        f"x0-param loss did not decrease meaningfully: {initial_loss:.3f} → {final_loss:.3f}"
    )

    # Sampler should now produce something on the trajectory scale, not noise.
    m.eval()
    out = sample_diffusion(m, cond=cond, steps=20)
    err = (out - target).abs().mean().item()
    assert err < 1.0, f"sampled trajectory off by {err:.2f}m, expected ≪ 1m"
