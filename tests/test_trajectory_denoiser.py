"""TrajectoryDenoiser + TimeEmbedding — shared backbone shape contract."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import TimeEmbedding, TrajectoryDenoiser


def test_time_embedding_shape():
    te = TimeEmbedding(dim=64)
    out = te(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, 64)


def test_time_embedding_deterministic():
    te = TimeEmbedding(dim=32)
    t = torch.tensor([0.3, 0.7])
    a, b = te(t), te(t)
    assert torch.allclose(a, b)


def test_denoiser_forward_shape():
    m = TrajectoryDenoiser(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    x_t = torch.zeros(4, 8, 2)
    t = torch.rand(4)
    cond = torch.zeros(4, 64)
    out = m(x_t, t, cond)
    assert out.shape == (4, 8, 2)


def test_denoiser_rejects_bad_x_shape():
    m = TrajectoryDenoiser(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    with pytest.raises(ValueError):
        m(torch.zeros(4, 7, 2), torch.rand(4), torch.zeros(4, 64))


def test_denoiser_rejects_bad_cond_shape():
    m = TrajectoryDenoiser(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    with pytest.raises(ValueError):
        m(torch.zeros(4, 8, 2), torch.rand(4), torch.zeros(4, 32))


def test_invalid_dims():
    with pytest.raises(ValueError):
        TrajectoryDenoiser(horizon=0, traj_dim=2, cond_dim=64, hidden_dim=128)
    with pytest.raises(ValueError):
        TimeEmbedding(dim=0)
