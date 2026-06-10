"""WaypointRegressionHead — direct MLP regression."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import WaypointRegressionHead


def test_output_shape():
    head = WaypointRegressionHead(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    out = head(torch.zeros(4, 64))
    assert out.shape == (4, 8, 2)


def test_deterministic_given_same_input():
    head = WaypointRegressionHead(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    head.eval()
    cond = torch.randn(2, 64)
    a, b = head(cond), head(cond)
    assert torch.allclose(a, b)


def test_rejects_bad_cond_shape():
    head = WaypointRegressionHead(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=128)
    with pytest.raises(ValueError):
        head(torch.zeros(4, 32))


def test_invalid_dims():
    with pytest.raises(ValueError):
        WaypointRegressionHead(horizon=0, traj_dim=2, cond_dim=64, hidden_dim=128)
    with pytest.raises(ValueError):
        WaypointRegressionHead(horizon=8, traj_dim=2, cond_dim=64, hidden_dim=0)
