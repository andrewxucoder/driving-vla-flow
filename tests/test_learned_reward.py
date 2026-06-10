"""evaluation.learned_reward — MLP scorer + Bradley-Terry loss."""

from __future__ import annotations

import pytest
import torch

from driving_vla.evaluation import LearnedRewardModel, bradley_terry_loss


def test_output_scalar_per_item():
    m = LearnedRewardModel(cond_dim=32, horizon=8, traj_dim=2)
    out = m(torch.zeros(4, 32), torch.zeros(4, 8, 2))
    assert out.shape == (4,)


def test_bad_cond_shape_raises():
    m = LearnedRewardModel(cond_dim=32, horizon=8, traj_dim=2)
    with pytest.raises(ValueError):
        m(torch.zeros(4, 16), torch.zeros(4, 8, 2))


def test_bad_traj_shape_raises():
    m = LearnedRewardModel(cond_dim=32, horizon=8, traj_dim=2)
    with pytest.raises(ValueError):
        m(torch.zeros(4, 32), torch.zeros(4, 5, 2))


def test_bradley_terry_loss_scalar():
    chosen = torch.tensor([1.0, 2.0])
    rejected = torch.tensor([0.0, 0.5])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_bradley_terry_gradient_flows():
    chosen = torch.tensor([0.5], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = bradley_terry_loss(chosen, rejected)
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None


def test_bradley_terry_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bradley_terry_loss(torch.zeros(3), torch.zeros(2))


def test_invalid_dims():
    with pytest.raises(ValueError):
        LearnedRewardModel(cond_dim=0, horizon=8, traj_dim=2)
