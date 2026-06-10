"""DrivingObservationEncoder — branches + shape contract."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import (
    DrivingObservationEncoder,
    HashedInstructionEncoder,
    MultiViewImageEncoder,
    RouteEncoder,
)


def test_minimum_branches_ego_state_plus_command():
    enc = DrivingObservationEncoder(state_dim=5, num_commands=4, latent_dim=64)
    out = enc({"ego_state": torch.zeros(2, 5)})
    assert out.shape == (2, 64)


def test_with_history():
    enc = DrivingObservationEncoder(
        state_dim=5,
        num_commands=4,
        latent_dim=64,
        history_dim=2,
    )
    out = enc(
        {
            "ego_state": torch.zeros(3, 5),
            "history": torch.zeros(3, 4, 2),
            "command_id": torch.tensor([0, 1, 2]),
        }
    )
    assert out.shape == (3, 64)


def test_missing_command_id_defaults_to_zero():
    enc = DrivingObservationEncoder(state_dim=5, num_commands=4, latent_dim=64)
    out_with = enc(
        {"ego_state": torch.zeros(1, 5), "command_id": torch.tensor([0])}
    )
    out_without = enc({"ego_state": torch.zeros(1, 5)})
    assert torch.allclose(out_with, out_without)


def test_with_images_branch():
    img_enc = MultiViewImageEncoder(latent_dim=32, image_size=64)
    enc = DrivingObservationEncoder(
        state_dim=5,
        num_commands=4,
        latent_dim=64,
        image_encoder=img_enc,
    )
    out = enc(
        {
            "ego_state": torch.zeros(2, 5),
            "images": torch.zeros(2, 3, 3, 64, 64),
        }
    )
    assert out.shape == (2, 64)


def test_with_instruction_branch():
    text_enc = HashedInstructionEncoder(latent_dim=24)
    enc = DrivingObservationEncoder(
        state_dim=5,
        num_commands=4,
        latent_dim=64,
        instruction_encoder=text_enc,
    )
    out = enc(
        {
            "ego_state": torch.zeros(2, 5),
            "instruction": ["change to the left lane", "slow down gently"],
        }
    )
    assert out.shape == (2, 64)


def test_missing_ego_state_raises():
    enc = DrivingObservationEncoder(state_dim=5, num_commands=4, latent_dim=64)
    with pytest.raises(KeyError):
        enc({})


def test_wrong_state_shape_raises():
    enc = DrivingObservationEncoder(state_dim=5, num_commands=4, latent_dim=64)
    with pytest.raises(ValueError):
        enc({"ego_state": torch.zeros(2, 3)})


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        DrivingObservationEncoder(state_dim=0, num_commands=4, latent_dim=64)
    with pytest.raises(ValueError):
        DrivingObservationEncoder(state_dim=5, num_commands=4, latent_dim=0)


def test_with_route_branch():
    route_enc = RouteEncoder(latent_dim=24, max_waypoints=16, hidden_dim=32, num_heads=2)
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=64, route_encoder=route_enc,
    )
    out = enc({
        "ego_state": torch.zeros(2, 5),
        "route": torch.randn(2, 12, 2),
    })
    assert out.shape == (2, 64)


def test_route_branch_missing_route_uses_zero_fill():
    route_enc = RouteEncoder(latent_dim=24, max_waypoints=16, hidden_dim=32, num_heads=2)
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=64, route_encoder=route_enc,
    )
    out = enc({"ego_state": torch.zeros(2, 5)})
    assert out.shape == (2, 64)


def test_route_branch_with_mask():
    route_enc = RouteEncoder(latent_dim=16, max_waypoints=16, hidden_dim=32, num_heads=2)
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=48, route_encoder=route_enc,
    )
    out = enc({
        "ego_state": torch.zeros(2, 5),
        "route": torch.randn(2, 16, 2),
        "route_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    })
    assert out.shape == (2, 48)


def test_route_branch_batch_mismatch_raises():
    route_enc = RouteEncoder(latent_dim=16, max_waypoints=16, hidden_dim=32, num_heads=2)
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=48, route_encoder=route_enc,
    )
    with pytest.raises(ValueError, match="route batch"):
        enc({"ego_state": torch.zeros(2, 5), "route": torch.randn(3, 8, 2)})


def test_route_branch_wrong_shape_raises():
    route_enc = RouteEncoder(latent_dim=16, max_waypoints=16, hidden_dim=32, num_heads=2)
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=48, route_encoder=route_enc,
    )
    with pytest.raises(ValueError, match=r"\[B, N, 2\]"):
        enc({"ego_state": torch.zeros(1, 5), "route": torch.randn(1, 8, 3)})
