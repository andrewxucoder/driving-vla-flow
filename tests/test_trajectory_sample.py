"""M1 smoke — DrivingTrajectorySample schema."""

from __future__ import annotations

import pytest
import torch

from driving_vla import DrivingTrajectorySample


def _state(t: int = 4, d: int = 5) -> torch.Tensor:
    return torch.zeros(t, d)


def _action(t: int = 8, d: int = 2) -> torch.Tensor:
    return torch.zeros(t, d)


def test_construct_minimum_fields():
    s = DrivingTrajectorySample(history=_state(), future=_action())
    assert s.history.shape == (4, 5)
    assert s.future.shape == (8, 2)
    assert s.instruction is None
    assert s.command_id is None
    assert s.ego_state is None
    assert s.images is None
    assert s.metadata == {}


def test_construct_full():
    s = DrivingTrajectorySample(
        history=_state(),
        future=_action(),
        instruction="Change to the left lane when it is safe.",
        command_id=1,
        ego_state=torch.zeros(5),
        images=torch.zeros(3, 3, 224, 224),
        metadata={"scenario_id": "scene-0001", "token": "abc"},
    )
    assert s.instruction.startswith("Change")
    assert s.command_id == 1
    assert s.ego_state is not None and s.ego_state.shape == (5,)
    assert s.images is not None and s.images.shape == (3, 3, 224, 224)
    assert s.metadata["scenario_id"] == "scene-0001"


def test_extra_field_forbidden():
    with pytest.raises(Exception):
        DrivingTrajectorySample(
            history=_state(),
            future=_action(),
            mystery_field="should fail",
        )


def test_history_must_be_2d():
    with pytest.raises(Exception):
        DrivingTrajectorySample(history=torch.zeros(5), future=_action())


def test_future_must_be_2d():
    with pytest.raises(Exception):
        DrivingTrajectorySample(history=_state(), future=torch.zeros(8, 2, 1))


def test_ego_state_must_be_1d():
    with pytest.raises(Exception):
        DrivingTrajectorySample(
            history=_state(), future=_action(), ego_state=torch.zeros(2, 5)
        )


def test_images_must_be_4d():
    with pytest.raises(Exception):
        DrivingTrajectorySample(
            history=_state(), future=_action(), images=torch.zeros(3, 224, 224)
        )


def test_route_optional_field():
    s = DrivingTrajectorySample(
        history=_state(), future=_action(), route=torch.zeros(20, 2)
    )
    assert s.route is not None and s.route.shape == (20, 2)


def test_route_default_none():
    s = DrivingTrajectorySample(history=_state(), future=_action())
    assert s.route is None


def test_route_must_be_2d_with_xy():
    with pytest.raises(Exception):
        DrivingTrajectorySample(
            history=_state(), future=_action(), route=torch.zeros(20)
        )
    with pytest.raises(Exception):
        DrivingTrajectorySample(
            history=_state(), future=_action(), route=torch.zeros(20, 3)
        )
