"""NAVSIM adapter — custom NavSimSceneLoader bypasses devkit imports."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from driving_vla.data import (
    NavSimScene,
    NavSimSceneFrame,
    NavSimSceneLoader,
    NavSimTrajectoryDataset,
)


class _FakeLoader:
    """Manufactures synthetic scenes for testing without navsim installed."""

    def __init__(self, scenes: list[NavSimScene]) -> None:
        self._scenes = {s.token: s for s in scenes}

    def tokens(self) -> Iterable[str]:
        return list(self._scenes.keys())

    def get_scene(self, token: str) -> NavSimScene:
        return self._scenes[token]


def _make_scene(token: str, *, n_history: int = 5, n_future: int = 8) -> NavSimScene:
    """Straight-line scene: ego moves +x at constant speed, no rotation."""
    total = n_history + 1 + n_future
    frames = []
    for i in range(total):
        frames.append(
            NavSimSceneFrame(
                ego_xy=np.array([float(i), 0.0]),
                ego_heading=0.0,
                velocity_xy=np.array([1.0, 0.0]),
                acceleration_xy=np.array([0.0, 0.0]),
                yaw_rate=0.0,
                camera_paths={},
            )
        )
    return NavSimScene(token=token, log_name="fake.log", frames=frames, current_index=n_history)


def test_fake_loader_satisfies_protocol():
    loader = _FakeLoader([_make_scene("t1")])
    assert isinstance(loader, NavSimSceneLoader)


def test_dataset_length_matches_loader():
    loader = _FakeLoader([_make_scene(f"t{i}") for i in range(3)])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        split="navtest",
        history_seconds=2.0,
        future_seconds=4.0,
        sample_rate_hz=2.0,
        scene_loader=loader,
    )
    assert len(ds) == 3


def test_sample_shape_and_keys():
    loader = _FakeLoader([_make_scene("t0", n_history=4, n_future=8)])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        split="navtest",
        history_seconds=2.0,    # 4 frames at 2 Hz
        future_seconds=4.0,     # 8 frames at 2 Hz
        sample_rate_hz=2.0,
        scene_loader=loader,
    )
    s = ds[0]
    assert s["future"].shape == (8, 2)
    assert s["history"].shape == (4, 2)
    assert s["ego_state"].shape == (5,)
    assert "condition" in s
    assert s["condition"]["command_id"].item() == 0  # straight line → keep_lane


def test_future_in_ego_frame():
    """Going straight forward → future xy should be (1,0), (2,0), ..."""
    loader = _FakeLoader([_make_scene("t0", n_history=2, n_future=4)])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        split="navtest",
        history_seconds=1.0,
        future_seconds=2.0,
        sample_rate_hz=2.0,
        scene_loader=loader,
    )
    s = ds[0]
    np.testing.assert_allclose(
        s["future"].numpy(),
        [[1, 0], [2, 0], [3, 0], [4, 0]],
        atol=1e-5,
    )


def test_scene_at_returns_raw():
    loader = _FakeLoader([_make_scene("t0")])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        scene_loader=loader,
    )
    raw = ds.scene_at(0)
    assert isinstance(raw, NavSimScene)
    assert raw.token == "t0"


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        NavSimTrajectoryDataset(
            data_root="/dev/null",
            split="invalid",
            scene_loader=_FakeLoader([_make_scene("t0")]),
        )


def test_too_few_future_frames_raises():
    # Scene only has 2 future frames but we request 8
    loader = _FakeLoader([_make_scene("t0", n_history=2, n_future=2)])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        future_seconds=4.0,    # 8 frames at 2 Hz
        sample_rate_hz=2.0,
        scene_loader=loader,
    )
    with pytest.raises(ValueError, match="future frames"):
        _ = ds[0]


def test_index_out_of_range_raises():
    loader = _FakeLoader([_make_scene("t0")])
    ds = NavSimTrajectoryDataset(
        data_root="/dev/null",
        scene_loader=loader,
    )
    with pytest.raises(IndexError):
        _ = ds[5]


def test_scenario_ids_returns_token_list():
    loader = _FakeLoader([_make_scene(f"t{i}") for i in range(3)])
    ds = NavSimTrajectoryDataset(data_root="/dev/null", scene_loader=loader)
    assert set(ds.scenario_ids()) == {"t0", "t1", "t2"}
