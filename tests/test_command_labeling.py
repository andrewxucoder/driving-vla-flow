"""command_labeling — vocab + heuristic command inference."""

from __future__ import annotations

import numpy as np

from driving_vla.data import COMMAND_TO_ID, COMMAND_VOCAB, infer_command


def test_vocab_has_four_classes():
    assert len(COMMAND_VOCAB) == 4
    assert set(COMMAND_VOCAB) == {"keep_lane", "change_left", "change_right", "slow_down"}


def test_vocab_to_id_consistent():
    for i, name in enumerate(COMMAND_VOCAB):
        assert COMMAND_TO_ID[name] == i


def test_keep_lane_straight():
    future = np.array([[i * 1.0, 0.0] for i in range(1, 9)], dtype=np.float32)
    assert infer_command(future) == "keep_lane"


def test_change_left_lateral_positive():
    future = np.stack(
        [np.arange(1, 9, dtype=np.float32), np.linspace(0.0, 3.0, 8, dtype=np.float32)],
        axis=1,
    )
    assert infer_command(future) == "change_left"


def test_change_right_lateral_negative():
    future = np.stack(
        [np.arange(1, 9, dtype=np.float32), np.linspace(0.0, -3.0, 8, dtype=np.float32)],
        axis=1,
    )
    assert infer_command(future) == "change_right"


def test_slow_down_decelerating():
    # Forward distance per step shrinks toward zero
    xs = np.cumsum(np.linspace(2.0, 0.1, 9))[1:].astype(np.float32)
    future = np.stack([xs, np.zeros_like(xs)], axis=1)
    assert infer_command(future) == "slow_down"


def test_short_trajectory_defaults_to_keep_lane():
    future = np.zeros((1, 2), dtype=np.float32)
    assert infer_command(future) == "keep_lane"
