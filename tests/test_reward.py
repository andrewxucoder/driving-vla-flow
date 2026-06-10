"""evaluation.reward — rule-based composite reward."""

from __future__ import annotations

import pytest
import torch

from driving_vla.evaluation import (
    LANE_CENTER_BY_COMMAND,
    RewardWeights,
    lane_center_from_command_id,
    lateral_penalty,
    trajectory_reward,
)
from driving_vla.evaluation.reward import expected_progress_penalty


def _line(y_end: float = 0.0, H: int = 8) -> torch.Tensor:
    xs = torch.linspace(1.0, 8.0, H)
    ys = torch.linspace(0.0, y_end, H)
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)


def test_lane_center_map_matches_constants():
    cmd = torch.tensor([0, 1, 2, 3])
    out = lane_center_from_command_id(cmd)
    assert out.tolist() == list(LANE_CENTER_BY_COMMAND)


def test_lane_center_clamps_out_of_range():
    out = lane_center_from_command_id(torch.tensor([99]))
    assert out.shape == (1,)


def test_lateral_penalty_is_zero_at_center():
    traj = _line(y_end=0.0)
    center = torch.tensor([0.0])
    assert lateral_penalty(traj, center).item() == pytest.approx(0.0)


def test_lateral_penalty_nonzero_offset():
    traj = _line(y_end=2.0)
    center = torch.tensor([0.0])
    pen = lateral_penalty(traj, center).item()
    assert pen > 0


def test_trajectory_reward_keep_lane():
    traj = _line(y_end=0.0)
    r = trajectory_reward(traj, command_id=torch.tensor([0]))
    # progress=+7, others ~ 0 → reward ≈ 7
    assert r.item() == pytest.approx(7.0, abs=0.1)


def test_trajectory_reward_collision_penalised():
    traj = _line(y_end=5.0)  # off-lane
    r_no_coll = trajectory_reward(_line(y_end=0.0), command_id=torch.tensor([0]))
    r_coll = trajectory_reward(traj, command_id=torch.tensor([0]))
    assert r_coll < r_no_coll


def test_trajectory_reward_change_left_centered():
    traj = _line(y_end=3.5)  # matches change_left expected lane center
    r = trajectory_reward(traj, command_id=torch.tensor([1]))  # change_left
    assert r.item() > 0


def test_lateral_penalty_shape_mismatch():
    with pytest.raises(ValueError):
        lateral_penalty(_line(), torch.zeros(2))


def test_reward_weights_dataclass():
    w = RewardWeights(progress=2.0)
    assert w.progress == 2.0


def test_expected_progress_penalty_zero_when_matching():
    # traj covers 7m forward over the horizon; ego_state vx=1.75 m/s * 4s = 7m.
    traj = _line(y_end=0.0)
    ego = torch.tensor([[1.75, 0.0, 0.0, 0.0, 0.0]])
    pen = expected_progress_penalty(traj, ego, horizon_seconds=4.0)
    assert pen.item() == pytest.approx(0.0, abs=1e-5)


def test_expected_progress_penalty_punishes_overshoot():
    traj = _line(y_end=0.0)  # 7m forward
    slow = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]])  # expected 2m
    pen = expected_progress_penalty(traj, slow, horizon_seconds=4.0)
    assert pen.item() == pytest.approx(5.0, abs=1e-5)


def test_progress_match_term_only_fires_with_ego_state():
    """Backwards-compat: omitting ego_state preserves the pre-fix reward value."""
    traj = _line(y_end=0.0)
    cmd = torch.tensor([0])
    baseline = trajectory_reward(traj, cmd).item()
    with_ego = trajectory_reward(
        traj,
        cmd,
        ego_state=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
        horizon_seconds=4.0,
    ).item()
    # baseline: progress=7 (no other penalties for a centered straight line).
    # with_ego (vx=0): expected progress=0 → penalty = 2.0 * |7−0| = 14 → score −7.
    assert baseline == pytest.approx(7.0, abs=0.1)
    assert with_ego == pytest.approx(-7.0, abs=0.1)
    assert with_ego < baseline  # the term actually changes the score


def test_progress_aware_reward_prefers_speed_matching_candidate():
    # Two same-shape candidates, only forward progress differs. ego implies 2m.
    fast = torch.stack([torch.linspace(1, 8, 8), torch.zeros(8)], dim=-1).unsqueeze(0)
    slow = torch.stack([torch.linspace(0.25, 2.0, 8), torch.zeros(8)], dim=-1).unsqueeze(0)
    cmd = torch.tensor([0])
    ego_slow = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]])  # 2m expected over 4s
    r_fast = trajectory_reward(
        fast, cmd, ego_state=ego_slow, horizon_seconds=4.0
    ).item()
    r_slow = trajectory_reward(
        slow, cmd, ego_state=ego_slow, horizon_seconds=4.0
    ).item()
    assert r_slow > r_fast, (
        f"speed-matched candidate should outscore overshoot; got slow={r_slow:.3f} fast={r_fast:.3f}"
    )
