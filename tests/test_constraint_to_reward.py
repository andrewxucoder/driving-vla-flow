"""M8.2 — constraint → reward translation + box_collision_proxy."""

from __future__ import annotations

import pytest
import torch

from driving_vla.evaluation import box_collision_proxy, trajectory_reward
from driving_vla.vla import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    trajectory_reward_with_constraints,
)


def _straight(y: float = 0.0, x_end: float = 8.0, H: int = 8) -> torch.Tensor:
    xs = torch.linspace(1.0, x_end, H)
    ys = torch.full((H,), y)
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)  # [1, H, 2]


# ── box_collision_proxy (geometry) ──────────────────────────────────────────


def test_box_collision_detects_waypoint_inside():
    traj = _straight(y=0.0)  # passes through x in [1,8], y=0
    boxes = torch.tensor([[3.0, 5.0, -1.0, 1.0]])  # covers x∈[3,5], y∈[-1,1]
    out = box_collision_proxy(traj, boxes)
    assert out.shape == (1,)
    assert out.item() == 1.0


def test_box_collision_misses_when_clear():
    traj = _straight(y=0.0)
    boxes = torch.tensor([[3.0, 5.0, 5.0, 7.0]])  # far in +y, traj at y=0
    assert box_collision_proxy(traj, boxes).item() == 0.0


def test_box_collision_empty_boxes_returns_zero():
    traj = _straight()
    out = box_collision_proxy(traj, torch.zeros(0, 4))
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_box_collision_multiple_boxes_any_hit():
    traj = _straight(y=0.0)
    boxes = torch.tensor([
        [3.0, 5.0, 5.0, 7.0],   # miss
        [6.0, 7.0, -0.5, 0.5],  # hit (traj passes x∈[6,7] at y=0)
    ])
    assert box_collision_proxy(traj, boxes).item() == 1.0


def test_box_collision_per_batch_boxes():
    traj = torch.stack([_straight(y=0.0)[0], _straight(y=3.0)[0]])  # [2, H, 2]
    # Per-batch boxes [B, K, 4]: first sample box at y=0 (hit), second at y=0 (miss for y=3 traj)
    boxes = torch.tensor([
        [[3.0, 5.0, -0.5, 0.5]],
        [[3.0, 5.0, -0.5, 0.5]],
    ])
    out = box_collision_proxy(traj, boxes)
    assert out[0].item() == 1.0  # y=0 traj enters box
    assert out[1].item() == 0.0  # y=3 traj clears box


def test_box_collision_rejects_bad_shape():
    with pytest.raises(ValueError):
        box_collision_proxy(torch.zeros(1, 8, 2), torch.zeros(3))  # boxes not [K,4]
    with pytest.raises(ValueError):
        box_collision_proxy(torch.zeros(8, 2), torch.zeros(1, 4))  # traj not [B,H,2]


# ── trajectory_reward_with_constraints ──────────────────────────────────────


def test_empty_constraints_equals_base_reward():
    traj = _straight(y=0.0)
    cmd = torch.tensor([0])
    base = trajectory_reward(traj, cmd)
    with_empty = trajectory_reward_with_constraints(traj, cmd, [])
    assert torch.allclose(base, with_empty)


@pytest.mark.parametrize("cmd_id", [0, 1, 2, 3])
def test_empty_constraints_equals_base_reward_all_commands(cmd_id):
    """With no constraints, the constraint-aware reward must reduce *exactly* to
    the base reward for every command_id — including change_left/right (center
    ±3.5), where the collision term must stay an absolute off-road check rather
    than being shifted by the command lane center."""
    cmd = torch.tensor([cmd_id])
    # A trajectory that rides the command-implied lane center, so the
    # change-lane cases actually exercise the off-road collision branch.
    for y in (0.0, 3.5, -3.5):
        traj = _straight(y=y)
        base = trajectory_reward(traj, cmd)
        with_empty = trajectory_reward_with_constraints(traj, cmd, [])
        assert torch.allclose(base, with_empty), f"cmd={cmd_id}, y={y}"


def test_lateral_offset_does_not_double_shift_change_lane_collision():
    """A lateral_offset on a change-lane command shifts collision by the offset
    only (not offset+center): a change_left trajectory at y=3.5 stays off-road,
    while a +offset that brings it back toward center relieves the penalty."""
    cmd = torch.tensor([1])  # change_left, base center +3.5
    traj = _straight(y=3.5)
    # offset that re-centers the collision check (3.5 - 3.5 = 0 → on-road)
    relief = [LateralOffsetConstraint(value=3.5, reason="treat as new center")]
    r_off = trajectory_reward_with_constraints(traj, cmd, [])
    r_on = trajectory_reward_with_constraints(traj, cmd, relief)
    # Relieving the off-road collision penalty raises the score.
    assert r_on > r_off


def test_lateral_offset_constraint_shifts_preference():
    """A trajectory at y=3 should score higher than y=0 when a +3 lateral
    offset constraint moves the target lane center to y=3 (keep_lane base=0)."""
    cmd = torch.tensor([0])  # keep_lane, base center = 0
    traj_center = _straight(y=0.0)
    traj_left = _straight(y=3.0)
    constraints = [LateralOffsetConstraint(value=3.0, reason="merge left")]

    # Without constraint: center (y=0) wins on lateral penalty
    r_center_base = trajectory_reward_with_constraints(traj_center, cmd, [])
    r_left_base = trajectory_reward_with_constraints(traj_left, cmd, [])
    assert r_center_base > r_left_base

    # With +3 offset: the y=3 trajectory now matches the shifted center
    r_center = trajectory_reward_with_constraints(traj_center, cmd, constraints)
    r_left = trajectory_reward_with_constraints(traj_left, cmd, constraints)
    assert r_left > r_center


def test_max_speed_constraint_penalises_overshoot():
    """ego moving at 5 m/s normally expects 20m over 4s; a max_speed=1 cap
    means the slow trajectory should now score higher than the fast one."""
    cmd = torch.tensor([0])
    ego = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])  # vx=5
    fast = _straight(x_end=20.0)   # ~19m progress
    slow = _straight(x_end=4.0)    # ~3m progress

    # No cap: fast matches expected (20m) → fast wins
    r_fast = trajectory_reward_with_constraints(
        fast, cmd, [], ego_state=ego, horizon_seconds=4.0
    )
    r_slow = trajectory_reward_with_constraints(
        slow, cmd, [], ego_state=ego, horizon_seconds=4.0
    )
    assert r_fast > r_slow

    # max_speed=1 → expected becomes ~4m → slow wins
    cap = [MaxSpeedConstraint(value=1.0, reason="construction")]
    r_fast_c = trajectory_reward_with_constraints(
        fast, cmd, cap, ego_state=ego, horizon_seconds=4.0
    )
    r_slow_c = trajectory_reward_with_constraints(
        slow, cmd, cap, ego_state=ego, horizon_seconds=4.0
    )
    assert r_slow_c > r_fast_c


def test_no_go_box_constraint_penalises_intruding_traj():
    cmd = torch.tensor([0])
    clear = _straight(y=0.0)
    # An intruding trajectory that dips through a box at (x∈[3,5], y∈[1.5,2.5])
    intrude = _straight(y=2.0)
    box = [NoGoBoxConstraint(x_min=1.0, x_max=8.0, y_min=1.5, y_max=2.5)]

    r_clear = trajectory_reward_with_constraints(clear, cmd, box)
    r_intrude = trajectory_reward_with_constraints(intrude, cmd, box)
    assert r_clear > r_intrude


def test_multiple_max_speed_constraints_take_minimum():
    cmd = torch.tensor([0])
    ego = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    fast = _straight(x_end=40.0)
    slow = _straight(x_end=4.0)
    # Two caps: 8 and 1 → effective cap is 1 → slow wins
    constraints = [MaxSpeedConstraint(value=8.0), MaxSpeedConstraint(value=1.0)]
    r_fast = trajectory_reward_with_constraints(
        fast, cmd, constraints, ego_state=ego, horizon_seconds=4.0
    )
    r_slow = trajectory_reward_with_constraints(
        slow, cmd, constraints, ego_state=ego, horizon_seconds=4.0
    )
    assert r_slow > r_fast
