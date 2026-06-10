"""Rule-based geometric reward over trajectory chunks.

Reward components (each producing one scalar per batch item):

- ``progress``         forward distance covered (raw)
- ``-lateral_pen``     absolute lateral deviation from the command-implied lane
- ``-collision_pen``   binary lane-off penalty
- ``-jerk_pen``        comfort proxy (mean |Δ²p|)
- ``-progress_match``  |actual progress − expected progress|, where expected
                       comes from current ego speed × horizon time. Only added
                       when an ``ego_state`` (with ``vx`` in slot 0) is supplied
                       to :func:`trajectory_reward`. Designed to prevent BoN
                       from selecting "fastest" candidates that overshoot the
                       speed implied by the observation — the failure mode that
                       made Round 0's NAVSIM BoN worse than greedy.

Composite reward is a fixed-weights linear combination; for unit-tests and
deterministic BoN baselines we use :class:`RewardWeights` defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from driving_vla.evaluation.metrics import (
    collision_proxy,
    forward_progress,
    jerk,
)


@dataclass
class RewardWeights:
    progress: float = 1.0
    lateral: float = 0.5
    collision: float = 5.0
    jerk: float = 0.1
    progress_match: float = 2.0
    """Weight on |actual − expected| forward progress. No effect unless
    ``ego_state`` is passed into :func:`trajectory_reward`. Tuned > base
    ``progress`` weight so the match term dominates BoN selection when the
    observation implies a specific speed (the failure mode that made Round 0
    NAVSIM BoN worse than greedy — fastest candidates kept winning regardless
    of the current vehicle speed)."""


# Per COMMAND_VOCAB index, the lane-center y the trajectory is expected to ride.
# keep_lane → 0, change_left → +3.5, change_right → -3.5, slow_down → 0.
LANE_CENTER_BY_COMMAND: tuple[float, ...] = (0.0, 3.5, -3.5, 0.0)


def lane_center_from_command_id(command_id: torch.Tensor) -> torch.Tensor:
    """Map ``[B]`` command_id indices to ``[B]`` expected lane-center y."""
    if command_id.ndim != 1:
        raise ValueError(f"command_id must be 1D, got {tuple(command_id.shape)}")
    centres = torch.tensor(
        LANE_CENTER_BY_COMMAND, dtype=torch.float32, device=command_id.device
    )
    return centres.index_select(0, command_id.clamp(min=0, max=len(LANE_CENTER_BY_COMMAND) - 1))


def lateral_penalty(traj: torch.Tensor, lane_center_y: torch.Tensor) -> torch.Tensor:
    """Mean |y - lane_center| over the horizon."""
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj expected [B, H, 2], got {tuple(traj.shape)}")
    if lane_center_y.ndim != 1 or lane_center_y.shape[0] != traj.shape[0]:
        raise ValueError(
            f"lane_center_y must be [B={traj.shape[0]}], got {tuple(lane_center_y.shape)}"
        )
    return (traj[..., 1] - lane_center_y.unsqueeze(-1)).abs().mean(dim=-1)


def box_collision_proxy(
    traj: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """``[B]`` 1.0 if any waypoint enters any box, else 0.0.

    Args:
        traj:  ``[B, H, 2]`` ego-frame trajectory.
        boxes: ``[K, 4]`` ego-frame axis-aligned boxes,
               each row = ``[x_min, x_max, y_min, y_max]`` in metres.

    Used by the M8 ``no_go_box`` constraint translator and by future evaluators
    that need a generic obstacle-collision proxy (without modelling agent
    dynamics). Boxes can be supplied per-batch via broadcasting: ``[B, K, 4]``
    is also accepted, where row ``b`` carries that sample's boxes.
    """
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj expected [B, H, 2], got {tuple(traj.shape)}")
    if boxes.ndim not in (2, 3) or boxes.shape[-1] != 4:
        raise ValueError(f"boxes expected [K, 4] or [B, K, 4], got {tuple(boxes.shape)}")
    batch = traj.shape[0]

    if boxes.numel() == 0:
        return torch.zeros(batch, device=traj.device, dtype=traj.dtype)

    if boxes.ndim == 2:
        boxes_b = boxes.unsqueeze(0).expand(batch, -1, -1)  # [B, K, 4]
    else:
        if boxes.shape[0] != batch:
            raise ValueError(
                f"per-batch boxes must have batch dim {batch}, got {boxes.shape[0]}"
            )
        boxes_b = boxes

    x = traj[..., 0].unsqueeze(1)  # [B, 1, H]
    y = traj[..., 1].unsqueeze(1)  # [B, 1, H]
    xmin = boxes_b[..., 0:1]       # [B, K, 1]
    xmax = boxes_b[..., 1:2]
    ymin = boxes_b[..., 2:3]
    ymax = boxes_b[..., 3:4]

    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)  # [B, K, H]
    return inside.any(dim=-1).any(dim=-1).to(dtype=traj.dtype)       # [B]


def expected_progress_penalty(
    traj: torch.Tensor,
    ego_state: torch.Tensor,
    horizon_seconds: float,
) -> torch.Tensor:
    """``[B]`` |actual_progress − ego_vx · horizon_seconds|.

    ``ego_state`` shape ``[B, ≥1]`` with ``vx`` in slot 0 (ego-frame forward
    speed in m/s). ``horizon_seconds`` is the wall-clock duration spanned by
    the trajectory chunk. Used as a soft prior in BoN to reject candidates
    whose terminal forward distance disagrees with the speed the observation
    implies.
    """
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj expected [B, H, 2], got {tuple(traj.shape)}")
    if ego_state.ndim != 2 or ego_state.shape[0] != traj.shape[0]:
        raise ValueError(
            f"ego_state must be [B={traj.shape[0]}, ≥1], got {tuple(ego_state.shape)}"
        )
    if horizon_seconds <= 0:
        raise ValueError(f"horizon_seconds must be > 0, got {horizon_seconds}")
    vx = ego_state[:, 0]
    expected = vx * horizon_seconds
    actual = forward_progress(traj)
    return (actual - expected).abs()


def trajectory_reward(
    traj: torch.Tensor,
    command_id: torch.Tensor,
    weights: RewardWeights | None = None,
    lane_half_width: float = 1.75,
    ego_state: torch.Tensor | None = None,
    horizon_seconds: float | None = None,
) -> torch.Tensor:
    """Composite scalar reward per batch item.

    Returns ``[B]``, higher = better. The ``progress_match`` term only fires
    when both ``ego_state`` and ``horizon_seconds`` are supplied — otherwise
    callers see the same numbers they saw before the term was added.
    """
    if weights is None:
        weights = RewardWeights()

    lane_center = lane_center_from_command_id(command_id)

    progress = forward_progress(traj)
    lateral_pen = lateral_penalty(traj, lane_center)
    collision_pen = collision_proxy(traj, lane_half_width=lane_half_width)
    jerk_pen = jerk(traj)

    score = (
        weights.progress * progress
        - weights.lateral * lateral_pen
        - weights.collision * collision_pen
        - weights.jerk * jerk_pen
    )
    if ego_state is not None and horizon_seconds is not None:
        score = score - weights.progress_match * expected_progress_penalty(
            traj, ego_state, horizon_seconds
        )
    return score


__all__ = [
    "LANE_CENTER_BY_COMMAND",
    "RewardWeights",
    "box_collision_proxy",
    "expected_progress_penalty",
    "lane_center_from_command_id",
    "lateral_penalty",
    "trajectory_reward",
]
