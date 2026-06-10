"""Translate ``TrajectoryConstraint`` lists into reward modifications (M8.2).

Bridges the VLA layer (constraint schema) and the evaluation layer (BoN
scoring + ``trajectory_reward``). The translation is *additive*: a constraint
list of length 0 produces the unconstrained score, so callers that don't use
the VLA stack see no behaviour change.

Three constraint types, three translation rules:

- ``LateralOffsetConstraint(value)`` → add ``value`` to every sample's
  ``lane_center`` before computing ``lateral_penalty``. Multiple offset
  constraints sum (their reasons are independent, e.g. "merge left for
  obstacle" + "merge slightly more left for VIP zone").
- ``MaxSpeedConstraint(value)`` → cap the sample's ``ego_vx`` at ``value``
  before computing ``expected_progress_penalty``. Multiple caps take the
  minimum (most restrictive wins).
- ``NoGoBoxConstraint(x_min, x_max, y_min, y_max)`` → ``box_collision_proxy``
  over the concatenated boxes, weighted by ``box_collision_weight``.

The companion :func:`build_constraint_aware_score_fn` returns a
``ScoreFn``-compatible callable that reads ``cond["constraints"]`` and
dispatches to :func:`trajectory_reward_with_constraints`. ``make_bon_predictor``
(M8.2 wiring) injects the constraints from a ``BaseConstraintExtractor``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Union

import torch

from driving_vla.evaluation.metrics import collision_proxy, forward_progress, jerk
from driving_vla.evaluation.reward import (
    RewardWeights,
    box_collision_proxy,
    expected_progress_penalty,
    lane_center_from_command_id,
)
from driving_vla.vla.constraint_schema import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
)

_AnyConstraint = Union[
    LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint
]


def _total_lateral_offset(constraints: Sequence[_AnyConstraint]) -> float:
    """Sum of all ``lateral_offset`` constraint values (0.0 if none apply)."""
    return sum(
        float(c.value) for c in constraints if isinstance(c, LateralOffsetConstraint)
    )


def _resolve_lane_center_with_offsets(
    command_id: torch.Tensor,
    constraints: Sequence[_AnyConstraint],
) -> torch.Tensor:
    """``[B]`` lane center after applying all lateral_offset overrides."""
    base = lane_center_from_command_id(command_id)
    total_offset = _total_lateral_offset(constraints)
    if total_offset == 0.0:
        return base
    return base + total_offset


def _resolve_capped_ego_state(
    ego_state: torch.Tensor | None,
    constraints: Sequence[_AnyConstraint],
) -> torch.Tensor | None:
    """Clone ``ego_state`` with ego_vx capped at the most restrictive max_speed."""
    if ego_state is None:
        return None
    speed_caps = [float(c.value) for c in constraints if isinstance(c, MaxSpeedConstraint)]
    if not speed_caps:
        return ego_state
    cap = min(speed_caps)
    capped = ego_state.clone()
    capped[..., 0] = torch.clamp(capped[..., 0], max=cap)
    return capped


def _stack_no_go_boxes(
    constraints: Sequence[_AnyConstraint],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[K, 4]`` ``[x_min, x_max, y_min, y_max]`` for all NoGoBoxConstraints."""
    boxes = [
        [c.x_min, c.x_max, c.y_min, c.y_max]
        for c in constraints
        if isinstance(c, NoGoBoxConstraint)
    ]
    if not boxes:
        return torch.zeros(0, 4, device=device, dtype=dtype)
    return torch.tensor(boxes, device=device, dtype=dtype)


def trajectory_reward_with_constraints(
    traj: torch.Tensor,
    command_id: torch.Tensor,
    constraints: Sequence[_AnyConstraint],
    *,
    weights: RewardWeights | None = None,
    lane_half_width: float = 1.75,
    ego_state: torch.Tensor | None = None,
    horizon_seconds: float | None = None,
    box_collision_weight: float = 5.0,
) -> torch.Tensor:
    """Composite reward with VLA constraints applied.

    Reduces *exactly* to :func:`driving_vla.evaluation.reward.trajectory_reward`
    when ``constraints=[]``, for every ``command_id`` (the collision term shifts
    by the constraint-induced lateral offset only, which is 0 here). Otherwise,
    modifies the inputs (lane_center, capped ego_vx) and adds a no-go-box penalty
    before composing the score.

    Returns ``[B]`` (higher = better).
    """
    if weights is None:
        weights = RewardWeights()

    center = _resolve_lane_center_with_offsets(command_id, constraints)
    capped_ego = _resolve_capped_ego_state(ego_state, constraints)
    lateral_offset = _total_lateral_offset(constraints)

    progress = forward_progress(traj)
    lateral_pen = (traj[..., 1] - center.unsqueeze(-1)).abs().mean(dim=-1)

    # Lane-departure collision is shifted by the *constraint-induced* lateral
    # offset only, not the full command lane center. A lateral_offset constraint
    # that legitimately moves the ego into an adjacent lane (e.g. "merge left to
    # avoid the obstacle") should not be spuriously penalised as off-road; but
    # the command-implied center (e.g. change_left → +3.5) must keep the same
    # absolute off-road treatment as the base reward. Shifting by `lateral_offset`
    # alone makes this term reduce to `trajectory_reward`'s absolute collision
    # check for *every* command_id when no lateral_offset constraint applies.
    if lateral_offset == 0.0:
        collision_pen = collision_proxy(traj, lane_half_width=lane_half_width)
    else:
        shifted = traj.clone()
        shifted[..., 1] = traj[..., 1] - lateral_offset
        collision_pen = collision_proxy(shifted, lane_half_width=lane_half_width)
    jerk_pen = jerk(traj)

    score = (
        weights.progress * progress
        - weights.lateral * lateral_pen
        - weights.collision * collision_pen
        - weights.jerk * jerk_pen
    )

    if capped_ego is not None and horizon_seconds is not None:
        score = score - weights.progress_match * expected_progress_penalty(
            traj, capped_ego, horizon_seconds
        )

    boxes = _stack_no_go_boxes(constraints, device=traj.device, dtype=traj.dtype)
    if boxes.shape[0] > 0:
        score = score - box_collision_weight * box_collision_proxy(traj, boxes)

    return score


def build_constraint_aware_score_fn(
    horizon_seconds: float = 4.0,
    *,
    box_collision_weight: float = 5.0,
) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """Return a BoN score_fn that reads ``cond["constraints"]`` if present.

    Drop-in replacement for ``evaluation.navsim_runner.build_default_score_fn``;
    behaviour is identical (for every ``command_id``) when no constraints are
    passed, and delegates to :func:`trajectory_reward_with_constraints` when they
    are.

    The score_fn handles the same broadcasting that the default does:
    ``command_id``/``ego_state`` of shape ``[1, ...]`` are repeat-interleaved
    to match ``samples.shape[0]``. The ``constraints`` list is *not* per-sample
    (constraints are per-observation, not per-trajectory candidate) so it is
    used as-is.
    """

    def _broadcast(t: torch.Tensor, n: int, *, name: str) -> torch.Tensor:
        if t.shape[0] == n:
            return t
        if n % t.shape[0] == 0:
            return t.repeat_interleave(n // t.shape[0], dim=0)
        raise ValueError(
            f"score_fn received {n} samples with {name} of length "
            f"{t.shape[0]} (not a divisor of {n})"
        )

    def _score(samples: torch.Tensor, cond: dict[str, Any]) -> torch.Tensor:
        n = samples.shape[0]
        command_id = cond.get("command_id")
        if command_id is None:
            command_id = torch.zeros(n, dtype=torch.long, device=samples.device)
        else:
            command_id = _broadcast(command_id, n, name="command_id").to(samples.device)

        ego_state = cond.get("ego_state")
        if ego_state is not None:
            ego_state = _broadcast(ego_state, n, name="ego_state").to(samples.device)

        constraints = cond.get("constraints") or []

        return trajectory_reward_with_constraints(
            samples,
            command_id,
            constraints,
            ego_state=ego_state,
            horizon_seconds=horizon_seconds if ego_state is not None else None,
            box_collision_weight=box_collision_weight,
        )

    return _score


__all__ = [
    "build_constraint_aware_score_fn",
    "trajectory_reward_with_constraints",
]
