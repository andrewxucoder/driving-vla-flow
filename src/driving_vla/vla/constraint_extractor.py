"""``BaseConstraintExtractor`` Protocol + non-VLM baselines (M8.1).

``BaseConstraintExtractor`` is the algorithmic seam: anything that maps
``(observation) → list[TrajectoryConstraint]`` is an extractor. The Qwen3-VL
backend (M8.3) is one such implementation; this module ships two non-VLM
baselines used for unit tests and ablations:

- :class:`MockConstraintExtractor`: returns a preset list verbatim. Used in
  tests and as the *oracle* baseline in the M8.4 evaluation (we hand-author
  the "right" constraints and check whether the BoN-under-constraints
  pipeline can in principle reach them).
- :class:`RuleBasedConstraintExtractor`: derives constraints from the
  observation via simple heuristics (low-speed history → emit a
  ``MaxSpeedConstraint``). A *non-VLM* baseline — if a rule-based extractor
  already beats the unconstrained baseline by a wide margin, the VLM has to
  beat the rules to justify its compute cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Union, runtime_checkable

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.vla.constraint_schema import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
)

_AnyConstraint = Union[
    LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint
]


@runtime_checkable
class BaseConstraintExtractor(Protocol):
    """Map a single observation to a list of trajectory constraints.

    Implementations must be deterministic w.r.t. a fixed observation (for
    reproducibility) but may rely on internal caches keyed by observation
    hash — the VLM backend uses this to avoid repeated inference calls in
    research / ablation runs.

    Returning an empty list is legal and means "no constraints derived" —
    downstream BoN simply falls back to the unconstrained reward.
    """

    def __call__(
        self, observation: DrivingTrajectorySample
    ) -> list[_AnyConstraint]:
        ...


class MockConstraintExtractor:
    """Returns the preset constraint list verbatim, ignoring the observation.

    Two modes:

    - ``constraints`` is a list → that list is returned on every call.
    - ``constraints`` is a callable ``(observation) → list[...]`` → the
      callable is invoked on every call (useful for test fixtures that need
      per-observation control).
    """

    def __init__(
        self,
        constraints: (
            list[_AnyConstraint]
            | Callable[[DrivingTrajectorySample], list[_AnyConstraint]]
        ),
    ) -> None:
        self._constraints = constraints

    def __call__(
        self, observation: DrivingTrajectorySample
    ) -> list[_AnyConstraint]:
        if callable(self._constraints):
            return list(self._constraints(observation))
        return list(self._constraints)


class RuleBasedConstraintExtractor:
    """Heuristic baseline derived from observation alone — no VLM, no images.

    Current rules (kept intentionally simple; this is a *baseline*, not the
    research target):

    - If the ego is moving slower than ``low_speed_threshold`` m/s (forward
      speed from ``ego_state[0]``), emit a ``MaxSpeedConstraint`` capping at
      ``low_speed_cap``. Approximates "if you're already slow, stay slow"
      — a useful sanity check against policies that snap back to high speed.
    - If ``command_id`` is the discrete change-lane signal (id 1 or 2 in
      ``COMMAND_VOCAB``), emit a ``LateralOffsetConstraint`` of ±
      ``lane_change_offset`` matching the command direction. Mirrors what
      a low-bandwidth navigation channel would emit.

    Returns ``[]`` when no rule fires.
    """

    _CHANGE_LEFT_ID: int = 1
    _CHANGE_RIGHT_ID: int = 2

    def __init__(
        self,
        low_speed_threshold: float = 1.0,
        low_speed_cap: float = 2.0,
        lane_change_offset: float = 3.5,
    ) -> None:
        if low_speed_threshold < 0:
            raise ValueError(
                f"low_speed_threshold must be ≥ 0, got {low_speed_threshold}"
            )
        if low_speed_cap < 0:
            raise ValueError(f"low_speed_cap must be ≥ 0, got {low_speed_cap}")
        if lane_change_offset <= 0:
            raise ValueError(
                f"lane_change_offset must be > 0, got {lane_change_offset}"
            )
        self.low_speed_threshold = float(low_speed_threshold)
        self.low_speed_cap = float(low_speed_cap)
        self.lane_change_offset = float(lane_change_offset)

    def __call__(
        self, observation: DrivingTrajectorySample
    ) -> list[_AnyConstraint]:
        out: list[_AnyConstraint] = []

        if observation.ego_state is not None and observation.ego_state.numel() >= 1:
            vx = float(observation.ego_state[0].item())
            if vx < self.low_speed_threshold:
                out.append(
                    MaxSpeedConstraint(
                        value=self.low_speed_cap,
                        reason=(
                            f"rule: ego_vx={vx:.2f} m/s < threshold "
                            f"{self.low_speed_threshold:.2f} m/s"
                        ),
                    )
                )

        if observation.command_id is not None:
            cid = int(observation.command_id)
            if cid == self._CHANGE_LEFT_ID:
                out.append(
                    LateralOffsetConstraint(
                        value=+self.lane_change_offset,
                        reason="rule: command_id=change_left",
                    )
                )
            elif cid == self._CHANGE_RIGHT_ID:
                out.append(
                    LateralOffsetConstraint(
                        value=-self.lane_change_offset,
                        reason="rule: command_id=change_right",
                    )
                )

        return out


__all__ = [
    "BaseConstraintExtractor",
    "MockConstraintExtractor",
    "RuleBasedConstraintExtractor",
]
