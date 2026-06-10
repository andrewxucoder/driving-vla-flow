"""Synthetic obstacle injection for VLA constraint evaluation (M8.4).

Real driving logs rarely contain the rare, safety-critical events the VLA layer
is meant to handle (cones, parked trucks, jaywalkers). This module *injects*
such events onto otherwise-normal observations so the M8.4 benchmark can run a
controlled 3-way comparison:

- **baseline** — unconstrained policy + BoN
- **mock-oracle** — the hand-authored "correct" constraint (the injection's
  ``oracle_constraints``) — measures the *ceiling* of the constraint mechanism
- **VLM** — constraints extracted by Qwen3-VL from the injected caption/image —
  measures how close real VLM reasoning gets to that ceiling

Each injection produces:

- ``caption`` — natural-language description, written into
  ``observation.metadata["injected_caption"]`` and read by the VLM extractor.
- ``oracle_constraints`` — the constraint(s) a perfect extractor *should* emit.
- ``boxes`` — ego-frame ``[x_min, x_max, y_min, y_max]`` footprints, used both
  for the no-go oracle and (later) collision scoring against the obstacle.

Image overlay is intentionally not done here: mapping an ego-frame box to camera
pixels needs intrinsics/extrinsics this layer doesn't carry, and a miscalibrated
overlay would mislead the VLM. The caption is the VLM's text channel; the image
remains the raw scene frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.vla.constraint_schema import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    TrajectoryConstraint,
)


@dataclass(frozen=True)
class ObstacleInjection:
    """A synthetic obstacle: its caption, oracle constraints, and footprint."""

    kind: str
    caption: str
    oracle_constraints: list[TrajectoryConstraint]
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def _side_word(y: float) -> str:
    return "left" if y >= 0 else "right"


def cone(x_ahead: float = 12.0, y_lateral: float = -1.0, *, half: float = 0.5) -> ObstacleInjection:
    """Traffic cone at ``(x_ahead, y_lateral)`` ego-frame metres (+y = left)."""
    box = (x_ahead - half, x_ahead + half, y_lateral - half, y_lateral + half)
    return ObstacleInjection(
        kind="cone",
        caption=(
            f"A traffic cone is about {x_ahead:.0f} m ahead and "
            f"{abs(y_lateral):.0f} m to the {_side_word(y_lateral)}."
        ),
        oracle_constraints=[
            NoGoBoxConstraint(
                x_min=box[0], x_max=box[1], y_min=box[2], y_max=box[3],
                reason="synthetic cone",
            )
        ],
        boxes=[box],
    )


def parked_truck(side: str = "right", *, offset: float = 2.0) -> ObstacleInjection:
    """A truck blocking ``side`` of the lane → merge to the opposite side."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    signed = -offset if side == "left" else offset  # merge away from the truck
    return ObstacleInjection(
        kind="parked_truck",
        caption=f"A truck is parked blocking the {side} side of the lane ahead.",
        oracle_constraints=[
            LateralOffsetConstraint(value=signed, reason=f"merge around parked truck on the {side}")
        ],
    )


def pedestrian(x_ahead: float = 10.0, y_lateral: float = 0.0, *, cap_speed: float = 2.0) -> ObstacleInjection:
    """A jaywalking pedestrian ahead → cap speed (and a no-go footprint)."""
    half = 0.6
    box = (x_ahead - half, x_ahead + half, y_lateral - half, y_lateral + half)
    return ObstacleInjection(
        kind="pedestrian",
        caption=f"A pedestrian is crossing the road about {x_ahead:.0f} m ahead.",
        oracle_constraints=[
            MaxSpeedConstraint(value=cap_speed, reason="pedestrian crossing"),
            NoGoBoxConstraint(
                x_min=box[0], x_max=box[1], y_min=box[2], y_max=box[3],
                reason="pedestrian footprint",
            ),
        ],
        boxes=[box],
    )


def default_suite() -> list[ObstacleInjection]:
    """A small standard set of injections for the M8.4 benchmark."""
    return [cone(), parked_truck("right"), pedestrian()]


def inject(
    observation: DrivingTrajectorySample, injection: ObstacleInjection
) -> DrivingTrajectorySample:
    """Return a copy of ``observation`` carrying the injection's caption.

    The caption lands in ``metadata["injected_caption"]`` (the VLM text channel);
    ``metadata["injected_obstacle"]`` records the kind for audit. The oracle
    constraints are *not* written into the observation — the eval feeds them
    through the mock-oracle path explicitly, keeping the VLM path blind to them.
    """
    meta = dict(observation.metadata)
    meta["injected_caption"] = injection.caption
    meta["injected_obstacle"] = injection.kind
    return observation.model_copy(update={"metadata": meta})


def oracle_boxes(injections: Sequence[ObstacleInjection]) -> list[tuple[float, float, float, float]]:
    """All obstacle footprints across ``injections`` (for collision scoring)."""
    return [box for inj in injections for box in inj.boxes]


__all__ = [
    "ObstacleInjection",
    "cone",
    "default_suite",
    "inject",
    "oracle_boxes",
    "parked_truck",
    "pedestrian",
]
