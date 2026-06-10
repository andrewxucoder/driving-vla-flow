"""``TrajectoryConstraint`` — pydantic discriminated-union schema (M8.1).

Three constraint types make up the M8 vocabulary:

- ``LateralOffsetConstraint``: shift the target lane center by ``value`` metres
  in the ego frame (+ = left, − = right). Implemented by re-writing the
  ``lane_center`` term in :func:`driving_vla.evaluation.reward.trajectory_reward`.
- ``MaxSpeedConstraint``: cap expected forward progress at ``value`` m/s.
  Implemented by overriding the ``expected_progress_penalty`` target.
- ``NoGoBoxConstraint``: an ego-frame axis-aligned rectangle the trajectory
  must avoid. Implemented by a new ``box_collision_proxy`` reward term.

The discriminated union ``TrajectoryConstraint`` is a type alias backed by a
pydantic ``TypeAdapter`` for round-tripping VLM JSON outputs. The three
concrete classes are also exported so callers can use ``isinstance``.

Shared fields on every constraint:

- ``type`` (str literal) — discriminator
- ``confidence`` (0..1) — VLM's self-reported certainty
- ``reason`` (str) — natural-language explanation (audit / debug only)
- ``valid_until_seconds`` (float > 0) — lifetime; the BoN scoring layer
  ignores constraints whose lifetime has elapsed since extraction
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _ConstraintBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="VLM-reported certainty; downstream may weight reward by this.",
    )
    reason: str = Field(
        default="",
        description="VLM's natural-language justification — used for audit/debug, "
                    "not consumed by any model head.",
    )
    valid_until_seconds: float = Field(
        default=4.0, gt=0.0,
        description="Lifetime from extraction. BoN ignores constraints past this.",
    )


class LateralOffsetConstraint(_ConstraintBase):
    """Shift the policy's preferred lane center by ``value`` metres (ego frame).

    Positive = left, negative = right. Typical use: "merge left to avoid the
    parked truck" → ``LateralOffsetConstraint(value=+2.0)``.
    """
    type: Literal["lateral_offset"] = "lateral_offset"
    value: float = Field(
        ..., description="Signed lateral offset in metres. + = left, − = right."
    )


class MaxSpeedConstraint(_ConstraintBase):
    """Cap the expected forward speed at ``value`` m/s.

    Typical use: "slow down because of construction" →
    ``MaxSpeedConstraint(value=5.0)``.
    """
    type: Literal["max_speed"] = "max_speed"
    value: float = Field(..., ge=0.0, description="Max forward speed in m/s.")


class NoGoBoxConstraint(_ConstraintBase):
    """Ego-frame axis-aligned box the trajectory must not enter.

    Coordinates are metres in the ego frame: ``x`` is forward, ``y`` is left.
    Typical use: "cone at (12m ahead, 1m right)" →
    ``NoGoBoxConstraint(x_min=11.0, x_max=13.0, y_min=-1.5, y_max=-0.5)``.
    """
    type: Literal["no_go_box"] = "no_go_box"
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def model_post_init(self, _ctx: Any) -> None:
        if self.x_max <= self.x_min:
            raise ValueError(
                f"NoGoBoxConstraint requires x_max ({self.x_max}) > x_min ({self.x_min})"
            )
        if self.y_max <= self.y_min:
            raise ValueError(
                f"NoGoBoxConstraint requires y_max ({self.y_max}) > y_min ({self.y_min})"
            )


# Discriminated union — Pydantic picks the right concrete class from the
# ``type`` field when deserialising VLM JSON output.
TrajectoryConstraint = Annotated[
    Union[LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint],
    Field(discriminator="type"),
]

_ConstraintAdapter: TypeAdapter[
    Union[LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint]
] = TypeAdapter(TrajectoryConstraint)


def parse_constraint(
    raw: dict[str, Any],
) -> Union[LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint]:
    """Validate + dispatch a single VLM JSON constraint."""
    return _ConstraintAdapter.validate_python(raw)


def parse_constraints(
    raw: list[dict[str, Any]],
) -> list[Union[LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint]]:
    """Validate + dispatch a list of constraints (typical VLM output shape)."""
    return [_ConstraintAdapter.validate_python(item) for item in raw]


__all__ = [
    "LateralOffsetConstraint",
    "MaxSpeedConstraint",
    "NoGoBoxConstraint",
    "TrajectoryConstraint",
    "parse_constraint",
    "parse_constraints",
]
