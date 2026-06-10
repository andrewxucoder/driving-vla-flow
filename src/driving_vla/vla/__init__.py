"""VLM-as-constraint algorithmic layer (M8).

Implements the algorithmic core of the project's "VLM reasoning → structured
trajectory constraints → reward injection → BoN-under-constraints" pipeline.

Three layers, increasing in dependency weight:

- :mod:`constraint_schema` — pydantic discriminated-union schema for the three
  constraint types (``lateral_offset`` / ``max_speed`` / ``no_go_box``).
- :mod:`constraint_extractor` — ``BaseConstraintExtractor`` Protocol +
  ``MockConstraintExtractor`` (preset list) + ``RuleBasedConstraintExtractor``
  (heuristic baseline). No VLM dependency.
- :mod:`constraint_to_reward` (M8.2) — translates a constraint list into reward
  modifications consumed by the existing ``trajectory_reward`` + BoN stack.

Design rationale (see ``docs/ROADMAP.md`` § M8): the project is an
**algorithmic research framework**, not a production driving stack. Trigger
mechanisms, latency budgets, and on-vehicle deployment are explicitly out of
scope and live in M9. ``BaseConstraintExtractor`` is designed to be reusable
across the sibling research projects ``robot-vla-flow`` and
``embodied-world-model``.
"""

from driving_vla.vla.constraint_extractor import (
    BaseConstraintExtractor,
    MockConstraintExtractor,
    RuleBasedConstraintExtractor,
)
from driving_vla.vla.constraint_schema import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    TrajectoryConstraint,
    parse_constraint,
    parse_constraints,
)
from driving_vla.vla.constraint_to_reward import (
    build_constraint_aware_score_fn,
    trajectory_reward_with_constraints,
)
from driving_vla.vla.qwen3_vl_extractor import (
    DEFAULT_MODEL_PATH,
    Qwen3VLConstraintExtractor,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "BaseConstraintExtractor",
    "LateralOffsetConstraint",
    "MaxSpeedConstraint",
    "MockConstraintExtractor",
    "NoGoBoxConstraint",
    "Qwen3VLConstraintExtractor",
    "RuleBasedConstraintExtractor",
    "TrajectoryConstraint",
    "build_constraint_aware_score_fn",
    "parse_constraint",
    "parse_constraints",
    "trajectory_reward_with_constraints",
]
