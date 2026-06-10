"""M8.2 — end-to-end: VLA constraints change BoN selection.

This is the M8.1/M8.2 acceptance test: a ``MockConstraintExtractor`` feeding a
constraint-aware score_fn must change which BoN candidate gets picked, relative
to the unconstrained baseline.

Uses a deterministic stub policy so the candidate set is fixed and the test
asserts on *which* candidate the BoN selects — not on stochastic sampling.
"""

from __future__ import annotations

import torch

from driving_vla.data import DrivingTrajectorySample
from driving_vla.evaluation import build_default_score_fn, make_bon_predictor
from driving_vla.vla import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    MockConstraintExtractor,
    NoGoBoxConstraint,
    build_constraint_aware_score_fn,
)


class _FixedCandidatePolicy:
    """Returns a fixed set of N candidate trajectories, ignoring the sample.

    Candidates differ so BoN selection is observable.
    """

    def __init__(self, candidates: torch.Tensor) -> None:
        # candidates: [N, H, 2]
        self._candidates = candidates

    def predict_chunk_n(self, sample: DrivingTrajectorySample, n: int) -> torch.Tensor:
        assert n == self._candidates.shape[0]
        return self._candidates.clone()


def _line(y: float, x_end: float = 8.0, H: int = 8) -> torch.Tensor:
    xs = torch.linspace(1.0, x_end, H)
    ys = torch.full((H,), y)
    return torch.stack([xs, ys], dim=-1)


def _obs(command_id: int = 0, ego_vx: float = 5.0) -> DrivingTrajectorySample:
    return DrivingTrajectorySample(
        history=torch.zeros(4, 2),
        future=torch.zeros(8, 2),
        command_id=command_id,
        ego_state=torch.tensor([ego_vx, 0.0, 0.0, 0.0, 0.0]),
    )


def test_lateral_offset_constraint_changes_bon_pick():
    # Two candidates: one centered (y=0), one shifted left (y=3).
    candidates = torch.stack([_line(y=0.0), _line(y=3.0)])  # [2, H, 2]
    policy = _FixedCandidatePolicy(candidates)
    obs = _obs(command_id=0)  # keep_lane → base center y=0

    # Baseline BoN (no constraints): centered candidate wins.
    baseline_predictor = make_bon_predictor(
        policy, n=2, score_fn=build_default_score_fn(),
    )
    baseline_pick = baseline_predictor(obs)
    assert baseline_pick[:, 1].mean().item() < 0.5  # ≈ y=0

    # Constraint BoN: +3 lateral offset → left candidate wins.
    extractor = MockConstraintExtractor(
        [LateralOffsetConstraint(value=3.0, reason="merge left for obstacle")]
    )
    constraint_predictor = make_bon_predictor(
        policy,
        n=2,
        score_fn=build_constraint_aware_score_fn(),
        constraint_extractor=extractor,
    )
    constraint_pick = constraint_predictor(obs)
    assert constraint_pick[:, 1].mean().item() > 2.5  # ≈ y=3


def test_max_speed_constraint_changes_bon_pick():
    # Fast vs slow candidate.
    candidates = torch.stack([_line(y=0.0, x_end=20.0), _line(y=0.0, x_end=4.0)])
    policy = _FixedCandidatePolicy(candidates)
    obs = _obs(command_id=0, ego_vx=5.0)  # expects ~20m over 4s

    # Baseline: fast candidate (matches expected progress) wins.
    baseline = make_bon_predictor(policy, n=2, score_fn=build_default_score_fn())
    fast_pick = baseline(obs)
    assert fast_pick[-1, 0].item() > 10.0  # reached far in +x

    # max_speed=1 → slow candidate wins.
    extractor = MockConstraintExtractor([MaxSpeedConstraint(value=1.0)])
    constrained = make_bon_predictor(
        policy, n=2,
        score_fn=build_constraint_aware_score_fn(),
        constraint_extractor=extractor,
    )
    slow_pick = constrained(obs)
    assert slow_pick[-1, 0].item() < 6.0  # stayed near origin


def test_no_go_box_constraint_changes_bon_pick():
    # Candidate A passes through a region; candidate B avoids it.
    through_box = _line(y=2.0)   # y=2 throughout
    avoids_box = _line(y=0.0)
    candidates = torch.stack([through_box, avoids_box])
    policy = _FixedCandidatePolicy(candidates)
    obs = _obs(command_id=0)

    extractor = MockConstraintExtractor(
        [NoGoBoxConstraint(x_min=1.0, x_max=8.0, y_min=1.5, y_max=2.5)]
    )
    predictor = make_bon_predictor(
        policy, n=2,
        score_fn=build_constraint_aware_score_fn(),
        constraint_extractor=extractor,
    )
    pick = predictor(obs)
    # Should pick the box-avoiding candidate (y=0), not the intruding one (y=2).
    assert pick[:, 1].mean().item() < 0.5


def test_no_constraints_matches_baseline_selection():
    """Empty extractor output → constraint-aware path picks same as baseline."""
    candidates = torch.stack([_line(y=0.0), _line(y=3.0)])
    policy = _FixedCandidatePolicy(candidates)
    obs = _obs(command_id=0)

    baseline = make_bon_predictor(policy, n=2, score_fn=build_default_score_fn())
    empty_extractor = MockConstraintExtractor([])
    constrained = make_bon_predictor(
        policy, n=2,
        score_fn=build_constraint_aware_score_fn(),
        constraint_extractor=empty_extractor,
    )
    assert torch.allclose(baseline(obs), constrained(obs))
