"""Synthetic obstacle injection (M8.4)."""

from __future__ import annotations

import pytest
import torch

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.vla import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    Qwen3VLConstraintExtractor,
)
from driving_vla.vla.synthetic_obstacles import (
    ObstacleInjection,
    cone,
    default_suite,
    inject,
    oracle_boxes,
    parked_truck,
    pedestrian,
)


def _sample() -> DrivingTrajectorySample:
    return DrivingTrajectorySample(history=torch.zeros(2, 2), future=torch.zeros(3, 2))


def test_cone_oracle_is_no_go_box_at_position():
    inj = cone(x_ahead=12.0, y_lateral=-1.0)
    assert inj.kind == "cone"
    assert len(inj.oracle_constraints) == 1
    c = inj.oracle_constraints[0]
    assert isinstance(c, NoGoBoxConstraint)
    assert c.x_min < 12.0 < c.x_max and c.y_min < -1.0 < c.y_max
    assert "right" in inj.caption  # y<0 -> right


def test_parked_truck_merges_away_from_blocked_side():
    right = parked_truck("right").oracle_constraints[0]
    left = parked_truck("left").oracle_constraints[0]
    assert isinstance(right, LateralOffsetConstraint)
    assert right.value > 0  # truck on right -> merge left (+)
    assert left.value < 0


def test_parked_truck_rejects_bad_side():
    with pytest.raises(ValueError):
        parked_truck("middle")


def test_pedestrian_caps_speed_and_marks_footprint():
    inj = pedestrian(x_ahead=10.0, cap_speed=2.0)
    types = {type(c) for c in inj.oracle_constraints}
    assert MaxSpeedConstraint in types and NoGoBoxConstraint in types
    speed = next(c for c in inj.oracle_constraints if isinstance(c, MaxSpeedConstraint))
    assert speed.value == 2.0
    assert len(inj.boxes) == 1


def test_inject_writes_caption_into_metadata_without_mutating_original():
    obs = _sample()
    inj = cone()
    out = inject(obs, inj)
    assert out.metadata["injected_caption"] == inj.caption
    assert out.metadata["injected_obstacle"] == "cone"
    # original untouched
    assert "injected_caption" not in obs.metadata


def test_injected_caption_feeds_vlm_extractor_prompt():
    seen = {}

    def capture(prompt, images):
        seen["prompt"] = prompt
        return "[]"

    obs = inject(_sample(), parked_truck("right"))
    Qwen3VLConstraintExtractor(generate_fn=capture)(obs)
    assert "truck is parked" in seen["prompt"]


def test_default_suite_and_oracle_boxes():
    suite = default_suite()
    assert {i.kind for i in suite} == {"cone", "parked_truck", "pedestrian"}
    # cone + pedestrian each contribute one box; truck contributes none
    assert len(oracle_boxes(suite)) == 2


def test_injection_is_frozen_dataclass():
    inj = ObstacleInjection(kind="x", caption="c", oracle_constraints=[])
    with pytest.raises(Exception):
        inj.kind = "y"  # frozen
