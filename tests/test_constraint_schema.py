"""M8.1 — TrajectoryConstraint pydantic discriminated union."""

from __future__ import annotations

import json

import pytest

from driving_vla.vla import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    parse_constraint,
    parse_constraints,
)


# ── Construction & field defaults ──────────────────────────────────────────


def test_lateral_offset_construction():
    c = LateralOffsetConstraint(value=2.0, reason="merge left")
    assert c.type == "lateral_offset"
    assert c.value == 2.0
    assert c.confidence == 1.0
    assert c.valid_until_seconds == 4.0


def test_max_speed_construction():
    c = MaxSpeedConstraint(value=5.0, reason="construction ahead", confidence=0.8)
    assert c.type == "max_speed"
    assert c.value == 5.0
    assert c.confidence == 0.8


def test_no_go_box_construction():
    c = NoGoBoxConstraint(x_min=10.0, x_max=15.0, y_min=-2.0, y_max=2.0)
    assert c.type == "no_go_box"
    assert c.x_min == 10.0 and c.x_max == 15.0


# ── Field-level validation ──────────────────────────────────────────────────


def test_confidence_out_of_range_rejected():
    with pytest.raises(Exception):
        LateralOffsetConstraint(value=1.0, confidence=1.5)
    with pytest.raises(Exception):
        MaxSpeedConstraint(value=5.0, confidence=-0.1)


def test_max_speed_negative_rejected():
    with pytest.raises(Exception):
        MaxSpeedConstraint(value=-3.0)


def test_valid_until_must_be_positive():
    with pytest.raises(Exception):
        LateralOffsetConstraint(value=1.0, valid_until_seconds=0.0)


def test_no_go_box_requires_x_max_gt_x_min():
    with pytest.raises(ValueError, match="x_max"):
        NoGoBoxConstraint(x_min=10.0, x_max=10.0, y_min=-1.0, y_max=1.0)


def test_no_go_box_requires_y_max_gt_y_min():
    with pytest.raises(ValueError, match="y_max"):
        NoGoBoxConstraint(x_min=10.0, x_max=15.0, y_min=1.0, y_max=-1.0)


def test_extra_field_forbidden():
    with pytest.raises(Exception):
        LateralOffsetConstraint(value=1.0, mystery="should fail")


# ── Discriminated-union dispatch ────────────────────────────────────────────


def test_parse_constraint_dispatches_lateral():
    c = parse_constraint({"type": "lateral_offset", "value": 1.5})
    assert isinstance(c, LateralOffsetConstraint)
    assert c.value == 1.5


def test_parse_constraint_dispatches_max_speed():
    c = parse_constraint({"type": "max_speed", "value": 3.0})
    assert isinstance(c, MaxSpeedConstraint)


def test_parse_constraint_dispatches_no_go_box():
    c = parse_constraint(
        {"type": "no_go_box", "x_min": 8.0, "x_max": 12.0, "y_min": -1.0, "y_max": 1.0}
    )
    assert isinstance(c, NoGoBoxConstraint)


def test_parse_constraint_unknown_type_rejected():
    with pytest.raises(Exception):
        parse_constraint({"type": "teleport_ego", "value": 1.0})


def test_parse_constraints_list():
    cs = parse_constraints([
        {"type": "max_speed", "value": 5.0, "reason": "construction"},
        {"type": "lateral_offset", "value": -1.5, "reason": "obstacle on left"},
    ])
    assert len(cs) == 2
    assert isinstance(cs[0], MaxSpeedConstraint)
    assert isinstance(cs[1], LateralOffsetConstraint)


def test_parse_constraint_extra_field_rejected():
    """Round-trips of VLM JSON must catch hallucinated fields rather than ignoring."""
    with pytest.raises(Exception):
        parse_constraint({"type": "max_speed", "value": 5.0, "fictional": 42})


# ── JSON round trip (what the VLM backend will do) ──────────────────────────


def test_json_round_trip_preserves_fields():
    original = NoGoBoxConstraint(
        x_min=8.0, x_max=12.0, y_min=-1.0, y_max=1.0,
        confidence=0.7, reason="cone cluster", valid_until_seconds=3.0,
    )
    restored = parse_constraint(json.loads(original.model_dump_json()))
    assert restored == original
