"""M8.1 — BaseConstraintExtractor Protocol + Mock/Rule baselines."""

from __future__ import annotations

import pytest
import torch

from driving_vla.data import DrivingTrajectorySample
from driving_vla.vla import (
    BaseConstraintExtractor,
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    MockConstraintExtractor,
    RuleBasedConstraintExtractor,
)


def _obs(
    *,
    ego_vx: float = 5.0,
    command_id: int | None = 0,
) -> DrivingTrajectorySample:
    return DrivingTrajectorySample(
        history=torch.zeros(4, 2),
        future=torch.zeros(8, 2),
        command_id=command_id,
        ego_state=torch.tensor([ego_vx, 0.0, 0.0, 0.0, 0.0]),
    )


# ── Protocol membership ─────────────────────────────────────────────────────


def test_mock_satisfies_protocol():
    assert isinstance(MockConstraintExtractor([]), BaseConstraintExtractor)


def test_rule_based_satisfies_protocol():
    assert isinstance(RuleBasedConstraintExtractor(), BaseConstraintExtractor)


# ── MockConstraintExtractor ─────────────────────────────────────────────────


def test_mock_returns_preset_list_verbatim():
    preset = [
        MaxSpeedConstraint(value=3.0, reason="static"),
        LateralOffsetConstraint(value=-1.0, reason="static"),
    ]
    extractor = MockConstraintExtractor(preset)
    out = extractor(_obs())
    assert out == preset
    # And on a second call with a different observation:
    assert extractor(_obs(ego_vx=20.0)) == preset


def test_mock_returns_copy_not_reference():
    """If the caller mutates the returned list, subsequent calls must be unaffected."""
    preset = [MaxSpeedConstraint(value=3.0)]
    extractor = MockConstraintExtractor(preset)
    out_a = extractor(_obs())
    out_a.append(LateralOffsetConstraint(value=1.0))
    out_b = extractor(_obs())
    assert len(out_b) == 1


def test_mock_accepts_callable():
    """``constraints=callable`` lets fixtures vary output per observation."""

    def per_obs(obs):
        # Return different constraint depending on ego speed
        vx = float(obs.ego_state[0].item())
        if vx < 2.0:
            return [MaxSpeedConstraint(value=2.0, reason="ego is slow")]
        return []

    extractor = MockConstraintExtractor(per_obs)
    slow_out = extractor(_obs(ego_vx=0.5))
    fast_out = extractor(_obs(ego_vx=10.0))
    assert len(slow_out) == 1 and isinstance(slow_out[0], MaxSpeedConstraint)
    assert fast_out == []


# ── RuleBasedConstraintExtractor ────────────────────────────────────────────


def test_rule_emits_max_speed_for_slow_ego():
    extractor = RuleBasedConstraintExtractor(
        low_speed_threshold=1.0, low_speed_cap=2.0
    )
    out = extractor(_obs(ego_vx=0.3))
    assert len(out) == 1
    assert isinstance(out[0], MaxSpeedConstraint)
    assert out[0].value == 2.0
    assert "ego_vx" in out[0].reason


def test_rule_skips_max_speed_for_fast_ego():
    extractor = RuleBasedConstraintExtractor(low_speed_threshold=1.0)
    out = extractor(_obs(ego_vx=10.0))
    # No max_speed emitted; only command-driven constraints (here command_id=0 → none)
    assert all(not isinstance(c, MaxSpeedConstraint) for c in out)


def test_rule_emits_lateral_offset_for_change_left():
    extractor = RuleBasedConstraintExtractor(lane_change_offset=3.5)
    out = extractor(_obs(ego_vx=10.0, command_id=1))  # change_left
    lats = [c for c in out if isinstance(c, LateralOffsetConstraint)]
    assert len(lats) == 1
    assert lats[0].value > 0


def test_rule_emits_negative_lateral_for_change_right():
    extractor = RuleBasedConstraintExtractor(lane_change_offset=3.5)
    out = extractor(_obs(ego_vx=10.0, command_id=2))
    lats = [c for c in out if isinstance(c, LateralOffsetConstraint)]
    assert len(lats) == 1
    assert lats[0].value < 0


def test_rule_emits_nothing_for_keep_lane_at_normal_speed():
    extractor = RuleBasedConstraintExtractor()
    out = extractor(_obs(ego_vx=10.0, command_id=0))  # keep_lane
    assert out == []


def test_rule_can_emit_two_constraints_simultaneously():
    """Slow ego + change-left command → both max_speed AND lateral_offset."""
    extractor = RuleBasedConstraintExtractor(low_speed_threshold=1.0)
    out = extractor(_obs(ego_vx=0.5, command_id=1))
    types = {type(c).__name__ for c in out}
    assert "MaxSpeedConstraint" in types
    assert "LateralOffsetConstraint" in types


def test_rule_constructor_rejects_negatives():
    with pytest.raises(ValueError):
        RuleBasedConstraintExtractor(low_speed_threshold=-1.0)
    with pytest.raises(ValueError):
        RuleBasedConstraintExtractor(low_speed_cap=-1.0)
    with pytest.raises(ValueError):
        RuleBasedConstraintExtractor(lane_change_offset=0.0)


def test_rule_no_ego_state_returns_only_command_constraints():
    obs = DrivingTrajectorySample(
        history=torch.zeros(4, 2),
        future=torch.zeros(8, 2),
        command_id=1,  # change_left
        ego_state=None,
    )
    extractor = RuleBasedConstraintExtractor()
    out = extractor(obs)
    # No max_speed (no ego_state), but lateral_offset from command_id fires
    assert all(not isinstance(c, MaxSpeedConstraint) for c in out)
    assert any(isinstance(c, LateralOffsetConstraint) for c in out)
