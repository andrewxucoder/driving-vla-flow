"""M1 smoke — verify v2 can import the cross-repo shared core.

Per R2 decision: only Episode / ClosedLoopSimulator / RolloutResult are imported
(Observation is deliberately NOT imported, as em-wm's v0 Observation schema is
not yet aligned across the three repos).
"""

from __future__ import annotations


def test_import_episode():
    from embodied_flow_core import Episode  # noqa: F401


def test_import_closed_loop_simulator():
    from embodied_flow_core import ClosedLoopSimulator  # noqa: F401


def test_import_rollout_result():
    from embodied_flow_core import RolloutResult  # noqa: F401


def test_three_aliases_resolvable_together():
    from embodied_flow_core import (
        ClosedLoopSimulator,
        Episode,
        RolloutResult,
    )

    assert Episode is not None
    assert ClosedLoopSimulator is not None
    assert RolloutResult is not None


def test_v2_does_not_import_observation():
    """R2: v2 must NOT pull in embodied_flow_core.Observation.

    This test enforces that no v2 module accidentally adopts em-wm's
    Observation schema (which has a `state` field; driving uses
    `history` + `future` instead via DrivingTrajectorySample).
    """
    import driving_vla

    seen = set()

    def _walk(mod):
        for attr in dir(mod):
            if attr == "Observation":
                seen.add(getattr(mod, attr))

    _walk(driving_vla)
    assert not seen, "v2 leaked an Observation symbol; should keep DrivingTrajectorySample only"
