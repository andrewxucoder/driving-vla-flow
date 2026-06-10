"""evaluation.simulator + Protocol compatibility with embodied_flow_core."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest
import torch

from embodied_flow_core import ClosedLoopSimulator

from driving_vla.data import NavSimScene, NavSimSceneFrame
from driving_vla.evaluation import (
    DrivingObservation,
    DrivingRolloutResult,
    NavSimClosedLoopSimulator,
)


class _FakeLoader:
    def __init__(self, scenes: list[NavSimScene]) -> None:
        self._scenes = {s.token: s for s in scenes}

    def tokens(self) -> Iterable[str]:
        return list(self._scenes.keys())

    def get_scene(self, token: str) -> NavSimScene:
        return self._scenes[token]


def _scene(token: str = "t0", *, n_history: int = 2, n_future: int = 5) -> NavSimScene:
    total = n_history + 1 + n_future
    frames = []
    for i in range(total):
        frames.append(NavSimSceneFrame(
            ego_xy=np.array([float(i), 0.0]),
            ego_heading=0.0,
        ))
    return NavSimScene(token=token, log_name="fake", frames=frames, current_index=n_history)


def test_simulator_satisfies_protocol_structurally():
    """The verifiable cross-repo consistency claim from ARCHITECTURE.md § 12.3."""
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    assert isinstance(sim, ClosedLoopSimulator)


def test_reset_returns_observation():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    obs = sim.reset()
    assert isinstance(obs, DrivingObservation)
    assert obs.scenario_id == "t0"
    assert obs.state.shape == (3,)


def test_step_advances_cursor_and_returns_done_on_horizon():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene(n_future=2)]), max_horizon=5)
    sim.reset()
    obs1, m1, done1 = sim.step(torch.tensor([1.0, 0.0]))
    assert not done1
    obs2, m2, done2 = sim.step(torch.tensor([2.0, 0.0]))
    assert done2 or sim._cursor >= len(sim._scene.frames)


def test_step_accepts_chunk_action():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    sim.reset()
    obs, metrics, done = sim.step(torch.zeros(8, 2))
    assert isinstance(obs, DrivingObservation)
    assert "predicted_x" in metrics


def test_step_rejects_bad_action_shape():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    sim.reset()
    with pytest.raises(ValueError):
        sim.step(torch.zeros(3, 4))


def test_step_before_reset_raises():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    with pytest.raises(RuntimeError):
        sim.step(torch.zeros(2))


def test_rollout_collects_observations_and_actions():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene(n_future=4)]), max_horizon=10)

    def policy(_obs: DrivingObservation) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    result = sim.rollout(policy, horizon=4)
    assert isinstance(result, DrivingRolloutResult)
    assert len(result.actions) >= 1
    assert len(result.observations) >= 2


def test_rollout_horizon_must_be_positive():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene()]))
    with pytest.raises(ValueError):
        sim.rollout(lambda _: torch.zeros(2), horizon=0)


def test_unknown_scenario_raises():
    sim = NavSimClosedLoopSimulator(_FakeLoader([_scene("t0")]))
    with pytest.raises(KeyError):
        sim.reset(scenario_id="ghost")


def test_empty_loader_raises():
    sim = NavSimClosedLoopSimulator(_FakeLoader([]))
    with pytest.raises(RuntimeError):
        sim.reset()
