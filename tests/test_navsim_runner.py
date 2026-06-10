"""evaluation.navsim_runner — multi-policy NAVSIM-style eval."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from driving_vla.data import NavSimScene, NavSimSceneFrame
from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.evaluation import (
    PolicyEntry,
    build_default_score_fn,
    evaluate_policies,
    make_bon_predictor,
)
from driving_vla.models import (
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    WaypointRegressionHead,
)
from driving_vla.policies import FlowPolicy, RegressionPolicy


class _FakeLoader:
    def __init__(self, scenes: list[NavSimScene]) -> None:
        self._scenes = {s.token: s for s in scenes}

    def tokens(self) -> Iterable[str]:
        return list(self._scenes.keys())

    def get_scene(self, token: str) -> NavSimScene:
        return self._scenes[token]


def _scene(token: str, *, n_history: int = 4, n_future: int = 8) -> NavSimScene:
    total = n_history + 1 + n_future
    frames = []
    for i in range(total):
        frames.append(NavSimSceneFrame(
            ego_xy=np.array([float(i), 0.0]),
            ego_heading=0.0,
            velocity_xy=np.array([1.0, 0.0]),
        ))
    return NavSimScene(token=token, log_name="fake", frames=frames, current_index=n_history)


def _zero_predict(_sample: DrivingTrajectorySample) -> torch.Tensor:
    return torch.zeros(8, 2)


def test_evaluate_single_policy():
    loader = _FakeLoader([_scene("t0"), _scene("t1")])
    entry = PolicyEntry(name="zero", predict=_zero_predict)
    report = evaluate_policies([entry], loader)
    assert "zero" in report.per_policy
    assert report.per_policy["zero"]["n_scenes"] == 2


def test_evaluate_multiple_policies():
    loader = _FakeLoader([_scene(f"t{i}") for i in range(3)])
    enc = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=32, history_dim=2,
    )
    reg = RegressionPolicy(
        encoder=enc,
        head=WaypointRegressionHead(horizon=8, traj_dim=2, cond_dim=32, hidden_dim=64),
    ).eval()
    flow_policy = FlowPolicy(
        encoder=enc,
        head=ConditionalFlowMatcher(horizon=8, traj_dim=2, cond_dim=32, hidden_dim=64),
        sampling_steps=4,
    ).eval()
    entries = [
        PolicyEntry(name="regression", predict=reg.predict_chunk),
        PolicyEntry(name="flow", predict=flow_policy.predict_chunk),
        PolicyEntry(
            name="flow_bon",
            predict=make_bon_predictor(flow_policy, n=3, score_fn=build_default_score_fn()),
        ),
    ]
    report = evaluate_policies(entries, loader)
    assert set(report.per_policy.keys()) == {"regression", "flow", "flow_bon"}
    for stats in report.per_policy.values():
        assert "ade_mean" in stats and "fde_mean" in stats


def test_default_score_fn_picks_best():
    score = build_default_score_fn()
    # Two samples: one centered, one off-lane
    samples = torch.stack([
        torch.stack([torch.linspace(1, 8, 8), torch.zeros(8)], dim=-1),
        torch.stack([torch.linspace(1, 8, 8), torch.full((8,), 5.0)], dim=-1),
    ])
    scores = score(samples, {})
    assert scores[0] > scores[1]


def test_default_score_fn_uses_ego_state_when_present():
    """progress_match term must engage and demote overshoot candidates."""
    score = build_default_score_fn(horizon_seconds=4.0)
    fast = torch.stack([torch.linspace(1, 8, 8), torch.zeros(8)], dim=-1)  # 7m fwd
    slow = torch.stack([torch.linspace(0.25, 2.0, 8), torch.zeros(8)], dim=-1)  # 1.75m fwd
    samples = torch.stack([fast, slow])  # [2, 8, 2]
    ego = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]])  # vx=0.5 → expected 2m
    scores = score(samples, {"ego_state": ego})
    assert scores[1] > scores[0], (
        f"speed-matched candidate should win; got {scores.tolist()}"
    )


def test_make_bon_predictor_passes_ego_state():
    """BoN wrapper must forward ego_state so progress_match can fire."""
    captured: dict = {}

    def _spy_score(samples: torch.Tensor, cond: dict) -> torch.Tensor:
        captured["cond"] = cond
        return samples[:, 0, 0]  # arbitrary deterministic score

    class _StubPolicy:
        def predict_chunk_n(self, sample, n):
            return torch.zeros(n, 8, 2)

    sample = DrivingTrajectorySample(
        history=torch.zeros(4, 2),
        future=torch.zeros(8, 2),
        command_id=2,
        ego_state=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    predictor = make_bon_predictor(_StubPolicy(), n=3, score_fn=_spy_score)  # type: ignore[arg-type]
    _ = predictor(sample)
    assert "ego_state" in captured["cond"]
    assert captured["cond"]["ego_state"].shape == (1, 5)
    assert int(captured["cond"]["command_id"].item()) == 2
