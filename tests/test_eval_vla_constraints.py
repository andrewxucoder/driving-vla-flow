"""Toy smoke test for scripts/eval_vla_constraints.py (M8.4).

Builds a tiny Flow checkpoint + a handful of nuPlan-val-shaped samples, runs the
3-way eval with a stubbed VLM extractor (no model load), and checks all arms
produce the expected metrics. The real run (retrained ckpt + real Qwen3-VL) is
out of scope here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from driving_vla.models import ConditionalFlowMatcher, DrivingObservationEncoder
from driving_vla.policies import FlowPolicy
from driving_vla.vla import Qwen3VLConstraintExtractor

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("eval_vla_constraints", _SCRIPTS / "eval_vla_constraints.py")
assert spec and spec.loader
eval_vla = importlib.util.module_from_spec(spec)
sys.modules["eval_vla_constraints"] = eval_vla
spec.loader.exec_module(eval_vla)

_DIMS = dict(horizon=4, traj_dim=2, latent_dim=8, hidden_dim=16, state_dim=5, num_commands=4)


def _toy_ckpt(path: Path) -> None:
    enc = DrivingObservationEncoder(
        state_dim=_DIMS["state_dim"], num_commands=_DIMS["num_commands"],
        latent_dim=_DIMS["latent_dim"], history_dim=_DIMS["traj_dim"],
    )
    head = ConditionalFlowMatcher(
        horizon=_DIMS["horizon"], traj_dim=_DIMS["traj_dim"],
        cond_dim=_DIMS["latent_dim"], hidden_dim=_DIMS["hidden_dim"],
    )
    policy = FlowPolicy(encoder=enc, head=head)
    torch.save(
        {"model_state": policy.state_dict(), "metadata": {"head": "flow", "args": dict(_DIMS)}},
        path,
    )


def _toy_data(path: Path, n: int = 6) -> None:
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(n):
        samples.append({
            "history": rng.standard_normal((2, 2)).astype("float32"),
            "future": np.cumsum(rng.standard_normal((4, 2)).astype("float32"), axis=0),
            "ego_state": np.array([5.0, 0, 0, 0.5, 0.0], dtype="float32"),
            "command_id": 0,
        })
    torch.save({"samples": samples, "command_vocab": ["keep_lane", "change_left", "change_right", "slow_down"]}, path)


def test_three_way_eval_smoke(tmp_path):
    ckpt = tmp_path / "flow.pt"
    data = tmp_path / "val.pt"
    out = tmp_path / "res.json"
    _toy_ckpt(ckpt)
    _toy_data(data)

    stub_vlm = Qwen3VLConstraintExtractor(
        generate_fn=lambda prompt, images: '[{"type":"max_speed","value":3.0}]'
    )
    rc = eval_vla.main(
        [
            "--data-path", str(data), "--flow-ckpt", str(ckpt), "--output", str(out),
            "--bon-n", "4", "--max-scenes", "6", "--device", "cpu",
            "--horizon", "4", "--traj-dim", "2", "--latent-dim", "8",
            "--hidden-dim", "16", "--state-dim", "5",
        ],
        vlm_extractor=stub_vlm,
    )
    assert rc == 0

    import json
    res = json.loads(out.read_text())["results"]
    assert set(res) == {"baseline", "mock_oracle", "vlm"}
    for arm in res.values():
        assert arm["n"] == 6
        for key in ("ade_mean", "obstacle_collision", "constraint_satisfaction"):
            assert key in arm
        assert 0.0 <= arm["constraint_satisfaction"] <= 1.0


def test_dry_run():
    assert eval_vla.main(["--dry-run"]) == 0
