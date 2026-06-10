"""M8.4 — VLM-as-constraint 3-way evaluation.

On normal nuPlan-val observations we inject synthetic obstacles (cone / parked
truck / pedestrian), then compare three BoN predictors built from the *same*
Flow policy:

- **baseline**    unconstrained reward (no obstacle knowledge).
- **mock-oracle** hand-authored "correct" constraints — the ceiling of the
                  constraint mechanism.
- **vlm**         constraints from Qwen3-VL reading the injected caption — how
                  close real VLM reasoning gets to that ceiling.

All three reuse the M8.2 wiring: ``make_bon_predictor(policy, score_fn=...,
constraint_extractor=...)``. Metrics per arm:

- ``ade_mean``               chosen trajectory vs logged GT future (realism).
- ``obstacle_collision``     fraction of chosen trajectories entering an
                             injected obstacle box (lower = safer).
- ``constraint_satisfaction`` fraction satisfying the oracle no-go boxes.

The VLM arm loads the 2B checkpoint; pass ``vlm_extractor=`` to :func:`main`
(tests do) to substitute a stub and keep the run offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import assert_ckpt_dims, device_from_arg

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.evaluation.metrics import ade
from driving_vla.evaluation.navsim_runner import build_default_score_fn, make_bon_predictor
from driving_vla.evaluation.reward import box_collision_proxy
from driving_vla.models import ConditionalFlowMatcher, DrivingObservationEncoder
from driving_vla.policies import FlowPolicy
from driving_vla.vla import Qwen3VLConstraintExtractor
from driving_vla.vla.constraint_to_reward import build_constraint_aware_score_fn
from driving_vla.vla.synthetic_obstacles import default_suite, inject


def _load_flow_policy(ckpt_path: str, args: argparse.Namespace, device: torch.device) -> FlowPolicy:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert_ckpt_dims(ckpt, args, expected_head="flow")
    state = ckpt.get("model_state", ckpt)
    head = ConditionalFlowMatcher(
        horizon=args.horizon, traj_dim=args.traj_dim,
        cond_dim=args.latent_dim, hidden_dim=args.hidden_dim,
    ).to(device)
    head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})
    enc = DrivingObservationEncoder(
        state_dim=args.state_dim, num_commands=args.num_commands,
        latent_dim=args.latent_dim, history_dim=args.traj_dim,
    ).to(device)
    enc_state = {k[8:]: v for k, v in state.items() if k.startswith("encoder.")}
    if enc_state:
        enc.load_state_dict(enc_state)
    return FlowPolicy(encoder=enc, head=head).to(device).eval()


def _obstacle_collision(traj: torch.Tensor, boxes: list[tuple]) -> float:
    if not boxes:
        return 0.0
    box_t = torch.tensor(boxes, dtype=traj.dtype, device=traj.device)
    return float(box_collision_proxy(traj.unsqueeze(0), box_t).item())


def evaluate_arm(
    predictor: Callable[[DrivingTrajectorySample], torch.Tensor],
    scenes: list[tuple[DrivingTrajectorySample, torch.Tensor, list[tuple]]],
) -> dict[str, float]:
    ade_sum = coll_sum = sat_sum = 0.0
    n = 0
    for obs, gt, boxes in scenes:
        pred = predictor(obs)
        h = min(pred.shape[0], gt.shape[0])
        ade_sum += float(ade(pred[:h].unsqueeze(0), gt[:h].unsqueeze(0)).item())
        coll = _obstacle_collision(pred, boxes)
        coll_sum += coll
        sat_sum += 1.0 - coll  # satisfaction = stayed out of the no-go footprint
        n += 1
    if n == 0:
        return {"n": 0.0}
    return {
        "n": float(n),
        "ade_mean": ade_sum / n,
        "obstacle_collision": coll_sum / n,
        "constraint_satisfaction": sat_sum / n,
    }


def build_arms(
    policy: FlowPolicy,
    *,
    bon_n: int,
    horizon_seconds: float,
    oracle_for: Callable[[DrivingTrajectorySample], list],
    vlm_extractor: Any | None,
) -> dict[str, Callable[[DrivingTrajectorySample], torch.Tensor]]:
    arms: dict[str, Callable] = {
        "baseline": make_bon_predictor(
            policy, n=bon_n, score_fn=build_default_score_fn(horizon_seconds)
        ),
        "mock_oracle": make_bon_predictor(
            policy, n=bon_n,
            score_fn=build_constraint_aware_score_fn(horizon_seconds),
            constraint_extractor=oracle_for,
        ),
    }
    if vlm_extractor is not None:
        arms["vlm"] = make_bon_predictor(
            policy, n=bon_n,
            score_fn=build_constraint_aware_score_fn(horizon_seconds),
            constraint_extractor=vlm_extractor,
        )
    return arms


def _build_scenes(
    samples: list[dict], max_scenes: int
) -> tuple[list[tuple], dict[str, list]]:
    """Inject obstacles (cycling the default suite) onto nuPlan-val samples.

    Returns (scenes, oracle_by_caption) where scenes is a list of
    (injected_observation, gt_future, obstacle_boxes) and oracle_by_caption maps
    each injected caption to its oracle constraint list.
    """
    suite = default_suite()
    scenes: list[tuple] = []
    oracle_by_caption: dict[str, list] = {}
    for i, s in enumerate(samples[:max_scenes]):
        injection = suite[i % len(suite)]
        obs = DrivingTrajectorySample(
            history=torch.tensor(s["history"]).float(),
            future=torch.tensor(s["future"]).float(),
            command_id=int(s["command_id"]),
            ego_state=torch.tensor(s["ego_state"]).float(),
        )
        obs = inject(obs, injection)
        gt = torch.tensor(s["future"]).float()
        scenes.append((obs, gt, injection.boxes))
        oracle_by_caption[injection.caption] = injection.oracle_constraints
    return scenes, oracle_by_caption


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M8.4 VLM-as-constraint 3-way eval.")
    p.add_argument("--data-path", type=str, default="data/nuplan/val.pt")
    p.add_argument("--flow-ckpt", type=str, required=False)
    p.add_argument("--output", type=str, default="outputs/m8_vla_constraints.json")
    p.add_argument("--bon-n", type=int, default=16)
    p.add_argument("--max-scenes", type=int, default=200)
    p.add_argument("--horizon-seconds", type=float, default=4.0)
    p.add_argument("--with-vlm", action="store_true", help="Run the Qwen3-VL arm.")
    p.add_argument("--vlm-model-path", type=str, default=None)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None, *, vlm_extractor: Any | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] eval_vla_constraints CLI parsed OK.")
        return 0
    if not args.flow_ckpt:
        build_argparser().error("--flow-ckpt is required (unless --dry-run)")

    device = device_from_arg(args.device)
    policy = _load_flow_policy(args.flow_ckpt, args, device)
    blob = torch.load(args.data_path, map_location="cpu", weights_only=False)
    scenes, oracle_by_caption = _build_scenes(blob["samples"], args.max_scenes)

    def oracle_for(obs: DrivingTrajectorySample) -> list:
        return oracle_by_caption.get(obs.metadata.get("injected_caption", ""), [])

    if vlm_extractor is None and args.with_vlm:
        kw: dict[str, Any] = {"device": str(device)}
        if args.vlm_model_path:
            kw["model_path"] = args.vlm_model_path
        vlm_extractor = Qwen3VLConstraintExtractor(**kw)

    arms = build_arms(
        policy, bon_n=args.bon_n, horizon_seconds=args.horizon_seconds,
        oracle_for=oracle_for, vlm_extractor=vlm_extractor,
    )
    results = {name: evaluate_arm(pred, scenes) for name, pred in arms.items()}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"[eval_vla_constraints] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
