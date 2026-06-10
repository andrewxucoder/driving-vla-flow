"""Run the NAVSIM 5-policy evaluation and write a JSON report.

Loads checkpoints for Regression / Flow / Diffusion, wraps Flow & Diffusion in
BoN variants, runs every scenario through each policy, and emits:

    outputs/round0_navsim.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import assert_ckpt_dims, device_from_arg

from driving_vla.data.navsim_adapter import NavSimSceneLoader, NavSimTrajectoryDataset
from driving_vla.evaluation import (
    PolicyEntry,
    build_default_score_fn,
    evaluate_policies,
    make_bon_predictor,
)
from driving_vla.models import (
    ConditionalDiffusionModel,
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    WaypointRegressionHead,
)
from driving_vla.policies import DiffusionPolicy, FlowPolicy, RegressionPolicy


def _load_ckpt(checkpoint: str, device: torch.device) -> dict:
    return torch.load(checkpoint, map_location=device, weights_only=False)


def _split_state(ckpt: dict) -> tuple[dict, dict]:
    if "model_state" not in ckpt:
        return ckpt, {}
    state = ckpt["model_state"]
    head_state = {
        k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")
    }
    enc_state = {
        k.replace("encoder.", ""): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    return head_state, enc_state


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NAVSIM 5-policy evaluation.")
    p.add_argument("--data-root", type=str, default="data/navsim")
    p.add_argument("--split", type=str, default="navtest")
    p.add_argument("--scenarios", type=int, default=200)
    p.add_argument("--output", type=str, default="outputs/round0_navsim.json")

    p.add_argument("--regression-ckpt", type=str, default=None)
    p.add_argument("--flow-ckpt", type=str, default=None)
    p.add_argument("--diffusion-ckpt", type=str, default=None)
    p.add_argument("--bon-n", type=int, default=8)

    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def _build_encoder(args: argparse.Namespace, device: torch.device) -> DrivingObservationEncoder:
    enc = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
    ).to(device)
    return enc


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] eval_navsim_heads CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = NavSimTrajectoryDataset(
        data_root=args.data_root,
        split=args.split,
        history_seconds=2.0,
        future_seconds=4.0,
        sample_rate_hz=2.0,
    )
    inner_loader: NavSimSceneLoader = dataset._loader  # type: ignore[attr-defined]

    class _LimitedLoader:
        """Wraps a scene loader to expose only the first N tokens."""

        def __init__(self, inner: NavSimSceneLoader, limit: int) -> None:
            self._inner = inner
            self._tokens = list(inner.tokens())[:limit]

        def tokens(self):
            return list(self._tokens)

        def get_scene(self, token):
            return self._inner.get_scene(token)

    scene_loader: NavSimSceneLoader = _LimitedLoader(inner_loader, args.scenarios)

    policies: list[PolicyEntry] = []

    if args.regression_ckpt is not None:
        ckpt = _load_ckpt(args.regression_ckpt, device)
        assert_ckpt_dims(ckpt, args, expected_head="regression")
        head_state, enc_state = _split_state(ckpt)
        head = WaypointRegressionHead(
            horizon=args.horizon,
            traj_dim=args.traj_dim,
            cond_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        head.load_state_dict(head_state)
        enc = _build_encoder(args, device)
        if enc_state:
            enc.load_state_dict(enc_state)
        reg_policy = RegressionPolicy(encoder=enc, head=head).to(device).eval()
        policies.append(PolicyEntry(name="regression", predict=reg_policy.predict_chunk))

    if args.flow_ckpt is not None:
        ckpt = _load_ckpt(args.flow_ckpt, device)
        assert_ckpt_dims(ckpt, args, expected_head="flow")
        head_state, enc_state = _split_state(ckpt)
        head = ConditionalFlowMatcher(
            horizon=args.horizon,
            traj_dim=args.traj_dim,
            cond_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        head.load_state_dict(head_state)
        enc = _build_encoder(args, device)
        if enc_state:
            enc.load_state_dict(enc_state)
        flow_policy = FlowPolicy(encoder=enc, head=head).to(device).eval()
        policies.append(PolicyEntry(name="flow", predict=flow_policy.predict_chunk))
        policies.append(
            PolicyEntry(
                name="flow_bon",
                predict=make_bon_predictor(
                    flow_policy, n=args.bon_n, score_fn=build_default_score_fn()
                ),
                metadata={"n": args.bon_n},
            )
        )

    if args.diffusion_ckpt is not None:
        ckpt = _load_ckpt(args.diffusion_ckpt, device)
        assert_ckpt_dims(ckpt, args, expected_head="diffusion")
        head_state, enc_state = _split_state(ckpt)
        head = ConditionalDiffusionModel(
            horizon=args.horizon,
            traj_dim=args.traj_dim,
            cond_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diffusion_steps,
        ).to(device)
        head.load_state_dict(head_state)
        enc = _build_encoder(args, device)
        if enc_state:
            enc.load_state_dict(enc_state)
        diff_policy = DiffusionPolicy(encoder=enc, head=head).to(device).eval()
        policies.append(PolicyEntry(name="diffusion", predict=diff_policy.predict_chunk))
        policies.append(
            PolicyEntry(
                name="diffusion_bon",
                predict=make_bon_predictor(
                    diff_policy, n=args.bon_n, score_fn=build_default_score_fn()
                ),
                metadata={"n": args.bon_n},
            )
        )

    if not policies:
        print("[eval_navsim_heads] no checkpoints provided — nothing to evaluate.", file=sys.stderr)
        return 1

    report = evaluate_policies(policies, scene_loader)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "per_policy": report.per_policy,
                "scenario_ids": report.scenario_ids[: args.scenarios],
                "horizon_frames": report.horizon_frames,
            },
            f,
            indent=2,
        )
    print(f"[eval_navsim_heads] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
