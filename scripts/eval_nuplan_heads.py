"""Evaluate 5 policies on a preprocessed nuPlan validation split.

Computes ADE / FDE / collision-proxy on the offline dataset (no closed-loop
simulator involved). Useful for quick sanity-checking before the NAVSIM run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import (
    assert_ckpt_dims,
    default_collate,
    device_from_arg,
    move_condition_to_device,
)

from driving_vla.data import NuPlanTrajectoryDataset
from driving_vla.evaluation import (
    ade,
    build_default_score_fn,
    collision_proxy,
    fde,
)
from driving_vla.models import (
    ConditionalDiffusionModel,
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    WaypointRegressionHead,
)
from driving_vla.policies import DiffusionPolicy, FlowPolicy, RegressionPolicy


def _load_split(
    checkpoint: str,
    device: torch.device,
    args: argparse.Namespace,
    expected_head: str,
) -> tuple[dict, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    assert_ckpt_dims(ckpt, args, expected_head=expected_head)
    state = ckpt.get("model_state", ckpt)
    head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    return head_state, enc_state


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline nuPlan ADE/FDE evaluation.")
    p.add_argument("--data-path", type=str, default="data/nuplan")
    p.add_argument("--output", type=str, default="outputs/round0_nuplan.json")
    p.add_argument("--regression-ckpt", type=str, default=None)
    p.add_argument("--flow-ckpt", type=str, default=None)
    p.add_argument("--diffusion-ckpt", type=str, default=None)
    p.add_argument("--bon-n", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)

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


def _evaluate_policy(name: str, predict_fn, loader, device: torch.device) -> dict:
    ade_sum, fde_sum, coll_sum, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        future = batch["future"].to(device)
        cond_batch = move_condition_to_device(batch["condition"], device)
        # Use the batch predictor for vectorised efficiency.
        pred = predict_fn(cond_batch)  # [B, H, 2]
        if pred.shape != future.shape:
            h = min(pred.shape[1], future.shape[1])
            pred = pred[:, :h]
            future = future[:, :h]
        ade_sum += float(ade(pred, future).sum().item())
        fde_sum += float(fde(pred, future).sum().item())
        coll_sum += float(collision_proxy(pred).sum().item())
        n += pred.shape[0]
    if n == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": n,
        "ade_mean": ade_sum / n,
        "fde_mean": fde_sum / n,
        "collision_rate": coll_sum / n,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] eval_nuplan_heads CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = NuPlanTrajectoryDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=default_collate)

    results: list[dict] = []

    def _enc() -> DrivingObservationEncoder:
        return DrivingObservationEncoder(
            state_dim=args.state_dim,
            num_commands=args.num_commands,
            latent_dim=args.latent_dim,
            history_dim=args.traj_dim,
        ).to(device)

    score_fn = build_default_score_fn()

    if args.regression_ckpt is not None:
        head_state, enc_state = _load_split(
            args.regression_ckpt, device, args, expected_head="regression"
        )
        head = WaypointRegressionHead(
            horizon=args.horizon, traj_dim=args.traj_dim,
            cond_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        ).to(device)
        head.load_state_dict(head_state)
        enc = _enc()
        if enc_state:
            enc.load_state_dict(enc_state)
        policy = RegressionPolicy(encoder=enc, head=head).to(device).eval()

        def predict(cond_batch: dict) -> torch.Tensor:
            return policy.head(policy.encoder(cond_batch))

        results.append(_evaluate_policy("regression", predict, loader, device))

    for name, head_cls, policy_cls, ckpt_arg in (
        ("flow", ConditionalFlowMatcher, FlowPolicy, args.flow_ckpt),
        ("diffusion", ConditionalDiffusionModel, DiffusionPolicy, args.diffusion_ckpt),
    ):
        if ckpt_arg is None:
            continue
        head_state, enc_state = _load_split(ckpt_arg, device, args, expected_head=name)
        head_kwargs: dict = dict(
            horizon=args.horizon, traj_dim=args.traj_dim,
            cond_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        )
        if name == "diffusion":
            head_kwargs["diffusion_steps"] = args.diffusion_steps
        head = head_cls(**head_kwargs).to(device)
        head.load_state_dict(head_state)
        enc = _enc()
        if enc_state:
            enc.load_state_dict(enc_state)
        policy = policy_cls(encoder=enc, head=head).to(device).eval()

        def predict(cond_batch: dict, _p=policy) -> torch.Tensor:
            return _p._sample_chunk(_p.encoder(cond_batch))  # type: ignore[no-any-return]

        results.append(_evaluate_policy(name, predict, loader, device))

        def predict_bon(cond_batch: dict, _p=policy) -> torch.Tensor:
            cond_latent = _p.encoder(cond_batch)
            samples = _p._sample_chunk_n(cond_latent, n=args.bon_n)  # [B, N, H, 2]
            score_cond: dict = {"command_id": cond_batch.get("command_id")}
            if cond_batch.get("ego_state") is not None:
                score_cond["ego_state"] = cond_batch["ego_state"]
            scores = score_fn(
                samples.reshape(-1, samples.shape[-2], samples.shape[-1]),
                score_cond,
            ).reshape(samples.shape[0], samples.shape[1])
            best_idx = scores.argmax(dim=-1)
            return samples[torch.arange(samples.shape[0]), best_idx]

        results.append(_evaluate_policy(f"{name}_bon", predict_bon, loader, device))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"[eval_nuplan_heads] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
