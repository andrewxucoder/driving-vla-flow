"""Generate (chosen, rejected) trajectory preference pairs.

Two source modes are supported:

1. **Perturbation mode** (no policy needed): take the expert ``future`` from a
   nuPlan/nuScenes dataset, apply a random :mod:`trajectory_perturbations`
   transform to make a noisy rejected candidate, and rank by rule-based reward.
   Round-0 DPO used this; the signal is weak because expert-vs-perturbed is
   too easy a discrimination task (DPO loss drops, ADE barely moves).

2. **BoN mode** (requires a trained Flow / Diffusion checkpoint): sample N
   independent candidates from the policy, score them with the default reward,
   and pick the highest + lowest as chosen / rejected. This trains DPO on the
   policy's *own* sampling distribution — preference signal is harder because
   the two candidates are realistic trajectories that share the same encoder
   conditioning, only differing in stochastic draw.

Outputs a ``.pt`` consumable by :class:`PreferenceDataset`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import assert_ckpt_dims, device_from_arg, move_condition_to_device

from driving_vla.data import (
    NuPlanTrajectoryDataset,
    random_perturb,
)
from driving_vla.evaluation import build_default_score_fn
from driving_vla.models import (
    ConditionalDiffusionModel,
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
)
from driving_vla.policies import DiffusionPolicy, FlowPolicy


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate trajectory preference pairs.")
    p.add_argument("--data-path", type=str, required=False, default="data/nuplan")
    p.add_argument("--output", type=str, default="data/preferences/pairs.pt")
    p.add_argument("--mode", choices=("perturb", "bon"), default="perturb")
    p.add_argument("--max-pairs", type=int, default=10_000)
    p.add_argument("--perturb-strength", type=float, default=0.5)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="(bon mode) trained policy checkpoint path.")
    p.add_argument("--policy-type", choices=("flow", "diffusion"), default="flow",
                   help="(bon mode) which head architecture the ckpt belongs to.")
    p.add_argument("--bon-n", type=int, default=8)
    p.add_argument("--min-reward-gap", type=float, default=0.5,
                   help="(bon mode) drop pairs whose reward_gap is below this threshold "
                        "— prevents near-duplicate chosen/rejected from polluting DPO.")
    # Structural dims (must match the checkpoint when --mode bon)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def _unsqueeze_condition(condition: dict) -> dict:
    """Add a leading batch dim to every tensor in ``condition``."""
    out: dict = {}
    for k, v in condition.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, dict):
            out[k] = [v]
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


def _build_policy(
    args: argparse.Namespace, device: torch.device
) -> FlowPolicy | DiffusionPolicy:
    encoder = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
    ).to(device)
    if args.policy_type == "flow":
        head = ConditionalFlowMatcher(
            horizon=args.horizon,
            traj_dim=args.traj_dim,
            cond_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        return FlowPolicy(encoder=encoder, head=head, sampling_steps=args.sampling_steps).to(device)
    head = ConditionalDiffusionModel(
        horizon=args.horizon,
        traj_dim=args.traj_dim,
        cond_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    return DiffusionPolicy(encoder=encoder, head=head, sampling_steps=args.sampling_steps).to(device)


def _load_policy_weights(policy: torch.nn.Module, ckpt: dict) -> None:
    state = ckpt.get("model_state", ckpt)
    enc_state = {
        k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")
    }
    head_state = {
        k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")
    }
    if enc_state:
        policy.encoder.load_state_dict(enc_state)
    if head_state:
        policy.head.load_state_dict(head_state)


def _generate_perturb_pairs(args: argparse.Namespace, device: torch.device) -> list[dict]:
    dataset = NuPlanTrajectoryDataset(args.data_path)
    score_fn = build_default_score_fn()
    pairs: list[dict] = []
    for idx in range(min(args.max_pairs, len(dataset))):
        sample = dataset[idx]
        expert = sample["future"].unsqueeze(0).to(device)
        cond_batch = move_condition_to_device(sample["condition"], device)
        for k, v in list(cond_batch.items()):
            if isinstance(v, torch.Tensor) and v.ndim == sample["condition"][k].ndim:
                cond_batch[k] = v.unsqueeze(0)
        noisy, _ = random_perturb(
            ["gaussian", "time_shift", "lane_jitter"],
            expert,
            strength=args.perturb_strength,
        )
        score_cond: dict = {"command_id": cond_batch.get("command_id")}
        if cond_batch.get("ego_state") is not None:
            score_cond["ego_state"] = cond_batch["ego_state"]
        scores = score_fn(torch.cat([expert, noisy], dim=0), score_cond)
        chosen_idx = int(scores.argmax().item())
        rejected_idx = int(scores.argmin().item())
        if chosen_idx == rejected_idx:
            continue
        traj_pair = torch.cat([expert, noisy], dim=0)
        pairs.append({
            "condition": sample["condition"],
            "chosen": traj_pair[chosen_idx].cpu(),
            "rejected": traj_pair[rejected_idx].cpu(),
            "chosen_reward": float(scores[chosen_idx].item()),
            "rejected_reward": float(scores[rejected_idx].item()),
            "reward_gap": float(scores[chosen_idx].item() - scores[rejected_idx].item()),
        })
    return pairs


def _generate_bon_pairs(args: argparse.Namespace, device: torch.device) -> list[dict]:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    assert_ckpt_dims(ckpt, args, expected_head=args.policy_type)

    policy = _build_policy(args, device)
    _load_policy_weights(policy, ckpt)
    policy.eval()

    dataset = NuPlanTrajectoryDataset(args.data_path)
    score_fn = build_default_score_fn()
    pairs: list[dict] = []
    dropped_dup = 0
    dropped_low_gap = 0

    with torch.no_grad():
        for idx in range(min(args.max_pairs, len(dataset))):
            sample = dataset[idx]
            cond_batch = _unsqueeze_condition(move_condition_to_device(sample["condition"], device))
            cond_latent = policy.encoder(cond_batch)
            chunks = policy._sample_chunk_n(cond_latent, n=args.bon_n)  # [1, N, H, D]
            chunks = chunks.squeeze(0)  # [N, H, D]

            score_cond: dict = {"command_id": cond_batch.get("command_id")}
            if cond_batch.get("ego_state") is not None:
                score_cond["ego_state"] = cond_batch["ego_state"]
            scores = score_fn(chunks, score_cond)

            chosen_idx = int(scores.argmax().item())
            rejected_idx = int(scores.argmin().item())
            if chosen_idx == rejected_idx:
                dropped_dup += 1
                continue

            reward_gap = float(scores[chosen_idx].item() - scores[rejected_idx].item())
            if reward_gap < args.min_reward_gap:
                dropped_low_gap += 1
                continue

            pairs.append({
                "condition": sample["condition"],
                "chosen": chunks[chosen_idx].cpu(),
                "rejected": chunks[rejected_idx].cpu(),
                "chosen_reward": float(scores[chosen_idx].item()),
                "rejected_reward": float(scores[rejected_idx].item()),
                "reward_gap": reward_gap,
            })

    print(
        f"[generate_preference_data] BoN: kept={len(pairs)}, "
        f"dropped_duplicate={dropped_dup}, dropped_low_gap={dropped_low_gap}",
        file=sys.stderr,
    )
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] generate_preference_data CLI parsed OK.")
        return 0

    if args.mode == "bon" and args.checkpoint is None:
        print("[generate_preference_data] --checkpoint required for --mode bon", file=sys.stderr)
        return 2

    device = device_from_arg(args.device)

    if args.mode == "perturb":
        pairs = _generate_perturb_pairs(args, device)
    else:
        pairs = _generate_bon_pairs(args, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pairs": pairs, "cfg": vars(args)}, out_path)
    print(
        f"[generate_preference_data] wrote {len(pairs)} pairs ({args.mode}) → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
