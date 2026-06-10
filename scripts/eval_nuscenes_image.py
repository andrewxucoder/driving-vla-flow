"""Evaluate the nuScenes image-conditioned Flow policy on the validation split."""

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
    default_collate,
    device_from_arg,
    move_condition_to_device,
)

from driving_vla.data import NuScenesTrajectoryDataset
from driving_vla.evaluation import ade, collision_proxy, fde
from driving_vla.models import (
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    MultiViewImageEncoder,
)
from driving_vla.policies import FlowPolicy


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate nuScenes flow+image policy.")
    p.add_argument("--data-path", type=str, default="data/nuscenes/mini.pt")
    p.add_argument("--checkpoint", type=str, required=False)
    p.add_argument("--output", type=str, default="outputs/round0_nuscenes_image.json")
    p.add_argument("--batch-size", type=int, default=4)

    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--image-latent", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] eval_nuscenes_image CLI parsed OK.")
        return 0

    if args.checkpoint is None:
        print("[eval_nuscenes_image] --checkpoint is required outside dry-run.", file=sys.stderr)
        return 2

    device = device_from_arg(args.device)
    dataset = NuScenesTrajectoryDataset(args.data_path, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=default_collate)

    image_encoder = MultiViewImageEncoder(
        latent_dim=args.image_latent, image_size=args.image_size
    )
    encoder = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
        image_encoder=image_encoder,
    ).to(device)
    head = ConditionalFlowMatcher(
        horizon=args.horizon,
        traj_dim=args.traj_dim,
        cond_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    policy = FlowPolicy(encoder=encoder, head=head).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    policy.load_state_dict(state)

    ade_sum, fde_sum, coll_sum, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        future = batch["future"].to(device)
        cond = move_condition_to_device(batch["condition"], device)
        with torch.no_grad():
            pred = policy._sample_chunk(policy.encoder(cond))
        h = min(pred.shape[1], future.shape[1])
        pred, future = pred[:, :h], future[:, :h]
        ade_sum += float(ade(pred, future).sum().item())
        fde_sum += float(fde(pred, future).sum().item())
        coll_sum += float(collision_proxy(pred).sum().item())
        n += pred.shape[0]

    report = {
        "n": n,
        "ade_mean": ade_sum / max(n, 1),
        "fde_mean": fde_sum / max(n, 1),
        "collision_rate": coll_sum / max(n, 1),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval_nuscenes_image] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
