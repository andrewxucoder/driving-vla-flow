"""Train the regression baseline on preprocessed nuPlan samples."""

from __future__ import annotations

import argparse
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

from driving_vla.data import NuPlanTrajectoryDataset
from driving_vla.models import DrivingObservationEncoder, WaypointRegressionHead
from driving_vla.policies import RegressionPolicy
from driving_vla.training import TrainConfig, build_lr_scheduler, build_optimizer, train


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train nuPlan regression head.")
    p.add_argument("--data-path", type=str, required=False, default="data/nuplan")
    p.add_argument("--output", type=str, default="outputs/nuplan_regression/model.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] regression train CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = NuPlanTrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate
    )

    encoder = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
    ).to(device)
    head = WaypointRegressionHead(
        horizon=args.horizon,
        traj_dim=args.traj_dim,
        cond_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    policy = RegressionPolicy(encoder=encoder, head=head).to(device)

    optimizer = build_optimizer(policy.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps)

    def step_fn(batch: dict) -> torch.Tensor:
        condition = move_condition_to_device(batch["condition"], device)
        future = batch["future"].to(device)
        return policy(condition, future)

    state = train(
        model=policy,
        optimizer=optimizer,
        data_loader=loader,
        step_fn=step_fn,
        config=TrainConfig(
            num_epochs=args.epochs,
            save_path=args.output,
        ),
        scheduler=scheduler,
        cfg_metadata={"head": "regression", "args": vars(args)},
    )
    print(f"[train_nuplan_regression] last_loss={state.last_loss:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
