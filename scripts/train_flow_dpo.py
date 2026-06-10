"""Fine-tune a Flow head with DPO over a PreferenceDataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import (
    assert_ckpt_dims,
    device_from_arg,
    move_condition_to_device,
)

from driving_vla.data import PreferenceDataset, collate_preference_batch
from driving_vla.models import ConditionalFlowMatcher, DrivingObservationEncoder
from driving_vla.training import (
    DPOConfig,
    TrainConfig,
    build_lr_scheduler,
    build_optimizer,
    flow_dpo_loss,
    train,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flow-DPO fine-tuning over preference pairs.")
    p.add_argument("--data-path", type=str, required=False, default="data/preferences/pairs.pt")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional path to a flow checkpoint to initialise both policy and reference heads.")
    p.add_argument("--output", type=str, default="outputs/nuplan_flow_dpo/model.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--timesteps-per-pair", type=int, default=4)
    p.add_argument("--with-reference", action="store_true",
                   help="Keep a frozen copy of the initial head as the DPO reference.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] flow_dpo CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = PreferenceDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_preference_batch,
    )

    encoder = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
    ).to(device)
    policy_head = ConditionalFlowMatcher(
        horizon=args.horizon,
        traj_dim=args.traj_dim,
        cond_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    reference_head: ConditionalFlowMatcher | None = None

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        assert_ckpt_dims(ckpt, args, expected_head="flow")
        if "model_state" in ckpt:
            policy_head.load_state_dict(
                {k.replace("head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("head.")}
            )
            enc_state = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model_state"].items()
                if k.startswith("encoder.")
            }
            if enc_state:
                encoder.load_state_dict(enc_state)
        if args.with_reference:
            reference_head = ConditionalFlowMatcher(
                horizon=args.horizon,
                traj_dim=args.traj_dim,
                cond_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
            ).to(device)
            reference_head.load_state_dict(policy_head.state_dict())
            for p in reference_head.parameters():
                p.requires_grad = False

    # Bundle encoder + head into one Module so train()'s grad_clip and
    # checkpoint save cover both. Saved keys: encoder.* and head.* — matches
    # the layout produced by FlowPolicy in train_nuplan_flow.py, so eval
    # scripts can load both back without code changes.
    trainable = nn.ModuleDict({"encoder": encoder, "head": policy_head})

    optimizer = build_optimizer(trainable.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dataset) // args.batch_size)
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps)
    dpo_cfg = DPOConfig(beta=args.beta, timesteps_per_pair=args.timesteps_per_pair)

    def step_fn(batch: dict) -> torch.Tensor:
        condition = move_condition_to_device(batch["condition"], device)
        chosen = batch["chosen"].to(device)
        rejected = batch["rejected"].to(device)
        cond_latent = encoder(condition)
        return flow_dpo_loss(
            policy_head=policy_head,
            reference_head=reference_head,
            chosen=chosen,
            rejected=rejected,
            cond=cond_latent,
            config=dpo_cfg,
        )

    state = train(
        model=trainable,
        optimizer=optimizer,
        data_loader=loader,
        step_fn=step_fn,
        config=TrainConfig(num_epochs=args.epochs, save_path=args.output),
        scheduler=scheduler,
        cfg_metadata={"head": "flow_dpo", "args": vars(args)},
    )
    print(f"[train_flow_dpo] last_loss={state.last_loss:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
