"""Train a Flow head with VLM-paraphrase instruction conditioning on nuScenes.

Two text encoders are selectable via ``--text-backbone``:

- ``hash`` (Round 0 baseline): deterministic SHA-256 hash → linear projection.
  No real language understanding; serves as a control for ablations.
- ``sentence-minilm-l6`` (M6.2 default): frozen all-MiniLM-L6-v2 (22M params,
  CPU-friendly) + projection. First run downloads HuggingFace weights — needs
  the ``[vlm]`` extra (``pip install -e .[vlm]``).
"""

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

from driving_vla.data import NuScenesTrajectoryDataset
from driving_vla.models import (
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    HashedInstructionEncoder,
    VLMTextEncoder,
    list_text_backbones,
)
from driving_vla.policies import FlowPolicy
from driving_vla.training import TrainConfig, build_lr_scheduler, build_optimizer, train


_TEXT_CHOICES = ["hash", *list_text_backbones()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train nuScenes flow head with VLM instructions.")
    p.add_argument("--data-path", type=str, default="data/nuscenes/mini.pt")
    p.add_argument("--output", type=str, default="outputs/nuscenes_vlm/model.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--text-latent", type=int, default=64)
    p.add_argument("--text-backbone", type=str, default="sentence-minilm-l6", choices=_TEXT_CHOICES,
                   help="hash (M3 baseline) or sentence-minilm-l6 (M6.2 default; needs [vlm] extras).")
    p.add_argument("--freeze-text", action="store_true", default=True,
                   help="Freeze pretrained LM weights (default; only projection trains).")
    p.add_argument("--no-freeze-text", dest="freeze_text", action="store_false")
    p.add_argument("--max-seq-len", type=int, default=64,
                   help="Max token length for the instruction encoder.")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def _build_text_encoder(args: argparse.Namespace) -> torch.nn.Module:
    if args.text_backbone == "hash":
        return HashedInstructionEncoder(latent_dim=args.text_latent)
    return VLMTextEncoder(
        latent_dim=args.text_latent,
        backbone=args.text_backbone,
        freeze_backbone=args.freeze_text,
        max_seq_len=args.max_seq_len,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] nuscenes_vlm train CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = NuScenesTrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate
    )

    text_encoder = _build_text_encoder(args)
    encoder = DrivingObservationEncoder(
        state_dim=args.state_dim,
        num_commands=args.num_commands,
        latent_dim=args.latent_dim,
        history_dim=args.traj_dim,
        instruction_encoder=text_encoder,
    ).to(device)
    head = ConditionalFlowMatcher(
        horizon=args.horizon,
        traj_dim=args.traj_dim,
        cond_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    policy = FlowPolicy(encoder=encoder, head=head, sampling_steps=args.sampling_steps).to(device)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, lr=args.lr)
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
        config=TrainConfig(num_epochs=args.epochs, save_path=args.output),
        scheduler=scheduler,
        cfg_metadata={
            "head": "flow+vlm",
            "text_backbone": args.text_backbone,
            "args": vars(args),
        },
    )
    print(f"[train_nuscenes_vlm] last_loss={state.last_loss:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
