"""Train a Flow head with multi-view image conditioning on nuScenes samples.

Two vision backbones are selectable via ``--vision-backbone``:

- ``small_cnn`` (Round 0 baseline): the original ~few-hundred-K-param CNN.
- ``dinov2-base`` / ``dinov2-small`` / ``siglip-base`` (M6.1): frozen
  pretrained ViT + learnable multi-view transformer fusion. Requires the
  ``[vlm]`` extra (``pip install -e .[vlm]``) for the first run to download
  HuggingFace weights.
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
    MultiViewImageEncoder,
    PretrainedVisionBackbone,
    list_backbones,
)
from driving_vla.policies import FlowPolicy
from driving_vla.training import TrainConfig, build_lr_scheduler, build_optimizer, train


_VISION_CHOICES = ["small_cnn", *list_backbones()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train nuScenes flow head with images.")
    p.add_argument("--data-path", type=str, default="data/nuscenes/mini.pt")
    p.add_argument("--output", type=str, default="outputs/nuscenes_image/model.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--state-dim", type=int, default=5)
    p.add_argument("--num-commands", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--image-latent", type=int, default=64,
                   help="Output dim of the vision encoder (becomes one branch of the obs encoder).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--vision-backbone", type=str, default="small_cnn", choices=_VISION_CHOICES,
                   help="small_cnn (M3 baseline) or any pretrained ViT (M6.1; needs [vlm] extras).")
    p.add_argument("--freeze-vision", action="store_true", default=True,
                   help="Freeze pretrained backbone weights (M6.1 default; only fusion+projection train).")
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--traj-dim", type=int, default=2)
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def _build_vision_encoder(args: argparse.Namespace) -> torch.nn.Module:
    if args.vision_backbone == "small_cnn":
        return MultiViewImageEncoder(
            latent_dim=args.image_latent, image_size=args.image_size
        )
    return PretrainedVisionBackbone(
        latent_dim=args.image_latent,
        backbone=args.vision_backbone,
        image_size=args.image_size,
        freeze_backbone=args.freeze_vision,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        print("[dry-run] nuscenes_image train CLI parsed OK.")
        return 0

    device = device_from_arg(args.device)
    dataset = NuScenesTrajectoryDataset(args.data_path, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate
    )

    image_encoder = _build_vision_encoder(args)
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
    policy = FlowPolicy(encoder=encoder, head=head, sampling_steps=args.sampling_steps).to(device)

    # Only train parameters that require grad — frozen backbones are skipped.
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
            "head": "flow+image",
            "vision_backbone": args.vision_backbone,
            "args": vars(args),
        },
    )
    print(f"[train_nuscenes_image] last_loss={state.last_loss:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
