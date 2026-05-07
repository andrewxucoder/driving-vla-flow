from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from driving_vla.data.toy_dataset import ToyDrivingConfig, ToyDrivingDataset
from driving_vla.evaluation.metrics import ade, fde, jerk_proxy
from driving_vla.models.flow_matching import ConditionalFlowMatcher, flow_matching_loss, sample_flow
from driving_vla.models.vla_encoder import DrivingVLAEncoder


def device_from_cfg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = device_from_cfg(cfg["training"].get("device", "auto"))

    data_cfg = ToyDrivingConfig(**cfg["data"])
    ds = ToyDrivingDataset(data_cfg, seed=seed)
    train_len = int(len(ds) * 0.9)
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    encoder = DrivingVLAEncoder(
        state_dim=cfg["data"]["state_dim"],
        num_commands=len(cfg["data"]["language_vocab"]),
        text_embed_dim=cfg["model"]["text_embed_dim"],
        latent_dim=cfg["model"]["latent_dim"],
    ).to(device)
    flow = ConditionalFlowMatcher(
        horizon=cfg["data"]["horizon"],
        traj_dim=2,
        cond_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(flow.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        encoder.train(); flow.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            state = batch["state"].to(device)
            cmd_id = batch["cmd_id"].to(device)
            traj = batch["trajectory"].to(device)
            cond = encoder(state, cmd_id)
            loss = flow_matching_loss(flow, traj, cond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(flow.parameters()), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        encoder.eval(); flow.eval()
        val_ade, val_fde, val_jerk = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                cmd_id = batch["cmd_id"].to(device)
                traj = batch["trajectory"].to(device)
                cond = encoder(state, cmd_id)
                pred = sample_flow(flow, cond, steps=cfg["model"]["flow_steps"])
                val_ade.append(float(ade(pred, traj).cpu()))
                val_fde.append(float(fde(pred, traj).cpu()))
                val_jerk.append(float(jerk_proxy(pred).cpu()))
        print({
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_ADE": round(float(np.mean(val_ade)), 4),
            "val_FDE": round(float(np.mean(val_fde)), 4),
            "val_jerk_proxy": round(float(np.mean(val_jerk)), 4),
        })

    torch.save({"encoder": encoder.state_dict(), "flow": flow.state_dict(), "cfg": cfg}, out_dir / "model.pt")
    print(f"saved to {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
