"""Regression: Flow-DPO training must save encoder weights alongside the head.

Round 0 silently dropped the fine-tuned encoder because the train script
passed only the policy_head to train(), so checkpoint save & grad-clip both
missed the encoder. The fix bundles them via nn.ModuleDict — this test
verifies the resulting checkpoint key layout, without spinning up the full
training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from driving_vla.models import ConditionalFlowMatcher, DrivingObservationEncoder
from driving_vla.training import TrainConfig, build_optimizer, train


def test_dpo_style_moduledict_persists_encoder(tmp_path: Path) -> None:
    encoder = DrivingObservationEncoder(
        state_dim=5, num_commands=4, latent_dim=16, history_dim=2,
    )
    head = ConditionalFlowMatcher(
        horizon=4, traj_dim=2, cond_dim=16, hidden_dim=32,
    )
    trainable = nn.ModuleDict({"encoder": encoder, "head": head})

    # Synthetic 1-batch dataset; loss is a stand-in (DPO loss tested separately).
    cond = {
        "ego_state": torch.zeros(2, 5),
        "history": torch.zeros(2, 2, 2),
        "command_id": torch.zeros(2, dtype=torch.long),
    }
    target = torch.zeros(2, 4, 2)
    batches = [{"condition": cond, "future": target}]

    def step_fn(batch):
        c = encoder(batch["condition"])
        x1 = batch["future"]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0])
        x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
        v = head(x_t, t, c)
        return ((v - (x1 - x0)) ** 2).mean()

    out_path = tmp_path / "dpo_test.pt"
    train(
        model=trainable,
        optimizer=build_optimizer(trainable.parameters(), lr=1e-4),
        data_loader=batches,
        step_fn=step_fn,
        config=TrainConfig(num_epochs=1, save_path=str(out_path), log_every=10),
    )

    ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
    keys = list(ckpt["model_state"].keys())
    assert any(k.startswith("encoder.") for k in keys), (
        f"checkpoint missing encoder.* keys; got {keys[:5]}..."
    )
    assert any(k.startswith("head.") for k in keys), (
        f"checkpoint missing head.* keys; got {keys[:5]}..."
    )
