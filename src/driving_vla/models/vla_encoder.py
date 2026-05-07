from __future__ import annotations

import torch
from torch import nn


class DrivingVLAEncoder(nn.Module):
    """Minimal VLA-style condition encoder.

    For MVP, visual/BEV features are represented by a compact state vector.
    Future versions can replace `state` with BEV tokens or multi-view features.
    """

    def __init__(self, state_dim: int, num_commands: int, text_embed_dim: int, latent_dim: int):
        super().__init__()
        self.cmd_embed = nn.Embedding(num_commands, text_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + text_embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
        )

    def forward(self, state: torch.Tensor, cmd_id: torch.Tensor) -> torch.Tensor:
        cmd = self.cmd_embed(cmd_id)
        return self.net(torch.cat([state, cmd], dim=-1))
