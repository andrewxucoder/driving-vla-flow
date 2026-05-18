from __future__ import annotations

import torch
from torch import nn


class DrivingVLAEncoder(nn.Module):
    """最小化的 VLA 风格条件编码器。

    在 MVP 中，视觉/BEV 特征由紧凑的状态向量表示。
    后续版本可以将 `state` 替换为 BEV token 或多视角特征。
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
