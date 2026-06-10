"""Instruction encoder — character-hash baseline + optional VLM hookup.

Default :class:`HashedInstructionEncoder` is a deterministic char-hash embedding
that requires no pretrained weights — useful for the M3 baseline and unit
tests. A heavier VLM (transformers + frozen LLM backbone) can be plugged in
later via the same ``latent_dim`` contract; see TODO at the bottom of this file.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn


class HashedInstructionEncoder(nn.Module):
    """Deterministic baseline that maps each instruction string to a latent.

    The string is reduced via SHA-256 to a fixed-length integer feature, then
    a learnable Linear projects it to ``latent_dim``. This is *not* a real VLM
    encoder — it serves as a pluggable placeholder while the M3 training
    pipeline is built. Replace via the same protocol when VLM training begins
    (M5 / Round 1+).
    """

    def __init__(self, latent_dim: int, feature_bytes: int = 32) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if feature_bytes <= 0:
            raise ValueError(f"feature_bytes must be > 0, got {feature_bytes}")
        self.latent_dim = latent_dim
        self.feature_bytes = feature_bytes
        self.proj = nn.Linear(feature_bytes, latent_dim)

    def _hash_to_features(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.feature_bytes]
        as_floats = [b / 255.0 for b in digest]
        return torch.tensor(as_floats, dtype=torch.float32)

    def forward(self, instructions: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not isinstance(instructions, (list, tuple)):
            raise TypeError(
                f"instructions must be list[str] / tuple[str], got {type(instructions).__name__}"
            )
        if not instructions:
            return torch.zeros(0, self.latent_dim, device=self.proj.weight.device)
        feats = torch.stack(
            [self._hash_to_features(s) for s in instructions], dim=0
        ).to(self.proj.weight.device)
        return self.proj(feats)


__all__ = ["HashedInstructionEncoder"]
