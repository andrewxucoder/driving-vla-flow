"""Pretrained vision backbone — DINOv2 / SigLIP + multi-view fusion.

Replaces the small CNN of :class:`MultiViewImageEncoder` with a frozen
pretrained vision transformer (DINOv2-base by default — self-supervised
geometric features align well with driving scenes; SigLIP-base alternative if
you want VLM-aligned features for downstream M8 VLA work).

Architecture::

    [B, V, 3, H, W]
        → flatten batch×view
        → backbone (frozen by default) → per-view [B*V, D_backbone]
        → reshape to [B, V, D_backbone]
        → + view-id positional embedding (learnable)
        → 1-layer transformer encoder (learnable, attends across views)
        → mean pool over views
        → MLP projection → [B, latent_dim]

The ``transformers`` package is loaded lazily inside ``_load_backbone`` so
``import driving_vla.models`` does not fail when the ``[vlm]`` extras are
absent. Pass ``backbone_module=<your nn.Module>`` to skip the registry-based
load entirely — used by the unit tests to avoid the ~85 MB HuggingFace
download in CI.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn


_BACKBONE_REGISTRY: dict[str, dict[str, Any]] = {
    "dinov2-base": {
        "hf_id": "facebook/dinov2-base",
        "embed_dim": 768,
        "uses_cls": True,
    },
    "dinov2-small": {
        "hf_id": "facebook/dinov2-small",
        "embed_dim": 384,
        "uses_cls": True,
    },
    "siglip-base": {
        "hf_id": "google/siglip-base-patch16-224",
        "embed_dim": 768,
        "uses_cls": False,
    },
}


def list_backbones() -> list[str]:
    """Names accepted by the ``backbone`` argument."""
    return sorted(_BACKBONE_REGISTRY.keys())


class PretrainedVisionBackbone(nn.Module):
    """Frozen pretrained vision tower + learnable multi-view fusion + projection."""

    def __init__(
        self,
        latent_dim: int,
        backbone: Literal["dinov2-base", "dinov2-small", "siglip-base"] = "dinov2-base",
        image_size: int = 224,
        freeze_backbone: bool = True,
        num_views_max: int = 8,
        fusion_heads: int = 4,
        embed_dim_override: int | None = None,
        backbone_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"backbone must be one of {list_backbones()}, got {backbone!r}"
            )
        if num_views_max <= 0:
            raise ValueError(f"num_views_max must be > 0, got {num_views_max}")

        spec = _BACKBONE_REGISTRY[backbone]
        embed_dim = embed_dim_override if embed_dim_override is not None else spec["embed_dim"]

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self._uses_cls: bool = bool(spec["uses_cls"])

        if backbone_module is not None:
            self._backbone = backbone_module
        else:
            self._backbone = self._load_backbone(spec["hf_id"])

        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()

        self.view_pos_embed = nn.Parameter(
            torch.randn(1, num_views_max, embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=fusion_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.view_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _load_backbone(hf_id: str) -> nn.Module:
        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PretrainedVisionBackbone needs `transformers` — install via "
                "`pip install -e .[vlm]`. Or pass `backbone_module=<nn.Module>` "
                "to skip the HuggingFace download (used by unit tests)."
            ) from exc
        return AutoModel.from_pretrained(hf_id)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone on ``[N, 3, H, W]``, return ``[N, embed_dim]``."""
        out = self._backbone(pixel_values=x)
        if self._uses_cls:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError(
                    f"{self.backbone_name} forward returned no last_hidden_state"
                )
            return hidden[:, 0, :]
        pooler = getattr(out, "pooler_output", None)
        if pooler is not None:
            return pooler
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                f"{self.backbone_name} forward exposed neither pooler_output "
                "nor last_hidden_state"
            )
        return hidden.mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"images expected [B, V, C, H, W], got {tuple(images.shape)}"
            )
        batch, views, c, h, w = images.shape
        if views > self.view_pos_embed.shape[1]:
            raise ValueError(
                f"got {views} views but view_pos_embed allocates only "
                f"{self.view_pos_embed.shape[1]}"
            )

        flat = images.reshape(batch * views, c, h, w)
        backbone_train_mode = any(p.requires_grad for p in self._backbone.parameters())
        with torch.set_grad_enabled(backbone_train_mode):
            per_view_flat = self._backbone_forward(flat)
        if per_view_flat.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"backbone produced dim {per_view_flat.shape[-1]} but "
                f"PretrainedVisionBackbone was configured with embed_dim={self.embed_dim}"
            )

        per_view = per_view_flat.reshape(batch, views, self.embed_dim)
        per_view = per_view + self.view_pos_embed[:, :views, :]
        fused = self.view_fusion(per_view)
        pooled = fused.mean(dim=1)
        return self.projection(pooled)


__all__ = ["PretrainedVisionBackbone", "list_backbones"]
