"""PretrainedVisionBackbone — multi-view fusion over a frozen vision tower.

We inject a stub backbone matching the HuggingFace AutoModel output structure
so the suite stays offline-friendly. Behaviour under a real DINOv2 / SigLIP is
validated separately via the M6 integration script.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from driving_vla.models import PretrainedVisionBackbone, list_backbones


class _StubDinoBackbone(nn.Module):
    """Mimics HF DINOv2: returns ``last_hidden_state`` [N, 1+P, D] with CLS at 0."""

    def __init__(self, embed_dim: int = 768, num_patches: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        # Trivial conv → embed: makes per-view embeddings depend on inputs.
        self.stem = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        # [N, embed_dim, H/16, W/16] → mean-pool spatially → CLS proxy
        feat = self.stem(pixel_values)
        cls = feat.mean(dim=(-1, -2))  # [N, embed_dim]
        patches = feat.flatten(2).transpose(1, 2)  # [N, P, embed_dim]
        if patches.shape[1] < self.num_patches:
            pad = torch.zeros(
                patches.shape[0], self.num_patches - patches.shape[1], self.embed_dim,
                device=patches.device, dtype=patches.dtype,
            )
            patches = torch.cat([patches, pad], dim=1)
        else:
            patches = patches[:, : self.num_patches]
        hidden = torch.cat([cls.unsqueeze(1), patches], dim=1)  # [N, 1+P, embed_dim]
        return SimpleNamespace(last_hidden_state=hidden)


class _StubSiglipBackbone(nn.Module):
    """Mimics HF SigLIP: returns ``pooler_output`` [N, D]."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        feat = self.stem(pixel_values).mean(dim=(-1, -2))
        return SimpleNamespace(pooler_output=feat, last_hidden_state=None)


def test_list_backbones_nonempty():
    names = list_backbones()
    assert "dinov2-base" in names
    assert "siglip-base" in names


def test_forward_shape_dino_cls_path():
    backbone = PretrainedVisionBackbone(
        latent_dim=64,
        backbone="dinov2-base",
        backbone_module=_StubDinoBackbone(embed_dim=768),
    )
    images = torch.randn(2, 6, 3, 224, 224)
    out = backbone(images)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_forward_shape_siglip_pooler_path():
    backbone = PretrainedVisionBackbone(
        latent_dim=32,
        backbone="siglip-base",
        backbone_module=_StubSiglipBackbone(embed_dim=768),
    )
    images = torch.randn(3, 4, 3, 224, 224)
    out = backbone(images)
    assert out.shape == (3, 32)


def test_freeze_backbone_blocks_grad():
    stub = _StubDinoBackbone(embed_dim=128)
    bb = PretrainedVisionBackbone(
        latent_dim=16,
        backbone="dinov2-base",
        embed_dim_override=128,
        freeze_backbone=True,
        backbone_module=stub,
    )
    images = torch.randn(1, 2, 3, 32, 32)
    loss = bb(images).sum()
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in stub.parameters()), (
        "frozen backbone should not accumulate gradient"
    )
    # Fusion + projection should receive gradient.
    has_fusion_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bb.view_fusion.parameters()
    )
    has_proj_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bb.projection.parameters()
    )
    assert has_fusion_grad and has_proj_grad


def test_unfrozen_backbone_passes_grad():
    stub = _StubDinoBackbone(embed_dim=128)
    bb = PretrainedVisionBackbone(
        latent_dim=16,
        backbone="dinov2-base",
        embed_dim_override=128,
        freeze_backbone=False,
        backbone_module=stub,
    )
    images = torch.randn(1, 2, 3, 32, 32)
    bb(images).sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in stub.parameters()
    )


def test_rejects_wrong_ndim():
    bb = PretrainedVisionBackbone(
        latent_dim=8, backbone="dinov2-base",
        embed_dim_override=64, backbone_module=_StubDinoBackbone(embed_dim=64),
    )
    with pytest.raises(ValueError, match=r"\[B, V, C, H, W\]"):
        bb(torch.randn(2, 3, 64, 64))  # missing view dim


def test_rejects_too_many_views():
    bb = PretrainedVisionBackbone(
        latent_dim=8, backbone="dinov2-base",
        embed_dim_override=64, num_views_max=4,
        backbone_module=_StubDinoBackbone(embed_dim=64),
    )
    with pytest.raises(ValueError, match="view_pos_embed"):
        bb(torch.randn(1, 5, 3, 32, 32))


def test_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="backbone must be one of"):
        PretrainedVisionBackbone(latent_dim=8, backbone="not-a-real-model")  # type: ignore[arg-type]


def test_embed_dim_mismatch_raises_at_forward():
    """Stub claims dim 128 but constructor expects 64 → mismatch surfaces clearly."""
    bb = PretrainedVisionBackbone(
        latent_dim=8, backbone="dinov2-base",
        embed_dim_override=64,
        backbone_module=_StubDinoBackbone(embed_dim=128),
    )
    with pytest.raises(RuntimeError, match="embed_dim"):
        bb(torch.randn(1, 2, 3, 32, 32))


def test_plugs_into_driving_observation_encoder():
    """Verifies the vision backbone satisfies the _FeatureEncoder protocol."""
    from driving_vla.models import DrivingObservationEncoder
    vision = PretrainedVisionBackbone(
        latent_dim=32, backbone="dinov2-base",
        embed_dim_override=64, backbone_module=_StubDinoBackbone(embed_dim=64),
    )
    enc = DrivingObservationEncoder(
        state_dim=5,
        num_commands=4,
        latent_dim=48,
        history_dim=2,
        image_encoder=vision,
    )
    out = enc({
        "ego_state": torch.zeros(2, 5),
        "history": torch.zeros(2, 4, 2),
        "command_id": torch.zeros(2, dtype=torch.long),
        "images": torch.randn(2, 6, 3, 224, 224),
    })
    assert out.shape == (2, 48)
