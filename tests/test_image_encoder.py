"""MultiViewImageEncoder — shape + batch/view collapse."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import MultiViewImageEncoder


def test_output_shape():
    enc = MultiViewImageEncoder(latent_dim=32, image_size=64)
    out = enc(torch.zeros(2, 3, 3, 64, 64))
    assert out.shape == (2, 32)


def test_single_view():
    enc = MultiViewImageEncoder(latent_dim=32, image_size=64)
    out = enc(torch.zeros(1, 1, 3, 64, 64))
    assert out.shape == (1, 32)


def test_six_view_typical_nuscenes():
    enc = MultiViewImageEncoder(latent_dim=64, image_size=64)
    out = enc(torch.randn(2, 6, 3, 64, 64))
    assert out.shape == (2, 64)


def test_rejects_4d_input():
    enc = MultiViewImageEncoder(latent_dim=32, image_size=64)
    with pytest.raises(ValueError):
        enc(torch.zeros(2, 3, 64, 64))


def test_invalid_latent_dim():
    with pytest.raises(ValueError):
        MultiViewImageEncoder(latent_dim=0)
