"""RouteEncoder — VectorNet-style polyline encoder + masked pooling."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import RouteEncoder


def test_forward_shape_no_mask():
    enc = RouteEncoder(latent_dim=32, max_waypoints=64, hidden_dim=64, num_heads=4)
    out = enc(torch.randn(3, 20, 2))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_forward_shape_with_mask():
    enc = RouteEncoder(latent_dim=16, max_waypoints=32, hidden_dim=32, num_heads=2)
    waypoints = torch.randn(2, 16, 2)
    mask = torch.zeros(2, 16, dtype=torch.int8)
    mask[:, :8] = 1  # first 8 are valid
    out = enc(waypoints, mask=mask)
    assert out.shape == (2, 16)


def test_masked_pool_ignores_padding():
    """A fully-padded suffix must not change the output relative to the
    truncated input — this validates the masked attention + masked pool."""
    enc = RouteEncoder(latent_dim=16, max_waypoints=32, hidden_dim=32, num_heads=2)
    enc.eval()
    base = torch.randn(1, 10, 2)
    padded = torch.cat([base, torch.randn(1, 22, 2) * 100.0], dim=1)  # noisy pad
    mask = torch.zeros(1, 32, dtype=torch.int8)
    mask[:, :10] = 1
    with torch.no_grad():
        a = enc(base)
        b = enc(padded, mask=mask)
    assert torch.allclose(a, b, atol=1e-5), (
        f"mask not respected: max diff = {(a - b).abs().max().item():.2e}"
    )


def test_rejects_too_many_waypoints():
    enc = RouteEncoder(latent_dim=8, max_waypoints=16, hidden_dim=16, num_heads=2)
    with pytest.raises(ValueError, match="max_waypoints"):
        enc(torch.randn(1, 20, 2))


def test_rejects_wrong_xy_dim():
    enc = RouteEncoder(latent_dim=8, max_waypoints=16, hidden_dim=16, num_heads=2)
    with pytest.raises(ValueError, match=r"\[B, N, 2\]"):
        enc(torch.randn(1, 10, 3))


def test_rejects_wrong_mask_shape():
    enc = RouteEncoder(latent_dim=8, max_waypoints=16, hidden_dim=16, num_heads=2)
    with pytest.raises(ValueError, match="mask shape"):
        enc(torch.randn(2, 10, 2), mask=torch.ones(2, 5, dtype=torch.int8))


def test_invalid_constructor_args():
    with pytest.raises(ValueError, match="latent_dim"):
        RouteEncoder(latent_dim=0)
    with pytest.raises(ValueError, match="max_waypoints"):
        RouteEncoder(latent_dim=8, max_waypoints=0)
    with pytest.raises(ValueError, match="divisible"):
        RouteEncoder(latent_dim=8, hidden_dim=33, num_heads=4)


def test_grad_flows_through_projection():
    enc = RouteEncoder(latent_dim=8, max_waypoints=16, hidden_dim=16, num_heads=2)
    out = enc(torch.randn(1, 10, 2))
    out.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.projection.parameters()
    )
