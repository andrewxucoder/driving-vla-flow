"""data/map_adapter.py — RouteProvider Protocol + route utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from driving_vla.data import RouteProvider, pad_route_to_length, route_from_future


def test_route_from_future_basic_shape():
    future = np.stack(
        [np.linspace(0.0, 30.0, 16), np.zeros(16)], axis=-1
    ).astype(np.float32)
    route = route_from_future(future, num_waypoints=20, lookahead_m=100.0)
    assert route.shape == (20, 2)
    assert route.dtype == np.float32
    assert np.isfinite(route).all()


def test_route_from_future_extrapolates_beyond_gt():
    future = np.stack(
        [np.linspace(0.0, 10.0, 6), np.zeros(6)], axis=-1
    ).astype(np.float32)
    route = route_from_future(future, num_waypoints=10, lookahead_m=50.0)
    # Last waypoint must lie well past the GT horizon (10m) along +x.
    assert route[-1, 0] > 30.0
    # Should remain on the y=0 line because extrapolation follows the last segment.
    assert abs(route[-1, 1]) < 1e-3


def test_route_from_future_handles_degenerate_short_future():
    future = np.zeros((1, 2), dtype=np.float32)
    route = route_from_future(future, num_waypoints=5, lookahead_m=20.0)
    assert route.shape == (5, 2)
    # Degenerate input → straight-ahead fallback
    assert route[-1, 0] == pytest.approx(20.0, abs=1e-3)


def test_route_from_future_handles_zero_motion():
    future = np.zeros((10, 2), dtype=np.float32)  # car stopped
    route = route_from_future(future, num_waypoints=5, lookahead_m=10.0)
    assert route.shape == (5, 2)
    assert np.isfinite(route).all()


def test_route_from_future_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        route_from_future(np.zeros((10,)), num_waypoints=5, lookahead_m=10.0)
    with pytest.raises(ValueError):
        route_from_future(np.zeros((10, 2)), num_waypoints=0, lookahead_m=10.0)
    with pytest.raises(ValueError):
        route_from_future(np.zeros((10, 2)), num_waypoints=5, lookahead_m=0.0)


def test_pad_route_to_length_padded_case():
    route = torch.tensor([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]])
    padded, mask = pad_route_to_length(route, max_n=8)
    assert padded.shape == (8, 2)
    assert mask.shape == (8,)
    assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert padded[3:].abs().sum().item() == 0


def test_pad_route_to_length_truncated_case():
    route = torch.randn(10, 2)
    padded, mask = pad_route_to_length(route, max_n=5)
    assert padded.shape == (5, 2)
    assert mask.tolist() == [1, 1, 1, 1, 1]
    assert torch.allclose(padded, route[:5])


def test_pad_route_to_length_from_numpy():
    route = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    padded, mask = pad_route_to_length(route, max_n=4)
    assert isinstance(padded, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert padded[0, 0].item() == pytest.approx(1.0)


def test_pad_route_to_length_rejects_invalid():
    with pytest.raises(ValueError):
        pad_route_to_length(torch.zeros(10, 3), max_n=8)
    with pytest.raises(ValueError):
        pad_route_to_length(torch.zeros(10, 2), max_n=0)


def test_route_provider_protocol_runtime_check():
    """Anything with ``get_route(scenario_id, ego_pose) -> ndarray`` satisfies it."""

    class _StubProvider:
        def get_route(self, *, scenario_id: str, ego_pose: np.ndarray) -> np.ndarray:
            return np.zeros((5, 2), dtype=np.float32)

    assert isinstance(_StubProvider(), RouteProvider)

    class _Wrong:
        def get_route(self, x):
            return None

    # Protocol doesn't enforce signature shape — runtime_checkable only checks
    # method existence — so we don't assert non-membership for _Wrong here.
    # The Protocol's value is documentation + IDE/mypy support.
    assert hasattr(_StubProvider(), "get_route")
