"""trajectory_perturbations — RM/DPO rejected candidate synthesisers."""

from __future__ import annotations

import pytest
import torch

from driving_vla.data import (
    PERTURBATIONS,
    gaussian_perturb,
    lane_jitter_perturb,
    perturb_by_name,
    random_perturb,
    time_shift_perturb,
)


def _traj(b: int = 2, h: int = 8, d: int = 2) -> torch.Tensor:
    return torch.arange(b * h * d, dtype=torch.float32).reshape(b, h, d)


def test_gaussian_changes_values():
    traj = _traj()
    g = torch.Generator().manual_seed(7)
    out = gaussian_perturb(traj, strength=0.5, generator=g)
    assert out.shape == traj.shape
    assert not torch.equal(out, traj)


def test_time_shift_keeps_shape():
    traj = _traj()
    g = torch.Generator().manual_seed(3)
    out = time_shift_perturb(traj, strength=0.25, generator=g)
    assert out.shape == traj.shape


def test_time_shift_requires_3d():
    with pytest.raises(ValueError):
        time_shift_perturb(torch.zeros(8, 2), strength=0.1)


def test_lane_jitter_offsets_only_y():
    traj = _traj()
    g = torch.Generator().manual_seed(11)
    out = lane_jitter_perturb(traj, strength=1.0, generator=g)
    # x channel untouched
    assert torch.allclose(out[..., 0], traj[..., 0])
    # y channel perturbed
    assert not torch.allclose(out[..., 1], traj[..., 1])


def test_lane_jitter_constant_offset_within_batch_item():
    traj = _traj()
    g = torch.Generator().manual_seed(11)
    out = lane_jitter_perturb(traj, strength=1.0, generator=g)
    # offset is constant across time within each batch item
    deltas = (out[..., 1] - traj[..., 1])
    for b in range(deltas.shape[0]):
        assert torch.allclose(deltas[b], deltas[b][0].expand_as(deltas[b]))


def test_registry_has_three_perturbations():
    assert set(PERTURBATIONS.keys()) == {"gaussian", "time_shift", "lane_jitter"}


def test_perturb_by_name_dispatch():
    traj = _traj()
    out = perturb_by_name("gaussian", traj, strength=0.0)
    # strength 0 with no generator: random noise, but std=0 → values unchanged within float precision
    assert torch.allclose(out, traj, atol=1e-6)


def test_perturb_by_name_unknown_raises():
    with pytest.raises(KeyError):
        perturb_by_name("nope", _traj(), strength=0.1)


def test_random_perturb_picks_one_per_item():
    traj = _traj(b=4)
    g = torch.Generator().manual_seed(13)
    out, names = random_perturb(["gaussian", "lane_jitter"], traj, 0.3, generator=g)
    assert out.shape == traj.shape
    assert len(names) == 4
    for n in names:
        assert n in ("gaussian", "lane_jitter")


def test_random_perturb_empty_names_raises():
    with pytest.raises(ValueError):
        random_perturb([], _traj(), 0.1)
