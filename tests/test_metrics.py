"""evaluation.metrics — trajectory metrics."""

from __future__ import annotations

import pytest
import torch

from driving_vla.evaluation import (
    ade,
    collision_proxy,
    fde,
    forward_progress,
    jerk,
    lateral_offset,
)


def _line(x_end: float = 8.0, y_end: float = 0.0, H: int = 8) -> torch.Tensor:
    xs = torch.linspace(1.0, x_end, H)
    ys = torch.linspace(0.0, y_end, H)
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)  # [1, H, 2]


def test_ade_zero_when_identical():
    t = _line()
    assert ade(t, t).item() == pytest.approx(0.0)


def test_ade_nonzero_when_offset():
    t = _line()
    shifted = t + torch.tensor([0.0, 1.0])
    assert ade(t, shifted).item() == pytest.approx(1.0)


def test_fde_uses_last_step():
    pred = _line(x_end=8.0)
    target = _line(x_end=8.0)
    assert fde(pred, target).item() == pytest.approx(0.0)


def test_collision_zero_for_centered():
    assert collision_proxy(_line()).item() == 0.0


def test_collision_one_for_large_lateral():
    t = _line(y_end=5.0)
    assert collision_proxy(t).item() == 1.0


def test_lateral_offset_max():
    t = _line(y_end=3.0)
    assert lateral_offset(t).item() == pytest.approx(3.0)


def test_forward_progress():
    t = _line(x_end=8.0)
    assert forward_progress(t).item() == pytest.approx(7.0)


def test_jerk_zero_for_linear():
    t = _line()
    assert jerk(t).item() == pytest.approx(0.0, abs=1e-5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ade(_line(), torch.zeros(1, 10, 2))


def test_rejects_wrong_dim():
    with pytest.raises(ValueError):
        ade(torch.zeros(4, 2), torch.zeros(4, 2))
