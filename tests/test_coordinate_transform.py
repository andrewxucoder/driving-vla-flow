"""coordinate_transform — global ↔ ego-frame rotation/translation."""

from __future__ import annotations

import numpy as np
import pytest

from driving_vla.data import global_to_ego_frame


def test_identity_when_ego_at_origin_facing_x():
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, -2.0]])
    ego = np.array([0.0, 0.0, 0.0])
    out = global_to_ego_frame(pts, ego)
    np.testing.assert_allclose(out, pts, atol=1e-6)


def test_translation_only():
    pts = np.array([[10.0, 5.0], [11.0, 5.0]])
    ego = np.array([10.0, 5.0, 0.0])
    out = global_to_ego_frame(pts, ego)
    np.testing.assert_allclose(out, [[0.0, 0.0], [1.0, 0.0]], atol=1e-6)


def test_rotation_only_90_deg():
    # ego facing +y; a point at world (1, 0) is to the ego's right, i.e. y_ego = -1
    pts = np.array([[1.0, 0.0]])
    ego = np.array([0.0, 0.0, np.pi / 2])
    out = global_to_ego_frame(pts, ego)
    np.testing.assert_allclose(out, [[0.0, -1.0]], atol=1e-6)


def test_combined_translation_and_rotation():
    pts = np.array([[2.0, 2.0]])
    ego = np.array([1.0, 1.0, np.pi / 4])
    out = global_to_ego_frame(pts, ego)
    np.testing.assert_allclose(out, [[np.sqrt(2), 0.0]], atol=1e-6)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        global_to_ego_frame(np.zeros((4, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        global_to_ego_frame(np.zeros((4, 2)), np.zeros(2))
