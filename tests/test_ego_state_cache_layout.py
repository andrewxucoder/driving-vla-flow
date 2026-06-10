"""Guard against the ego_state layout regression (tech_report §7.5).

The v1-inherited nuPlan cache encoded ego_state as [0, 0, 0, |v|, |a|] — the
signed vx/vy/ax/yaw components were dropped, which silently broke every
speed-conditioned consumer and was the true cause of the offline→NAVSIM gap. No
test caught it. This one does: whenever a local nuPlan cache is present, it
asserts the kinematic columns carry signal and that vx behaves like a forward
speed. It skips cleanly in environments without the cache (CI), so it never
blocks unrelated work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_CACHE = Path(__file__).resolve().parents[1] / "data" / "nuplan" / "train.pt"


@pytest.mark.skipif(not _CACHE.exists(), reason="no local nuPlan cache")
def test_nuplan_cache_ego_state_is_populated():
    blob = torch.load(_CACHE, map_location="cpu", weights_only=False)
    ego = np.stack(
        [np.asarray(s["ego_state"], dtype=np.float64) for s in blob["samples"]], axis=0
    )
    assert ego.shape[1] == 5, "ego_state must be [vx, vy, ax, ay, yaw_rate]"

    # The exact regression: a kinematic column identically zero everywhere.
    for idx, name in enumerate(("vx", "vy", "ax", "ay", "yaw_rate")):
        assert np.any(np.abs(ego[:, idx]) > 1e-6), f"ego_state column {name!r} is all-zero"

    # vx (slot 0) is forward speed: predominantly non-negative, plausible scale.
    vx = ego[:, 0]
    assert np.mean(vx >= -0.5) > 0.95, "vx should be a (mostly non-negative) forward speed"
    assert 0.5 < np.percentile(vx, 50) < 30.0, "median vx outside a plausible m/s range"
