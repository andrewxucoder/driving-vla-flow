"""High-level driving command vocabulary and heuristic labeller.

The command vocabulary is a small discrete set used as the primary conditioning
signal for non-VLM heads. It is intentionally narrow (4 classes) — enough to
exercise conditional policy heads without committing to a full driving ontology.
"""

from __future__ import annotations

import numpy as np


COMMAND_VOCAB: tuple[str, ...] = (
    "keep_lane",
    "change_left",
    "change_right",
    "slow_down",
)

COMMAND_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(COMMAND_VOCAB)}

# Standard nuScenes 6-camera rig channel names.
CAMERA_CHANNELS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def infer_command(
    future: np.ndarray,
    slow_down_ratio: float = 0.65,
    lane_change_threshold: float = 1.5,
) -> str:
    """Infer a high-level command from an ego-frame future trajectory.

    Priority: slow_down > change_left/right > keep_lane.

    Args:
        future: shape [N, 2], x is forward, y is lateral, in metres.
        slow_down_ratio: if late-segment mean step distance falls below this
            fraction of the early-segment mean, classify as slow_down.
        lane_change_threshold: absolute lateral displacement (metres) over the
            horizon above which we classify a left/right lane change.
    """
    if future.shape[0] < 2:
        return "keep_lane"

    step = np.diff(future, axis=0)
    step_dist = np.linalg.norm(step, axis=-1)

    if step_dist.shape[0] >= 4:
        third = max(1, step_dist.shape[0] // 3)
        early_mean = float(np.mean(step_dist[:third]))
        late_mean = float(np.mean(step_dist[-third:]))
        if early_mean > 1e-3 and late_mean < slow_down_ratio * early_mean:
            return "slow_down"

    lateral_delta = float(future[-1, 1] - future[0, 1])
    if lateral_delta > lane_change_threshold:
        return "change_left"
    if lateral_delta < -lane_change_threshold:
        return "change_right"
    return "keep_lane"


__all__ = [
    "CAMERA_CHANNELS",
    "COMMAND_TO_ID",
    "COMMAND_VOCAB",
    "infer_command",
]
