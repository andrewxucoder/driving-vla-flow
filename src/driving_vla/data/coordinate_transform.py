"""Global-to-ego-frame 2D coordinate transforms.

Used by NAVSIM / nuPlan adapters to express past + future poses in the ego's
current heading frame, which is the convention every policy head expects.
"""

from __future__ import annotations

import numpy as np


def global_to_ego_frame(
    points_xy: np.ndarray,
    ego_pose: np.ndarray,
) -> np.ndarray:
    """Rotate-and-translate global-frame XY points into the ego frame.

    Args:
        points_xy: shape [N, 2], world-frame xy in metres.
        ego_pose: shape [3], [ego_x, ego_y, ego_heading_rad]. Heading is the
            angle of the ego's forward axis relative to world +x.

    Returns:
        shape [N, 2], with x = forward distance, y = leftward distance
        relative to the ego at its current pose.
    """
    if points_xy.ndim != 2 or points_xy.shape[-1] != 2:
        raise ValueError(f"points_xy must be [N, 2], got {tuple(points_xy.shape)}")
    if ego_pose.shape != (3,):
        raise ValueError(f"ego_pose must be [3], got {tuple(ego_pose.shape)}")

    cos_h = float(np.cos(ego_pose[2]))
    sin_h = float(np.sin(ego_pose[2]))

    dx = points_xy[:, 0] - float(ego_pose[0])
    dy = points_xy[:, 1] - float(ego_pose[1])

    ego_x = cos_h * dx + sin_h * dy
    ego_y = -sin_h * dx + cos_h * dy

    return np.stack([ego_x, ego_y], axis=-1)


__all__ = ["global_to_ego_frame"]
