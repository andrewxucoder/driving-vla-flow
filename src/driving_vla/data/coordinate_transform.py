from __future__ import annotations

import numpy as np


def _pose_components(ego_pose: np.ndarray, points_ndim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ego_pose.shape[-1] != 3:
        raise ValueError("ego_pose 的最后一维必须是 [x, y, heading]")

    if ego_pose.ndim == 1:
        return ego_pose[0], ego_pose[1], ego_pose[2]

    extra_dims = max(points_ndim - ego_pose.ndim, 0)
    view_shape = ego_pose.shape[:-1] + (1,) * extra_dims
    return (
        ego_pose[..., 0].reshape(view_shape),
        ego_pose[..., 1].reshape(view_shape),
        ego_pose[..., 2].reshape(view_shape),
    )


def global_to_ego_frame(points: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """将全局坐标点转换到 ego 坐标系。

    `points` 的最后一维为 `[x, y]`，`ego_pose` 的最后一维为
    `[x, y, heading]`。支持形如 `[N, 2]`、`[B, N, 2]`
    的点集，以及 `[3]`、`[B, 3]` 的 ego 位姿。
    """
    points_arr = np.asarray(points, dtype=np.float64)
    ego_pose_arr = np.asarray(ego_pose, dtype=np.float64)
    if points_arr.shape[-1] != 2:
        raise ValueError("points 的最后一维必须是 [x, y]")

    ego_x, ego_y, heading = _pose_components(ego_pose_arr, points_arr.ndim)
    dx = points_arr[..., 0] - ego_x
    dy = points_arr[..., 1] - ego_y
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    ego_points = np.empty_like(points_arr, dtype=np.float64)
    ego_points[..., 0] = cos_h * dx + sin_h * dy
    ego_points[..., 1] = -sin_h * dx + cos_h * dy
    return ego_points


def ego_to_global_frame(points: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """将 ego 坐标系点转换回全局坐标。"""
    points_arr = np.asarray(points, dtype=np.float64)
    ego_pose_arr = np.asarray(ego_pose, dtype=np.float64)
    if points_arr.shape[-1] != 2:
        raise ValueError("points 的最后一维必须是 [x, y]")

    ego_x, ego_y, heading = _pose_components(ego_pose_arr, points_arr.ndim)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    global_points = np.empty_like(points_arr, dtype=np.float64)
    global_points[..., 0] = cos_h * points_arr[..., 0] - sin_h * points_arr[..., 1] + ego_x
    global_points[..., 1] = sin_h * points_arr[..., 0] + cos_h * points_arr[..., 1] + ego_y
    return global_points
