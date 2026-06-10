"""Route data abstraction across map providers.

The :class:`RouteProvider` Protocol decouples the policy stack from any
specific map devkit. A real loader (nuPlan lanelet2, nuScenes map_api,
OpenDRIVE) will produce route waypoints from the global plan + lane graph.

This module also ships two non-devkit utilities:

- :func:`route_from_future` derives a pseudo-route from the ground-truth
  future trajectory by subsampling + linear extrapolation. Useful for
  ablation studies and unit tests *before* the real map adapter is wired in.
  **Do not use this in production training** — it cheats by exposing future
  GT to the encoder, which would leak the answer.

- :func:`pad_route_to_length` produces a fixed-length ``[max_n, 2]`` tensor
  plus a ``[max_n]`` validity mask, which the M7 ``RouteEncoder`` consumes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class RouteProvider(Protocol):
    """Anything that turns an observation into a route polyline.

    Implementations live in the data adapter layer (one per dataset). The
    encoder layer only cares that the returned shape is ``[N, 2]`` in the
    ego frame.
    """

    def get_route(self, *, scenario_id: str, ego_pose: np.ndarray) -> np.ndarray:
        """Returns ``[N, 2]`` ego-frame waypoints.

        ``ego_pose`` is ``[ego_x, ego_y, ego_heading_rad]`` in the global
        frame. Implementations should return at least one waypoint; if the
        scenario has no route info, return ``np.zeros((0, 2), dtype=float32)``
        (the encoder treats empty routes as "no route available").
        """
        ...


def route_from_future(
    future: np.ndarray,
    *,
    num_waypoints: int = 20,
    lookahead_m: float = 100.0,
) -> np.ndarray:
    """Derive a pseudo-route from a GT future trajectory.

    Subsamples ``future`` to ``num_waypoints`` evenly along arc length, then
    linearly extrapolates the last segment out to ``lookahead_m`` total
    distance from the start. The result lives in the same ego frame as
    ``future``.

    Args:
        future: ``[T, 2]`` ego-frame ground-truth future.
        num_waypoints: target output length. Output is exactly this size.
        lookahead_m: total along-route distance from waypoint 0 to the last
            extrapolated point. If the GT future already covers more than
            this, only the prefix is used.

    Returns:
        ``[num_waypoints, 2]`` float32 ego-frame polyline.
    """
    if future.ndim != 2 or future.shape[-1] != 2:
        raise ValueError(f"future expected [T, 2], got {tuple(future.shape)}")
    if num_waypoints <= 0:
        raise ValueError(f"num_waypoints must be > 0, got {num_waypoints}")
    if lookahead_m <= 0:
        raise ValueError(f"lookahead_m must be > 0, got {lookahead_m}")
    if future.shape[0] < 2:
        # Degenerate input — emit a straight ahead route.
        x = np.linspace(0.0, lookahead_m, num_waypoints, dtype=np.float32)
        y = np.zeros(num_waypoints, dtype=np.float32)
        return np.stack([x, y], axis=-1)

    fut = future.astype(np.float64, copy=False)
    seg = np.diff(fut, axis=0)
    seg_lens = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    arc_total = float(cum[-1])

    if arc_total < 1e-3:
        # Future barely moves — stretch a straight line forward.
        x = np.linspace(0.0, lookahead_m, num_waypoints, dtype=np.float32)
        y = np.zeros(num_waypoints, dtype=np.float32)
        return np.stack([x, y], axis=-1)

    # Sample evenly along arc length out to lookahead_m. Points inside the GT
    # arc length are interpolated; points beyond it are linearly extrapolated
    # along the final GT segment below. When the GT future is longer than
    # lookahead_m this samples only its prefix (no extrapolation needed).
    sample_arcs = np.linspace(0.0, lookahead_m, num_waypoints, dtype=np.float64)
    xs = np.interp(sample_arcs, cum, fut[:, 0])
    ys = np.interp(sample_arcs, cum, fut[:, 1])

    if lookahead_m > arc_total + 1e-3:
        # np.interp clamps out-of-range samples to the last GT point; replace
        # those with a linear extrapolation along the final GT segment.
        dx = float(fut[-1, 0] - fut[-2, 0])
        dy = float(fut[-1, 1] - fut[-2, 1])
        d = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / d, dy / d
        beyond = sample_arcs > arc_total
        delta = sample_arcs[beyond] - arc_total
        xs[beyond] = fut[-1, 0] + ux * delta
        ys[beyond] = fut[-1, 1] + uy * delta
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def pad_route_to_length(
    route: np.ndarray | torch.Tensor,
    max_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a route to a fixed length and return ``(route, mask)``.

    Returns:
        - ``route``: ``[max_n, 2]`` float32 tensor (truncated if longer).
        - ``mask``:  ``[max_n]`` int8 tensor, 1 for valid waypoints, 0 for pad.
    """
    if isinstance(route, np.ndarray):
        route = torch.from_numpy(route)
    if route.ndim != 2 or route.shape[-1] != 2:
        raise ValueError(f"route expected [N, 2], got {tuple(route.shape)}")
    if max_n <= 0:
        raise ValueError(f"max_n must be > 0, got {max_n}")

    n = route.shape[0]
    if n >= max_n:
        return (
            route[:max_n].to(dtype=torch.float32).contiguous(),
            torch.ones(max_n, dtype=torch.int8),
        )

    padded = torch.zeros(max_n, 2, dtype=torch.float32)
    padded[:n] = route.to(dtype=torch.float32)
    mask = torch.zeros(max_n, dtype=torch.int8)
    mask[:n] = 1
    return padded, mask


__all__ = ["RouteProvider", "pad_route_to_length", "route_from_future"]
