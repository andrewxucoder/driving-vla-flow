"""NAVSIM 5-policy evaluator entry point.

Drives a :class:`BasePolicy` (Regression / Flow / Flow-BoN / Diffusion /
Diffusion-BoN) through every scenario in a NAVSIM scene loader, accumulating
per-scenario metrics into a single :class:`NavSimEvalReport`.

The runner is dataset-shape agnostic — it gets its scenes from any
:class:`NavSimSceneLoader` (real devkit-backed or a fake test loader). Each
policy variant is described by a :class:`PolicyEntry`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from driving_vla.data.command_labeling import COMMAND_TO_ID, infer_command
from driving_vla.data.coordinate_transform import global_to_ego_frame
from driving_vla.data.navsim_adapter import (
    NavSimScene,
    NavSimSceneFrame,
    NavSimSceneLoader,
    ego_state_from_frame,
)
from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.evaluation.metrics import ade, collision_proxy, fde
from driving_vla.evaluation.reward import trajectory_reward
from driving_vla.policies.base_policy import BasePolicy


@dataclass
class PolicyEntry:
    """Describes one of the 5 policies under evaluation.

    ``predict`` produces an ``[H, 2]`` ego-frame trajectory. For BoN routes the
    caller wraps a stochastic policy + reward to pick the best of N draws.
    """

    name: str
    predict: Callable[[DrivingTrajectorySample], torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavSimEvalReport:
    per_policy: dict[str, dict[str, float]] = field(default_factory=dict)
    scenario_ids: list[str] = field(default_factory=list)
    horizon_frames: int = 0


def make_bon_predictor(
    policy: BasePolicy,
    *,
    n: int,
    score_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
    constraint_extractor: Callable[[DrivingTrajectorySample], list[Any]] | None = None,
) -> Callable[[DrivingTrajectorySample], torch.Tensor]:
    """Wrap a stochastic policy as a BoN predictor.

    Draws ``n`` samples, scores them with ``score_fn``, returns the best.

    The condition dict passed to ``score_fn`` carries ``command_id`` and (when
    available on the sample) ``ego_state`` — the latter lets a progress-aware
    score reject overshoot candidates that ignore the current vehicle speed.

    M8.2: if ``constraint_extractor`` is supplied, its per-observation output
    (a ``list[TrajectoryConstraint]``) is placed in ``cond["constraints"]``. A
    constraint-aware ``score_fn`` (see
    ``driving_vla.vla.build_constraint_aware_score_fn``) reads it and rewards
    constraint-satisfying candidates. The extractor is typed loosely
    (``Callable``) so the evaluation layer stays free of a hard import on the
    VLA layer — dependency flows vla → evaluation, never the reverse.
    """

    def _predict(sample: DrivingTrajectorySample) -> torch.Tensor:
        chunks = policy.predict_chunk_n(sample, n=n)  # [N, H, 2]
        cond: dict[str, Any] = {}
        if sample.command_id is not None:
            cond["command_id"] = torch.tensor([int(sample.command_id)], dtype=torch.long)
        if sample.ego_state is not None:
            cond["ego_state"] = sample.ego_state.unsqueeze(0)  # [1, state_dim]
        if constraint_extractor is not None:
            cond["constraints"] = constraint_extractor(sample)
        scores = score_fn(chunks, cond)
        return chunks[scores.argmax()]

    return _predict


def build_default_score_fn(
    horizon_seconds: float = 4.0,
) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """Default BoN score = :func:`trajectory_reward`.

    Broadcasts ``command_id`` and ``ego_state`` to match the per-sample axis.
    When ``ego_state`` is present in ``cond``, the ``progress_match`` term in
    ``trajectory_reward`` is enabled with ``horizon_seconds`` (default 4.0,
    matching the NAVSIM 4-second prediction window).

    Handles three call shapes for the per-condition tensors:

    - missing key                  → fill (``command_id=0`` / ``ego_state=zeros``).
    - shape[0] == samples.shape[0] → 1-to-1 match (single-shot BoN).
    - shape[0] divides samples     → repeat-interleave per batch item (callers
      that reshape ``[B, N, H, D]`` → ``[B*N, H, D]``).
    """

    def _broadcast_to_n(t: torch.Tensor, n: int, *, name: str) -> torch.Tensor:
        if t.shape[0] == n:
            return t
        if n % t.shape[0] == 0:
            return t.repeat_interleave(n // t.shape[0], dim=0)
        raise ValueError(
            f"score_fn received {n} samples with {name} of length "
            f"{t.shape[0]} (not a divisor of {n})"
        )

    def _score(samples: torch.Tensor, cond: dict[str, Any]) -> torch.Tensor:
        n = samples.shape[0]
        command_id = cond.get("command_id")
        if command_id is None:
            command_id = torch.zeros(n, dtype=torch.long, device=samples.device)
        else:
            command_id = _broadcast_to_n(command_id, n, name="command_id").to(samples.device)

        ego_state = cond.get("ego_state")
        if ego_state is not None:
            ego_state = _broadcast_to_n(ego_state, n, name="ego_state").to(samples.device)

        return trajectory_reward(
            samples,
            command_id,
            ego_state=ego_state,
            horizon_seconds=horizon_seconds if ego_state is not None else None,
        )

    return _score


def _scene_to_sample(scene: NavSimScene, *, history_frames: int) -> DrivingTrajectorySample:
    cur = scene.frames[scene.current_index]
    cur_pose = np.array(
        [float(cur.ego_xy[0]), float(cur.ego_xy[1]), float(cur.ego_heading)],
        dtype=np.float64,
    )
    history_np = _build_history(scene, cur_pose, history_frames=history_frames)
    history_t = (
        torch.from_numpy(history_np)
        if history_np is not None
        else torch.zeros(0, 2, dtype=torch.float32)
    )
    # Canonical [vx, vy, ax, ay, yaw_rate] ego-frame layout, shared with the
    # nuPlan training cache and the NavSim dataset adapter via the single
    # ego_state_from_frame helper (see tech_report §7.5 — the prior layout
    # mismatch was the true cause of the offline→NAVSIM gap).
    ego_state = torch.from_numpy(ego_state_from_frame(cur))
    # Heuristic command from a short look-ahead future stub (8 steps) of GT.
    look = _build_future(scene, cur_pose, future_frames=min(8, len(scene.frames) - scene.current_index - 1))
    cmd_name = infer_command(look) if look.size > 0 else "keep_lane"
    return DrivingTrajectorySample(
        history=history_t,
        future=torch.from_numpy(_build_future(scene, cur_pose, future_frames=8)).float(),
        instruction=None,
        command_id=COMMAND_TO_ID[cmd_name],
        ego_state=ego_state,
        metadata={"scenario_id": scene.token, "log_name": scene.log_name},
    )


def _build_history(
    scene: NavSimScene,
    cur_pose: np.ndarray,
    *,
    history_frames: int,
) -> np.ndarray | None:
    if history_frames == 0:
        return None
    start = max(0, scene.current_index - history_frames)
    past = scene.frames[start : scene.current_index]
    if not past:
        return np.zeros((history_frames, 2), dtype=np.float32)
    past_xy = np.stack([f.ego_xy for f in past], axis=0).astype(np.float64)
    past_ego = global_to_ego_frame(past_xy, cur_pose).astype(np.float32)
    if past_ego.shape[0] < history_frames:
        pad = np.zeros((history_frames - past_ego.shape[0], 2), dtype=np.float32)
        past_ego = np.concatenate([pad, past_ego], axis=0)
    return past_ego


def _build_future(
    scene: NavSimScene,
    cur_pose: np.ndarray,
    *,
    future_frames: int,
) -> np.ndarray:
    end = scene.current_index + 1 + future_frames
    fut = scene.frames[scene.current_index + 1 : end]
    if not fut:
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.stack([f.ego_xy for f in fut], axis=0).astype(np.float64)
    return global_to_ego_frame(xy, cur_pose).astype(np.float32)


def _resample_horizon(pred: torch.Tensor, target_horizon: int) -> torch.Tensor:
    """Evenly-spaced index resample from ``[H_src, D]`` to ``[H_dst, D]``.

    Both endpoints (t=0 boundary and t=4s boundary) align. Used when the
    policy's prediction horizon differs from the eval ground-truth horizon
    even though both span the same wall-clock duration — common when models
    train at one Hz and evaluate at another (e.g. nuPlan 7.5 Hz → NAVSIM 2 Hz).
    """
    h_src = pred.shape[0]
    if h_src == target_horizon:
        return pred
    idx = torch.linspace(0, h_src - 1, target_horizon).round().long()
    return pred.index_select(0, idx)


def evaluate_policies(
    policies: list[PolicyEntry],
    scene_loader: NavSimSceneLoader,
    *,
    history_frames: int = 4,
    future_frames: int = 8,
) -> NavSimEvalReport:
    """Run every ``policies`` entry over every scene in ``scene_loader``."""
    tokens = list(scene_loader.tokens())
    report = NavSimEvalReport(scenario_ids=list(tokens), horizon_frames=future_frames)

    for entry in policies:
        ade_sum = 0.0
        fde_sum = 0.0
        coll_sum = 0.0
        n_scenes = 0
        for token in tokens:
            scene = scene_loader.get_scene(token)
            sample = _scene_to_sample(scene, history_frames=history_frames)
            gt_future = torch.from_numpy(
                _build_future(scene, np.array(
                    [float(scene.frames[scene.current_index].ego_xy[0]),
                     float(scene.frames[scene.current_index].ego_xy[1]),
                     float(scene.frames[scene.current_index].ego_heading)],
                    dtype=np.float64,
                ), future_frames=future_frames),
            ).float()
            if gt_future.shape[0] < future_frames:
                # Skip scenes too short to evaluate; still count toward total.
                continue
            pred = entry.predict(sample)
            if pred.shape != gt_future.shape:
                # Predictions and ground truth span the same duration but may
                # use different sampling rates — uniformly resample the
                # prediction onto the GT grid before scoring.
                pred = _resample_horizon(pred, gt_future.shape[0])
            ade_sum += float(ade(pred.unsqueeze(0), gt_future.unsqueeze(0)).item())
            fde_sum += float(fde(pred.unsqueeze(0), gt_future.unsqueeze(0)).item())
            coll_sum += float(collision_proxy(pred.unsqueeze(0)).item())
            n_scenes += 1

        if n_scenes == 0:
            report.per_policy[entry.name] = {"n_scenes": 0.0}
            continue
        report.per_policy[entry.name] = {
            "n_scenes": float(n_scenes),
            "ade_mean": ade_sum / n_scenes,
            "fde_mean": fde_sum / n_scenes,
            "collision_rate": coll_sum / n_scenes,
            "geometric_score": (
                1.0 - min(1.0, ade_sum / n_scenes / 5.0) - 0.5 * coll_sum / n_scenes
            ),
        }
    return report


__all__ = [
    "NavSimEvalReport",
    "NavSimSceneFrame",
    "PolicyEntry",
    "build_default_score_fn",
    "evaluate_policies",
    "make_bon_predictor",
]
