"""Closed-loop simulator for NAVSIM-style geometric eval.

Implements ``embodied_flow_core.ClosedLoopSimulator`` structurally (Protocol
membership verifiable via ``isinstance``) without importing
``embodied_flow_core.Observation`` — per R2, v2 keeps its own driving-specific
observation dataclass to avoid pulling in em-wm's ``state``-field schema.

The simulator advances through a pre-recorded NAVSIM scene one frame at a time.
Each ``step(action)`` logs the policy action and proceeds to the next ground-
truth frame; per-step metrics (collision proxy, lateral offset, etc.) are
computed on the cumulative predicted trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from driving_vla.data.navsim_adapter import NavSimScene, NavSimSceneLoader


@dataclass
class DrivingObservation:
    """v2's driving-specific observation (deliberately decoupled from em-wm's)."""

    state: torch.Tensor                          # [state_dim], ego frame
    images: torch.Tensor | None = None           # [V, 3, H, W] when cameras present
    instruction: str | None = None
    scenario_id: str | None = None
    timestep: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DrivingRolloutResult:
    observations: list[DrivingObservation] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    failed: bool = False
    failure_reason: str | None = None


class NavSimClosedLoopSimulator:
    """ClosedLoopSimulator over a NAVSIM scene sequence.

    Action semantics: each ``step(action)`` accepts a ``[H, 2]`` planned chunk
    or ``[2]`` single waypoint; the simulator stores the first step and treats
    the next ground-truth frame as the new observation. This is "closed-loop"
    in name only — NAVSIM scenes are pre-recorded, so it is actually a
    geometric-proxy evaluation aligned with the upstream NAVSIM convention.
    """

    def __init__(
        self,
        scene_loader: NavSimSceneLoader,
        *,
        max_horizon: int = 40,
    ) -> None:
        if max_horizon <= 0:
            raise ValueError(f"max_horizon must be > 0, got {max_horizon}")
        self.scene_loader = scene_loader
        self.max_horizon = max_horizon
        self._scene: NavSimScene | None = None
        self._cursor: int = 0
        self._predicted: list[torch.Tensor] = []

    def reset(self, scenario_id: str | None = None) -> DrivingObservation:
        tokens = list(self.scene_loader.tokens())
        if not tokens:
            raise RuntimeError("scene_loader produced no tokens")
        token = scenario_id if scenario_id is not None else tokens[0]
        if token not in tokens:
            raise KeyError(f"scenario_id {token!r} not found in loader tokens")

        scene = self.scene_loader.get_scene(token)
        self._scene = scene
        self._cursor = scene.current_index
        self._predicted = []
        return self._frame_to_observation(self._cursor, timestep=0)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[DrivingObservation, dict[str, float], bool]:
        if self._scene is None:
            raise RuntimeError("step() called before reset()")
        if action.ndim == 2:
            waypoint = action[0]
        elif action.ndim == 1:
            waypoint = action
        else:
            raise ValueError(f"action expected [H, 2] or [2], got {tuple(action.shape)}")
        if waypoint.shape[-1] != 2:
            raise ValueError(f"action last-dim must be 2, got {tuple(waypoint.shape)}")

        self._predicted.append(waypoint.detach().cpu())
        self._cursor += 1
        timestep = len(self._predicted)

        # done means "no further step is possible": cursor sits on the last
        # frame, so the next step() would advance out of bounds.
        done = (
            self._cursor >= len(self._scene.frames) - 1
            or timestep >= self.max_horizon
        )
        obs = self._frame_to_observation(
            min(self._cursor, len(self._scene.frames) - 1),
            timestep=timestep,
        )
        per_step_metrics = {
            "predicted_x": float(waypoint[0].item()),
            "predicted_y": float(waypoint[1].item()),
        }
        return obs, per_step_metrics, done

    def rollout(
        self,
        policy: Callable[[DrivingObservation], torch.Tensor],
        scenario_id: str | None = None,
        horizon: int = 20,
    ) -> DrivingRolloutResult:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        obs = self.reset(scenario_id)
        result = DrivingRolloutResult()
        result.observations.append(obs)
        for _ in range(horizon):
            action = policy(obs)
            try:
                obs, _, done = self.step(action)
            except Exception as exc:  # noqa: BLE001 — record and continue
                result.failed = True
                result.failure_reason = repr(exc)
                break
            result.observations.append(obs)
            result.actions.append(action.detach().cpu())
            if done:
                break

        if self._predicted:
            traj = torch.stack(self._predicted, dim=0)
            result.metrics["predicted_steps"] = float(traj.shape[0])
            result.metrics["lateral_max"] = float(traj[:, 1].abs().max().item())
        return result

    def _frame_to_observation(self, idx: int, *, timestep: int) -> DrivingObservation:
        assert self._scene is not None
        frame = self._scene.frames[idx]
        state = torch.tensor(
            [
                float(frame.ego_xy[0]),
                float(frame.ego_xy[1]),
                float(frame.ego_heading),
            ],
            dtype=torch.float32,
        )
        return DrivingObservation(
            state=state,
            instruction=None,
            scenario_id=self._scene.token,
            timestep=timestep,
            metadata={"log_name": self._scene.log_name},
        )


__all__ = [
    "DrivingObservation",
    "DrivingRolloutResult",
    "NavSimClosedLoopSimulator",
]
