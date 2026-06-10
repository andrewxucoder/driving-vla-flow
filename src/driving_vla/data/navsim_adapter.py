"""NAVSIM scene → trajectory sample adapter.

The adapter never imports the ``navsim`` package at module load. NAVSIM's
``Scene`` / ``Frame`` objects are normalised into the internal dataclasses
:class:`NavSimScene` / :class:`NavSimSceneFrame` by a loader that satisfies the
:class:`NavSimSceneLoader` ``Protocol``. Tests construct these dataclasses
directly and exercise the full coordinate transform + command-labelling path
without needing the navsim devkit installed.

The real wrapper :class:`_NavSimDevkitLoader` is only instantiated when the
caller does *not* pass a ``scene_loader``; that path lazy-imports navsim and
raises a clear :class:`ImportError` with setup pointers if it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import Dataset

from driving_vla.data.command_labeling import COMMAND_TO_ID, infer_command
from driving_vla.data.coordinate_transform import global_to_ego_frame
from driving_vla.data.instruction_templates import resolve_instruction


_VALID_SPLITS: frozenset[str] = frozenset({"navtest", "navtrain", "mini"})


@dataclass
class NavSimSceneFrame:
    """Per-frame fields the adapter consumes. Coordinates are in the global frame."""

    ego_xy: np.ndarray              # shape (2,), metres
    ego_heading: float              # radians
    velocity_xy: np.ndarray | None = None
    acceleration_xy: np.ndarray | None = None
    yaw_rate: float = 0.0
    camera_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class NavSimScene:
    token: str
    log_name: str
    frames: list[NavSimSceneFrame]
    current_index: int  # index in `frames` that is "now"


@runtime_checkable
class NavSimSceneLoader(Protocol):
    """Anything that produces :class:`NavSimScene` objects."""

    def tokens(self) -> Iterable[str]: ...
    def get_scene(self, token: str) -> NavSimScene: ...


def ego_state_from_frame(frame: NavSimSceneFrame) -> np.ndarray:
    """``[vx, vy, ax, ay, yaw_rate]`` in the ego frame.

    NavSim's ``velocity_xy`` / ``acceleration_xy`` are *already* in the ego
    (vehicle-body) frame — verified empirically: for forward-driving scenes
    ``velocity_xy ≈ (speed, ~0)`` and matches the +x future motion. This is the
    same convention as the nuPlan training cache (devkit
    ``rear_axle_velocity_2d``), so the components are used directly. (An earlier
    version rotated them by the heading, which double-transformed an already
    ego-frame vector and scrambled vx — the cause of the Round-2 NAVSIM
    regression; see tech_report §7.5.) Both the dataset and the eval runner go
    through this one function so train/eval layouts can't silently diverge.
    """
    vel = (
        np.asarray(frame.velocity_xy, dtype=np.float64)
        if frame.velocity_xy is not None
        else np.zeros(2, dtype=np.float64)
    )
    acc = (
        np.asarray(frame.acceleration_xy, dtype=np.float64)
        if frame.acceleration_xy is not None
        else np.zeros(2, dtype=np.float64)
    )
    return np.array(
        [vel[0], vel[1], acc[0], acc[1], float(frame.yaw_rate)],
        dtype=np.float32,
    )


class NavSimTrajectoryDataset(Dataset):
    """NAVSIM-backed dataset yielding dicts with the same shape as nuPlan/nuScenes adapters."""

    STATE_DIM: int = 5      # vx, vy, ax, ay, yaw_rate (ego frame)
    HISTORY_DIM: int = 2    # xy in ego frame

    def __init__(
        self,
        data_root: str | Path,
        split: str = "navtest",
        history_seconds: float = 2.0,
        future_seconds: float = 4.0,
        sample_rate_hz: float = 2.0,
        include_camera: bool = False,
        scene_loader: NavSimSceneLoader | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
        if history_seconds < 0:
            raise ValueError(f"history_seconds must be ≥ 0, got {history_seconds}")
        if future_seconds <= 0:
            raise ValueError(f"future_seconds must be > 0, got {future_seconds}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")

        self.data_root = Path(data_root)
        self.split = split
        self.history_seconds = float(history_seconds)
        self.future_seconds = float(future_seconds)
        self.sample_rate_hz = float(sample_rate_hz)
        self.history_frames = int(round(history_seconds * sample_rate_hz))
        self.future_frames = int(round(future_seconds * sample_rate_hz))
        self.include_camera = bool(include_camera)

        if scene_loader is None:
            scene_loader = _NavSimDevkitLoader(
                data_root=self.data_root,
                split=split,
                history_frames=self.history_frames,
                future_frames=self.future_frames,
                include_camera=self.include_camera,
            )
        self._loader: NavSimSceneLoader = scene_loader
        self._tokens: list[str] = list(scene_loader.tokens())

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < len(self._tokens):
            raise IndexError(f"index {idx} out of range for size {len(self._tokens)}")
        scene = self._loader.get_scene(self._tokens[idx])
        return self._scene_to_sample(scene)

    def scene_at(self, idx: int) -> NavSimScene:
        """Return the raw :class:`NavSimScene` without the tensor packing overhead."""
        if not 0 <= idx < len(self._tokens):
            raise IndexError(f"index {idx} out of range for size {len(self._tokens)}")
        return self._loader.get_scene(self._tokens[idx])

    def scenario_ids(self) -> list[str]:
        return list(self._tokens)

    def _scene_to_sample(self, scene: NavSimScene) -> dict[str, Any]:
        n_frames = len(scene.frames)
        if not 0 <= scene.current_index < n_frames:
            raise ValueError(
                f"scene {scene.token!r}: current_index={scene.current_index} "
                f"out of bounds [0, {n_frames})"
            )
        cur = scene.frames[scene.current_index]
        cur_pose = np.array(
            [float(cur.ego_xy[0]), float(cur.ego_xy[1]), float(cur.ego_heading)],
            dtype=np.float64,
        )

        history_np = self._build_history(scene, cur_pose)
        future_np = self._build_future(scene, cur_pose)
        ego_state_np = self._build_ego_state(cur)

        command_name = infer_command(future_np)
        command_id_int = COMMAND_TO_ID[command_name]

        metadata: dict[str, Any] = {
            "scenario_id": scene.token,
            "log_name": scene.log_name,
            "source": "navsim",
            "split": self.split,
            "command_name": command_name,
        }
        if self.include_camera:
            metadata["camera_paths"] = dict(cur.camera_paths)

        history = torch.from_numpy(history_np) if history_np is not None else None
        future = torch.from_numpy(future_np)
        ego_state = torch.from_numpy(ego_state_np)
        command_id = torch.tensor(command_id_int, dtype=torch.long)
        instruction = resolve_instruction(
            raw=None,
            command_id=command_id_int,
            seed=hash(scene.token) & 0xFFFFFFFF,
        )

        condition: dict[str, Any] = {
            "ego_state": ego_state,
            "history": history,
            "command_id": command_id,
            "instruction": instruction,
            "metadata": metadata,
        }
        return {
            "condition": condition,
            "future": future,
            "ego_state": ego_state,
            "history": history,
            "command_id": command_id,
            "instruction": instruction,
            "metadata": metadata,
        }

    def _build_history(self, scene: NavSimScene, cur_pose: np.ndarray) -> np.ndarray | None:
        if self.history_frames == 0:
            return None
        start = max(0, scene.current_index - self.history_frames)
        past = scene.frames[start : scene.current_index]
        if not past:
            return np.zeros((self.history_frames, self.HISTORY_DIM), dtype=np.float32)
        past_xy = np.stack([f.ego_xy for f in past], axis=0).astype(np.float64)
        past_ego = global_to_ego_frame(past_xy, cur_pose).astype(np.float32)
        if past_ego.shape[0] < self.history_frames:
            pad = np.zeros(
                (self.history_frames - past_ego.shape[0], self.HISTORY_DIM),
                dtype=np.float32,
            )
            past_ego = np.concatenate([pad, past_ego], axis=0)
        return past_ego

    def _build_future(self, scene: NavSimScene, cur_pose: np.ndarray) -> np.ndarray:
        end = scene.current_index + 1 + self.future_frames
        future = scene.frames[scene.current_index + 1 : end]
        if len(future) < self.future_frames:
            raise ValueError(
                f"scene {scene.token!r}: only {len(future)} future frames after "
                f"current index {scene.current_index} (need {self.future_frames})"
            )
        future_xy = np.stack([f.ego_xy for f in future], axis=0).astype(np.float64)
        return global_to_ego_frame(future_xy, cur_pose).astype(np.float32)

    def _build_ego_state(self, frame: NavSimSceneFrame) -> np.ndarray:
        return ego_state_from_frame(frame)


class _NavSimDevkitLoader:
    """Wraps the real ``navsim`` SceneLoader and converts scenes to :class:`NavSimScene`."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        history_frames: int,
        future_frames: int,
        include_camera: bool,
    ) -> None:
        try:
            from navsim.common.dataclasses import SceneFilter, SensorConfig  # type: ignore
            from navsim.common.dataloader import SceneLoader  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "NavSimTrajectoryDataset requires `navsim` + `nuplan-devkit`. "
                "See docs/DATASETS.md for setup. "
                f"Underlying error: {exc}"
            ) from exc

        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(
                f"OpenScene data_root missing: {data_root}. "
                "Set it to a directory containing navsim_logs/{split}/ and sensor_blobs/{split}/."
            )

        logs_dir = data_root / "navsim_logs" / split
        sensor_dir = data_root / "sensor_blobs" / split
        if not logs_dir.exists():
            raise FileNotFoundError(f"NAVSIM logs missing: {logs_dir}")
        if include_camera and not sensor_dir.exists():
            raise FileNotFoundError(
                f"sensor_blobs missing while include_camera=True: {sensor_dir}"
            )
        if not sensor_dir.exists():
            sensor_dir.mkdir(parents=True, exist_ok=True)

        scene_filter = SceneFilter(
            num_history_frames=history_frames + 1,
            num_future_frames=future_frames,
        )
        sensor_config = (
            SensorConfig.build_all_sensors() if include_camera else SensorConfig.build_no_sensors()
        )
        self._scene_loader = SceneLoader(
            data_path=logs_dir,
            original_sensor_path=sensor_dir,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
        self._history_frames = history_frames
        self._include_camera = include_camera

    def tokens(self) -> Iterable[str]:
        raw = getattr(self._scene_loader, "tokens")
        return list(raw() if callable(raw) else raw)

    def get_scene(self, token: str) -> NavSimScene:
        scene = self._scene_loader.get_scene_from_token(token)
        meta = getattr(scene, "scene_metadata", scene)
        log_name = str(getattr(meta, "log_name", "") or "")
        scene_token = (
            getattr(meta, "scene_token", None)
            or getattr(meta, "initial_token", None)
            or token
        )
        frames = [self._convert_frame(f) for f in scene.frames]
        return NavSimScene(
            token=str(scene_token),
            log_name=log_name,
            frames=frames,
            current_index=self._history_frames,
        )

    def _convert_frame(self, frame: Any) -> NavSimSceneFrame:
        ego_status = getattr(frame, "ego_status", frame)
        pose = np.asarray(ego_status.ego_pose, dtype=np.float64)
        if pose.shape[-1] != 3:
            raise ValueError(f"unexpected navsim ego_pose shape {pose.shape}; expected (..., 3)")
        velocity = np.asarray(getattr(ego_status, "ego_velocity", np.zeros(2)), dtype=np.float64)
        accel = np.asarray(getattr(ego_status, "ego_acceleration", np.zeros(2)), dtype=np.float64)
        yaw_rate = float(getattr(ego_status, "ego_yaw_rate", 0.0))

        camera_paths: dict[str, str] = {}
        if self._include_camera:
            cameras = getattr(frame, "cameras", None)
            if cameras is not None:
                for channel in getattr(cameras, "__dataclass_fields__", {}):
                    cam = getattr(cameras, channel, None)
                    img_path = getattr(cam, "image_path", None) if cam is not None else None
                    if img_path is not None:
                        camera_paths[channel] = str(img_path)

        return NavSimSceneFrame(
            ego_xy=pose[:2].copy(),
            ego_heading=float(pose[2]),
            velocity_xy=velocity[:2].copy() if velocity.size >= 2 else None,
            acceleration_xy=accel[:2].copy() if accel.size >= 2 else None,
            yaw_rate=yaw_rate,
            camera_paths=camera_paths,
        )


__all__ = [
    "NavSimScene",
    "NavSimSceneFrame",
    "NavSimSceneLoader",
    "NavSimTrajectoryDataset",
]
