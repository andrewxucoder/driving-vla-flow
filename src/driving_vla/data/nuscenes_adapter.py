"""nuScenes trajectory + multi-view image dataset adapter.

Reads preprocessed ``.pt`` files written by
``scripts/preprocess_nuscenes.py``. Each sample dict carries the standard
trajectory fields plus a ``metadata["image_paths"]`` list of per-camera image
paths. The adapter lazily loads + CLIP-normalises images on ``__getitem__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from driving_vla.data.command_labeling import CAMERA_CHANNELS
from driving_vla.data.instruction_templates import resolve_instruction


_CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32).view(3, 1, 1)
_CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32).view(3, 1, 1)


def _load_clip_image(path: str | Path, size: int) -> torch.Tensor:
    """Load a single image and CLIP-normalise to ``[3, size, size]``."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (tensor - _CLIP_MEAN) / _CLIP_STD


class NuScenesTrajectoryDataset(Dataset):
    """PyTorch ``Dataset`` over preprocessed nuScenes ``.pt`` samples.

    Each sample yields a dict with the standard trajectory fields and, when
    ``metadata["image_paths"]`` is present, a stacked tensor ``images`` of shape
    ``[V, 3, image_size, image_size]`` where ``V`` is the number of cameras
    listed (typically the 6 of :data:`CAMERA_CHANNELS`).
    """

    def __init__(self, data_path: str | Path, image_size: int = 224) -> None:
        self.data_path = Path(data_path)
        self.image_size = int(image_size)
        if not self.data_path.exists():
            raise FileNotFoundError(f"nuScenes preprocessed data not found: {self.data_path}")

        loaded = torch.load(self.data_path, map_location="cpu", weights_only=False)
        self._samples: list[dict[str, Any]] = self._normalize_loaded(loaded)

    def _normalize_loaded(self, loaded: Any) -> list[dict[str, Any]]:
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            if "samples" in loaded:
                inner = loaded["samples"]
                if not isinstance(inner, list):
                    raise TypeError("'samples' must be a list of sample dicts")
                return inner
            return [loaded]
        raise TypeError(
            "nuScenes .pt file must contain a sample dict, a list of dicts, or "
            "a dict with a 'samples' key."
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._samples[idx]

        future = _as_float_tensor(raw["future"])
        history = _as_float_tensor(raw["history"]) if raw.get("history") is not None else None
        ego_state = _as_float_tensor(raw["ego_state"]) if raw.get("ego_state") is not None else None
        command_id = _as_long_scalar(raw["command_id"])
        metadata = dict(raw.get("metadata") or {})
        instruction = resolve_instruction(
            raw=raw.get("instruction"),
            command_id=int(command_id.item()),
            seed=idx,
        )

        image_paths = metadata.get("image_paths")
        images: torch.Tensor | None = None
        if image_paths:  # only stack when the list is non-empty
            images = torch.stack(
                [_load_clip_image(p, self.image_size) for p in image_paths]
            )

        condition: dict[str, Any] = {
            "ego_state": ego_state,
            "history": history,
            "command_id": command_id,
            "instruction": instruction,
            "metadata": metadata,
        }
        if images is not None:
            condition["images"] = images

        return {
            "condition": condition,
            "future": future,
            "ego_state": ego_state,
            "history": history,
            "command_id": command_id,
            "instruction": instruction,
            "metadata": metadata,
            "images": images,
        }


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _as_long_scalar(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.long).reshape(())
    return torch.tensor(int(value), dtype=torch.long)


__all__ = ["CAMERA_CHANNELS", "NuScenesTrajectoryDataset"]
