"""nuPlan trajectory dataset adapter.

Reads preprocessed ``.pt`` sample files produced by
``scripts/preprocess_nuplan_mini.py``. The adapter never imports nuplan-devkit;
preprocessing is a one-shot offline pass that lives in the script layer.

Per-sample payload conforms to :class:`DrivingTrajectorySample` (history /
future / command_id / instruction / ego_state / metadata).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from driving_vla.data.instruction_templates import resolve_instruction


class NuPlanTrajectoryDataset(Dataset):
    """PyTorch ``Dataset`` over preprocessed nuPlan ``.pt`` samples.

    Two on-disk layouts are supported:

    1. A directory of per-sample ``*.pt`` files (one sample per file).
    2. A single ``.pt`` file containing either a ``list[dict]`` of samples, a
       ``dict`` with a ``"samples"`` list, or a single sample ``dict``.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"nuPlan preprocessed data not found: {self.data_path}")

        self._sample_files: list[Path] = []
        self._samples: list[dict[str, Any]] | None = None

        if self.data_path.is_dir():
            self._sample_files = sorted(self.data_path.glob("*.pt"))
            if not self._sample_files:
                raise FileNotFoundError(f"no .pt sample files in {self.data_path}")
        else:
            loaded = torch.load(self.data_path, map_location="cpu", weights_only=False)
            self._samples = self._normalize_loaded(loaded)

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
            "nuPlan .pt file must contain a sample dict, a list of sample dicts, "
            "or a dict with a 'samples' key."
        )

    def _load_raw(self, idx: int) -> dict[str, Any]:
        if self._samples is not None:
            return self._samples[idx]
        sample = torch.load(self._sample_files[idx], map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"sample file must contain a dict: {self._sample_files[idx]}")
        return sample

    def __len__(self) -> int:
        if self._samples is not None:
            return len(self._samples)
        return len(self._sample_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._load_raw(idx)

        history = _as_float_tensor(raw["history"]) if raw.get("history") is not None else None
        future = _as_float_tensor(raw["future"])
        ego_state = _as_float_tensor(raw["ego_state"]) if raw.get("ego_state") is not None else None
        command_id = _as_long_scalar(raw["command_id"])
        instruction = resolve_instruction(
            raw=raw.get("instruction"),
            command_id=int(command_id.item()),
            seed=idx,
        )
        metadata = dict(raw.get("metadata") or {})

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


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _as_long_scalar(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.long).reshape(())
    return torch.tensor(int(value), dtype=torch.long)


__all__ = ["NuPlanTrajectoryDataset"]
