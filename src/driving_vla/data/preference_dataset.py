"""Preference-pair dataset for trajectory RM / DPO training.

Each ``.pt`` file produced by ``scripts/generate_preference_data.py`` carries:

- ``pairs``: list of dicts with keys ``condition`` / ``chosen`` / ``rejected`` /
  ``chosen_reward`` / ``rejected_reward`` / ``reward_gap``
- ``cfg``: dict capturing the generation parameters for reproducibility

``collate_preference_batch`` stacks the per-pair tensors and keeps text /
metadata fields as Python lists so they pass through ``DataLoader`` cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "pairs" not in data:
            raise TypeError(
                f"preference data at {self.path} must be a dict with a 'pairs' key"
            )
        pairs = data["pairs"]
        if not isinstance(pairs, list):
            raise TypeError("'pairs' must be a list of pair dicts")
        self.pairs: list[dict[str, Any]] = pairs
        self.cfg: dict[str, Any] = data.get("cfg", {})

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.pairs[idx]


def collate_preference_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep string / dict fields as Python lists."""
    if not samples:
        raise ValueError("samples must be non-empty")
    conds = [s["condition"] for s in samples]

    histories = [c.get("history") for c in conds]
    condition: dict[str, Any] = {
        "ego_state": torch.stack([c["ego_state"] for c in conds]),
        "history": None
        if any(h is None for h in histories)
        else torch.stack(histories),
        "command_id": torch.stack([c["command_id"] for c in conds]).reshape(-1),
        "instruction": [c.get("instruction", "") for c in conds],
        "metadata": [c.get("metadata", {}) for c in conds],
    }
    return {
        "condition": condition,
        "chosen": torch.stack([s["chosen"] for s in samples]),
        "rejected": torch.stack([s["rejected"] for s in samples]),
        "chosen_reward": torch.tensor([float(s["chosen_reward"]) for s in samples]),
        "rejected_reward": torch.tensor([float(s["rejected_reward"]) for s in samples]),
        "reward_gap": torch.tensor([float(s["reward_gap"]) for s in samples]),
    }


__all__ = ["PreferenceDataset", "collate_preference_batch"]
