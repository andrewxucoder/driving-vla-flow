"""nuScenes adapter — synthetic .pt fixtures (no images)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from driving_vla.data import NuScenesTrajectoryDataset


def _make_sample(*, command_id: int = 0, with_images: bool = False) -> dict:
    metadata: dict = {"scenario_id": "ns-0001"}
    if with_images:
        # Image paths intentionally missing; testing that image_paths field is present.
        metadata["image_paths"] = []
    return {
        "ego_state": torch.zeros(5),
        "history": torch.zeros(4, 2),
        "future": torch.linspace(1.0, 8.0, 8 * 2).reshape(8, 2),
        "command_id": command_id,
        "metadata": metadata,
    }


def test_loads_basic_samples_without_images(tmp_path: Path):
    path = tmp_path / "samples.pt"
    torch.save([_make_sample() for _ in range(2)], path)
    ds = NuScenesTrajectoryDataset(path)
    assert len(ds) == 2
    s = ds[0]
    assert s["images"] is None
    assert "condition" in s and "future" in s
    assert s["condition"]["command_id"].item() == 0


def test_samples_dict_wrapper(tmp_path: Path):
    path = tmp_path / "wrapped.pt"
    torch.save({"samples": [_make_sample(command_id=3)]}, path)
    ds = NuScenesTrajectoryDataset(path)
    assert len(ds) == 1
    assert ds[0]["command_id"].item() == 3


def test_empty_image_paths_treated_as_no_images(tmp_path: Path):
    path = tmp_path / "samples.pt"
    torch.save([_make_sample(with_images=True)], path)
    ds = NuScenesTrajectoryDataset(path)
    s = ds[0]
    # Empty image_paths list is equivalent to "no images for this sample".
    assert s["images"] is None
    assert "images" not in s["condition"]


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        NuScenesTrajectoryDataset(tmp_path / "missing.pt")
