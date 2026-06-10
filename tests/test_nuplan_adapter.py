"""nuPlan adapter — synthetic .pt fixtures (no devkit required)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from driving_vla.data import NuPlanTrajectoryDataset


def _make_sample(*, command_id: int = 0) -> dict:
    return {
        "ego_state": torch.zeros(5),
        "history": torch.zeros(4, 2),
        "future": torch.linspace(1.0, 8.0, 8 * 2).reshape(8, 2),
        "command_id": command_id,
        "language_command": "",
        "metadata": {"scenario_id": "scene-0001"},
    }


def test_single_pt_file_with_list(tmp_path: Path):
    path = tmp_path / "samples.pt"
    torch.save([_make_sample() for _ in range(3)], path)
    ds = NuPlanTrajectoryDataset(path)
    assert len(ds) == 3
    s = ds[0]
    assert "condition" in s and "future" in s
    assert s["condition"]["command_id"].item() == 0
    assert s["future"].shape == (8, 2)


def test_single_pt_file_with_samples_dict(tmp_path: Path):
    path = tmp_path / "wrapped.pt"
    torch.save({"samples": [_make_sample(command_id=2)]}, path)
    ds = NuPlanTrajectoryDataset(path)
    assert len(ds) == 1
    assert ds[0]["command_id"].item() == 2


def test_directory_of_pt_files(tmp_path: Path):
    d = tmp_path / "samples_dir"
    d.mkdir()
    for i in range(2):
        torch.save(_make_sample(command_id=i), d / f"s_{i:03d}.pt")
    ds = NuPlanTrajectoryDataset(d)
    assert len(ds) == 2
    cids = sorted(int(ds[i]["command_id"].item()) for i in range(2))
    assert cids == [0, 1]


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        NuPlanTrajectoryDataset(tmp_path / "does_not_exist.pt")


def test_empty_directory_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        NuPlanTrajectoryDataset(empty)


def test_instruction_resolved_to_natural_language(tmp_path: Path):
    path = tmp_path / "samples.pt"
    torch.save([_make_sample(command_id=1)], path)
    ds = NuPlanTrajectoryDataset(path)
    s = ds[0]
    # empty raw → paraphrase
    assert isinstance(s["instruction"], str) and len(s["instruction"]) > 0
    assert s["instruction"] not in ("change_left", "")  # not the raw vocab token
