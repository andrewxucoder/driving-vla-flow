"""preference_dataset — load + collate semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from driving_vla.data import PreferenceDataset, collate_preference_batch


def _make_pair(*, chosen_reward: float = 0.8, rejected_reward: float = 0.2) -> dict:
    return {
        "condition": {
            "ego_state": torch.zeros(5),
            "history": torch.zeros(4, 2),
            "command_id": torch.tensor(0, dtype=torch.long),
            "instruction": "Hold the current lane and drive straight ahead.",
            "metadata": {"scenario_id": "x"},
        },
        "chosen": torch.zeros(8, 2),
        "rejected": torch.ones(8, 2) * 0.5,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_gap": chosen_reward - rejected_reward,
    }


def test_load_pairs(tmp_path: Path):
    path = tmp_path / "prefs.pt"
    torch.save({"pairs": [_make_pair(), _make_pair()], "cfg": {"strength": 0.5}}, path)
    ds = PreferenceDataset(path)
    assert len(ds) == 2
    assert ds.cfg["strength"] == 0.5
    s = ds[0]
    assert s["chosen"].shape == (8, 2)
    assert s["rejected"].shape == (8, 2)


def test_missing_pairs_key_raises(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"oops": []}, path)
    with pytest.raises(TypeError):
        PreferenceDataset(path)


def test_collate_stacks_tensors():
    samples = [_make_pair(chosen_reward=0.9, rejected_reward=0.1) for _ in range(3)]
    batch = collate_preference_batch(samples)
    assert batch["chosen"].shape == (3, 8, 2)
    assert batch["rejected"].shape == (3, 8, 2)
    assert batch["condition"]["ego_state"].shape == (3, 5)
    assert batch["condition"]["history"].shape == (3, 4, 2)
    assert batch["condition"]["command_id"].shape == (3,)
    assert batch["reward_gap"].tolist() == [pytest.approx(0.8)] * 3


def test_collate_keeps_instruction_as_list():
    samples = [_make_pair(), _make_pair()]
    batch = collate_preference_batch(samples)
    assert isinstance(batch["condition"]["instruction"], list)
    assert len(batch["condition"]["instruction"]) == 2


def test_collate_history_none_propagates():
    s = _make_pair()
    s["condition"]["history"] = None
    batch = collate_preference_batch([s, _make_pair()])
    assert batch["condition"]["history"] is None


def test_collate_empty_raises():
    with pytest.raises(ValueError):
        collate_preference_batch([])
