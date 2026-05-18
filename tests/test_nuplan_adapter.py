import torch

from driving_vla.data.nuplan_adapter import NuPlanTrajectoryDataset


def test_nuplan_adapter_loads_preprocessed_pt_file(tmp_path):
    sample = {
        "ego_state": [1.0, 0.0, 0.0],
        "history": [[0.0, 0.0], [0.5, 0.0]],
        "command_id": 1,
        "language_command": "change lane left safely",
        "future": [[1.0, 0.0], [2.0, 0.5]],
        "metadata": {"scenario_id": "sample-1"},
    }
    path = tmp_path / "nuplan_samples.pt"
    torch.save({"samples": [sample]}, path)

    ds = NuPlanTrajectoryDataset(path)
    item = ds[0]

    assert len(ds) == 1
    assert item["future"].shape == (2, 2)
    assert item["condition"]["ego_state"].shape == (3,)
    assert item["condition"]["history"].shape == (2, 2)
    assert item["condition"]["command_id"].item() == 1
    assert item["condition"]["language_command"] == "change lane left safely"
    assert item["condition"]["metadata"]["scenario_id"] == "sample-1"


def test_nuplan_adapter_loads_preprocessed_pt_directory(tmp_path):
    sample = {
        "ego_state": torch.zeros(3),
        "history": torch.zeros(2, 2),
        "command_id": torch.tensor(0),
        "language_command": "keep lane smoothly",
        "future": torch.zeros(3, 2),
        "metadata": {"scenario_id": "sample-0"},
    }
    torch.save(sample, tmp_path / "sample_000.pt")

    ds = NuPlanTrajectoryDataset(tmp_path)
    item = ds[0]

    assert len(ds) == 1
    assert item["future"].shape == (3, 2)
    assert item["condition"]["metadata"]["scenario_id"] == "sample-0"
