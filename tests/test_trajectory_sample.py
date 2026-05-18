import torch

from driving_vla.data.trajectory_sample import DrivingTrajectorySample, trajectory_sample_to_tensors


def test_toy_sample_dict_to_tensors_keeps_legacy_keys():
    sample = {
        "state": torch.zeros(8),
        "cmd_id": torch.tensor(1),
        "trajectory": torch.zeros(30, 2),
    }

    tensors = trajectory_sample_to_tensors(sample)

    assert tensors["ego_state"].shape == (8,)
    assert tensors["future"].shape == (30, 2)
    assert tensors["command_id"].dtype == torch.long
    assert torch.equal(tensors["state"], tensors["ego_state"])
    assert torch.equal(tensors["trajectory"], tensors["future"])
    assert torch.equal(tensors["cmd_id"], tensors["command_id"])


def test_driving_trajectory_sample_to_tensors():
    sample = DrivingTrajectorySample(
        ego_state=[1.0, 2.0],
        history=[[0.0, 0.0], [1.0, 0.1]],
        future=[[2.0, 0.2], [3.0, 0.3]],
        command_id=2,
        language_command="change lane right safely",
        metadata={"source": "unit_test"},
    )

    tensors = trajectory_sample_to_tensors(sample)

    assert tensors["ego_state"].dtype == torch.float32
    assert tensors["history"].shape == (2, 2)
    assert tensors["future"].shape == (2, 2)
    assert tensors["command_id"].item() == 2
    assert tensors["language_command"] == "change lane right safely"
    assert tensors["metadata"]["source"] == "unit_test"
