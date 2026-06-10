"""BasePolicy contract + three policy wrappers."""

from __future__ import annotations

import pytest
import torch

from driving_vla import DrivingTrajectorySample
from driving_vla.models import (
    ConditionalDiffusionModel,
    ConditionalFlowMatcher,
    DrivingObservationEncoder,
    WaypointRegressionHead,
)
from driving_vla.policies import (
    BasePolicy,
    DiffusionPolicy,
    FlowPolicy,
    RegressionPolicy,
    observation_to_condition,
)

STATE_DIM = 5
NUM_COMMANDS = 4
LATENT_DIM = 32
HORIZON = 4
TRAJ_DIM = 2
HIDDEN_DIM = 64


def _encoder() -> DrivingObservationEncoder:
    return DrivingObservationEncoder(
        state_dim=STATE_DIM,
        num_commands=NUM_COMMANDS,
        latent_dim=LATENT_DIM,
        history_dim=TRAJ_DIM,
    )


def _sample() -> DrivingTrajectorySample:
    return DrivingTrajectorySample(
        history=torch.zeros(3, TRAJ_DIM),
        future=torch.zeros(HORIZON, TRAJ_DIM),
        instruction="Hold the current lane and drive straight ahead.",
        command_id=0,
        ego_state=torch.zeros(STATE_DIM),
        metadata={"scenario_id": "t0"},
    )


def _condition_batch(batch: int = 3) -> dict:
    return {
        "ego_state": torch.zeros(batch, STATE_DIM),
        "history": torch.zeros(batch, 3, TRAJ_DIM),
        "command_id": torch.zeros(batch, dtype=torch.long),
    }


def _target(batch: int = 3) -> torch.Tensor:
    return torch.zeros(batch, HORIZON, TRAJ_DIM)


# ---- observation_to_condition ----------------------------------------------


def test_observation_to_condition_adds_batch_dim():
    cond = observation_to_condition(_sample())
    assert cond["ego_state"].shape == (1, STATE_DIM)
    assert cond["history"].shape == (1, 3, TRAJ_DIM)
    assert cond["command_id"].shape == (1,)
    assert cond["instruction"] == ["Hold the current lane and drive straight ahead."]


def test_observation_to_condition_fallback_to_history_last():
    s = DrivingTrajectorySample(
        history=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        future=torch.zeros(HORIZON, TRAJ_DIM),
    )
    cond = observation_to_condition(s)
    assert cond["ego_state"].shape == (1, 2)
    # Last history row [4, 5]
    assert torch.equal(cond["ego_state"].squeeze(0), torch.tensor([4.0, 5.0]))


def test_observation_to_condition_rejects_empty_history_and_no_state():
    bad = DrivingTrajectorySample(
        history=torch.zeros(0, 2),
        future=torch.zeros(HORIZON, 2),
    )
    with pytest.raises(ValueError):
        observation_to_condition(bad)


# ---- BasePolicy contract enforcement ---------------------------------------


def test_base_policy_rejects_invalid_dims():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    with pytest.raises(ValueError):
        BasePolicy(encoder=enc, head=head, action_dim=0, horizon=HORIZON)


def test_base_policy_forward_rejects_bad_target_shape():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    with pytest.raises(ValueError):
        policy(_condition_batch(), torch.zeros(3, HORIZON, TRAJ_DIM + 1))


# ---- RegressionPolicy ------------------------------------------------------


def test_regression_policy_loss_scalar():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    loss = policy(_condition_batch(), _target())
    assert loss.ndim == 0


def test_regression_policy_predict_chunk_shape():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    policy.eval()
    chunk = policy.predict_chunk(_sample())
    assert chunk.shape == (HORIZON, TRAJ_DIM)


def test_regression_policy_act_step_shape():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    policy.eval()
    action = policy.act_step(_sample())
    assert action.shape == (TRAJ_DIM,)


# ---- FlowPolicy ------------------------------------------------------------


def test_flow_policy_loss_and_predict():
    enc = _encoder()
    head = ConditionalFlowMatcher(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = FlowPolicy(encoder=enc, head=head, sampling_steps=4)
    loss = policy(_condition_batch(), _target())
    assert loss.ndim == 0
    policy.eval()
    chunk = policy.predict_chunk(_sample())
    assert chunk.shape == (HORIZON, TRAJ_DIM)


def test_flow_policy_predict_chunk_n_independent():
    enc = _encoder()
    head = ConditionalFlowMatcher(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = FlowPolicy(encoder=enc, head=head, sampling_steps=4)
    policy.eval()
    chunks = policy.predict_chunk_n(_sample(), n=4)
    assert chunks.shape == (4, HORIZON, TRAJ_DIM)
    diffs = [
        (chunks[i] - chunks[j]).abs().sum().item()
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    assert max(diffs) > 1e-6, "FlowPolicy should sample independently in BoN"


# ---- DiffusionPolicy -------------------------------------------------------


def test_diffusion_policy_loss_and_predict():
    enc = _encoder()
    head = ConditionalDiffusionModel(
        horizon=HORIZON,
        traj_dim=TRAJ_DIM,
        cond_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        diffusion_steps=20,
    )
    policy = DiffusionPolicy(encoder=enc, head=head, sampling_steps=5)
    loss = policy(_condition_batch(), _target())
    assert loss.ndim == 0
    policy.eval()
    chunk = policy.predict_chunk(_sample())
    assert chunk.shape == (HORIZON, TRAJ_DIM)


def test_diffusion_policy_bon_independent():
    enc = _encoder()
    head = ConditionalDiffusionModel(
        horizon=HORIZON,
        traj_dim=TRAJ_DIM,
        cond_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        diffusion_steps=20,
    )
    policy = DiffusionPolicy(encoder=enc, head=head, sampling_steps=5)
    policy.eval()
    chunks = policy.predict_chunk_n(_sample(), n=3)
    assert chunks.shape == (3, HORIZON, TRAJ_DIM)


# ---- BoN n>0 contract ------------------------------------------------------


def test_predict_chunk_n_rejects_zero_n():
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    policy.eval()
    with pytest.raises(ValueError):
        policy.predict_chunk_n(_sample(), n=0)


def test_regression_policy_n_broadcast():
    """RegressionPolicy is deterministic — all N samples should be identical."""
    enc = _encoder()
    head = WaypointRegressionHead(
        horizon=HORIZON, traj_dim=TRAJ_DIM, cond_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    policy = RegressionPolicy(encoder=enc, head=head)
    policy.eval()
    chunks = policy.predict_chunk_n(_sample(), n=3)
    assert chunks.shape == (3, HORIZON, TRAJ_DIM)
    assert torch.allclose(chunks[0], chunks[1]) and torch.allclose(chunks[1], chunks[2])
