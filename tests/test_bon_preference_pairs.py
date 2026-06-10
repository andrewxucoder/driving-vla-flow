"""M7.2 — generate_preference_data.py --mode bon end-to-end smoke.

Builds a tiny toy ``.pt`` dataset + Flow checkpoint, invokes the CLI, and
verifies the resulting pair file conforms to :class:`PreferenceDataset`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
import torch

from driving_vla.data import PreferenceDataset
from driving_vla.models import ConditionalFlowMatcher, DrivingObservationEncoder
from driving_vla.policies import FlowPolicy


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _save_toy_dataset(path: Path, n_samples: int = 4) -> None:
    """Make a minimal nuPlan-style .pt file the CLI can ingest."""
    samples = []
    for i in range(n_samples):
        samples.append({
            "ego_state": torch.tensor([1.0 + i * 0.1, 0.0, 0.0, 0.0, 0.0]),
            "history": torch.zeros(4, 2),
            "future": torch.stack(
                [torch.linspace(0.5, 4.0, 8), torch.zeros(8)], dim=-1
            ).contiguous(),
            "command_id": torch.tensor(i % 4, dtype=torch.long),
            "instruction": None,
            "metadata": {"token": f"tok-{i}"},
        })
    torch.save({"samples": samples}, path)


def _save_toy_flow_ckpt(
    path: Path,
    *,
    horizon: int,
    state_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_commands: int,
    traj_dim: int,
) -> None:
    """Train a Flow policy for zero steps and save it — we just need valid
    weights of the right shape, not converged ones."""
    encoder = DrivingObservationEncoder(
        state_dim=state_dim,
        num_commands=num_commands,
        latent_dim=latent_dim,
        history_dim=traj_dim,
    )
    head = ConditionalFlowMatcher(
        horizon=horizon, traj_dim=traj_dim, cond_dim=latent_dim, hidden_dim=hidden_dim,
    )
    policy = FlowPolicy(encoder=encoder, head=head, sampling_steps=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "metadata": {
                "head": "flow",
                "args": {
                    "horizon": horizon,
                    "state_dim": state_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "num_commands": num_commands,
                    "traj_dim": traj_dim,
                },
            },
        },
        path,
    )


def _run_cli(argv: list[str]) -> int:
    sys.argv = ["generate_preference_data.py", *argv]
    try:
        runpy.run_path(
            str(_SCRIPTS / "generate_preference_data.py"), run_name="__main__"
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_bon_mode_generates_loadable_pairs(tmp_path: Path) -> None:
    horizon, state_dim, latent_dim, hidden_dim = 8, 5, 16, 32
    num_commands, traj_dim = 4, 2

    data_path = tmp_path / "toy_nuplan.pt"
    ckpt_path = tmp_path / "toy_flow_ckpt.pt"
    out_path = tmp_path / "bon_pairs.pt"

    _save_toy_dataset(data_path, n_samples=8)
    _save_toy_flow_ckpt(
        ckpt_path,
        horizon=horizon, state_dim=state_dim, latent_dim=latent_dim,
        hidden_dim=hidden_dim, num_commands=num_commands, traj_dim=traj_dim,
    )

    rc = _run_cli([
        "--mode", "bon",
        "--data-path", str(data_path),
        "--checkpoint", str(ckpt_path),
        "--output", str(out_path),
        "--bon-n", "4",
        "--max-pairs", "8",
        "--min-reward-gap", "0.0",   # accept anything for the smoke
        "--state-dim", str(state_dim),
        "--num-commands", str(num_commands),
        "--latent-dim", str(latent_dim),
        "--hidden-dim", str(hidden_dim),
        "--horizon", str(horizon),
        "--traj-dim", str(traj_dim),
        "--sampling-steps", "4",
        "--device", "cpu",
    ])
    assert rc == 0
    assert out_path.exists()

    pdataset = PreferenceDataset(out_path)
    assert len(pdataset) > 0
    sample = pdataset[0]
    assert sample["chosen"].shape == (horizon, traj_dim)
    assert sample["rejected"].shape == (horizon, traj_dim)
    assert sample["reward_gap"] >= 0.0


def test_bon_mode_rejects_dim_mismatched_ckpt(tmp_path: Path) -> None:
    """Bug-4 guard: trying to BoN-sample with an arg-dim mismatch must fail
    fast, not load partially and produce garbage trajectories."""
    data_path = tmp_path / "toy_nuplan.pt"
    ckpt_path = tmp_path / "toy_flow_ckpt.pt"
    out_path = tmp_path / "bon_pairs.pt"

    _save_toy_dataset(data_path, n_samples=2)
    _save_toy_flow_ckpt(
        ckpt_path,
        horizon=8, state_dim=5, latent_dim=16, hidden_dim=32,
        num_commands=4, traj_dim=2,
    )

    with pytest.raises(ValueError, match="horizon"):
        # Mismatched horizon (16 vs ckpt's 8) — must raise from assert_ckpt_dims.
        _run_cli([
            "--mode", "bon",
            "--data-path", str(data_path),
            "--checkpoint", str(ckpt_path),
            "--output", str(out_path),
            "--bon-n", "2",
            "--max-pairs", "1",
            "--horizon", "16",     # wrong
            "--state-dim", "5",
            "--latent-dim", "16",
            "--hidden-dim", "32",
            "--traj-dim", "2",
            "--device", "cpu",
        ])


def test_bon_mode_filters_low_gap_pairs(tmp_path: Path) -> None:
    """``--min-reward-gap`` very large → all pairs filtered → empty output."""
    data_path = tmp_path / "toy.pt"
    ckpt_path = tmp_path / "ckpt.pt"
    out_path = tmp_path / "bon_pairs.pt"

    _save_toy_dataset(data_path, n_samples=4)
    _save_toy_flow_ckpt(
        ckpt_path, horizon=8, state_dim=5, latent_dim=16,
        hidden_dim=32, num_commands=4, traj_dim=2,
    )

    rc = _run_cli([
        "--mode", "bon",
        "--data-path", str(data_path),
        "--checkpoint", str(ckpt_path),
        "--output", str(out_path),
        "--bon-n", "2",
        "--max-pairs", "4",
        "--min-reward-gap", "1e6",  # impossibly large
        "--state-dim", "5",
        "--num-commands", "4",
        "--latent-dim", "16",
        "--hidden-dim", "32",
        "--horizon", "8",
        "--traj-dim", "2",
        "--sampling-steps", "4",
        "--device", "cpu",
    ])
    assert rc == 0
    pdataset = PreferenceDataset(out_path)
    assert len(pdataset) == 0
