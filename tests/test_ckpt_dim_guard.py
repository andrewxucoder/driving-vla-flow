"""scripts/_common.assert_ckpt_dims — fail-fast guard for eval scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import assert_ckpt_dims  # noqa: E402


def _args(**kw) -> argparse.Namespace:
    base = dict(
        horizon=30,
        traj_dim=2,
        state_dim=5,
        latent_dim=128,
        hidden_dim=256,
        num_commands=4,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _ckpt(args_overrides: dict | None = None, head: str | None = "flow") -> dict:
    saved = dict(
        horizon=30,
        traj_dim=2,
        state_dim=5,
        latent_dim=128,
        hidden_dim=256,
        num_commands=4,
    )
    if args_overrides:
        saved.update(args_overrides)
    meta: dict = {"args": saved}
    if head is not None:
        meta["head"] = head
    return {"model_state": {}, "metadata": meta}


def test_matching_dims_pass():
    # Should not raise
    assert_ckpt_dims(_ckpt(), _args(), expected_head="flow")


def test_horizon_mismatch_raises():
    with pytest.raises(ValueError, match="horizon"):
        assert_ckpt_dims(_ckpt({"horizon": 30}), _args(horizon=8), expected_head="flow")


def test_latent_dim_mismatch_raises():
    with pytest.raises(ValueError, match="latent_dim"):
        assert_ckpt_dims(_ckpt({"latent_dim": 128}), _args(latent_dim=64))


def test_head_label_mismatch_raises():
    with pytest.raises(ValueError, match="head"):
        assert_ckpt_dims(_ckpt(head="diffusion"), _args(), expected_head="flow")


def test_legacy_ckpt_without_metadata_skips_check():
    # Legacy weights-only ckpt: no metadata at all → skip cleanly (no raise).
    assert_ckpt_dims({"model_state": {}}, _args())


def test_missing_args_block_in_metadata_skips_check():
    assert_ckpt_dims({"model_state": {}, "metadata": {}}, _args())


def test_pre_x0_diffusion_ckpt_rejected():
    """Round-0 diffusion ckpts predate the x0 fix and must be refused."""
    pre_x0 = _ckpt(head="diffusion")  # no parameterization tag
    with pytest.raises(ValueError, match="parameterization"):
        assert_ckpt_dims(pre_x0, _args(), expected_head="diffusion")


def test_post_x0_diffusion_ckpt_accepted():
    post_x0 = _ckpt(head="diffusion")
    post_x0["metadata"]["parameterization"] = "x0"
    assert_ckpt_dims(post_x0, _args(), expected_head="diffusion")
