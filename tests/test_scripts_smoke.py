"""Smoke tests for CLI scripts under scripts/.

The preprocess scripts are stubs (real devkit invocation deferred), so we only
verify their argparse surface + dry-run path. setup_data is checked in --check
mode against an empty data/ dir to exercise the "missing" branch cleanly.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_preprocess_nuplan_dry_run(capsys):
    sys.argv = ["preprocess_nuplan_mini.py", "--dry-run"]
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPTS / "preprocess_nuplan_mini.py"), run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower()


def test_preprocess_nuscenes_dry_run(capsys):
    sys.argv = ["preprocess_nuscenes.py", "--dry-run"]
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPTS / "preprocess_nuscenes.py"), run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower()


def test_setup_data_check_runs(capsys):
    # --check on a clean tree returns non-zero (no symlinks yet) but must not raise.
    sys.argv = ["setup_data.py", "--check"]
    try:
        runpy.run_path(str(_SCRIPTS / "setup_data.py"), run_name="__main__")
    except SystemExit as exc:
        # Non-zero is expected when no symlinks exist; zero is fine too.
        assert exc.code in (0, 1)


@pytest.mark.parametrize(
    "script",
    [
        "train_nuplan_regression.py",
        "train_nuplan_flow.py",
        "train_nuplan_diffusion.py",
        "train_nuscenes_image.py",
        "train_nuscenes_vlm.py",
        "train_flow_dpo.py",
        "eval_navsim_heads.py",
        "eval_nuplan_heads.py",
        "eval_nuscenes_image.py",
        "generate_preference_data.py",
    ],
)
def test_train_eval_scripts_dry_run(script: str):
    sys.argv = [script, "--dry-run"]
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPTS / script), run_name="__main__")
    assert exc_info.value.code == 0
