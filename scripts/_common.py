"""Shared helpers for train/eval scripts.

Centralises the dataset → encoder → head construction sequence so each script
can stay focused on the policy variant it owns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Structural dims that, if mismatched, will silently produce wrong-shape
# weights on load and either crash with a confusing message or worse, partially
# load. We check these explicitly when a ckpt carries its training-time args.
_CHECKED_DIMS: tuple[str, ...] = (
    "horizon",
    "traj_dim",
    "state_dim",
    "latent_dim",
    "hidden_dim",
    "num_commands",
    "diffusion_steps",
)


def assert_ckpt_dims(
    ckpt: dict[str, Any],
    args: argparse.Namespace,
    *,
    expected_head: str | None = None,
) -> None:
    """Fail fast if eval-time structural args don't match training-time args.

    ``ckpt`` is the dict produced by ``train()`` (carries ``metadata.args``
    written by every train_*.py script). If metadata is absent (e.g. legacy
    weights-only file), the check is skipped with a warning.

    ``expected_head`` (e.g. ``"flow"``, ``"diffusion"``) cross-checks the
    saved head label — guards against accidentally pointing a flow-eval at a
    regression checkpoint.
    """
    meta = ckpt.get("metadata") if isinstance(ckpt, dict) else None
    if not meta or "args" not in meta:
        print(
            "[assert_ckpt_dims] ckpt has no training-time metadata — skipping dim check.",
            file=sys.stderr,
        )
        return
    saved = meta["args"]
    mismatches: list[str] = []
    for dim in _CHECKED_DIMS:
        if dim not in saved or not hasattr(args, dim):
            continue
        want = saved[dim]
        got = getattr(args, dim)
        if want != got:
            mismatches.append(f"  {dim}: ckpt={want}, args={got}")
    if expected_head is not None and meta.get("head") not in (None, expected_head):
        mismatches.append(
            f"  head: ckpt={meta.get('head')!r}, expected={expected_head!r}"
        )
    # Hard guard: a diffusion ckpt missing the parameterization tag was trained
    # before the x0-parameterization fix and is *not* compatible with the new
    # sampler — silent acceptance would produce nonsense predictions.
    if (
        expected_head == "diffusion"
        and meta.get("head") == "diffusion"
        and "parameterization" not in meta
    ):
        mismatches.append(
            "  parameterization: ckpt is pre-x0 (Round 0). Retrain with the "
            "current train_nuplan_diffusion.py before re-evaluating."
        )
    if mismatches:
        details = "\n".join(mismatches)
        raise ValueError(
            "Checkpoint / eval-args structural mismatch — refusing to load to "
            f"avoid silent shape errors. Differences:\n{details}"
        )


def device_from_arg(arg: str) -> torch.device:
    """Resolve a device string. ``auto`` prefers CUDA → MPS → CPU."""
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def default_collate(batch: list[dict]) -> dict:
    """Light dict-of-dicts collate used by all v2 trajectory adapters."""
    condition_keys = list(batch[0]["condition"].keys())
    out_condition: dict = {}
    for k in condition_keys:
        vals = [b["condition"][k] for b in batch]
        # Stack tensors; pass through strings / dicts / None.
        if isinstance(vals[0], torch.Tensor):
            out_condition[k] = torch.stack(vals, dim=0)
        else:
            out_condition[k] = list(vals)

    out: dict = {"condition": out_condition}
    if "future" in batch[0]:
        out["future"] = torch.stack([b["future"] for b in batch], dim=0)
    return out


def move_condition_to_device(
    condition: dict, device: torch.device
) -> dict:
    out: dict = {}
    for k, v in condition.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


__all__ = [
    "assert_ckpt_dims",
    "default_collate",
    "device_from_arg",
    "move_condition_to_device",
]
