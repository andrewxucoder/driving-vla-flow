"""Trajectory perturbation primitives.

Used by ``scripts/generate_preference_data.py`` to synthesise rejected
trajectories from expert futures (RM / DPO training signal). Each perturbation
is a small independent function so they compose freely via :data:`PERTURBATIONS`.
"""

from __future__ import annotations

from typing import Callable

import torch


PerturbFn = Callable[[torch.Tensor, float, "torch.Generator | None"], torch.Tensor]


def gaussian_perturb(
    traj: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-timestep IID Gaussian noise (same std on x and y)."""
    noise = torch.empty_like(traj).normal_(mean=0.0, std=strength, generator=generator)
    return traj + noise


def time_shift_perturb(
    traj: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shift the trajectory along the time axis by k frames, edge-clamped.

    ``strength`` is the maximum shift as a fraction of the horizon; the actual
    shift is drawn uniformly from ``[-K, K]`` where ``K = max(1, round(strength * H))``.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj expected [B, H, D], got {tuple(traj.shape)}")
    _, horizon, _ = traj.shape
    max_k = max(1, int(round(strength * horizon)))
    if generator is None:
        k = int(torch.randint(-max_k, max_k + 1, (1,)).item())
    else:
        k = int(torch.randint(-max_k, max_k + 1, (1,), generator=generator).item())
    if k == 0:
        return traj.clone()
    idx = (torch.arange(horizon, device=traj.device) - k).clamp(min=0, max=horizon - 1)
    return traj.index_select(dim=1, index=idx)


def lane_jitter_perturb(
    traj: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add a constant lateral offset to every timestep (per-batch-item)."""
    if traj.ndim != 3:
        raise ValueError(f"traj expected [B, H, D], got {tuple(traj.shape)}")
    batch, _, d = traj.shape
    if d < 2:
        raise ValueError(f"lane_jitter needs D ≥ 2 (lateral axis), got D={d}")
    offsets = torch.empty(batch, device=traj.device).normal_(
        mean=0.0, std=strength, generator=generator
    )
    out = traj.clone()
    out[..., 1] = out[..., 1] + offsets.unsqueeze(-1)
    return out


PERTURBATIONS: dict[str, PerturbFn] = {
    "gaussian": gaussian_perturb,
    "time_shift": time_shift_perturb,
    "lane_jitter": lane_jitter_perturb,
}


def perturb_by_name(
    name: str,
    traj: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    try:
        fn = PERTURBATIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown perturbation {name!r}; available: {sorted(PERTURBATIONS)}"
        ) from exc
    return fn(traj, strength, generator)


def random_perturb(
    names: list[str],
    traj: torch.Tensor,
    strength: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """For each batch item, uniformly pick a perturbation from ``names`` and apply it."""
    if not names:
        raise ValueError("names list must not be empty")
    batch = traj.shape[0]
    if generator is not None:
        picks_idx = torch.randint(0, len(names), (batch,), generator=generator)
    else:
        picks_idx = torch.randint(0, len(names), (batch,))
    out = traj.clone()
    chosen: list[str] = []
    for i, k in enumerate(picks_idx.tolist()):
        name = names[k]
        chosen.append(name)
        out[i : i + 1] = perturb_by_name(name, traj[i : i + 1], strength, generator)
    return out, chosen


__all__ = [
    "PERTURBATIONS",
    "gaussian_perturb",
    "lane_jitter_perturb",
    "perturb_by_name",
    "random_perturb",
    "time_shift_perturb",
]
