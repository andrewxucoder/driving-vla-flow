"""Best-of-N selector over policy samples.

Given ``[N, H, D]`` samples and a scoring function, return the highest-scoring
sample. Used by ``flow_bon`` / ``diffusion_bon`` evaluation routes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


ScoreFn = Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]
"""``(samples [N, H, D], condition) -> scores [N]`` — higher is better."""


def best_of_n_select(
    samples: torch.Tensor,
    score_fn: ScoreFn,
    condition: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Pick the highest-scoring sample.

    Args:
        samples: ``[N, H, D]`` candidate trajectories for a single observation.
        score_fn: receives ``(samples, condition)`` and returns ``[N]`` scores.
        condition: optional dict passed to ``score_fn`` for context-aware rewards.

    Returns:
        ``(best_sample [H, D], best_idx, all_scores [N])``
    """
    if samples.ndim != 3:
        raise ValueError(f"samples expected [N, H, D], got {tuple(samples.shape)}")
    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one candidate")

    scores = score_fn(samples, condition or {})
    if scores.ndim != 1 or scores.shape[0] != samples.shape[0]:
        raise ValueError(
            f"score_fn must return [N={samples.shape[0]}], got {tuple(scores.shape)}"
        )

    best_idx = int(scores.argmax().item())
    return samples[best_idx], best_idx, scores


def best_of_n_batch(
    samples: torch.Tensor,
    score_fn: ScoreFn,
    condition: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised BoN over a batch.

    Args:
        samples: ``[B, N, H, D]``.

    Returns:
        ``(best_samples [B, H, D], best_idx [B], all_scores [B, N])``
    """
    if samples.ndim != 4:
        raise ValueError(f"samples expected [B, N, H, D], got {tuple(samples.shape)}")
    b, n, h, d = samples.shape
    flat = samples.reshape(b * n, h, d)
    cond = condition or {}
    scores = score_fn(flat, cond).reshape(b, n)  # [B, N]
    best_idx = scores.argmax(dim=-1)  # [B]
    best = samples[torch.arange(b, device=samples.device), best_idx]  # [B, H, D]
    return best, best_idx, scores


__all__ = ["ScoreFn", "best_of_n_batch", "best_of_n_select"]
