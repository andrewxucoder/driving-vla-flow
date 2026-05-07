from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ToyDrivingConfig:
    num_samples: int = 4096
    horizon: int = 30
    state_dim: int = 8
    language_vocab: List[str] | None = None


class ToyDrivingDataset(Dataset):
    """Synthetic driving trajectories conditioned on ego state and language command.

    This dataset is intentionally simple. It is used to validate the algorithmic
    pipeline before moving to nuPlan/nuScenes/CARLA adapters.
    """

    def __init__(self, cfg: ToyDrivingConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.vocab = cfg.language_vocab or [
            "keep lane smoothly",
            "change lane left safely",
            "change lane right safely",
            "slow down and yield",
        ]
        self.samples = [self._make_sample(i) for i in range(cfg.num_samples)]

    def _make_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        cmd_id = int(self.rng.integers(0, len(self.vocab)))
        speed = float(self.rng.uniform(4.0, 15.0))
        curvature = float(self.rng.normal(0.0, 0.02))
        gap = float(self.rng.uniform(5.0, 40.0))
        route_bias = float(self.rng.normal(0.0, 1.0))

        state = np.zeros(self.cfg.state_dim, dtype=np.float32)
        state[:4] = [speed, curvature, gap, route_bias]
        state[4:] = self.rng.normal(0, 0.1, size=self.cfg.state_dim - 4)

        t = np.linspace(0.1, 3.0, self.cfg.horizon, dtype=np.float32)
        x = speed * t
        y = np.zeros_like(x)

        if cmd_id == 1:  # left lane change
            y = 3.5 / (1.0 + np.exp(-3 * (t - 1.5)))
        elif cmd_id == 2:  # right lane change
            y = -3.5 / (1.0 + np.exp(-3 * (t - 1.5)))
        elif cmd_id == 3:  # yield / slow down
            x = speed * (1 - np.exp(-0.9 * t)) / 0.9
        else:  # keep lane with slight curvature
            y = curvature * x**2

        traj = np.stack([x, y], axis=-1)
        traj += self.rng.normal(0, 0.05, size=traj.shape).astype(np.float32)

        return {
            "state": torch.tensor(state, dtype=torch.float32),
            "cmd_id": torch.tensor(cmd_id, dtype=torch.long),
            "trajectory": torch.tensor(traj, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
