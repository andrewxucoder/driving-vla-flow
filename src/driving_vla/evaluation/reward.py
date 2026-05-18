from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardWeights:
    progress: float = 0.05
    lane: float = 1.0
    smoothness: float = 0.5


def reward_components(traj: torch.Tensor, lane_center: float = 0.0) -> dict[str, torch.Tensor]:
    """计算轨迹奖励的分项：前向进展、车道偏离和平滑性。"""
    progress = traj[:, -1, 0] - traj[:, 0, 0]
    lane_eval = traj[:, max(1, traj.shape[1] // 2) :, 1]
    lane_deviation = torch.abs(lane_eval - lane_center).mean(dim=1)
    if traj.shape[1] < 4:
        smoothness = torch.zeros(traj.shape[0], device=traj.device)
    else:
        vel = traj[:, 1:] - traj[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        smoothness = torch.norm(jerk, dim=-1).mean(dim=1)
    return {
        "progress": progress,
        "lane_deviation": lane_deviation,
        "smoothness": smoothness,
    }


def trajectory_reward(traj: torch.Tensor, lane_center: float = 0.0, weights: RewardWeights | None = None) -> torch.Tensor:
    weights = weights or RewardWeights()
    components = reward_components(traj, lane_center=lane_center)
    return (
        weights.progress * components["progress"]
        - weights.lane * components["lane_deviation"]
        - weights.smoothness * components["smoothness"]
    )


def rank_trajectories(traj: torch.Tensor, lane_center: float = 0.0, weights: RewardWeights | None = None) -> tuple[int, torch.Tensor, dict[str, torch.Tensor]]:
    scores = trajectory_reward(traj, lane_center=lane_center, weights=weights)
    components = reward_components(traj, lane_center=lane_center)
    best_idx = int(torch.argmax(scores).item())
    return best_idx, scores, components
