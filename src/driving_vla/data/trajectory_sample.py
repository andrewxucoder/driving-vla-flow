from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class DrivingTrajectorySample:
    """真实驾驶数据集和 toy 数据集共用的轨迹样本结构。"""

    ego_state: torch.Tensor | np.ndarray | list[float]
    history: torch.Tensor | np.ndarray | list[list[float]] | None
    future: torch.Tensor | np.ndarray | list[list[float]]
    command_id: torch.Tensor | int
    language_command: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _to_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def trajectory_sample_to_tensors(sample: DrivingTrajectorySample | Mapping[str, Any]) -> dict[str, Any]:
    """将统一轨迹样本或旧版 toy 样本字典转换为张量字典。

    兼容旧版 toy 输出：
    - `state` 会映射到 `ego_state`
    - `trajectory` 会映射到 `future`
    - `cmd_id` 会映射到 `command_id`

    返回结果同时保留旧版别名，方便现有训练脚本继续使用。
    """
    if isinstance(sample, DrivingTrajectorySample):
        raw = {
            "ego_state": sample.ego_state,
            "history": sample.history,
            "future": sample.future,
            "command_id": sample.command_id,
            "language_command": sample.language_command,
            "metadata": sample.metadata,
        }
    else:
        raw = dict(sample)

    ego_state = raw.get("ego_state", raw.get("state"))
    future = raw.get("future", raw.get("trajectory"))
    command_id = raw.get("command_id", raw.get("cmd_id"))
    if ego_state is None:
        raise KeyError("样本缺少 `ego_state` 或兼容字段 `state`")
    if future is None:
        raise KeyError("样本缺少 `future` 或兼容字段 `trajectory`")
    if command_id is None:
        raise KeyError("样本缺少 `command_id` 或兼容字段 `cmd_id`")

    history = raw.get("history")
    language_command = raw.get("language_command", "")
    metadata = raw.get("metadata") or {}

    tensor_dict: dict[str, Any] = {
        "ego_state": _to_float_tensor(ego_state),
        "history": None if history is None else _to_float_tensor(history),
        "future": _to_float_tensor(future),
        "command_id": _to_long_tensor(command_id),
        "language_command": language_command,
        "metadata": dict(metadata),
    }
    tensor_dict["state"] = tensor_dict["ego_state"]
    tensor_dict["trajectory"] = tensor_dict["future"]
    tensor_dict["cmd_id"] = tensor_dict["command_id"]
    return tensor_dict


def sample_dict_to_tensors(sample: DrivingTrajectorySample | Mapping[str, Any]) -> dict[str, Any]:
    """`trajectory_sample_to_tensors` 的兼容别名。"""
    return trajectory_sample_to_tensors(sample)
