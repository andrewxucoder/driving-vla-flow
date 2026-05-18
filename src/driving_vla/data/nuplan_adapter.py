from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from driving_vla.data.trajectory_sample import trajectory_sample_to_tensors


class NuPlanTrajectoryDataset(Dataset):
    """基于预处理 `.pt` 文件的 nuPlan 轨迹数据集适配器。

    该类不直接读取原始 nuPlan DB，也不在导入时依赖 nuplan-devkit。
    预处理文件中的每个样本应包含：
    `ego_state`、`history`、`command_id`、`language_command`、
    `future`、`metadata`。
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"未找到 nuPlan 预处理数据：{self.data_path}")

        self.sample_files: list[Path] = []
        self.samples: list[dict[str, Any]] | None = None
        if self.data_path.is_dir():
            self.sample_files = sorted(self.data_path.glob("*.pt"))
            if not self.sample_files:
                raise FileNotFoundError(f"目录中没有 `.pt` 样本文件：{self.data_path}")
        else:
            loaded = torch.load(self.data_path, map_location="cpu", weights_only=False)
            self.samples = self._normalize_loaded_samples(loaded)

    def _normalize_loaded_samples(self, loaded: Any) -> list[dict[str, Any]]:
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            if "samples" in loaded:
                samples = loaded["samples"]
                if not isinstance(samples, list):
                    raise TypeError("`samples` 字段必须是样本 dict 列表")
                return samples
            return [loaded]
        raise TypeError("nuPlan 预处理 `.pt` 文件应保存样本 dict、样本 list，或包含 `samples` 的 dict")

    def _load_raw_sample(self, idx: int) -> dict[str, Any]:
        if self.samples is not None:
            return self.samples[idx]
        sample = torch.load(self.sample_files[idx], map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"样本文件必须保存 dict：{self.sample_files[idx]}")
        return sample

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = trajectory_sample_to_tensors(self._load_raw_sample(idx))
        condition = {
            "ego_state": sample["ego_state"],
            "history": sample["history"],
            "command_id": sample["command_id"],
            "language_command": sample["language_command"],
            "metadata": sample["metadata"],
        }
        return {
            "condition": condition,
            "future": sample["future"],
            "ego_state": sample["ego_state"],
            "history": sample["history"],
            "command_id": sample["command_id"],
            "language_command": sample["language_command"],
            "metadata": sample["metadata"],
        }


def preprocess_raw_nuplan_to_pt(*args: Any, **kwargs: Any) -> None:
    """预留的原始 nuPlan DB 预处理入口。

    只有调用该函数时才检查 nuplan-devkit，避免训练/读取预处理数据时
    对 nuplan-devkit 产生硬依赖。
    """
    try:
        import nuplan  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError("原始 nuPlan DB 预处理需要先安装 nuplan-devkit") from exc

    raise NotImplementedError("请在此处实现从原始 nuPlan DB 到预处理 `.pt` 样本的转换逻辑")
