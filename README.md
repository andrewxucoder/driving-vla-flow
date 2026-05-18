# Driving-VLA-Flow

用于探索**智能驾驶中的视觉-语言-动作建模，以及基于 Flow Matching 的轨迹生成**。


## 核心思路

```text
多模态驾驶上下文
(相机/BEV 代理特征 + 自车状态 + 路线指令 + 语言指令)
        ↓
Driving-VLA 状态编码器
        ↓
条件 Flow Matching 轨迹头
        ↓
未来路径点 / 轨迹片段
        ↓
基于奖励的评估：ADE/FDE、平滑度、碰撞代理、路线一致性
```

## MVP 阶段

1. 玩具轨迹数据集 + 语言条件机动标签。
2. 基线路径点回归头。
3. 条件 Flow Matching 轨迹头。
4. 用于轨迹排序的偏好/奖励评分器。
5. 可选迁移到 nuPlan / nuScenes / CARLA。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_toy_regression.py --config configs/toy_flow.yaml
python scripts/train_toy_flow.py --config configs/toy_flow.yaml
python scripts/visualize_toy_flow.py --config configs/toy_flow.yaml --model all --num-samples 8
python scripts/eval_toy_flow.py --config configs/toy_flow.yaml
```

## 项目结构

```text
configs/                 # 实验配置
src/driving_vla/data/    # 玩具数据集与未来数据集适配器
src/driving_vla/models/  # VLA 编码器、Flow Matching 头、基线头
src/driving_vla/training/# 训练循环与损失函数
src/driving_vla/evaluation/ # 指标与奖励评分器
scripts/                 # 训练/评估入口
tests/                   # 单元测试
```
