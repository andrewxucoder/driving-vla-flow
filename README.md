# driving-vla-flow

> 面向**驾驶轨迹规划**的分层 VLA(Vision-Language-Action)**研究框架**:低频语言 / VLM 产生**结构化约束**,高频 Flow / Diffusion policy 生成候选轨迹,再经 `reward · Best-of-N · Flow-DPO` 完成**约束下的轨迹选择与策略优化**。三仓 Physical Intelligence 生态中的**驾驶域**(与 robot-vla-flow 机器人操作域平级,共享 embodied-world-model 枢纽)。

**系统级主线**:生成式策略「提议」、外部验证器「把关」——本仓让 **VLM / 几何约束**充当轨迹的外部验证器(出约束、不出轨迹);跨仓再由 world model 作统一**评价器 / 偏好信号源**,把「生成 + 验证」闭环回流到 driving 与 robot 两域。

## TL;DR — 是什么 / 不是什么

**是**:**把 VLM 当「出约束的」而非「出轨迹的」的分层 driving 研究框架**——让你快速搭起并对比 driving planning 想法:三类轨迹生成策略、语言/VLM 约束建模、候选轨迹 Best-of-N 选择、偏好 / DPO 优化,以及与共享 world model 的闭环对接,全部用统一 schema 与 policy 接口串起来。整套在一台 M4 Pro 笔记本上分钟级可跑通。

**不是**:不是完整自动驾驶系统(不覆盖感知 / 预测 / 规控全栈);不是端到端「VLM 直接输出轨迹」——走双频分层 VLA(VLM 出约束、policy 出轨迹),理由见 § 方法框架。

## 这套系统能做什么(能力 × 可开展的实验 × 验证状态)

每一行 = 一项**已实现的能力** + 它**支持开展的实验** + 当前**想法验证状态**(✅ 已验证 / 🟡 跑通待量化 / 🟢 接口就位待真跑):


| 能力                                                               | 能开展的实验                                       | 状态                  |
| ---------------------------------------------------------------- | -------------------------------------------- | ------------------- |
| 三类轨迹策略(Regression / Diffusion / Flow,共享 `TrajectoryDenoiser` 骨干) | 同骨干下三策略**公平对比**                              | ✅                   |
| Best-of-N 候选采样 + reward 选择                                       | BoN 对 ADE / 碰撞 / 约束满足的影响                     | ✅                   |
| VLM-as-constraint(语言 → 结构化 `TrajectoryConstraint` → reward)      | 约束机制有效性;oracle vs 真 VLM                      | ✅                   |
| Preference 构造 + Flow-DPO                                         | perturbation pair vs **WM-based pair** 的偏好优化 | 🟡 before/after 待重训 |
| 跨数据集 adapter(nuPlan / nuScenes / NAVSIM)                         | offline ↔ 闭环 gap 分析、跨集泛化                     | ✅                   |
| 与 world model 闭环对接(共享 `ClosedLoopSimulator` Protocol)            | WM 作闭环 sim / 偏好信号源                           | 🟡 待真跑              |
| route / 导航 prior 注入(`RouteEncoder`)                              | route-aware vs no-route ablation             | 🟢                  |
| 真感知(DINOv2 / SigLIP)+ 真 VLM text encoder                         | camera vs state-only、真文本 steering            | 🟢                  |


研究入口的具体落点:训练 / 评测 / DPO 走 `scripts/` 下 14 个 CLI。

## 这套系统验证了哪些想法

把它当成一个「假设 → 能不能跑通 / 成不成立」的验证台,目前的结论(数字只作**证据**,不作 leaderboard):

- **「结构化约束 → reward → 候选选择」机制成不成立?** → 成立:oracle 约束把注入障碍碰撞 `0.10 → 0.033`、约束满足 `0.90 → 0.967`。真 VLM(Qwen3-VL-2B)当前持平 baseline(几何精度不足,**诚实负结果**;下一步喂精确坐标 + 真图像)。
- **Diffusion 在米级轨迹上到底能不能训好?** → 能;此前「训不动」的根因是 noise schedule(`T=100` 时信号没被打碎),改 `T=1000` 后 nuPlan val ADE `6.4 → 0.47`。
- **offline 指标能不能迁移到闭环?** → 此前 23× 的 offline→NAVSIM gap 主要是 `ego_state` 布局 bug、非建模上限;根治后 NAVSIM regression `6.77 → 2.44`——评测一致性比刷分更关键。

## Quick Start

```bash
# 1. 安装(editable;会带出 embodied-world-model 的 path 依赖)
pip install -e .                     # 可选 extras: .[dev] / .[nuplan] / .[nuscenes] / .[vlm]

# 2. 数据(软链已有的磁盘缓存,零重复占用)
python scripts/setup_data.py         # 链接所有可用数据集
python scripts/setup_data.py --check # 校验软链

# 3. 训练 3 个 head(nuPlan 90k × 3 epoch,M4 Pro MPS 总计 ~78s)
python scripts/train_nuplan_regression.py --data-path data/nuplan/train.pt \
    --output outputs/nuplan_regression/model.pt --epochs 3 --batch-size 64 \
    --horizon 30 --traj-dim 2 --state-dim 5
python scripts/train_nuplan_flow.py      --data-path data/nuplan/train.pt \
    --output outputs/nuplan_flow/model.pt --epochs 3 --batch-size 64 \
    --horizon 30 --traj-dim 2 --state-dim 5
python scripts/train_nuplan_diffusion.py --data-path data/nuplan/train.pt \
    --output outputs/nuplan_diffusion/model.pt --epochs 3 --batch-size 64 \
    --horizon 30 --traj-dim 2 --state-dim 5 --diffusion-steps 1000

# 4. 评测 / VLM 约束 / DPO —— 完整命令与 ~3min 一键复现见 reports/tech_report.md Appendix A
python scripts/eval_nuplan_heads.py --data-path data/nuplan/val.pt \
    --regression-ckpt outputs/nuplan_regression/model.pt \
    --flow-ckpt outputs/nuplan_flow/model.pt \
    --diffusion-ckpt outputs/nuplan_diffusion/model.pt \
    --bon-n 8 --horizon 30 --traj-dim 2 --state-dim 5 --output outputs/round2_nuplan.json
```

## 方法框架

驾驶域走**双频分层 VLA**,而非端到端「VLM→轨迹」——因为驾驶缺 RT-X 量级的 `(vision, language, action)` 配对数据,且 7B+ VLM 单次推理 200–500ms 撑不起 10–20Hz 控制环(Waymo / Wayve / 小鹏都在收敛到该范式)。

```
低频 (1-3 Hz):Camera + Route → VLM → 结构化 TrajectoryConstraint
                                              │
高频 (10-20 Hz):Flow / Diffusion policy  ──► Best-of-N under constraints ──► 轨迹
```

四层实现:**数据与样本抽象**(nuPlan / nuScenes / NAVSIM adapter → `DrivingTrajectorySample`)→ **轨迹生成策略**(Regression / Diffusion / Flow,共享 `TrajectoryDenoiser` 骨干保证公平对比)→ **结构化约束与选择**(`TrajectoryConstraint` schema → `constraint_to_reward` → BoN)→ **偏好优化**(perturbation / WM-based preference pairs → Flow-DPO)。

## 三仓 Physical Intelligence 生态

三个平级模块 + 一个共享枢纽,统一探索「智能体如何用语言、约束、世界模型与偏好信号,在物理任务中形成可执行、可评估、可优化的策略」:


| repo                                             | 角色         | 关注                                                             |
| ------------------------------------------------ | ---------- | -------------------------------------------------------------- |
| **driving-vla-flow**(本仓)                         | 驾驶域 VLA 规划 | trajectory flow/diffusion · VLM-as-constraint · BoN · Flow-DPO |
| [robot-vla-flow](https://github.com/andrewxucoder/robot-vla-flow)             | 机器人操作域 VLA | manipulation policy · learned RM · DPO · 闭环 rollout            |
| [embodied-world-model](https://github.com/andrewxucoder/embodied-world-model/tree/dev) | **共享枢纽**   | 跨域 latent world model · 闭环 simulator · WM-based 偏好信号源          |


## 结果边界(读数前必读)

- **碰撞指标来自 eval 脚本的注入式障碍模型**,nuPlan mini 数据本身无 per-agent 障碍框,**不等于**完整真实交通参与者仿真。
- NAVSIM 的 `geometric_score` 是 toy 公式(`1 - ade/5 - 0.5·collision`),**非真 PDM-Score**(待接 navsim 官方 PDMScore 管线)。
- 真 VLM(Qwen3-VL-2B)约束**当前持平 baseline**(N=30,方向性结论);几何精度提升与真图像输入在路线图上。
- Flow-DPO pipeline 已打通,**before/after 需在同一 eval set 重训刷新**。
- 当前评测为 state-only;camera / 真感知(DINOv2 / SigLIP)与真 VLM text encoder 的**接口已就位,待真训对比**。

## 项目结构

```text
src/driving_vla/
├── data/          # nuPlan / nuScenes / NAVSIM adapters + DrivingTrajectorySample + 扰动/偏好数据
├── models/        # encoders + regression/diffusion/flow heads + 共享 TrajectoryDenoiser + route/vision/VLM 编码器
├── policies/      # BasePolicy 契约 + 三个 head 的 policy wrapper
├── training/      # train loop + optim + Flow-DPO
├── evaluation/    # metrics · reward · Best-of-N · learned reward · simulator · NAVSIM runner
├── simulation/    # 闭环 sim glue(实现在 evaluation/)
└── vla/           # TrajectoryConstraint schema · extractor · constraint→reward · Qwen3-VL · 合成障碍

scripts/           # 14 个 CLI:preprocess / train_{reg,flow,diffusion,vlm,image} / eval_{nuplan,navsim,vla} / DPO
outputs/           # run artifacts
tests/             # 338 passed(ruff + mypy clean)
```

