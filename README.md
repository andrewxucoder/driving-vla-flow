# Driving-VLA-Flow

A research-oriented GitHub project for **Vision-Language-Action modeling with Flow Matching trajectory generation for intelligent driving**.

> Scope: algorithm research prototype, not a production autonomous driving stack.

## Core idea

```text
Multi-modal driving context
(camera/BEV proxy + ego state + route command + language instruction)
        ↓
Driving-VLA state encoder
        ↓
Conditional Flow Matching trajectory head
        ↓
Future waypoints / trajectory chunks
        ↓
Reward-based evaluation: ADE/FDE, smoothness, collision proxy, route adherence
```

## MVP stages

1. Toy trajectory dataset + language-conditioned maneuver labels.
2. Baseline waypoint regression head.
3. Conditional Flow Matching trajectory head.
4. Preference/reward scorer for trajectory ranking.
5. Optional migration to nuPlan / nuScenes / CARLA.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_toy_flow.py --config configs/toy_flow.yaml
python scripts/eval_toy_flow.py --config configs/toy_flow.yaml
```

## Repository layout

```text
configs/                 # experiment configs
src/driving_vla/data/    # toy and future dataset adapters
src/driving_vla/models/  # VLA encoder, flow matching head, baseline head
src/driving_vla/training/# training loops and losses
src/driving_vla/evaluation/ # metrics and reward scorer
scripts/                 # train/eval entry points
tests/                   # unit tests
```
