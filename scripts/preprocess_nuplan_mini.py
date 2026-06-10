"""Preprocess nuPlan mini scenarios into ``.pt`` sample files.

Reads raw nuPlan ``.db`` logs (via nuplan-devkit) and emits one preprocessed
sample dict per scenario, in the format consumed by
:class:`driving_vla.data.NuPlanTrajectoryDataset` (``{"samples": [...],
"command_vocab": [...]}`` with ``train.pt`` / ``val.pt`` / ``metadata.json``).

v2 owns this pipeline outright — it imports only ``driving_vla`` helpers and has
no dependency on v1. Each sample's ``ego_state`` is the **full kinematic
vector** ``[vx, vy, ax, ay, yaw_rate]`` in the ego (vehicle-body) frame:

- ``vx, vy``     longitudinal / lateral velocity (m/s), from the devkit's
                 ``dynamic_car_state.rear_axle_velocity_2d``
- ``ax, ay``     longitudinal / lateral acceleration (m/s²), from
                 ``rear_axle_acceleration_2d``
- ``yaw_rate``   angular velocity (rad/s), from ``angular_velocity``

This matches the schema documented in ``ARCHITECTURE.md`` and assumed by the
reward (``expected_progress_penalty`` reads ``vx`` from slot 0), the M8
constraint extractors, and ``navsim_runner``. (The v1 cache that previously
backed ``data/nuplan`` encoded ``[0, 0, 0, |v|, |a|]`` — only speed/accel
*magnitudes*, with vx/vy/ax/yaw all dropped — which silently broke every
speed-conditioned consumer; see tech_report §7.5.)

Usage::

    python scripts/preprocess_nuplan_mini.py \\
        --raw-data-root /path/to/nuplan \\
        --output-dir    data/nuplan_processed/ \\
        --max-scenarios 200          # subset for a quick validation run

    python scripts/preprocess_nuplan_mini.py --dry-run   # CLI surface only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driving_vla.data.command_labeling import COMMAND_TO_ID, COMMAND_VOCAB, infer_command
from driving_vla.data.coordinate_transform import global_to_ego_frame
from driving_vla.data.instruction_templates import sample_instruction


def _progress_iter(iterable: Iterable[Any], desc: str, total: int | None = None) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="scenario")


def _require_nuplan_devkit() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_parallel import (
            SingleMachineParallelExecutor,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "nuplan-devkit is required to preprocess raw nuPlan logs. See docs/DATASETS.md."
        ) from exc
    return NuPlanScenarioBuilder, ScenarioFilter, SingleMachineParallelExecutor


def _resolve_db_files(raw_data_root: Path, db_files: str | None) -> str:
    if db_files is not None:
        return db_files
    for cand in (
        raw_data_root,
        raw_data_root / "nuplan-v1.1" / "splits" / "mini",
        raw_data_root / "nuplan-v1.1" / "mini",
        raw_data_root / "splits" / "mini",
        raw_data_root / "mini",
    ):
        if cand.is_file() and cand.suffix == ".db":
            return str(cand)
        if cand.is_dir() and any(cand.glob("*.db")):
            return str(cand)
    raise FileNotFoundError(f"no nuPlan .db files under {raw_data_root}")


def _resolve_map_root(raw_data_root: Path, map_root: str | None) -> str:
    if map_root is not None:
        return map_root
    for cand in (raw_data_root / "maps", raw_data_root.parent / "maps"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"no nuPlan maps directory under {raw_data_root}")


def _filter_parameters(limit_total_scenarios: int | None, shuffle: bool) -> dict[str, Any]:
    return {
        "scenario_types": None,
        "scenario_tokens": None,
        "log_names": None,
        "map_names": None,
        "num_scenarios_per_type": None,
        "limit_total_scenarios": limit_total_scenarios,
        "timestamp_threshold_s": None,
        "ego_displacement_minimum_m": None,
        "expand_scenarios": False,
        "remove_invalid_goals": False,
        "shuffle": shuffle,
    }


def _build_scenarios(args: argparse.Namespace) -> list[Any]:
    NuPlanScenarioBuilder, ScenarioFilter, SingleMachineParallelExecutor = _require_nuplan_devkit()
    raw_root = Path(args.raw_data_root).expanduser()
    db_files = _resolve_db_files(raw_root, args.db_files)
    map_root = _resolve_map_root(raw_root, args.map_root)
    args.resolved_db_files = db_files
    args.resolved_map_root = map_root
    print(
        {"stage": "resolve_paths", "raw_data_root": str(raw_root),
         "map_root": map_root, "db_files": db_files},
        flush=True,
    )
    builder = NuPlanScenarioBuilder(
        str(raw_root), map_root, args.sensor_root, db_files, args.map_version
    )
    scenario_filter = ScenarioFilter(
        **_filter_parameters(args.max_scenarios, args.shuffle_scenarios)
    )
    worker = SingleMachineParallelExecutor(use_process_pool=args.use_process_pool)
    print({"stage": "load_scenarios", "max_scenarios": args.max_scenarios}, flush=True)
    scenarios = list(
        _progress_iter(
            builder.get_scenarios(scenario_filter, worker),
            desc="load_scenarios",
            total=args.max_scenarios,
        )
    )
    print({"stage": "loaded_scenarios", "num_scenarios": len(scenarios)}, flush=True)
    return scenarios


def _xy_heading(ego_state: Any) -> tuple[float, float, float]:
    pose = getattr(ego_state, "center", None) or getattr(ego_state, "rear_axle", None)
    if pose is None:
        raise AttributeError("EgoState lacks a center/rear_axle pose")
    return float(pose.x), float(pose.y), float(pose.heading)


def _vec2(dynamic_state: Any, *names: str) -> tuple[float, float]:
    """Return (x, y) of the first present 2D vector attribute, else (0, 0)."""
    for name in names:
        v = getattr(dynamic_state, name, None)
        if v is not None:
            return float(v.x), float(v.y)
    return 0.0, 0.0


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _ego_kinematics(current: Any, next_state: Any | None, dt: float) -> np.ndarray:
    """``[vx, vy, ax, ay, yaw_rate]`` in the ego (vehicle-body) frame.

    vx/vy/ax/ay come from the devkit's recorded dynamic state. yaw_rate is
    *derived* from the heading change to ``next_state`` over ``dt`` — raw
    nuPlan recorded states leave ``angular_velocity`` at 0 (it's populated by
    the bicycle model only during simulation), so a finite-difference of the
    logged heading is the only honest source from recorded data.
    """
    dyn = getattr(current, "dynamic_car_state", None)
    if dyn is None:
        return np.zeros(5, dtype=np.float32)
    vx, vy = _vec2(dyn, "rear_axle_velocity_2d", "center_velocity_2d")
    ax, ay = _vec2(dyn, "rear_axle_acceleration_2d", "center_acceleration_2d")
    yaw_rate = 0.0
    if next_state is not None and dt > 0:
        yaw_rate = _wrap_angle(_xy_heading(next_state)[2] - _xy_heading(current)[2]) / dt
    return np.array([vx, vy, ax, ay, yaw_rate], dtype=np.float32)


def _states_to_ego_points(states: Iterable[Any], ego_pose: np.ndarray) -> np.ndarray:
    xy = [_xy_heading(s)[:2] for s in states]
    if not xy:
        return np.zeros((0, 2), dtype=np.float32)
    return global_to_ego_frame(np.array(xy, dtype=np.float32), ego_pose).astype(np.float32)


def _scenario_metadata(scenario: Any) -> dict[str, Any]:
    md: dict[str, Any] = {}
    for name in ("token", "scenario_name", "scenario_type", "log_name", "map_name"):
        value = getattr(scenario, name, None)
        if value is not None:
            md[name] = str(value)
    return md


def _extract_sample(scenario: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    current = scenario.get_ego_state_at_iteration(0)
    ego_x, ego_y, ego_heading = _xy_heading(current)
    ego_pose = np.array([ego_x, ego_y, ego_heading], dtype=np.float32)
    past_states = list(
        scenario.get_ego_past_trajectory(
            iteration=0, time_horizon=args.past_horizon, num_samples=args.past_samples
        )
    )
    future_states = list(
        scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=args.future_horizon, num_samples=args.future_samples
        )
    )
    if len(future_states) == 0:
        return None

    history = _states_to_ego_points(past_states, ego_pose)
    future = _states_to_ego_points(future_states, ego_pose)
    command = infer_command(
        future,
        slow_down_ratio=args.slow_down_ratio,
        lane_change_threshold=args.lane_change_threshold,
    )
    command_id = COMMAND_TO_ID[command]
    metadata = _scenario_metadata(scenario)
    metadata.update(
        {
            "global_ego_pose": ego_pose.tolist(),
            "past_horizon": args.past_horizon,
            "future_horizon": args.future_horizon,
        }
    )
    seed = hash((metadata.get("token", ""), command_id)) & 0xFFFFFFFF
    yaw_dt = args.future_horizon / max(args.future_samples, 1)
    return {
        "ego_state": _ego_kinematics(current, future_states[0], yaw_dt),
        "history": history,
        "future": future,
        "command_id": command_id,
        "language_command": command,
        "instruction": sample_instruction(command_id, seed=seed),
        "metadata": metadata,
    }


def _split_samples(
    samples: list[dict[str, Any]], val_ratio: float, seed: int, split_by: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    if split_by == "log_name":
        groups: dict[str, list] = defaultdict(list)
        for s in samples:
            groups[s.get("metadata", {}).get("log_name", "")].append(s)
        log_names = sorted(groups.keys())
        if len(log_names) > 1:
            rng.shuffle(log_names)
            val_count = max(1, round(len(log_names) * val_ratio))
            val_set = set(log_names[:val_count])
            train = [s for s in samples if s.get("metadata", {}).get("log_name", "") not in val_set]
            val = [s for s in samples if s.get("metadata", {}).get("log_name", "") in val_set]
            return train, val
        print("warning: only 1 log_name group — falling back to random split", flush=True)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_count = min(max(int(round(len(shuffled) * val_ratio)), 1 if len(shuffled) > 1 else 0), len(shuffled))
    return shuffled[val_count:], shuffled[:val_count]


def _save_outputs(
    samples: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace, skipped: int
) -> None:
    import torch

    # Fail-fast guard against the regression that backed the old v1 cache, where
    # ego_state collapsed to [0, 0, 0, |v|, |a|] — the velocity/accel components
    # were silently dropped. Assert the signed kinematic columns carry signal.
    ego = np.stack([np.asarray(s["ego_state"], dtype=np.float64) for s in samples], axis=0)
    degenerate = [
        name for idx, name in enumerate(("vx", "vy", "ax", "ay", "yaw_rate"))
        if np.all(np.abs(ego[:, idx]) < 1e-6)
    ]
    if degenerate:
        raise RuntimeError(
            f"ego_state columns {degenerate} are identically zero across all "
            f"{len(samples)} samples — kinematic extraction is broken. "
            "Expected layout [vx, vy, ax, ay, yaw_rate]."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train, val = _split_samples(samples, args.val_ratio, args.seed, args.split_by)
    torch.save({"samples": train, "command_vocab": COMMAND_VOCAB}, output_dir / "train.pt")
    torch.save({"samples": val, "command_vocab": COMMAND_VOCAB}, output_dir / "val.pt")
    metadata = {
        "num_samples": len(samples),
        "num_train": len(train),
        "num_val": len(val),
        "num_skipped": skipped,
        "ego_state_layout": ["vx", "vy", "ax", "ay", "yaw_rate"],
        "command_vocab": COMMAND_VOCAB,
        "command_counts": dict(Counter(s["language_command"] for s in samples)),
        "raw_data_root": args.raw_data_root,
        "map_root": getattr(args, "resolved_map_root", args.map_root),
        "db_files": getattr(args, "resolved_db_files", args.db_files),
        "map_version": args.map_version,
        "past_horizon": args.past_horizon,
        "past_samples": args.past_samples,
        "future_horizon": args.future_horizon,
        "future_samples": args.future_samples,
        "val_ratio": args.val_ratio,
        "split_by": args.split_by,
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("[dry-run] nuPlan preprocessing CLI parsed OK; devkit not invoked.")
        return 0

    args.raw_data_root = args.data_root
    scenarios = _build_scenarios(args)
    samples: list[dict[str, Any]] = []
    skipped = 0
    for scenario in _progress_iter(scenarios, desc="extract_samples", total=len(scenarios)):
        sample = _extract_sample(scenario, args)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    if not samples:
        raise RuntimeError("no nuPlan mini samples extracted")

    output_dir = Path(args.output_dir)
    _save_outputs(samples, output_dir, args, skipped=skipped)
    print(
        {"processed": len(samples), "skipped": skipped, "output_dir": str(output_dir)},
        flush=True,
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess nuPlan mini logs into .pt samples.")
    p.add_argument("data_root", nargs="?", type=str, help="Path to the nuPlan root.")
    p.add_argument("--data-root", "--raw-data-root", dest="data_root_arg", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="data/nuplan_processed")
    p.add_argument("--db-files", type=str, default=None, help="Explicit .db path or directory.")
    p.add_argument("--map-root", type=str, default=None)
    p.add_argument("--sensor-root", type=str, default=None)
    p.add_argument("--map-version", type=str, default="nuplan-maps-v1.0")
    p.add_argument("--past-horizon", type=float, default=2.0)
    p.add_argument("--future-horizon", type=float, default=4.0)
    p.add_argument("--past-samples", type=int, default=10)
    p.add_argument("--future-samples", type=int, default=30)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scenarios", type=int, default=None,
                   help="Limit total scenarios (use a small value for a quick validation run).")
    p.add_argument("--shuffle-scenarios", action="store_true")
    p.add_argument("--use-process-pool", action="store_true")
    p.add_argument("--split-by", type=str, default="log_name", choices=["log_name", "random"])
    p.add_argument("--lane-change-threshold", type=float, default=1.5)
    p.add_argument("--slow-down-ratio", type=float, default=0.65)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the CLI surface without invoking the devkit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.data_root = args.data_root_arg or args.data_root
    if not args.dry_run and args.data_root is None:
        build_argparser().error("nuPlan data root required: pass --data-root or the positional arg")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
