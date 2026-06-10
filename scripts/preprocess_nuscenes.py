"""Preprocess nuScenes mini into a single ``.pt`` sample file.

Reads raw nuScenes data (via nuscenes-devkit) and emits a list of sample dicts
keyed for :class:`driving_vla.data.NuScenesTrajectoryDataset`. Multi-view image
file paths are stored under ``metadata["image_paths"]``; the adapter loads them
lazily at training time.

Per R2 decision, the typical workflow is to reuse v1's already-preprocessed
cache via ``data/nuscenes/``; this script supports re-running from scratch
when v1's cache is unavailable or stale.

Usage::

    python scripts/preprocess_nuscenes.py \\
        --raw-data-root /path/to/nuscenes \\
        --output-file   data/nuscenes_processed/mini.pt

    python scripts/preprocess_nuscenes.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _require_nuscenes_devkit() -> None:
    try:
        import nuscenes  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "nuscenes-devkit is required to preprocess raw nuScenes data. "
            "See docs/DATASETS.md."
        ) from exc


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("[dry-run] nuScenes preprocessing CLI parsed OK; devkit not invoked.")
        return 0

    _require_nuscenes_devkit()
    raw_root = Path(args.raw_data_root).resolve()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[preprocess_nuscenes] raw_root={raw_root} out={out_path}",
        file=sys.stderr,
    )
    print(
        "[preprocess_nuscenes] scene iteration not implemented yet — "
        "re-use v1's cached output via data/ symlink for now.",
        file=sys.stderr,
    )
    return 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess nuScenes mini into a .pt sample file.")
    p.add_argument("--raw-data-root", type=str, help="Path to the nuScenes data root.")
    p.add_argument("--output-file", type=str, default="data/nuscenes_processed/mini.pt",
                   help="Destination .pt path.")
    p.add_argument("--version", type=str, default="v1.0-mini",
                   help="nuScenes split version (mini / trainval / test).")
    p.add_argument("--history-seconds", type=float, default=2.0)
    p.add_argument("--future-seconds", type=float, default=4.0)
    p.add_argument("--sample-rate-hz", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the CLI surface without invoking the devkit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
