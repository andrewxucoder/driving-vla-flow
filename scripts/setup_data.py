"""Create symlinks from v2's data/ into v1's already-downloaded dataset caches.

Per R2 decision (2026-05-27), v2 reuses v1's nuPlan / nuScenes / NAVSIM caches
to avoid ~30GB of re-downloading. The v2 codebase is independent of v1's code;
only the bytes on disk are shared.

Usage::

    python scripts/setup_data.py             # link all three datasets
    python scripts/setup_data.py --check     # verify existing symlinks
    python scripts/setup_data.py --force     # overwrite existing symlinks

If v1 has not populated a dataset yet, the corresponding symlink is skipped
with a warning (not an error).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_V1_ROOT = Path("/Users/andrew/Documents/project/driving-vla-flow-backup")
_SHARED_DATASET_ROOT = Path("/Users/andrew/Documents/project/dataset")
_V2_ROOT = Path(__file__).resolve().parents[1]
_DATASETS: tuple[str, ...] = ("nuplan", "nuscenes", "navsim")


def _v1_candidates(name: str) -> tuple[Path, ...]:
    """Candidate source paths for ``name``'s cache.

    First check the shared ``/dataset`` tree (preferred since 2026-05-28), then
    fall back to v1's local copies.
    """
    shared: tuple[Path, ...]
    if name == "nuplan":
        shared = (_SHARED_DATASET_ROOT / "nuplan" / "processed",)
    elif name == "nuscenes":
        shared = (_SHARED_DATASET_ROOT / "nuscenes",)
    elif name == "navsim":
        shared = (_SHARED_DATASET_ROOT / "navsim" / "dataset",)
    else:
        shared = ()
    return shared + (
        _V1_ROOT / "data" / "processed" / f"{name}_mini",
        _V1_ROOT / "data" / f"{name}_mini",
        _V1_ROOT / "data" / name,
    )


def _link(name: str, *, force: bool) -> tuple[bool, str]:
    dst = _V2_ROOT / "data" / name

    src: Path | None = None
    for cand in _v1_candidates(name):
        if cand.exists():
            src = cand
            break
    if src is None:
        return False, f"v1 cache missing for {name} (checked: {[str(c) for c in _v1_candidates(name)]})"

    if dst.exists() or dst.is_symlink():
        if not force:
            return False, f"already present: {dst} (use --force to overwrite)"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            return False, f"refusing to overwrite real directory: {dst}"

    dst.symlink_to(src.resolve())
    return True, f"linked: {dst} -> {src.resolve()}"


def _check(name: str) -> tuple[bool, str]:
    dst = _V2_ROOT / "data" / name
    if not dst.exists() and not dst.is_symlink():
        return False, f"missing: {dst}"
    if not dst.is_symlink():
        return False, f"not a symlink: {dst}"
    target = dst.resolve()
    return True, f"ok: {dst} -> {target}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Symlink v1 dataset caches into v2/data/.")
    parser.add_argument("--check", action="store_true", help="Just verify symlinks; do not create.")
    parser.add_argument("--force", action="store_true", help="Replace existing symlinks.")
    args = parser.parse_args(argv)

    (_V2_ROOT / "data").mkdir(parents=True, exist_ok=True)

    ok_all = True
    for name in _DATASETS:
        if args.check:
            ok, msg = _check(name)
        else:
            ok, msg = _link(name, force=args.force)
        prefix = "OK" if ok else "..."
        print(f"  [{prefix}] {msg}", file=sys.stderr)
        if not ok and not args.check:
            # v1 cache may legitimately be absent for some datasets; we don't
            # fail the whole command — return non-zero only if --check fails.
            pass
        if args.check and not ok:
            ok_all = False

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
