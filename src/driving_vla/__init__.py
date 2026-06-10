"""driving-vla-flow — driving policies on NAVSIM, aligned with robot-vla-flow + embodied-world-model."""

__version__ = "0.1.0"

from driving_vla.data.trajectory_sample import DrivingTrajectorySample

__all__ = [
    "DrivingTrajectorySample",
    "__version__",
]
