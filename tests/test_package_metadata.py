"""M1 smoke — public API surface and version metadata."""

from __future__ import annotations


def test_package_version():
    import driving_vla

    assert isinstance(driving_vla.__version__, str)
    assert driving_vla.__version__ == "0.1.0"


def test_public_exports():
    import driving_vla

    assert "DrivingTrajectorySample" in driving_vla.__all__


def test_subpackages_importable():
    """All M1 skeleton subpackages should be importable as namespaces."""
    import driving_vla.data
    import driving_vla.evaluation
    import driving_vla.models
    import driving_vla.policies
    import driving_vla.simulation
    import driving_vla.training

    for pkg in (
        driving_vla.data,
        driving_vla.evaluation,
        driving_vla.models,
        driving_vla.policies,
        driving_vla.simulation,
        driving_vla.training,
    ):
        assert pkg is not None
