"""HashedInstructionEncoder — deterministic baseline."""

from __future__ import annotations

import pytest
import torch

from driving_vla.models import HashedInstructionEncoder


def test_output_shape():
    enc = HashedInstructionEncoder(latent_dim=24)
    out = enc(["change to the left lane", "slow down"])
    assert out.shape == (2, 24)


def test_deterministic_same_input():
    enc = HashedInstructionEncoder(latent_dim=24)
    out1 = enc(["foo"])
    out2 = enc(["foo"])
    assert torch.allclose(out1, out2)


def test_different_inputs_different_outputs():
    enc = HashedInstructionEncoder(latent_dim=24)
    out_a = enc(["alpha"])
    out_b = enc(["beta"])
    assert not torch.allclose(out_a, out_b)


def test_empty_list_returns_empty_tensor():
    enc = HashedInstructionEncoder(latent_dim=24)
    out = enc([])
    assert out.shape == (0, 24)


def test_rejects_non_list():
    enc = HashedInstructionEncoder(latent_dim=24)
    with pytest.raises(TypeError):
        enc("just a string")  # type: ignore[arg-type]


def test_tuple_input_accepted():
    enc = HashedInstructionEncoder(latent_dim=16)
    out = enc(("a", "b", "c"))
    assert out.shape == (3, 16)
