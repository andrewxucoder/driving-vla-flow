import torch
from driving_vla.evaluation.metrics import ade, fde


def test_zero_metrics():
    x = torch.zeros(2, 3, 2)
    assert ade(x, x).item() == 0.0
    assert fde(x, x).item() == 0.0
