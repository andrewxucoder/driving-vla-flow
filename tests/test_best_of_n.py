"""evaluation.best_of_n — selector + batch selector."""

from __future__ import annotations

import pytest
import torch

from driving_vla.evaluation import best_of_n_batch, best_of_n_select


def test_select_picks_argmax():
    samples = torch.zeros(3, 4, 2)
    samples[1, 0, 0] = 42.0

    def score(s, _):
        # score = sum of first column at first timestep
        return s[:, 0, 0]

    best, idx, scores = best_of_n_select(samples, score)
    assert idx == 1
    assert scores.shape == (3,)
    assert best.shape == (4, 2)


def test_select_empty_raises():
    with pytest.raises(ValueError):
        best_of_n_select(torch.zeros(0, 4, 2), lambda s, _: torch.zeros(0))


def test_select_bad_shape_raises():
    with pytest.raises(ValueError):
        best_of_n_select(torch.zeros(4, 2), lambda s, _: torch.zeros(4))


def test_select_wrong_score_shape_raises():
    samples = torch.zeros(3, 4, 2)
    with pytest.raises(ValueError):
        best_of_n_select(samples, lambda s, _: torch.zeros(2))


def test_batch_returns_best_per_item():
    samples = torch.zeros(2, 3, 4, 2)
    samples[0, 2, 0, 0] = 99.0
    samples[1, 1, 0, 0] = 50.0

    def score(s, _):
        return s[:, 0, 0]

    best, idx, scores = best_of_n_batch(samples, score)
    assert idx.tolist() == [2, 1]
    assert best.shape == (2, 4, 2)
    assert scores.shape == (2, 3)


def test_batch_bad_shape_raises():
    with pytest.raises(ValueError):
        best_of_n_batch(torch.zeros(3, 4, 2), lambda s, _: torch.zeros(3))
