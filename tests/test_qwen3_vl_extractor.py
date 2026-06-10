"""Qwen3VLConstraintExtractor — offline tests via injected stub VLM (M8.3).

No HuggingFace download / model load: every test injects ``generate_fn`` so the
parse / retry / cache / schema-conformance logic is exercised on canned strings.
"""

from __future__ import annotations

import torch

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.vla import (
    BaseConstraintExtractor,
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    Qwen3VLConstraintExtractor,
)


def _sample(instruction: str | None = None, **kw) -> DrivingTrajectorySample:
    return DrivingTrajectorySample(
        history=torch.zeros(2, 2), future=torch.zeros(3, 2), instruction=instruction, **kw
    )


def test_satisfies_base_protocol():
    ex = Qwen3VLConstraintExtractor(generate_fn=lambda p, i: "[]")
    assert isinstance(ex, BaseConstraintExtractor)


def test_parses_each_constraint_type():
    raw = (
        '[{"type":"max_speed","value":5.0},'
        '{"type":"lateral_offset","value":2.0},'
        '{"type":"no_go_box","x_min":1,"x_max":2,"y_min":-1,"y_max":1}]'
    )
    ex = Qwen3VLConstraintExtractor(generate_fn=lambda p, i: raw)
    out = ex(_sample("merge left, slow down, cone ahead"))
    assert [type(c) for c in out] == [
        MaxSpeedConstraint,
        LateralOffsetConstraint,
        NoGoBoxConstraint,
    ]
    assert out[0].value == 5.0


def test_empty_array_means_no_constraints():
    ex = Qwen3VLConstraintExtractor(generate_fn=lambda p, i: "[]")
    assert ex(_sample("open highway")) == []


def test_strips_surrounding_prose():
    raw = 'Sure! Here you go:\n[{"type":"max_speed","value":3.0}]\nLet me know if...'
    ex = Qwen3VLConstraintExtractor(generate_fn=lambda p, i: raw)
    out = ex(_sample("school zone"))
    assert len(out) == 1 and isinstance(out[0], MaxSpeedConstraint)


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt, images):
        calls["n"] += 1
        return "not json" if calls["n"] < 3 else '[{"type":"max_speed","value":2.0}]'

    ex = Qwen3VLConstraintExtractor(generate_fn=flaky, max_retries=3)
    out = ex(_sample("construction"))
    assert calls["n"] == 3
    assert len(out) == 1


def test_returns_empty_when_all_retries_fail():
    calls = {"n": 0}

    def always_bad(prompt, images):
        calls["n"] += 1
        return "no json here"

    ex = Qwen3VLConstraintExtractor(generate_fn=always_bad, max_retries=2)
    assert ex(_sample("garbled")) == []
    assert calls["n"] == 2


def test_rejects_hallucinated_fields():
    # `extra="forbid"` in the schema rejects an unknown key -> parse fails -> []
    raw = '[{"type":"max_speed","value":5.0,"made_up_field":99}]'
    ex = Qwen3VLConstraintExtractor(generate_fn=lambda p, i: raw, max_retries=1)
    assert ex(_sample("x")) == []


def test_observation_cache_avoids_second_call():
    calls = {"n": 0}

    def once(prompt, images):
        calls["n"] += 1
        return '[{"type":"max_speed","value":4.0}]'

    ex = Qwen3VLConstraintExtractor(generate_fn=once)
    s = _sample("same scene", command_id=1)
    ex(s)
    ex(s)
    assert calls["n"] == 1  # second call served from cache


def test_caption_in_metadata_feeds_prompt():
    seen = {}

    def capture(prompt, images):
        seen["prompt"] = prompt
        return "[]"

    ex = Qwen3VLConstraintExtractor(generate_fn=capture)
    ex(_sample(metadata={"injected_caption": "a fallen ladder blocks the lane"}))
    assert "fallen ladder" in seen["prompt"]


def test_max_retries_validation():
    import pytest

    with pytest.raises(ValueError):
        Qwen3VLConstraintExtractor(generate_fn=lambda p, i: "[]", max_retries=0)
