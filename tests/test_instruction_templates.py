"""instruction_templates — paraphrase coverage + resolve semantics."""

from __future__ import annotations

import pytest

from driving_vla.data import (
    COMMAND_VOCAB,
    INSTRUCTION_TEMPLATES,
    resolve_instruction,
    sample_instruction,
)
from driving_vla.data.instruction_templates import (
    all_paraphrases,
    assert_template_coverage,
)


def test_all_four_commands_have_at_least_12_paraphrases():
    for cid in range(len(COMMAND_VOCAB)):
        assert len(INSTRUCTION_TEMPLATES[cid]) >= 12, f"command_id {cid} short on paraphrases"


def test_coverage_assertion_passes():
    assert_template_coverage(min_paraphrases=12)


def test_paraphrases_unique_per_command():
    for cid in range(len(COMMAND_VOCAB)):
        pool = INSTRUCTION_TEMPLATES[cid]
        assert len(set(pool)) == len(pool), f"duplicates in command_id {cid}"


def test_sample_deterministic_with_seed():
    a = sample_instruction(command_id=1, seed=42)
    b = sample_instruction(command_id=1, seed=42)
    assert a == b


def test_sample_indexed_mode():
    a = sample_instruction(command_id=2, paraphrase_idx=0)
    b = sample_instruction(command_id=2, paraphrase_idx=0)
    assert a == b
    assert a in INSTRUCTION_TEMPLATES[2]


def test_resolve_uses_raw_when_natural_language():
    out = resolve_instruction("Take the second exit at the roundabout.", command_id=0, seed=0)
    assert out == "Take the second exit at the roundabout."


def test_resolve_paraphrases_when_raw_is_vocab_token():
    out = resolve_instruction("keep_lane", command_id=0, seed=1)
    assert out in INSTRUCTION_TEMPLATES[0]


def test_resolve_paraphrases_when_raw_empty():
    out = resolve_instruction("", command_id=3, seed=5)
    assert out in INSTRUCTION_TEMPLATES[3]


def test_resolve_paraphrases_when_raw_none():
    out = resolve_instruction(None, command_id=2, seed=5)
    assert out in INSTRUCTION_TEMPLATES[2]


def test_unknown_command_id_raises():
    with pytest.raises(KeyError):
        sample_instruction(command_id=99)


def test_all_paraphrases_returns_full_pool():
    pool = all_paraphrases(0)
    assert tuple(pool) == INSTRUCTION_TEMPLATES[0]
