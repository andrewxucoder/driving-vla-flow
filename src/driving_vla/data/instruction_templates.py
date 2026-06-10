"""Natural-language instruction paraphrases keyed by command_id.

Each command in :data:`COMMAND_VOCAB` has at least 12 paraphrases covering tone
and length variants. This is a placeholder approach suited to mini datasets —
for production, paraphrases should be VLM-generated or human-authored per scene.
"""

from __future__ import annotations

import random
from typing import Sequence

from driving_vla.data.command_labeling import COMMAND_VOCAB


# At least 12 paraphrases per command id. Order is not significant; deterministic
# sampling is controlled by `sample_instruction(seed=...)`.
INSTRUCTION_TEMPLATES: dict[int, tuple[str, ...]] = {
    0: (  # keep_lane
        "Hold the current lane and drive straight ahead.",
        "Stay in this lane; no maneuver required.",
        "Continue forward inside the existing lane.",
        "Keep lane and maintain current speed.",
        "Track the lane center and keep moving.",
        "Drive on in the same lane.",
        "Carry on along the current lane.",
        "Stay the course in this lane.",
        "No lane change — just keep going.",
        "Press ahead in the current lane.",
        "Remain in lane and proceed normally.",
        "Continue straight along this lane.",
    ),
    1: (  # change_left
        "Move one lane to the left.",
        "Merge into the left-hand lane.",
        "Indicate and steer over to the left lane.",
        "Take the lane on your left side.",
        "Switch into the lane immediately to your left.",
        "Make a smooth lane change to the left.",
        "Signal left and slide into the next lane over.",
        "Drift gently into the left lane.",
        "Perform a left lane change now.",
        "Move over one lane towards the left.",
        "Step into the left lane carefully.",
        "Transition to the lane on your left.",
    ),
    2: (  # change_right
        "Move one lane to the right.",
        "Merge into the right-hand lane.",
        "Indicate and steer over to the right lane.",
        "Take the lane on your right side.",
        "Switch into the lane immediately to your right.",
        "Make a smooth lane change to the right.",
        "Signal right and slide into the next lane over.",
        "Drift gently into the right lane.",
        "Perform a right lane change now.",
        "Move over one lane towards the right.",
        "Step into the right lane carefully.",
        "Transition to the lane on your right.",
    ),
    3: (  # slow_down
        "Ease the speed downward.",
        "Reduce velocity gradually.",
        "Lift off the throttle and decelerate.",
        "Bring the speed down for what is ahead.",
        "Apply gentle braking and slow the vehicle.",
        "Decelerate smoothly before continuing.",
        "Lower the cruising speed now.",
        "Take it slower from this point on.",
        "Coast and let the speed bleed off.",
        "Soft-brake and reduce your pace.",
        "Trim the speed for safety.",
        "Begin slowing the vehicle.",
    ),
}


_MIN_PARAPHRASES = 12


def _validate_command_id(command_id: int) -> int:
    if command_id not in INSTRUCTION_TEMPLATES:
        raise KeyError(
            f"unknown command_id={command_id}; valid {sorted(INSTRUCTION_TEMPLATES.keys())} "
            f"(vocab={list(COMMAND_VOCAB)})"
        )
    return command_id


def sample_instruction(
    command_id: int,
    seed: int | None = None,
    paraphrase_idx: int | None = None,
) -> str:
    """Draw a paraphrase for ``command_id``.

    - With ``paraphrase_idx`` set, return that fixed index (mod pool size).
    - Otherwise build a :class:`random.Random` from ``seed`` and pick uniformly.

    Both modes are deterministic given identical inputs.
    """
    cid = _validate_command_id(int(command_id))
    pool = INSTRUCTION_TEMPLATES[cid]
    if paraphrase_idx is not None:
        return pool[int(paraphrase_idx) % len(pool)]
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(pool)


def assert_template_coverage(min_paraphrases: int = _MIN_PARAPHRASES) -> None:
    for cid, pool in INSTRUCTION_TEMPLATES.items():
        if len(pool) < min_paraphrases:
            raise AssertionError(
                f"command_id={cid} has only {len(pool)} paraphrases, need ≥ {min_paraphrases}"
            )


def all_paraphrases(command_id: int) -> Sequence[str]:
    return INSTRUCTION_TEMPLATES[_validate_command_id(int(command_id))]


def resolve_instruction(
    raw: str | None,
    command_id: int,
    seed: int | None = None,
) -> str:
    """Upgrade a raw command label into a natural-language paraphrase.

    Rules:
    - If ``raw`` is non-empty and is NOT a COMMAND_VOCAB token, return it as-is
      (it is already natural language).
    - Otherwise (empty / None / vocab token), draw a paraphrase via
      :func:`sample_instruction`.
    """
    if raw and raw not in COMMAND_VOCAB:
        return raw
    return sample_instruction(int(command_id), seed=seed)


__all__ = [
    "INSTRUCTION_TEMPLATES",
    "all_paraphrases",
    "assert_template_coverage",
    "resolve_instruction",
    "sample_instruction",
]
