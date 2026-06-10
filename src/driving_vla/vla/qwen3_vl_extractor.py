"""Qwen3-VL constraint extractor (M8.3).

A :class:`BaseConstraintExtractor` backed by the local Qwen3-VL-2B-Instruct
checkpoint. It maps a driving observation (front camera + instruction/caption)
to a list of :class:`TrajectoryConstraint` by prompting the VLM with a few-shot
template and parsing its JSON output through the M8.1 schema.

Design choices (consistent with the rest of the model layer):

- **Lazy heavy deps.** ``transformers`` + the 2B checkpoint are only loaded on
  the first *real* inference call (:meth:`_ensure_loaded`). The base install
  and the unit tests never touch them.
- **Dependency injection for tests.** Pass ``generate_fn=(prompt, images) ->
  str`` to bypass the model entirely; the extractor's parsing / retry / cache
  logic is then testable offline with canned VLM strings (mirrors the
  ``backbone_module`` injection in :mod:`driving_vla.models.vlm_text_encoder`).
- **Constrained decoding via retry.** ``outlines`` would give grammar-level JSON
  guarantees but is an optional extra and not assumed installed; the shipped
  fallback is parse-and-retry (up to ``max_retries``), with strict pydantic
  validation (``extra="forbid"``) rejecting hallucinated fields. An empty list
  is returned if every attempt fails — "no constraints derived", which the BoN
  layer treats as the unconstrained case.
- **Observation cache.** Repeated calls on the same observation (common in
  ablation sweeps) reuse the first result instead of re-invoking the VLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any, Union

from driving_vla.data.trajectory_sample import DrivingTrajectorySample
from driving_vla.vla.constraint_schema import (
    LateralOffsetConstraint,
    MaxSpeedConstraint,
    NoGoBoxConstraint,
    parse_constraints,
)

_AnyConstraint = Union[LateralOffsetConstraint, MaxSpeedConstraint, NoGoBoxConstraint]

DEFAULT_MODEL_PATH = "/Users/andrew/Documents/project/dataset/model/Qwen3-VL-2B-Instruct"

_SYSTEM_PROMPT = (
    "You are a driving co-pilot. Given a front-camera view and a short scene "
    "description, output ONLY a JSON array of trajectory constraints the ego "
    "vehicle should obey. Each constraint is one of:\n"
    '  {"type":"lateral_offset","value":<metres, + left / - right>}\n'
    '  {"type":"max_speed","value":<m/s, >= 0>}\n'
    '  {"type":"no_go_box","x_min":..,"x_max":..,"y_min":..,"y_max":..}'
    "  (ego frame, x forward, y left, metres)\n"
    "Optional fields on any constraint: confidence (0..1), reason (string), "
    "valid_until_seconds (>0). Output [] if no constraint applies. Output the "
    "JSON array and nothing else."
)

# Text-only few-shot examples (scene description -> constraint JSON). The real
# query additionally carries the camera image; examples stay text-only so the
# prompt does not require per-example images.
_FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("Construction zone ahead, workers on the right shoulder.",
     '[{"type":"max_speed","value":5.0,"reason":"construction zone"}]'),
    ("A parked truck blocks the right half of my lane.",
     '[{"type":"lateral_offset","value":2.0,"reason":"merge left around parked truck"}]'),
    ("Traffic cone at roughly 12 m ahead and 1 m to my right.",
     '[{"type":"no_go_box","x_min":11.0,"x_max":13.0,"y_min":-1.5,"y_max":-0.5,'
     '"reason":"cone"}]'),
    ("Open highway, clear lane, nothing unusual.",
     "[]"),
    ("School zone, children near a crosswalk on the left.",
     '[{"type":"max_speed","value":4.0,"reason":"school zone"},'
     '{"type":"lateral_offset","value":-0.5,"reason":"edge away from crosswalk"}]'),
)


class Qwen3VLConstraintExtractor:
    """``(observation) -> list[TrajectoryConstraint]`` via Qwen3-VL-2B."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        max_retries: int = 3,
        device: str = "auto",
        max_new_tokens: int = 256,
        generate_fn: Callable[[str, list[Any]], str] | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.model_path = model_path
        self.max_retries = max_retries
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._generate_fn = generate_fn
        self._model: Any = None
        self._processor: Any = None
        self._cache: dict[str, list[_AnyConstraint]] = {}

    # ---- public API -------------------------------------------------------

    def __call__(self, observation: DrivingTrajectorySample) -> list[_AnyConstraint]:
        key = self._obs_key(observation)
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)

        prompt = self._build_prompt(observation)
        images = self._extract_images(observation)
        result: list[_AnyConstraint] = []
        for _ in range(self.max_retries):
            raw = self._generate(prompt, images)
            parsed = self._try_parse(raw)
            if parsed is not None:
                result = parsed
                break
        self._cache[key] = list(result)
        return result

    # ---- prompt / parsing -------------------------------------------------

    def _build_prompt(self, observation: DrivingTrajectorySample) -> str:
        caption = ""
        if observation.metadata:
            caption = str(observation.metadata.get("injected_caption", "") or "")
        instruction = observation.instruction or ""
        scene = caption or instruction or "Front camera view; describe and constrain."
        shots = "\n".join(f"Scene: {s}\nConstraints: {c}" for s, c in _FEW_SHOT)
        return f"{_SYSTEM_PROMPT}\n\n{shots}\n\nScene: {scene}\nConstraints:"

    @staticmethod
    def _try_parse(raw: str) -> list[_AnyConstraint] | None:
        """Extract the first JSON array from ``raw`` and validate it.

        Returns ``None`` (triggering a retry) on malformed JSON or any field
        that violates the constraint schema.
        """
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if match is None:
            return None
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list):
            return None
        try:
            return list(parse_constraints(items))
        except Exception:
            return None

    # ---- VLM backend (injectable) ----------------------------------------

    def _generate(self, prompt: str, images: list[Any]) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt, images)
        self._ensure_loaded()
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - exercised only with real deps
            raise ImportError(
                "Qwen3VLConstraintExtractor needs the `[vlm]` extra (transformers). "
                "Install it, or pass generate_fn=... for an offline stub."
            ) from exc
        dtype = torch.float32 if self.device == "cpu" else "auto"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=dtype, device_map=self.device
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _extract_images(observation: DrivingTrajectorySample) -> list[Any]:
        """Front-view image(s) as a list (empty when the sample is state-only)."""
        imgs = observation.images
        if imgs is None:
            return []
        try:
            from torchvision.transforms.functional import to_pil_image
        except ImportError:  # pragma: no cover - torchvision optional
            return []
        # images is [V, C, H, W]; use the first (front) view.
        front = imgs[0] if imgs.ndim == 4 else imgs
        return [to_pil_image(front.float().clamp(0, 1) if front.max() <= 1.0 else front.byte())]

    @staticmethod
    def _obs_key(observation: DrivingTrajectorySample) -> str:
        h = hashlib.sha256()
        h.update((observation.instruction or "").encode())
        h.update(str(observation.command_id).encode())
        if observation.metadata:
            h.update(str(observation.metadata.get("injected_caption", "")).encode())
        if observation.ego_state is not None:
            h.update(observation.ego_state.detach().cpu().numpy().round(4).tobytes())
        if observation.images is not None:
            im = observation.images
            h.update(str(tuple(im.shape)).encode())
            h.update(str(round(float(im.float().sum()), 2)).encode())
        return h.hexdigest()


__all__ = ["DEFAULT_MODEL_PATH", "Qwen3VLConstraintExtractor"]
