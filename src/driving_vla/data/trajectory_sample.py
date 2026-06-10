"""Canonical sample schema for v2.

Per user decision R2 (2026-05-27):
- Field naming is hybrid: `history` / `future` keep the driving-industry term
  (nuPlan / NAVSIM papers use these directly), while `instruction` is renamed
  from v1's `language_command` to align with robot-vla-flow.
- This is the *only* sample-level schema. All data adapters output this. All
  policy heads consume this.

M2 extension: `command_id` is added as an optional top-level int field. It is
the discrete high-level driving command (see `command_labeling.COMMAND_VOCAB`)
and is the primary conditioning signal for non-VLM heads. `instruction` is the
natural-language paraphrase derived from `command_id` for the VLM pathway.

M7 extension: `route` is added as an optional `[N, 2]` ego-frame polyline of
look-ahead waypoints (the global plan ahead of the ego, beyond the immediate
`future` horizon). Encodes long-range navigation intent — necessary for the
policy to anticipate turns / merges that lie further than the prediction
window.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DrivingTrajectorySample(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=False,
    )

    history: torch.Tensor = Field(
        ...,
        description="Past ego states, shape [T_hist, D_state]",
    )
    future: torch.Tensor = Field(
        ...,
        description="Target future trajectory, shape [T_fut, D_action]",
    )
    instruction: str | None = Field(
        default=None,
        description="Natural-language driving command (e.g., 'change to the left lane').",
    )
    command_id: int | None = Field(
        default=None,
        description="Discrete high-level driving command id (index into COMMAND_VOCAB). Primary condition for non-VLM heads.",
    )
    ego_state: torch.Tensor | None = Field(
        default=None,
        description="Current ego state vector, shape [D_state]. Optional convenience separate from history.",
    )
    images: torch.Tensor | None = Field(
        default=None,
        description="Multi-view camera images, shape [V, C, H, W]. None if running state-only.",
    )
    route: torch.Tensor | None = Field(
        default=None,
        description=(
            "Ego-frame route polyline ahead of the ego, shape [N, 2]. "
            "Captures beyond-horizon navigation intent. None if no route is available."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Scenario id, token, timestamp — free-form, not consumed by model heads.",
    )

    @field_validator("history", "future")
    @classmethod
    def _require_2d(cls, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2:
            raise ValueError(f"expected 2D tensor [T, D], got shape {tuple(v.shape)}")
        return v

    @field_validator("ego_state")
    @classmethod
    def _require_1d_state(cls, v: torch.Tensor | None) -> torch.Tensor | None:
        if v is not None and v.ndim != 1:
            raise ValueError(f"ego_state must be 1D [D], got shape {tuple(v.shape)}")
        return v

    @field_validator("images")
    @classmethod
    def _require_4d_images(cls, v: torch.Tensor | None) -> torch.Tensor | None:
        if v is not None and v.ndim != 4:
            raise ValueError(f"images must be 4D [V, C, H, W], got shape {tuple(v.shape)}")
        return v

    @field_validator("route")
    @classmethod
    def _require_2d_route(cls, v: torch.Tensor | None) -> torch.Tensor | None:
        if v is not None and (v.ndim != 2 or v.shape[-1] != 2):
            raise ValueError(f"route must be 2D [N, 2], got shape {tuple(v.shape)}")
        return v
