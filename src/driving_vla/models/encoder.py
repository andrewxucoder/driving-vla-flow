"""DrivingObservationEncoder — condition dict → latent vector.

Accepts a flexible condition mapping with optional fields and produces a single
``[B, latent_dim]`` tensor consumed by trajectory heads.

Mandatory: ``ego_state`` ``[B, state_dim]``
Optional:
  - ``command_id``  ``[B]``                — discrete driving command
  - ``history``     ``[B, T_hist, state_dim]`` — past states (mean-pooled in time)
  - ``images``      ``[B, V, C, H, W]``    — multi-view camera (via image encoder)
  - ``instruction`` ``list[str]``          — natural-language command (via VLM encoder)
  - ``route``       ``[B, N, 2]``           — ego-frame look-ahead route waypoints
  - ``route_mask``  ``[B, N]``              — 1 = valid waypoint, 0 = padding (optional)

Image / VLM / route encoders are pluggable (any ``nn.Module`` with a
``latent_dim`` int attribute). When absent, the corresponding branch is
skipped. The route encoder additionally accepts a ``mask`` keyword for padded
batches; the other encoders use only positional input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class _FeatureEncoder(Protocol):
    latent_dim: int

    def __call__(self, x: Any) -> torch.Tensor: ...


class DrivingObservationEncoder(nn.Module):
    """Lightweight encoder for driving condition dicts.

    The output latent dimension is reported via :attr:`latent_dim` so it can
    feed directly into the trajectory head's ``cond_dim``.
    """

    def __init__(
        self,
        state_dim: int,
        num_commands: int,
        latent_dim: int,
        history_dim: int | None = None,
        history_embed_dim: int | None = None,
        command_embed_dim: int | None = None,
        image_encoder: _FeatureEncoder | None = None,
        instruction_encoder: _FeatureEncoder | None = None,
        route_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_commands <= 0 or latent_dim <= 0:
            raise ValueError(
                f"state_dim/num_commands/latent_dim must all be > 0; "
                f"got {state_dim}/{num_commands}/{latent_dim}"
            )
        if route_encoder is not None and not hasattr(route_encoder, "latent_dim"):
            raise TypeError("route_encoder must expose a `latent_dim` int attribute")

        self.state_dim = state_dim
        self.num_commands = num_commands
        self.latent_dim = latent_dim
        self.history_dim = history_dim
        self.history_embed_dim = history_embed_dim or latent_dim
        self.command_embed_dim = command_embed_dim or latent_dim

        # Always-on: ego_state projection
        self.state_proj = nn.Linear(state_dim, latent_dim)

        # Always-on: command_id embedding (one slot per vocab entry)
        self.command_embed = nn.Embedding(num_commands, self.command_embed_dim)

        # Optional history MLP — pre-mean-pool MLP over per-timestep features
        if history_dim is None:
            self.history_proj: nn.Module | None = None
        else:
            self.history_proj = nn.Sequential(
                nn.Linear(history_dim, self.history_embed_dim),
                nn.SiLU(),
                nn.Linear(self.history_embed_dim, self.history_embed_dim),
            )

        # Optional pluggable image / instruction / route encoders
        self.image_encoder = image_encoder
        self.instruction_encoder = instruction_encoder
        self.route_encoder = route_encoder

        # Fuse all available branches → latent_dim
        fused_dim = latent_dim + self.command_embed_dim
        if self.history_proj is not None:
            fused_dim += self.history_embed_dim
        if image_encoder is not None:
            fused_dim += image_encoder.latent_dim
        if instruction_encoder is not None:
            fused_dim += instruction_encoder.latent_dim
        if route_encoder is not None:
            fused_dim += cast(int, route_encoder.latent_dim)

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, condition: Mapping[str, Any]) -> torch.Tensor:
        ego_state = condition.get("ego_state")
        if ego_state is None:
            raise KeyError("condition missing required 'ego_state'")
        if not isinstance(ego_state, torch.Tensor):
            raise TypeError(f"'ego_state' must be a tensor, got {type(ego_state).__name__}")
        if ego_state.ndim != 2 or ego_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"ego_state must be [B, {self.state_dim}], got {tuple(ego_state.shape)}"
            )
        batch = ego_state.shape[0]

        parts: list[torch.Tensor] = [self.state_proj(ego_state)]

        # command_id → embedding (defaults to 0 / "keep_lane" if missing)
        command_id = condition.get("command_id")
        if command_id is None:
            command_id = torch.zeros(batch, dtype=torch.long, device=ego_state.device)
        elif command_id.dtype != torch.long:
            command_id = command_id.to(dtype=torch.long)
        if command_id.ndim == 0:
            command_id = command_id.expand(batch)
        parts.append(self.command_embed(command_id))

        # Optional history branch
        if self.history_proj is not None:
            history = condition.get("history")
            if history is None:
                parts.append(
                    torch.zeros(batch, self.history_embed_dim, device=ego_state.device)
                )
            else:
                if history.ndim != 3:
                    raise ValueError(
                        f"history must be [B, T, D], got {tuple(history.shape)}"
                    )
                per_step = self.history_proj(history)  # [B, T, embed]
                parts.append(per_step.mean(dim=1))

        # Optional image branch
        if self.image_encoder is not None:
            images = condition.get("images")
            if images is None:
                parts.append(
                    torch.zeros(batch, self.image_encoder.latent_dim, device=ego_state.device)
                )
            else:
                parts.append(self.image_encoder(images))

        # Optional instruction branch
        if self.instruction_encoder is not None:
            instruction = condition.get("instruction")
            if instruction is None:
                parts.append(
                    torch.zeros(
                        batch, self.instruction_encoder.latent_dim, device=ego_state.device
                    )
                )
            else:
                parts.append(self.instruction_encoder(instruction))

        # Optional route branch (look-ahead navigation polyline)
        if self.route_encoder is not None:
            route = condition.get("route")
            route_dim = cast(int, self.route_encoder.latent_dim)
            if route is None:
                parts.append(torch.zeros(batch, route_dim, device=ego_state.device))
            else:
                if not isinstance(route, torch.Tensor):
                    raise TypeError(
                        f"'route' must be a tensor, got {type(route).__name__}"
                    )
                if route.ndim != 3 or route.shape[-1] != 2:
                    raise ValueError(
                        f"route must be [B, N, 2], got {tuple(route.shape)}"
                    )
                if route.shape[0] != batch:
                    raise ValueError(
                        f"route batch {route.shape[0]} ≠ ego_state batch {batch}"
                    )
                route_mask = condition.get("route_mask")
                if route_mask is not None and not isinstance(route_mask, torch.Tensor):
                    raise TypeError(
                        f"'route_mask' must be a tensor or None, got {type(route_mask).__name__}"
                    )
                parts.append(self.route_encoder(route, mask=route_mask))  # type: ignore[arg-type]

        fused = torch.cat(parts, dim=-1)
        return self.fuse(fused)


__all__ = ["DrivingObservationEncoder"]
