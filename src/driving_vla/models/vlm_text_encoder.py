"""Frozen pretrained text encoder for driving instructions.

Replaces :class:`HashedInstructionEncoder` (SHA-256 placeholder) with a real
language model so the ``instruction`` channel of
:class:`DrivingObservationEncoder` actually carries meaning. Default backbone is
``sentence-transformers/all-MiniLM-L6-v2`` — 22M parameters, fits comfortably on
M-series silicon, doesn't need a GPU.

The interface matches the M3 placeholder:

    encoder(instructions: list[str]) -> Tensor[B, latent_dim]

so it can drop into :class:`DrivingObservationEncoder` via the existing
``instruction_encoder`` argument.

Lazy import of ``transformers`` (in :meth:`_load`) keeps the base install free
of the ``[vlm]`` extra. Pass ``backbone_module``/``tokenizer_module`` to
short-circuit the registry-based load — used by unit tests to avoid the
HuggingFace download in CI.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from torch import nn


_TEXT_REGISTRY: dict[str, dict[str, Any]] = {
    "sentence-minilm-l6": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_dim": 384,
        "pooling": "mean",
    },
}


def list_text_backbones() -> list[str]:
    return sorted(_TEXT_REGISTRY.keys())


class VLMTextEncoder(nn.Module):
    """Frozen pretrained LM + mean/pooler pooling + MLP projection."""

    def __init__(
        self,
        latent_dim: int,
        backbone: Literal["sentence-minilm-l6"] = "sentence-minilm-l6",
        freeze_backbone: bool = True,
        max_seq_len: int = 64,
        embed_dim_override: int | None = None,
        backbone_module: nn.Module | None = None,
        tokenizer_module: Any | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if backbone not in _TEXT_REGISTRY:
            raise ValueError(
                f"backbone must be one of {list_text_backbones()}, got {backbone!r}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

        spec = _TEXT_REGISTRY[backbone]
        embed_dim = embed_dim_override if embed_dim_override is not None else spec["embed_dim"]

        self.latent_dim = latent_dim
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self._pooling: str = spec["pooling"]

        if backbone_module is not None and tokenizer_module is not None:
            self._tokenizer = tokenizer_module
            self._backbone = backbone_module
        else:
            self._tokenizer, self._backbone = self._load(spec["hf_id"])

        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _load(hf_id: str) -> tuple[Any, nn.Module]:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "VLMTextEncoder needs `transformers` — install via "
                "`pip install -e .[vlm]`. Or pass `backbone_module=<nn.Module>` "
                "+ `tokenizer_module=<tokenizer>` to skip the HuggingFace "
                "download (used by unit tests)."
            ) from exc
        tok = AutoTokenizer.from_pretrained(hf_id)
        mod = AutoModel.from_pretrained(hf_id)
        return tok, mod

    def _pool(
        self, output: Any, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self._pooling == "pooler":
            pooler = getattr(output, "pooler_output", None)
            if pooler is not None:
                return pooler
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                f"{self.backbone_name} returned neither pooler_output nor "
                "last_hidden_state"
            )
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _device(self) -> torch.device:
        return cast(torch.device, self.projection[0].weight.device)

    def forward(self, instructions: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not isinstance(instructions, (list, tuple)):
            raise TypeError(
                f"instructions must be list[str]/tuple[str], got "
                f"{type(instructions).__name__}"
            )
        if not instructions:
            return torch.zeros(0, self.latent_dim, device=self._device())

        encoded = self._tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        device = self._device()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        backbone_train_mode = any(p.requires_grad for p in self._backbone.parameters())
        with torch.set_grad_enabled(backbone_train_mode):
            output = self._backbone(**encoded)
        pooled = self._pool(output, encoded["attention_mask"])
        if pooled.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"backbone produced pooled dim {pooled.shape[-1]} but "
                f"VLMTextEncoder was configured with embed_dim={self.embed_dim}"
            )
        return self.projection(pooled)


__all__ = ["VLMTextEncoder", "list_text_backbones"]
