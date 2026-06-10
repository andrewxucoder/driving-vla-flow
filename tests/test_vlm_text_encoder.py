"""VLMTextEncoder — frozen pretrained LM + projection.

Stubs the tokenizer + backbone to keep the suite offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from driving_vla.models import VLMTextEncoder, list_text_backbones


class _StubTokenizer:
    """Mimics HF tokenizer's __call__ signature.

    Returns deterministic per-string token IDs by hashing each character —
    same input string → same IDs across calls.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        seqs = []
        for t in texts:
            ids = [(ord(c) % (self.vocab_size - 1)) + 1 for c in t[:max_length]]
            seqs.append(ids)
        max_len = max(len(s) for s in seqs) if truncation else max_length
        input_ids, attn = [], []
        for s in seqs:
            pad = [0] * (max_len - len(s))
            input_ids.append(s + pad)
            attn.append([1] * len(s) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _StubLM(nn.Module):
    """Mimics HF AutoModel: returns ``last_hidden_state`` [B, T, D]."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 384) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.ln(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=None)


def _make_encoder(latent_dim: int = 32) -> VLMTextEncoder:
    return VLMTextEncoder(
        latent_dim=latent_dim,
        backbone="sentence-minilm-l6",
        embed_dim_override=64,
        backbone_module=_StubLM(embed_dim=64),
        tokenizer_module=_StubTokenizer(),
    )


def test_list_text_backbones_nonempty():
    assert "sentence-minilm-l6" in list_text_backbones()


def test_forward_shape():
    enc = _make_encoder(latent_dim=16)
    out = enc(["turn left", "keep lane", "slow down now"])
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_empty_list_returns_zero_batch():
    enc = _make_encoder(latent_dim=16)
    out = enc([])
    assert out.shape == (0, 16)


def test_rejects_non_list():
    enc = _make_encoder()
    with pytest.raises(TypeError, match="list"):
        enc("turn left")  # type: ignore[arg-type]


def test_different_instructions_yield_different_embeddings():
    """The whole point of M6.2: switching instructions must change the latent."""
    enc = _make_encoder(latent_dim=16)
    a = enc(["change lane to the left"])
    b = enc(["maintain current lane"])
    diff = (a - b).abs().sum().item()
    assert diff > 1e-3, f"text encoder collapsed: |a-b|={diff:.2e}"


def test_same_instruction_yields_same_embedding():
    enc = _make_encoder(latent_dim=16)
    enc.eval()
    a = enc(["slow down"])
    b = enc(["slow down"])
    assert torch.allclose(a, b, atol=1e-5)


def test_freeze_backbone_blocks_grad():
    stub = _StubLM(embed_dim=64)
    enc = VLMTextEncoder(
        latent_dim=8,
        backbone="sentence-minilm-l6",
        embed_dim_override=64,
        freeze_backbone=True,
        backbone_module=stub,
        tokenizer_module=_StubTokenizer(),
    )
    enc(["keep lane"]).sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in stub.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.projection.parameters()
    )


def test_plugs_into_driving_observation_encoder():
    from driving_vla.models import DrivingObservationEncoder
    text = _make_encoder(latent_dim=32)
    obs = DrivingObservationEncoder(
        state_dim=5,
        num_commands=4,
        latent_dim=48,
        history_dim=2,
        instruction_encoder=text,
    )
    out = obs({
        "ego_state": torch.zeros(2, 5),
        "history": torch.zeros(2, 4, 2),
        "command_id": torch.tensor([0, 1], dtype=torch.long),
        "instruction": ["keep this lane", "merge into the left lane"],
    })
    assert out.shape == (2, 48)
