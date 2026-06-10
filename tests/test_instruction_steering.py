"""M6.2 acceptance: switching instruction actually steers policy output.

This is the *behavioural* test the project was missing in Round 0: with a hash
encoder the instruction channel carried no signal, so changing the instruction
had no effect on the policy. With :class:`VLMTextEncoder` (or any real text
encoder), the policy should be able to learn instruction-conditioned behaviour.

We construct a toy 3-class regression problem (turn-left / keep / turn-right),
train a tiny policy for a handful of steps, then verify the predicted
trajectory's lateral sign matches the instruction's intent.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from driving_vla.models import (
    DrivingObservationEncoder,
    VLMTextEncoder,
    WaypointRegressionHead,
)
from driving_vla.policies import RegressionPolicy


class _StubTokenizer:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        seqs = [
            [(ord(c) % (self.vocab_size - 1)) + 1 for c in t[:max_length]]
            for t in texts
        ]
        max_len = max(len(s) for s in seqs)
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
    def __init__(self, vocab_size: int = 64, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=self.ln(self.embed(input_ids)),
            pooler_output=None,
        )


def _build_policy(latent_dim: int = 32, horizon: int = 4) -> RegressionPolicy:
    text_encoder = VLMTextEncoder(
        latent_dim=16,
        backbone="sentence-minilm-l6",
        embed_dim_override=64,
        freeze_backbone=False,  # let the stub embed table adapt — proxy for fine-tuning
        backbone_module=_StubLM(vocab_size=64, embed_dim=64),
        tokenizer_module=_StubTokenizer(vocab_size=64),
    )
    encoder = DrivingObservationEncoder(
        state_dim=2,
        num_commands=3,
        latent_dim=latent_dim,
        history_dim=2,
        instruction_encoder=text_encoder,
    )
    head = WaypointRegressionHead(
        horizon=horizon, traj_dim=2, cond_dim=latent_dim, hidden_dim=64,
    )
    return RegressionPolicy(encoder=encoder, head=head)


def _make_batch(instructions: list[str], futures: torch.Tensor) -> dict:
    n = len(instructions)
    return {
        "condition": {
            "ego_state": torch.zeros(n, 2),
            "history": torch.zeros(n, 2, 2),
            "command_id": torch.zeros(n, dtype=torch.long),
            "instruction": instructions,
        },
        "future": futures,
    }


def test_paired_counterfactual_instruction_steers_lateral_sign():
    """With matching instructions in training data, the policy must learn that
    'turn left' implies +lateral and 'turn right' implies −lateral.
    Pre-fix (HashedInstructionEncoder) this never converges — the hash gave
    identical-looking features for unrelated strings, so the policy fell back to
    predicting the mean (≈ 0 lateral) regardless of instruction."""
    torch.manual_seed(0)
    policy = _build_policy(horizon=4)

    horizon = 4
    left_future = torch.stack(
        [torch.arange(1.0, horizon + 1), torch.full((horizon,), 1.5)], dim=-1
    ).unsqueeze(0)      # [1, H, 2]: forward + lateral=+1.5
    right_future = left_future.clone()
    right_future[..., 1] = -1.5                              # lateral=−1.5
    keep_future = left_future.clone()
    keep_future[..., 1] = 0.0                                # lateral=0

    instructions = ["turn left", "turn right", "keep lane"]
    futures = torch.cat([left_future, right_future, keep_future], dim=0)
    batch = _make_batch(instructions, futures)

    optim = torch.optim.Adam(policy.parameters(), lr=2e-2)
    for _ in range(300):
        loss = policy(batch["condition"], batch["future"])
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate counterfactuals: identical observation, three instructions.
    policy.eval()
    with torch.no_grad():
        eval_batch = _make_batch(["turn left", "turn right", "keep lane"], futures)
        cond_latent = policy.encoder(eval_batch["condition"])
        pred = policy.head(cond_latent)  # [3, H, 2]

    left_y = pred[0, :, 1].mean().item()
    right_y = pred[1, :, 1].mean().item()
    keep_y = pred[2, :, 1].mean().item()

    assert left_y > 0.5, f"'turn left' should yield +lateral; got {left_y:.3f}"
    assert right_y < -0.5, f"'turn right' should yield −lateral; got {right_y:.3f}"
    assert abs(keep_y) < 0.7, f"'keep lane' should be near zero; got {keep_y:.3f}"
    assert left_y - right_y > 1.5, (
        f"instruction must visibly steer policy; diff(left,right)={left_y - right_y:.3f}"
    )
