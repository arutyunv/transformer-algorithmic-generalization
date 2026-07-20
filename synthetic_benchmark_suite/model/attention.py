"""Unified attention module covering axes 1, 2 and 4 of the ablation
(softmax approximation, attention pattern, KV/projection design), plus hooks for axis 6
(positional encoding) since RoPE/ALiBi are applied inside attention
rather than at the embedding layer.

Axis 3 (attention approximation - Linformer/Reformer/Performer/
Nystromformer) is intentionally not implemented here because of a
qualitatively different mechanism (they replace this O(n^2) module
with a different algorithm entirely) and
belong in a separate `attention_approx.py` with using this file's 
`forward` signature as the interface to match (same input/output shapes)
so it drops into the same DecoderBlock unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import AttentionConfig
from .positional_encoding import PositionalEncoding


def _taylor_softmax(scores: Tensor, mask_bias: Tensor) -> Tensor:
    """2nd-order Taylor approximation of softmax (axis 1, "softmax
    approximation" ablation): exp(x) ~= 1 + x + x^2/2, clamped to stay
    non-negative, then normalized.
    This is a simplified didactic approximation, good enough to test the 
    ablation's qualitative prediction (blurrier, higher-entropy attention 
    than exact softmax) without pulling in an external dependency.
    """
    x = scores + mask_bias
    approx = (1.0 + x + 0.5 * x.pow(2)).clamp(min=1e-6)
    # masked (-inf) positions: mask_bias makes x -> -inf, approx already
    # clamped to ~0 there, so no separate re-masking is needed.
    approx = torch.where(torch.isfinite(x), approx, torch.zeros_like(approx))
    denom = approx.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return approx / denom


def _build_pattern_mask(
    seq_len: int, pattern: str, window_size: int | None, n_global_tokens: int, device: torch.device
) -> Tensor:
    """Returns an additive (T, T) mask: 0.0 where attention is allowed,
    -inf where forbidden. Always causal (upper triangle forbidden)."""
    causal_forbidden = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    if pattern == "full":
        forbidden = causal_forbidden
    elif pattern == "local":
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        too_far = (i - j) >= window_size
        forbidden = causal_forbidden | too_far
        if n_global_tokens > 0:
            is_global_key = torch.zeros(seq_len, dtype=torch.bool, device=device)
            is_global_key[:n_global_tokens] = True
            forbidden = forbidden & ~is_global_key.unsqueeze(0)  # global keys always allowed (if not future)
            forbidden = forbidden | causal_forbidden  # but never break causality
    else:
        raise ValueError(f"Unknown attention pattern: {pattern!r}")

    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(forbidden, float("-inf"))
    return mask


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, config: AttentionConfig, pos_enc: PositionalEncoding):
        super().__init__()
        self.config = config
        self.pos_enc = pos_enc
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        self.w_q = nn.Linear(d_model, q_out, bias=False)
        self.w_k = nn.Linear(d_model, kv_out, bias=False)
        self.w_v = None if config.share_kv else nn.Linear(d_model, kv_out, bias=False)
        self.w_o = nn.Linear(q_out, d_model, bias=False)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.n_rep == 1:
            return x
        b, h_kv, t, d = x.shape
        x = x.unsqueeze(2).expand(b, h_kv, self.n_rep, t, d)
        return x.reshape(b, h_kv * self.n_rep, t, d)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """x: (B, T, d_model). positions: (B, T) integer positions
        (usually just arange, but kept explicit for looped-model reuse
        across timesteps)."""
        b, t, _ = x.shape
        device = x.device

        q = self.w_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_src = self.w_k(x) if self.w_v is None else self.w_v(x)
        v = v_src.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # RoPE (no-op for every other scheme)
        q, k = self.pos_enc.rotate_qk(q, k, positions)

        pattern_mask = _build_pattern_mask(
            t, self.config.pattern, self.config.window_size, self.config.n_global_tokens, device
        )
        alibi_bias = self.pos_enc.attention_bias(t, self.n_heads, device)  # None for non-ALiBi
        mask_bias = pattern_mask.unsqueeze(0) if alibi_bias is None else pattern_mask.unsqueeze(0) + alibi_bias

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.config.softmax_kind == "standard":
            attn = F.softmax(scores + mask_bias, dim=-1)
        elif self.config.softmax_kind == "taylor":
            attn = _taylor_softmax(scores, mask_bias)
        else:
            raise ValueError(f"Unknown softmax_kind: {self.config.softmax_kind!r}")

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.w_o(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
