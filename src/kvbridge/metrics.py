"""Attention-aware and output-aware metrics for cache-transfer validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from kvbridge.cache import KVCache


@dataclass(frozen=True, slots=True)
class AttentionCosineReport:
    """Cosine agreement of candidate and reference attention outputs."""

    mean: float
    minimum: float
    per_layer: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expand_grouped_kv(tensor: Tensor, query_heads: int) -> Tensor:
    kv_heads = tensor.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return tensor.repeat_interleave(query_heads // kv_heads, dim=1)


def attention_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> Tensor:
    """Compute stable FP32 grouped-query attention for evaluation.

    Query is ``[batch, query_heads, query_tokens, head_dim]`` and K/V are
    ``[batch, kv_heads, key_tokens, head_dim]``. The routine intentionally
    avoids backend-specific fused kernels so experiment metrics are portable.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and cache batch/head dimensions are incompatible")
    query32 = query.float()
    key32 = _expand_grouped_kv(key.float(), query.shape[1])
    value32 = _expand_grouped_kv(value.float(), query.shape[1])
    scores = torch.matmul(query32, key32.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if key_padding_mask is not None:
        if key_padding_mask.shape != (key.shape[0], key.shape[2]):
            raise ValueError("key_padding_mask must be [batch, key_tokens]")
        scores = scores.masked_fill(~key_padding_mask.bool()[:, None, None, :], -torch.inf)
    if causal:
        query_tokens, key_tokens = query.shape[2], key.shape[2]
        if query_tokens > key_tokens:
            raise ValueError("causal evaluation cannot have more query than key tokens")
        offset = key_tokens - query_tokens
        query_positions = torch.arange(query_tokens, device=query.device) + offset
        key_positions = torch.arange(key_tokens, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value32)


def attention_output_cosine(
    queries: Sequence[Tensor],
    candidate: KVCache,
    reference: KVCache,
    *,
    key_padding_mask: Tensor | None = None,
    causal: bool = False,
) -> AttentionCosineReport:
    """Measure per-layer attention-output cosine under identical target queries."""
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference caches must have identical shapes")
    if len(queries) != candidate.shape[0]:
        raise ValueError("one target-query tensor is required per cache layer")
    per_layer: list[float] = []
    for layer, query in enumerate(queries):
        candidate_output = attention_output(
            query,
            candidate.keys[layer],
            candidate.values[layer],
            key_padding_mask=key_padding_mask,
            causal=causal,
        )
        reference_output = attention_output(
            query,
            reference.keys[layer],
            reference.values[layer],
            key_padding_mask=key_padding_mask,
            causal=causal,
        )
        similarity = F.cosine_similarity(candidate_output, reference_output, dim=-1, eps=1e-8)
        per_layer.append(float(similarity.mean().item()))
    return AttentionCosineReport(
        mean=sum(per_layer) / len(per_layer),
        minimum=min(per_layer),
        per_layer=tuple(per_layer),
    )


def logit_kl_divergence(
    candidate_logits: Tensor,
    reference_logits: Tensor,
    *,
    temperature: float = 1.0,
    max_suffix_tokens: int | None = None,
    symmetric: bool = False,
) -> float:
    """Return mean token KL(reference || candidate), optionally symmetric."""
    if candidate_logits.shape != reference_logits.shape or candidate_logits.ndim < 2:
        raise ValueError("candidate and reference logits must have identical rank >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_suffix_tokens is not None:
        if max_suffix_tokens <= 0:
            raise ValueError("max_suffix_tokens must be positive")
        candidate_logits = candidate_logits[..., -max_suffix_tokens:, :]
        reference_logits = reference_logits[..., -max_suffix_tokens:, :]
    candidate_log = F.log_softmax(candidate_logits.float() / temperature, dim=-1)
    reference_log = F.log_softmax(reference_logits.float() / temperature, dim=-1)
    forward = F.kl_div(candidate_log, reference_log, log_target=True, reduction="none").sum(-1)
    value = forward.mean()
    if symmetric:
        reverse = F.kl_div(reference_log, candidate_log, log_target=True, reduction="none").sum(-1)
        value = 0.5 * (value + reverse.mean())
    return float((value * temperature**2).item())
