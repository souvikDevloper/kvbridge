"""Optional Hugging Face adapter for cache capture, conversion, and live handoff.

Imports are lazy so the numerics-only package stays usable without Transformers.
The adapter currently targets decoder-only models exposing a model-level rotary
embedding and the standard DynamicCache protocol (Qwen/Llama-style models).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.config import ModelSignature
from kvbridge.errors import CompatibilityError
from kvbridge.mapper import CrossModelKVMapper


@dataclass(frozen=True, slots=True)
class HFCapture:
    """A target prefill capture used for attention-aware evaluation."""

    cache: KVCache
    queries: tuple[Tensor, ...]
    logits: Tensor | None


def _metadata_model(model: Any) -> Any:
    """Return the underlying HF model used for configuration and layer access.

    PyTorch/XLA FSDPv2 keeps the wrapped module in ``_orig_module`` and does
    not proxy Hugging Face helpers such as ``get_input_embeddings``.  Forward
    calls must still use the wrapper, while metadata and hooks must target the
    original module whose parameters have already been moved to XLA.
    """
    seen: set[int] = set()
    while hasattr(model, "_orig_module") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_module
    return model


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the vocabulary and special-token contract, independent of model name."""
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens_map": tokenizer.special_tokens_map,
        "added_tokens": sorted(
            (str(key), str(value)) for key, value in tokenizer.get_added_vocab().items()
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_signature(
    model: Any,
    tokenizer: Any,
    *,
    revision: str,
    attention_kind: str = "dense",
) -> ModelSignature:
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    return ModelSignature(
        model_id=getattr(config, "_name_or_path", config.model_type),
        revision=revision,
        tokenizer_hash=tokenizer_fingerprint(tokenizer),
        num_layers=config.num_hidden_layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        attention_kind=attention_kind,  # type: ignore[arg-type]
        architecture=(config.architectures or [config.model_type])[0],
    )


def _rotary_module(model: Any) -> Any:
    model = _metadata_model(model)
    candidates = [
        getattr(model, "rotary_emb", None),
        getattr(getattr(model, "model", None), "rotary_emb", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "rotary_emb", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise CompatibilityError("model does not expose a supported model-level rotary embedding")


@torch.no_grad()
def capture_rotary_factors(model: Any, position_ids: Tensor) -> RotaryFactors:
    metadata_model = _metadata_model(model)
    rotary = _rotary_module(metadata_model)
    device = metadata_model.get_input_embeddings().weight.device
    dtype = metadata_model.get_input_embeddings().weight.dtype
    positions = position_ids.to(device)
    dummy = torch.empty((*positions.shape, 1), device=device, dtype=dtype)
    output = rotary(dummy, positions)
    if not isinstance(output, tuple) or len(output) < 2:
        raise CompatibilityError("rotary module did not return (cos, sin)")
    return RotaryFactors(output[0], output[1], interleaved=False)


def _legacy_layers(cache: Any) -> list[tuple[Tensor, Tensor]]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if isinstance(cache, tuple | list):
        return [(layer[0], layer[1]) for layer in cache]
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    raise CompatibilityError("unsupported Transformers cache representation")


def _decoder_layers(model: Any) -> Any:
    model = _metadata_model(model)
    candidates = [
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise CompatibilityError("model does not expose supported decoder layers")


def _canonical_query(output: Any, *, query_heads: int, head_dim: int) -> Tensor:
    if isinstance(output, tuple | list):
        output = output[0]
    if not isinstance(output, Tensor):
        raise CompatibilityError("query hook did not emit a tensor")
    if output.ndim == 3:
        batch, tokens, features = output.shape
        if features != query_heads * head_dim:
            raise CompatibilityError("query projection width differs from model configuration")
        return output.reshape(batch, tokens, query_heads, head_dim).permute(0, 2, 1, 3)
    if output.ndim == 4 and output.shape[-2:] == (query_heads, head_dim):
        return output.permute(0, 2, 1, 3)
    if output.ndim == 4 and output.shape[1] == query_heads and output.shape[-1] == head_dim:
        return output
    raise CompatibilityError(f"unsupported query tensor shape: {tuple(output.shape)}")


@torch.no_grad()
def capture_cache(
    model: Any, input_ids: Tensor, *, attention_mask: Tensor | None = None
) -> KVCache:
    """Run prefill and capture rotated K/V plus exact model-produced RoPE factors."""
    metadata_model = _metadata_model(model)
    device = metadata_model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    batch, tokens = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch, tokens), dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    layers = _legacy_layers(outputs.past_key_values)
    rotary = capture_rotary_factors(model, position_ids)
    return KVCache([layer[0] for layer in layers], [layer[1] for layer in layers], rotary)


@torch.no_grad()
def capture_cache_with_queries(
    model: Any,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    retain_logits: bool = True,
) -> HFCapture:
    """Capture target K/V, RoPE-applied Q, and logits in one reference prefill.

    Qwen3 exposes per-head query normalization, so the hook is placed after
    ``q_norm``. Llama-style models without that module fall back to ``q_proj``.
    """
    metadata_model = _metadata_model(model)
    device = metadata_model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    batch, tokens = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch, tokens), dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    query_heads = int(metadata_model.config.num_attention_heads)
    head_dim = int(
        getattr(
            metadata_model.config,
            "head_dim",
            metadata_model.config.hidden_size // query_heads,
        )
    )
    raw_queries: list[Tensor | None] = [None] * int(
        metadata_model.config.num_hidden_layers
    )
    handles = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        attention = layer.self_attn
        query_module = getattr(attention, "q_norm", None) or attention.q_proj

        def capture_query(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            raw_queries[index] = _canonical_query(
                output, query_heads=query_heads, head_dim=head_dim
            )

        handles.append(query_module.register_forward_hook(capture_query))
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(query is None for query in raw_queries):
        raise CompatibilityError("not every target layer emitted a query tensor")
    rotary = capture_rotary_factors(model, position_ids)
    queries = tuple(rotary.apply(query) for query in raw_queries if query is not None)
    layers = _legacy_layers(outputs.past_key_values)
    cache = KVCache([layer[0] for layer in layers], [layer[1] for layer in layers], rotary)
    return HFCapture(
        cache=cache,
        queries=queries,
        logits=outputs.logits if retain_logits else None,
    )


def to_dynamic_cache(cache: KVCache, model: Any) -> Any:
    try:
        from transformers import DynamicCache  # type: ignore[import-not-found, unused-ignore]
    except ImportError as error:  # pragma: no cover - exercised in HF integration environment
        raise RuntimeError("install kvbridge[hf] to use the Hugging Face adapter") from error
    metadata_model = _metadata_model(model)
    try:
        dynamic = DynamicCache(config=metadata_model.config)
    except TypeError:
        dynamic = DynamicCache()
    for layer, (key, value) in enumerate(zip(cache.keys, cache.values, strict=True)):
        dynamic.update(key, value, layer)
    return dynamic


@torch.no_grad()
def suffix_logits_from_cache(
    model: Any,
    cache: KVCache,
    suffix_input_ids: Tensor,
    *,
    batch_sharder: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Consume a short suffix against an existing cache and return its logits."""
    if suffix_input_ids.ndim != 2 or suffix_input_ids.shape[1] < 1:
        raise ValueError("suffix_input_ids must be rank-2 with at least one token")
    metadata_model = _metadata_model(model)
    device = metadata_model.get_input_embeddings().weight.device
    suffix_input_ids = suffix_input_ids.to(device)
    if cache.keys[0].device != device:
        cache = cache.to(device)
    past = to_dynamic_cache(cache, model)
    batch, suffix_tokens = suffix_input_ids.shape
    prefix_tokens = cache.shape[3]
    attention_mask = torch.ones(
        (batch, prefix_tokens + suffix_tokens), dtype=torch.long, device=device
    )
    position_ids = torch.arange(
        prefix_tokens, prefix_tokens + suffix_tokens, device=device
    ).unsqueeze(0).expand(batch, -1)
    cache_position = torch.arange(
        prefix_tokens, prefix_tokens + suffix_tokens, device=device
    )
    if batch_sharder is not None:
        suffix_input_ids = batch_sharder(suffix_input_ids)
        attention_mask = batch_sharder(attention_mask)
        position_ids = batch_sharder(position_ids)
    kwargs = {
        "input_ids": suffix_input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past,
        "use_cache": False,
        "return_dict": True,
    }
    try:
        outputs = model(cache_position=cache_position, **kwargs)
    except TypeError:
        outputs = model(**kwargs)
    return cast(Tensor, outputs.logits)


@torch.no_grad()
def greedy_handoff_generate(
    *,
    source_model: Any,
    target_model: Any,
    mapper: CrossModelKVMapper,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Tensor:
    """Prefill on the source, map the prefix cache, then decode on the target.

    The final prompt token is intentionally held back: the target consumes it
    against the mapped prefix to produce the first target-model logits.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("handoff generation requires a rank-2 prompt with at least two tokens")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    source_prefix = input_ids[:, :-1]
    source_cache = capture_cache(source_model, source_prefix)
    target_metadata_model = _metadata_model(target_model)
    target_device = target_metadata_model.get_input_embeddings().weight.device
    batch, prefix_tokens = source_prefix.shape
    target_positions = (
        torch.arange(prefix_tokens, device=target_device).unsqueeze(0).expand(batch, -1)
    )
    target_rotary = capture_rotary_factors(target_model, target_positions)
    mapped = mapper.map(source_cache, target_rotary=target_rotary).to(target_device)
    past = to_dynamic_cache(mapped, target_model)

    generated = input_ids.to(target_device)
    current = generated[:, -1:]
    attention_mask = torch.ones((batch, prefix_tokens + 1), dtype=torch.long, device=target_device)
    cache_position = torch.tensor([prefix_tokens], device=target_device)
    for _ in range(max_new_tokens):
        kwargs = {
            "input_ids": current,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
        }
        try:
            outputs = target_model(cache_position=cache_position, **kwargs)
        except TypeError:
            outputs = target_model(**kwargs)
        current = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, current), dim=1)
        past = outputs.past_key_values
        if eos_token_id is not None and bool((current == eos_token_id).all()):
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones((batch, 1), dtype=torch.long, device=target_device)), dim=1
        )
        cache_position = cache_position + 1
    return generated
