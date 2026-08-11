"""Optional Hugging Face adapter for cache capture, conversion, and live handoff.

Imports are lazy so the numerics-only package stays usable without Transformers.
The adapter currently targets decoder-only models exposing a model-level rotary
embedding and the standard DynamicCache protocol (Qwen/Llama-style models).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from kvbridge.cache import KVCache, RotaryFactors
from kvbridge.config import ModelSignature
from kvbridge.errors import CompatibilityError
from kvbridge.mapper import CrossModelKVMapper


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
    candidates = [
        getattr(getattr(model, "model", None), "rotary_emb", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "rotary_emb", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise CompatibilityError("model does not expose a supported model-level rotary embedding")


@torch.inference_mode()
def capture_rotary_factors(model: Any, position_ids: Tensor) -> RotaryFactors:
    rotary = _rotary_module(model)
    device = model.get_input_embeddings().weight.device
    dtype = model.get_input_embeddings().weight.dtype
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


@torch.inference_mode()
def capture_cache(
    model: Any, input_ids: Tensor, *, attention_mask: Tensor | None = None
) -> KVCache:
    """Run prefill and capture rotated K/V plus exact model-produced RoPE factors."""
    device = model.get_input_embeddings().weight.device
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


def to_dynamic_cache(cache: KVCache, model: Any) -> Any:
    try:
        from transformers import DynamicCache
    except ImportError as error:  # pragma: no cover - exercised in HF integration environment
        raise RuntimeError("install kvbridge[hf] to use the Hugging Face adapter") from error
    try:
        dynamic = DynamicCache(config=model.config)
    except TypeError:
        dynamic = DynamicCache()
    for layer, (key, value) in enumerate(zip(cache.keys, cache.values, strict=True)):
        dynamic.update(key, value, layer)
    return dynamic


@torch.inference_mode()
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
    target_device = target_model.get_input_embeddings().weight.device
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
