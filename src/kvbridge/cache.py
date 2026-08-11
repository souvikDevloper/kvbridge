"""Canonical KV-cache representation and exact RoPE transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from kvbridge.errors import CacheValidationError


def _rotate_half(x: Tensor, *, interleaved: bool) -> Tensor:
    if x.shape[-1] % 2:
        raise CacheValidationError("RoPE requires an even head dimension")
    if interleaved:
        paired = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
        return rotated.flatten(-2)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@dataclass(frozen=True, slots=True)
class RotaryFactors:
    """Cosine/sine values emitted by the model's own rotary module.

    Capturing model-produced factors avoids silently reimplementing scaling
    variants (Llama-3, YaRN, dynamic NTK) incorrectly.
    """

    cos: Tensor
    sin: Tensor
    interleaved: bool = False

    def __post_init__(self) -> None:
        if self.cos.shape != self.sin.shape:
            raise CacheValidationError("RoPE cosine and sine tensors must have identical shapes")
        if self.cos.ndim not in {2, 3}:
            raise CacheValidationError("RoPE factors must be [tokens, dim] or [batch, tokens, dim]")

    def apply(self, x: Tensor, *, inverse: bool = False) -> Tensor:
        """Apply or exactly invert RoPE on `[batch, heads, tokens, dim]`."""
        if x.ndim != 4:
            raise CacheValidationError("RoPE input must be [batch, heads, tokens, dim]")
        cos, sin = (
            self.cos.to(device=x.device, dtype=x.dtype),
            self.sin.to(device=x.device, dtype=x.dtype),
        )
        if cos.ndim == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        if cos.shape[-2:] != x.shape[-2:]:
            raise CacheValidationError(
                f"RoPE factors {tuple(cos.shape)} do not match cache {tuple(x.shape)}"
            )
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        sign = -1.0 if inverse else 1.0
        return x * cos + sign * _rotate_half(x, interleaved=self.interleaved) * sin


@dataclass(frozen=True, slots=True)
class KVCache:
    """Model-independent cache; every layer is `[batch, kv_heads, tokens, head_dim]`."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    rotary: RotaryFactors | None = None
    keys_are_content: bool = False

    def __init__(
        self,
        keys: Sequence[Tensor],
        values: Sequence[Tensor],
        rotary: RotaryFactors | None = None,
        keys_are_content: bool = False,
    ) -> None:
        object.__setattr__(self, "keys", tuple(keys))
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "rotary", rotary)
        object.__setattr__(self, "keys_are_content", keys_are_content)
        self.validate()

    def validate(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise CacheValidationError("cache must contain equally many non-empty K and V layers")
        reference = self.keys[0].shape
        if len(reference) != 4:
            raise CacheValidationError("cache tensors must be [batch, kv_heads, tokens, head_dim]")
        for index, (key, value) in enumerate(zip(self.keys, self.values, strict=True)):
            if key.shape != value.shape:
                raise CacheValidationError(f"K/V shape mismatch at layer {index}")
            if key.shape != reference:
                raise CacheValidationError("all layers must share a cache shape")
            if not key.is_floating_point() or not value.is_floating_point():
                raise CacheValidationError("cache tensors must be floating point")
        if self.rotary is not None and self.rotary.cos.shape[-2:] != (reference[2], reference[3]):
            raise CacheValidationError("rotary factors do not match cache token/head dimensions")

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        batch, heads, tokens, dim = self.keys[0].shape
        return len(self.keys), batch, heads, tokens, dim

    def to_content_space(self) -> KVCache:
        if self.keys_are_content:
            return self
        if self.rotary is None:
            raise CacheValidationError("content-space mapping requires captured RoPE factors")
        return replace(
            self,
            keys=tuple(self.rotary.apply(key, inverse=True) for key in self.keys),
            keys_are_content=True,
        )

    def apply_rotary(self, factors: RotaryFactors) -> KVCache:
        if not self.keys_are_content:
            raise CacheValidationError("cannot apply target RoPE to keys already in position space")
        return replace(
            self,
            keys=tuple(factors.apply(key) for key in self.keys),
            rotary=factors,
            keys_are_content=False,
        )

    def detach(self, *, device: str | torch.device = "cpu") -> KVCache:
        rotary = None
        if self.rotary is not None:
            rotary = RotaryFactors(
                self.rotary.cos.detach().to(device),
                self.rotary.sin.detach().to(device),
                self.rotary.interleaved,
            )
        return KVCache(
            [item.detach().to(device) for item in self.keys],
            [item.detach().to(device) for item in self.values],
            rotary,
            self.keys_are_content,
        )

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        rotary = None
        if self.rotary is not None:
            rotary = RotaryFactors(
                self.rotary.cos.to(device=device, dtype=dtype),
                self.rotary.sin.to(device=device, dtype=dtype),
                self.rotary.interleaved,
            )
        return KVCache(
            [item.to(device=device, dtype=dtype) for item in self.keys],
            [item.to(device=device, dtype=dtype) for item in self.values],
            rotary,
            self.keys_are_content,
        )
