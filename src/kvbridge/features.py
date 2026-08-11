"""Tensor layout helpers shared by fitting and inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def flatten_tokens(tensor: Tensor) -> Tensor:
    """Convert `[batch, heads, tokens, dim]` to `[batch*tokens, heads*dim]`."""
    batch, heads, tokens, dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch * tokens, heads * dim)


def flatten_head_tokens(tensor: Tensor, head: int) -> Tensor:
    """Convert one head to `[batch*tokens, dim]`."""
    return tensor[:, head].reshape(-1, tensor.shape[-1])


def selected_layer_features(layers: Sequence[Tensor], selected: Sequence[int]) -> Tensor:
    """Concatenate all heads of selected layers at each token."""
    return torch.cat([flatten_tokens(layers[layer]) for layer in selected], dim=-1)
