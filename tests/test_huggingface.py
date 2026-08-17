from types import SimpleNamespace

import torch

from kvbridge.huggingface import (
    _canonical_query,
    _decoder_layers,
    _metadata_model,
    _rotary_module,
    tokenizer_fingerprint,
)


class FakeTokenizer:
    special_tokens_map = {"eos_token": "</s>"}

    def get_vocab(self):
        return {"hello": 1, "world": 2}

    def get_added_vocab(self):
        return {"</s>": 3}


def test_tokenizer_fingerprint_is_deterministic() -> None:
    assert tokenizer_fingerprint(FakeTokenizer()) == tokenizer_fingerprint(FakeTokenizer())


def test_canonical_query_handles_normalized_and_projected_layouts() -> None:
    normalized = torch.arange(2 * 5 * 4 * 8).reshape(2, 5, 4, 8)
    projected = normalized.reshape(2, 5, 32)
    expected = normalized.permute(0, 2, 1, 3)

    torch.testing.assert_close(
        _canonical_query(normalized, query_heads=4, head_dim=8), expected
    )
    torch.testing.assert_close(
        _canonical_query(projected, query_heads=4, head_dim=8), expected
    )


def test_bare_and_fsdp_wrapped_hf_model_metadata_is_discovered() -> None:
    layers = [object(), object()]
    rotary = object()
    bare = SimpleNamespace(layers=layers, rotary_emb=rotary)
    wrapped = SimpleNamespace(_orig_module=bare)

    assert _metadata_model(wrapped) is bare
    assert _decoder_layers(wrapped) is layers
    assert _rotary_module(wrapped) is rotary
