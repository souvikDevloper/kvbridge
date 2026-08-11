import pytest

from kvbridge.config import ModelSignature
from kvbridge.errors import CompatibilityError


def signature(**overrides):
    values = {
        "model_id": "model",
        "revision": "abc123",
        "tokenizer_hash": "same",
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
    }
    values.update(overrides)
    return ModelSignature(**values)


def test_pair_requires_shared_tokenizer() -> None:
    with pytest.raises(CompatibilityError, match="tokenizers differ"):
        signature().validate_pair(signature(tokenizer_hash="different"))


def test_pair_rejects_hybrid_attention() -> None:
    with pytest.raises(CompatibilityError, match="dense full-attention"):
        signature().validate_pair(signature(attention_kind="hybrid"))


def test_fingerprint_changes_with_revision() -> None:
    assert signature().fingerprint != signature(revision="def456").fingerprint
