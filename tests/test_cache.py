import pytest
import torch

from kvbridge.cache import KVCache
from kvbridge.errors import CacheValidationError
from kvbridge.synthetic import rotary_factors


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_round_trip(interleaved: bool) -> None:
    tensor = torch.randn(2, 3, 11, 8)
    factors = rotary_factors(11, 8, theta=10_000.0, batch=2, interleaved=interleaved)

    recovered = factors.apply(factors.apply(tensor), inverse=True)

    torch.testing.assert_close(recovered, tensor, atol=2e-6, rtol=2e-6)


def test_cache_rejects_mismatched_layers() -> None:
    with pytest.raises(CacheValidationError, match="K/V shape mismatch"):
        KVCache([torch.zeros(1, 2, 3, 4)], [torch.zeros(1, 2, 2, 4)])


def test_content_space_requires_factors() -> None:
    cache = KVCache([torch.zeros(1, 2, 3, 4)], [torch.zeros(1, 2, 3, 4)])
    with pytest.raises(CacheValidationError, match="requires captured RoPE"):
        cache.to_content_space()


def test_token_sampling_keeps_rotary_alignment() -> None:
    factors = rotary_factors(9, 4, theta=10_000.0)
    cache = KVCache(
        [torch.randn(1, 2, 9, 4)],
        [torch.randn(1, 2, 9, 4)],
        factors,
    )

    sampled = cache.sample_tokens(3)

    assert sampled.shape == (1, 1, 2, 3, 4)
    torch.testing.assert_close(sampled.rotary.cos, factors.cos[:, ::3])
    with pytest.raises(ValueError, match="positive"):
        cache.sample_tokens(0)
