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
