import pytest
import torch

from kvbridge.ridge import RidgeAccumulator


def test_ridge_recovers_affine_map_and_merges_shards() -> None:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(300, 5, generator=generator)
    expected_weight = torch.randn(5, 3, generator=generator)
    expected_bias = torch.randn(3, generator=generator)
    y = x @ expected_weight + expected_bias

    left = RidgeAccumulator(5, 3)
    right = RidgeAccumulator(5, 3)
    left.update(x[:130], y[:130])
    right.update(x[130:], y[130:])
    solution = left.merge(right).solve(alpha=1e-9)

    torch.testing.assert_close(solution.weight.float(), expected_weight, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(solution.bias.float(), expected_bias, atol=1e-5, rtol=1e-5)
    assert solution.r2 > 0.999999
    assert solution.observations == 300


def test_ridge_rejects_empty_solve() -> None:
    accumulator = RidgeAccumulator(2, 1)
    try:
        accumulator.solve(0.01)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("empty accumulator unexpectedly solved")


def test_ridge_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host")
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        RidgeAccumulator(2, 1, device="cuda")
