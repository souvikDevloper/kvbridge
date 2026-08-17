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


def test_float32_centering_is_stable_with_large_feature_offsets() -> None:
    generator = torch.Generator().manual_seed(7)
    centered = torch.randn(2048, 4, generator=generator)
    x = centered + 10_000.0
    expected_weight = torch.tensor([[0.5, -0.2], [0.3, 0.7], [-0.4, 0.1], [0.8, -0.6]])
    expected_bias = torch.tensor([1.25, -2.5])
    y = centered @ expected_weight + expected_bias
    accumulator = RidgeAccumulator(4, 2, dtype=torch.float32)

    for shard in x.split(64):
        start = accumulator.count
        accumulator.update(shard, y[start : start + len(shard)])
    solution = accumulator.solve(alpha=0.01)

    torch.testing.assert_close(solution.weight, expected_weight, atol=2e-3, rtol=2e-3)
    assert solution.r2 > 0.999


def test_float32_solve_is_finite_across_extreme_feature_scales() -> None:
    generator = torch.Generator().manual_seed(19)
    scales = torch.tensor([1e-3, 1e-1, 1.0, 1e2, 1e4])
    x = torch.randn(4096, 5, generator=generator) * scales
    expected_weight = torch.tensor([[2.0], [-0.7], [0.4], [0.003], [-0.00002]])
    y = x @ expected_weight + 0.25
    accumulator = RidgeAccumulator(5, 1, dtype=torch.float32)

    for x_shard, y_shard in zip(x.split(128), y.split(128), strict=True):
        accumulator.update(x_shard, y_shard)
    solution = accumulator.solve(alpha=0.01)

    assert torch.isfinite(solution.weight).all()
    assert torch.isfinite(solution.bias).all()
    assert solution.r2 > 0.999


def test_ridge_rejects_non_finite_statistics_before_factorization() -> None:
    accumulator = RidgeAccumulator(2, 1, dtype=torch.float32)
    accumulator.update(torch.ones(2, 2), torch.ones(2, 1))
    accumulator.centered_xtx[0, 0] = torch.inf

    with pytest.raises(FloatingPointError, match="XTX"):
        accumulator.solve(alpha=0.01)


def test_ridge_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host")
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        RidgeAccumulator(2, 1, device="cuda")
