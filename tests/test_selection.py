import pytest
import torch

from kvbridge.ridge import RidgeAccumulator
from kvbridge.selection import BatchedSelectionAccumulator


def test_batched_selection_matches_independent_ridge_accumulators() -> None:
    generator = torch.Generator().manual_seed(73)
    source = torch.randn(3, 2, 97, 4, generator=generator)
    target = torch.randn(2, 2, 97, 4, generator=generator)
    target[0, 0] = source[2, 0] @ torch.randn(4, 4, generator=generator) + 0.3
    target[1, 1] = source[0, 1] @ torch.randn(4, 4, generator=generator) - 0.7

    batched = BatchedSelectionAccumulator(3, 2, 2, 4, 4, dtype=torch.float64)
    batched.update(source[:, :, :31], target[:, :, :31])
    checkpoint = batched.state_dict()
    restored = BatchedSelectionAccumulator(3, 2, 2, 4, 4, dtype=torch.float64)
    restored.load_state_dict(checkpoint)
    restored.update(source[:, :, 31:], target[:, :, 31:])
    actual = restored.solve(alpha=0.01)

    expected = torch.empty_like(actual.r2)
    for target_layer in range(2):
        for source_layer in range(3):
            for head in range(2):
                accumulator = RidgeAccumulator(4, 4)
                accumulator.update(source[source_layer, head], target[target_layer, head])
                expected[target_layer, source_layer, head] = accumulator.solve(0.01).r2

    torch.testing.assert_close(actual.r2, expected, atol=1e-10, rtol=1e-10)
    assert actual.observations == 97
    _, indices = actual.topk(1)
    assert indices[0].item() == 2
    assert indices[1].item() == 0


def test_batched_selection_validates_shapes_and_checkpoint() -> None:
    accumulator = BatchedSelectionAccumulator(2, 1, 1, 3, 3)
    with pytest.raises(ValueError, match="source must have shape"):
        accumulator.update(torch.zeros(2, 1, 4, 2), torch.zeros(1, 1, 4, 3))
    with pytest.raises(ValueError, match="missing tensors"):
        accumulator.load_state_dict({"count": torch.tensor(0)})


def test_xla_accumulation_fails_clearly_without_torch_xla() -> None:
    try:
        import torch_xla  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="torch_xla is unavailable"):
            BatchedSelectionAccumulator(1, 1, 1, 2, 2, device="xla")
