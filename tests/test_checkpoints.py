from pathlib import Path

import pytest
import torch

from kvbridge.cache import KVCache
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.errors import ArtifactError
from kvbridge.fit import CalibrationPair, fit_mapper


def _signature(name: str, layers: int) -> ModelSignature:
    return ModelSignature(name, "revision", "tokenizer", layers, 1, 2)


def _pairs() -> list[CalibrationPair]:
    generator = torch.Generator().manual_seed(123)
    pairs = []
    for _ in range(3):
        source_keys = [torch.randn(1, 1, 12, 2, generator=generator) for _ in range(2)]
        source_values = [torch.randn(1, 1, 12, 2, generator=generator) for _ in range(2)]
        target_keys = [source_keys[1] * 0.7 + 0.2]
        target_values = [source_values[0] * -0.4 - 0.1]
        pairs.append(
            CalibrationPair(
                KVCache(source_keys, source_values, keys_are_content=True),
                KVCache(target_keys, target_values, keys_are_content=True),
            )
        )
    return pairs


def test_fit_checkpoint_resume_does_not_reiterate_calibration(tmp_path: Path) -> None:
    source = _signature("source", 2)
    target = _signature("target", 1)
    config = FitConfig(top_k=1, content_space=True, accumulation_dtype="float64")
    calls = 0

    def examples() -> list[CalibrationPair]:
        nonlocal calls
        calls += 1
        return _pairs()

    first = fit_mapper(
        examples, source, target, config, checkpoint_dir=tmp_path / "checkpoints"
    )
    calls_after_first = calls
    second = fit_mapper(
        examples, source, target, config, checkpoint_dir=tmp_path / "checkpoints"
    )

    assert calls == calls_after_first
    assert first.selected_layers == second.selected_layers
    torch.testing.assert_close(first.key_weights[0], second.key_weights[0])
    torch.testing.assert_close(first.value_weights[0], second.value_weights[0])


def test_checkpoint_contract_rejects_changed_fit(tmp_path: Path) -> None:
    source = _signature("source", 2)
    target = _signature("target", 1)
    root = tmp_path / "checkpoints"
    fit_mapper(_pairs(), source, target, FitConfig(top_k=1), checkpoint_dir=root)

    with pytest.raises(ArtifactError, match="contract differs"):
        fit_mapper(_pairs(), source, target, FitConfig(top_k=2), checkpoint_dir=root)


def test_pre_sampled_pairs_are_not_strided_twice() -> None:
    source = _signature("source", 2)
    target = _signature("target", 1)
    pairs = _pairs()
    pre_sampled = [
        CalibrationPair(pair.source.sample_tokens(2), pair.target.sample_tokens(2), 2)
        for pair in pairs
    ]

    reference = fit_mapper(pairs, source, target, FitConfig(top_k=1, token_stride=2))
    actual = fit_mapper(pre_sampled, source, target, FitConfig(top_k=1, token_stride=2))

    assert reference.selected_layers == actual.selected_layers
    torch.testing.assert_close(reference.key_weights[0], actual.key_weights[0])
