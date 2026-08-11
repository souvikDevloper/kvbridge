from pathlib import Path

import torch

from kvbridge.io import calibration_shard_factory, load_calibration_shard, save_calibration_shard
from kvbridge.synthetic import make_problem


def test_calibration_shard_round_trip(tmp_path: Path) -> None:
    pair = make_problem(calibration_pairs=2, tokens=8).calibration[0]
    path = save_calibration_shard(tmp_path / "000.safetensors", pair, sequence_id="seq-0")

    restored = load_calibration_shard(path)

    torch.testing.assert_close(restored.source.keys[0], pair.source.keys[0])
    torch.testing.assert_close(restored.target.values[-1], pair.target.values[-1])
    assert restored.source.rotary is not None


def test_factory_is_reiterable(tmp_path: Path) -> None:
    problem = make_problem(calibration_pairs=2, tokens=8)
    for index, pair in enumerate(problem.calibration):
        save_calibration_shard(tmp_path / f"{index:03}.safetensors", pair, sequence_id=str(index))
    factory = calibration_shard_factory(tmp_path)

    assert len(list(factory())) == 2
    assert len(list(factory())) == 2
