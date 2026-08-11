from pathlib import Path

import pytest

from kvbridge.errors import ArtifactError
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.synthetic import cache_r2, fit_demo, make_problem


@pytest.fixture(scope="module")
def fitted():
    problem = make_problem(calibration_pairs=5, tokens=18)
    return problem, fit_demo(problem)


def test_fit_selects_predictive_layers_and_generalizes(fitted) -> None:
    problem, mapper = fitted
    assert mapper.selected_layers == [[2], [0]]

    mapped = mapper.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)

    assert cache_r2(mapped, problem.evaluation.target) > 0.9999


def test_artifact_round_trip(fitted, tmp_path: Path) -> None:
    problem, mapper = fitted
    artifact = mapper.save(tmp_path / "artifact")
    loaded = CrossModelKVMapper.load(artifact)

    mapped = loaded.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)

    assert loaded.selected_layers == mapper.selected_layers
    assert cache_r2(mapped, problem.evaluation.target) > 0.9999


def test_artifact_detects_tampering(fitted, tmp_path: Path) -> None:
    _, mapper = fitted
    artifact = mapper.save(tmp_path / "artifact")
    weights = artifact / "mapper.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(ArtifactError, match="checksum"):
        CrossModelKVMapper.load(artifact)
