import pytest

from kvbridge.statistics import bootstrap_mean_interval, paired_bootstrap_difference


def test_bootstrap_interval_is_deterministic_and_contains_center() -> None:
    first = bootstrap_mean_interval([0.2, 0.4, 0.8], resamples=500, seed=17)
    second = bootstrap_mean_interval([0.2, 0.4, 0.8], resamples=500, seed=17)

    assert first == second
    assert first.low <= first.center <= first.high
    assert first.center == pytest.approx(1.4 / 3)


def test_paired_bootstrap_preserves_pairing() -> None:
    interval = paired_bootstrap_difference(
        [0.7, 0.8, 0.9], [0.5, 0.6, 0.7], resamples=200, seed=9
    )

    assert interval.low == pytest.approx(0.2)
    assert interval.center == pytest.approx(0.2)
    assert interval.high == pytest.approx(0.2)


def test_bootstrap_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_interval([])
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="1,000,000"):
        bootstrap_mean_interval([1.0], resamples=1_000_001)
