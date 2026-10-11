import pytest

from bot import aggregate, numeric


def test_binary_median_and_clip():
    assert aggregate.aggregate_binary([0.2, 0.3, 0.9], 0.02) == pytest.approx(0.3)
    assert aggregate.aggregate_binary([0.001, 0.0, 0.005], 0.02) == pytest.approx(0.02)
    assert aggregate.aggregate_binary([0.999, 1.0, 0.99], 0.02) == pytest.approx(0.98)


def test_mc_floor_and_exact_sum():
    dists = [{"a": 1.0, "b": 0.0, "c": 0.0}, {"a": 0.9, "b": 0.1, "c": 0.0}]
    out = aggregate.aggregate_multiple_choice(dists, ["a", "b", "c"], 0.01)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-12)
    assert out["c"] >= 0.0099


def test_cdf_mixture_stays_valid():
    a = numeric.percentiles_to_cdf({0.05: 1, 0.1: 2, 0.5: 5, 0.9: 8, 0.95: 9, 0.2: 3, 0.8: 7}, 0, 10, None, False, False)
    b = numeric.percentiles_to_cdf({0.05: 5, 0.1: 6, 0.5: 8, 0.9: 9.5, 0.95: 9.8, 0.2: 7, 0.8: 9}, 0, 10, None, False, False)
    mix = aggregate.aggregate_cdfs([a, b])
    numeric.validate_cdf(mix, False, False)
