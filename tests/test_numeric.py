import numpy as np
import pytest

from bot import numeric

PCTS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def random_percentiles(rng, lo, hi, spread_frac, center_frac):
    center = lo + (hi - lo) * center_frac
    spread = (hi - lo) * spread_frac
    z = np.array([-1.645, -1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816, 1.645])
    skew = rng.uniform(0.5, 2.0)
    vals = center + spread * np.where(z < 0, z, z * skew)
    return dict(zip(PCTS, vals.tolist()))


@pytest.mark.parametrize("open_lower", [False, True])
@pytest.mark.parametrize("open_upper", [False, True])
@pytest.mark.parametrize("log_scale", [False, True])
def test_random_distributions_always_produce_valid_cdfs(open_lower, open_upper, log_scale):
    rng = np.random.default_rng(42)
    for _ in range(300):
        lo = float(rng.uniform(1, 100)) if log_scale else float(rng.uniform(-100, 100))
        hi = lo + float(rng.uniform(1, 10_000))
        zero = lo - float(rng.uniform(0.01, 50)) if log_scale else None
        # Includes very narrow, very wide and off-range distributions.
        pct = random_percentiles(rng, lo, hi, spread_frac=10 ** rng.uniform(-3.5, 0.3), center_frac=rng.uniform(-0.3, 1.3))
        cdf = numeric.percentiles_to_cdf(pct, lo, hi, zero, open_lower, open_upper, 201)
        assert len(cdf) == 201
        numeric.validate_cdf(cdf, open_lower, open_upper)


@pytest.mark.parametrize("size", [11, 21, 51, 101])
def test_discrete_sizes(size):
    pct = {0.05: 2, 0.1: 3, 0.2: 4, 0.3: 5, 0.4: 5, 0.5: 6, 0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9, 0.95: 10}
    cdf = numeric.percentiles_to_cdf(pct, -0.5, size - 1.5, None, False, True, size)
    assert len(cdf) == size
    numeric.validate_cdf(cdf, False, True)


def test_median_is_preserved_for_reasonable_forecast():
    pct = {0.05: 20, 0.1: 25, 0.2: 32, 0.3: 37, 0.4: 41, 0.5: 45, 0.6: 49, 0.7: 53, 0.8: 58, 0.9: 65, 0.95: 70}
    cdf = numeric.percentiles_to_cdf(pct, 0, 100, None, False, False)
    assert abs(numeric.cdf_quantile(cdf, 0.5, 0, 100, None) - 45) < 1.5
    assert abs(numeric.cdf_quantile(cdf, 0.1, 0, 100, None) - 25) < 2.5


def test_log_scaled_median_is_preserved():
    pct = {0.05: 1_000, 0.1: 2_000, 0.2: 4_000, 0.3: 6_000, 0.4: 8_000, 0.5: 10_000, 0.6: 13_000, 0.7: 17_000, 0.8: 25_000, 0.9: 40_000, 0.95: 60_000}
    cdf = numeric.percentiles_to_cdf(pct, 100, 1_000_000, 0, False, True)
    med = numeric.cdf_quantile(cdf, 0.5, 100, 1_000_000, 0)
    assert 8_500 < med < 11_500


def test_location_roundtrip():
    for zero in (None, 0.0, -5.0):
        for v in (1.0, 10.0, 999.0):
            loc = numeric.value_to_location(v, 1.0, 1000.0, zero)
            assert abs(numeric.location_to_value(loc, 1.0, 1000.0, zero) - v) < 1e-6
