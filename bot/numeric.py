"""Turn a handful of percentiles into the CDF format the Metaculus API requires.

API rules (see metaculus.com/api "CDF generation details"):
- `cdf_size` points (201 for numeric, inbound_outcome_count + 1 for discrete),
  evaluated at evenly spaced locations between range_min and range_max
  (log-spaced in value when the question has a zero_point).
- Closed bound: cdf is exactly 0 at the lower / 1 at the upper end.
- Open bound: at least 0.001 of mass beyond it (cdf[0] >= 0.001, cdf[-1] <= 0.999).
- Strictly increasing by at least 0.01 / (cdf_size - 1) per step, and no single
  step may exceed 0.2 * 200 / (cdf_size - 1).
"""
from __future__ import annotations

import numpy as np

MAX_PMF_AT_200 = 0.2


def value_to_location(value: float, lo: float, hi: float, zero_point: float | None) -> float:
    if zero_point is None:
        return (value - lo) / (hi - lo)
    ratio = (hi - zero_point) / (lo - zero_point)
    value = max(value, zero_point + 1e-12 * max(1.0, abs(hi - lo)))
    return float(np.log((value - lo) * (ratio - 1) + (hi - lo)) - np.log(hi - lo)) / float(np.log(ratio))


def location_to_value(loc: float, lo: float, hi: float, zero_point: float | None) -> float:
    if zero_point is None:
        return lo + (hi - lo) * loc
    ratio = (hi - zero_point) / (lo - zero_point)
    return lo + (hi - lo) * (ratio**loc - 1) / (ratio - 1)


def _max_pmf(cdf_size: int) -> float:
    return MAX_PMF_AT_200 * 200 / (cdf_size - 1)


def standardize(cdf: np.ndarray, open_lower: bool, open_upper: bool) -> np.ndarray:
    """Apply the API's minimum-slope and maximum-step rules (mirrors Metaculus' reference code)."""
    cdf = np.asarray(cdf, dtype=float).copy()
    n = len(cdf)
    lower_to = 0.0 if open_lower else cdf[0]
    upper_to = 1.0 if open_upper else cdf[-1]
    span = upper_to - lower_to
    locs = np.linspace(0, 1, n)
    rescaled = (cdf - lower_to) / span
    if open_lower and open_upper:
        cdf = 0.988 * rescaled + 0.01 * locs + 0.001
    elif open_lower:
        cdf = 0.989 * rescaled + 0.01 * locs + 0.001
    elif open_upper:
        cdf = 0.989 * rescaled + 0.01 * locs
    else:
        cdf = 0.99 * rescaled + 0.01 * locs

    pmf = np.diff(cdf, prepend=0.0, append=1.0)
    cap = _max_pmf(n) * 0.95

    def capped(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    lo_s, hi_s = 1.0, 1.0
    while capped(hi_s).sum() < 1.0:
        hi_s *= 1.2
    scale = 1.0
    for _ in range(100):
        scale = 0.5 * (lo_s + hi_s)
        s = capped(scale).sum()
        if s < 1.0:
            lo_s = scale
        else:
            hi_s = scale
        if hi_s - lo_s < 2e-6:
            break
    pmf = capped(scale)
    pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
    return np.round(np.cumsum(pmf)[:-1], 10)


def percentiles_to_cdf(
    percentiles: dict[float, float],
    lo: float,
    hi: float,
    zero_point: float | None,
    open_lower: bool,
    open_upper: bool,
    cdf_size: int = 201,
) -> list[float]:
    """Piecewise-linear CDF (in cdf-location space) through the declared percentiles."""
    pts = sorted(percentiles.items())
    xs = np.array([value_to_location(v, lo, hi, zero_point) for _, v in pts], dtype=float)
    ys = np.array([p for p, _ in pts], dtype=float)
    # De-duplicate identical locations (models sometimes repeat a value).
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6

    # Slope of the outermost segments, used to extend the tails.
    left_slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
    right_slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)

    # Value of the CDF at the range edges.
    if open_lower:
        y0 = float(np.interp(0.0, xs, ys)) if xs[0] < 0 else ys[0] - left_slope * xs[0]
        y0 = min(max(y0, 0.0), ys[0])
    else:
        y0 = 0.0
    if open_upper:
        y1 = float(np.interp(1.0, xs, ys)) if xs[-1] > 1 else ys[-1] + right_slope * (1 - xs[-1])
        y1 = max(min(y1, 1.0), ys[-1])
    else:
        y1 = 1.0

    inside = (xs > 0) & (xs < 1)
    knots_x = np.concatenate([[0.0], xs[inside], [1.0]])
    knots_y = np.concatenate([[y0], ys[inside], [y1]])
    # Closed bounds: if all mass the model gave is squeezed against a bound,
    # interpolation above still yields a valid (if steep) CDF; standardize() caps steps.
    locs = np.linspace(0, 1, cdf_size)
    raw = np.interp(locs, knots_x, knots_y)
    raw = np.maximum.accumulate(raw)
    cdf = standardize(raw, open_lower, open_upper)
    validate_cdf(cdf, open_lower, open_upper)
    return cdf.tolist()


def validate_cdf(cdf: list[float] | np.ndarray, open_lower: bool, open_upper: bool) -> None:
    c = np.asarray(cdf, dtype=float)
    n = len(c)
    steps = np.diff(c)
    min_step = 0.01 / (n - 1) * 0.99
    if np.any(steps < min_step - 1e-9):
        raise ValueError(f"CDF not increasing enough (min step {steps.min():.2e})")
    if np.any(steps > _max_pmf(n) + 1e-9):
        raise ValueError(f"CDF step too large ({steps.max():.3f})")
    if open_lower and c[0] < 0.001 - 1e-9:
        raise ValueError("open lower bound needs >= 0.001 mass below range")
    if not open_lower and abs(c[0]) > 1e-9:
        raise ValueError("closed lower bound must start at 0")
    if open_upper and c[-1] > 0.999 + 1e-9:
        raise ValueError("open upper bound needs >= 0.001 mass above range")
    if not open_upper and abs(c[-1] - 1) > 1e-9:
        raise ValueError("closed upper bound must end at 1")


def cdf_quantile(cdf: list[float], q: float, lo: float, hi: float, zero_point: float | None) -> float:
    """Inverse CDF, used for human-readable summaries in comments."""
    c = np.asarray(cdf)
    locs = np.linspace(0, 1, len(c))
    loc = float(np.interp(q, c, locs))
    return location_to_value(loc, lo, hi, zero_point)
