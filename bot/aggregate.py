"""Combine the ensemble's forecasts into one submission."""
from __future__ import annotations

import numpy as np


def aggregate_binary(probs: list[float], clip: float) -> float:
    # Median: robust to one model misreading the question (the dominant failure mode).
    p = float(np.median(np.asarray(probs, dtype=float)))
    return float(min(max(p, clip), 1 - clip))


def aggregate_multiple_choice(dists: list[dict[str, float]], options: list[str], floor: float) -> dict[str, float]:
    mean = {opt: float(np.mean([d[opt] for d in dists])) for opt in options}
    floored = {opt: max(v, floor) for opt, v in mean.items()}
    total = sum(floored.values())
    out = {opt: v / total for opt, v in floored.items()}
    # Fix float drift so the API sees an exact 1.0 sum.
    drift = 1.0 - sum(out.values())
    top = max(out, key=out.get)
    out[top] += drift
    return out


def aggregate_cdfs(cdfs: list[list[float]]) -> list[float]:
    # Mixture (pointwise mean) keeps the ensemble's disagreement as extra spread;
    # bot makers' most common numeric error was under-estimating variance.
    return np.round(np.mean(np.asarray(cdfs, dtype=float), axis=0), 10).tolist()
