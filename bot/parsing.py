"""Robust extraction of final answers from free-text model output."""
from __future__ import annotations

import re

from .prompts import NUMERIC_PERCENTILES

_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+"


class ParseError(ValueError):
    pass


def _to_float(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-"))


def parse_binary(text: str) -> float:
    """Return P(yes) in [0, 1] from 'Probability: XX%' (last occurrence wins)."""
    matches = re.findall(rf"probability\s*[:=]\s*\**\s*({_NUM})\s*%", text, re.I)
    if not matches:
        # Fall back to the last percentage in the final 300 characters.
        matches = re.findall(rf"({_NUM})\s*%", text[-300:])
    if not matches:
        raise ParseError("no probability found")
    value = _to_float(matches[-1])
    if not 0 <= value <= 100:
        raise ParseError(f"probability out of range: {value}")
    return value / 100.0


_SUFFIX = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9, "bn": 1e9, "t": 1e12, "trillion": 1e12}


def parse_percentiles(text: str) -> dict[float, float]:
    """Return {percentile_fraction: value}. Uses the last block of 'Percentile N: value' lines."""
    found: dict[float, float] = {}
    pattern = re.compile(
        rf"percentile\s*(\d{{1,2}}(?:\.\d+)?)\s*[:=]\s*\**\s*\$?\s*({_NUM})\s*(k|thousand|m|million|bn|b|billion|t|trillion)?\b",
        re.I,
    )
    for line in text.splitlines():
        m = pattern.search(line.replace("−", "-"))
        if not m:
            continue
        pct = float(m.group(1))
        value = _to_float(m.group(2))
        if m.group(3):
            value *= _SUFFIX[m.group(3).lower()]
        found[pct / 100.0] = value  # later lines overwrite earlier drafts
    expected = {p / 100.0 for p in NUMERIC_PERCENTILES}
    if len(found.keys() & expected) < 5:
        raise ParseError(f"only {len(found)} percentiles found")
    pcts = sorted(found)
    values = sorted(found[p] for p in pcts)  # repair non-monotonic answers by sorting values
    return dict(zip(pcts, values))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Return {option: probability} normalised to sum to 1."""
    lines = [re.sub(r"[*_`#>]", "", ln).strip() for ln in text.splitlines() if "%" in ln]
    result: dict[str, float] = {}
    # Prefer exact "Option: XX%" lines, searching from the end (final answer block).
    for opt in sorted(options, key=len, reverse=True):  # longest first avoids prefix collisions
        target = _norm(opt)
        for ln in reversed(lines):
            head, _, tail = ln.rpartition(":")
            if head and _norm(head).endswith(target) and opt not in result:
                nums = re.findall(rf"({_NUM})\s*%", tail)
                if nums:
                    result[opt] = _to_float(nums[-1])
                    break
    if len(result) != len(options):
        # Fallback: the last len(options) percentage lines, in order.
        tail_lines = lines[-len(options):]
        if len(tail_lines) == len(options):
            result = {}
            for opt, ln in zip(options, tail_lines):
                nums = re.findall(rf"({_NUM})\s*%", ln)
                if not nums:
                    break
                result[opt] = _to_float(nums[-1])
    if len(result) != len(options):
        raise ParseError(f"matched {len(result)}/{len(options)} options")
    total = sum(result.values())
    if total <= 0:
        raise ParseError("probabilities sum to zero")
    return {opt: result[opt] / total for opt in options}
