"""Forecasting prompts.

The rules in SYSTEM encode the failure modes bot makers reported as their
*largest* single-question losses (literal resolution criteria, the "assume it
hasn't happened yet" convention, forward-looking "before <date>" questions,
stale evidence) plus base-rate-first reasoning, which one top bot maker called
the single biggest lever.
"""
from __future__ import annotations

import datetime as dt

from .metaculus import Question

SYSTEM = """You are an elite superforecaster competing in a forecasting tournament. Forecasts are scored with a proper logarithmic scoring rule relative to other forecasters, so the only winning strategy is honest, well-calibrated probabilities.

Ground rules (tournament conventions - follow them exactly):
1. The question resolves ONLY by its resolution criteria and fine print, read literally. Do not substitute what "should" count or make convenient "reasonable assumptions". If a specific source is named, what matters is what that source will report, by the stated date.
2. Unless the research clearly shows otherwise, assume the outcome had not yet occurred when the question opened. Questions phrased "before <date>" or "by <date>" are about what happens between now and that deadline.
3. Check every date: today's date, the close and resolution dates, and the date of each piece of evidence. Stale or misdated evidence causes the biggest misses.
4. Your training data has a cutoff. For anything recent, trust dated research over memory. Never invent facts; if research is thin, lean on base rates.
5. The world usually changes slowly. The status-quo outcome deserves substantial weight, especially when little time remains.
6. Calibration: go below 3% or above 97% only when current facts essentially settle the outcome AND the resolution mechanics are unambiguous.
7. Prediction markets are useful evidence when they match the question's exact wording, threshold and deadline; discount them when they do not, and treat thin or play-money markets as weak evidence."""

METHOD = """Work through these steps, briefly and concretely:
1. RESOLUTION: Restate exactly what must happen (and by when, per which source) for each outcome. Note any edge cases in the fine print.
2. STATUS QUO: What is true right now, per the most recent dated evidence? What happens if nothing changes?
3. OUTSIDE VIEW: Name a reference class and give an explicit base rate or historical distribution. Adjust it for the time remaining ({time_left}).
4. INSIDE VIEW: The strongest specific evidence for and against, citing the research. How much should each move you from the base rate?
5. MARKETS: Do any prediction markets match closely enough to use? If so, how do you weigh them?
6. SANITY CHECK: Consider how you could be wrong (surprises, resolution technicalities, stale data) and whether you are over- or under-confident."""


def _time_left(q: Question) -> str:
    def parse(ts: str | None) -> dt.datetime | None:
        if not ts:
            return None
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    now = dt.datetime.now(dt.timezone.utc)
    resolve = parse(q.resolve_time)
    if not resolve:
        return "unknown"
    days = (resolve - now).total_seconds() / 86400
    return f"about {days:.0f} days until scheduled resolution ({resolve:%Y-%m-%d})" if days >= 1 else "less than a day until scheduled resolution"


def _header(q: Question, research: str) -> str:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return f"""Today is {today}.

QUESTION: {q.title}

BACKGROUND:
{q.description.strip() or '(none)'}

RESOLUTION CRITERIA:
{q.resolution_criteria.strip() or '(none)'}

FINE PRINT:
{q.fine_print.strip() or '(none)'}

Question opened: {q.open_time}. Forecasting closes: {q.close_time}. Scheduled resolution: {q.resolve_time}.

RESEARCH (gathered just now; check dates and relevance yourself):
{research}
"""


def binary_prompt(q: Question, research: str) -> str:
    return (
        _header(q, research)
        + "\n"
        + METHOD.format(time_left=_time_left(q))
        + "\n\nEnd your answer with a final line exactly in this form (one number, decimals allowed):\nProbability: XX%"
    )


def _bounds_text(q: Question) -> str:
    lo, hi = q.range_min, q.range_max
    lines = []
    if q.open_lower_bound:
        lines.append(f"The question's range starts at {lo}, but outcomes below {lo} are possible (open lower bound).")
    else:
        lines.append(f"The outcome cannot be lower than {lo} (closed lower bound).")
    if q.open_upper_bound:
        lines.append(f"The question's range ends at {hi}, but outcomes above {hi} are possible (open upper bound).")
    else:
        lines.append(f"The outcome cannot be higher than {hi} (closed upper bound).")
    if q.type == "discrete":
        lines.append("This is a discrete question: the outcome is a whole number (or one of evenly spaced values).")
    return "\n".join(lines)


NUMERIC_PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def numeric_prompt(q: Question, research: str) -> str:
    pct_lines = "\n".join(f"Percentile {p}: X" for p in NUMERIC_PERCENTILES)
    return (
        _header(q, research)
        + f"\nUNITS: {q.unit or 'not stated - infer from the question'}\n{_bounds_text(q)}\n\n"
        + METHOD.format(time_left=_time_left(q))
        + "\n7. DISTRIBUTION: Start from the latest value and the historical volatility over a comparable horizon. "
        "Uncertainty grows with time to resolution; underestimating spread is the most common numeric mistake, "
        "so make sure your 5th-95th percentile range would contain the outcome roughly 90% of the time.\n\n"
        "End your answer with your percentiles in exactly this form, in the question's units, increasing, "
        "plain numbers only (no units, no commas, no scientific notation):\n" + pct_lines
    )


def multiple_choice_prompt(q: Question, research: str) -> str:
    options = q.options or []
    option_lines = "\n".join(f"{opt}: XX%" for opt in options)
    return (
        _header(q, research)
        + f"\nOPTIONS (in order): {options}\n\n"
        + METHOD.format(time_left=_time_left(q))
        + "\nLeave at least a small probability on every option - surprises happen.\n\n"
        "End your answer with one line per option, in this exact order and format, summing to 100%:\n" + option_lines
    )


def prompt_for(q: Question, research: str) -> str:
    if q.type == "binary":
        return binary_prompt(q, research)
    if q.type in ("numeric", "discrete"):
        return numeric_prompt(q, research)
    if q.type == "multiple_choice":
        return multiple_choice_prompt(q, research)
    raise ValueError(f"unsupported question type {q.type}")


REPAIR_PROMPT = """Extract the final forecast from the analysis below. Do not change any numbers.
{instructions}

ANALYSIS:
{text}"""


def repair_instructions(q: Question) -> str:
    if q.type == "binary":
        return "Reply with exactly one line: Probability: XX%"
    if q.type in ("numeric", "discrete"):
        return "Reply with exactly these lines and nothing else:\n" + "\n".join(
            f"Percentile {p}: X" for p in NUMERIC_PERCENTILES
        )
    return "Reply with exactly one line per option, in this order, nothing else:\n" + "\n".join(
        f"{opt}: XX%" for opt in (q.options or [])
    )
