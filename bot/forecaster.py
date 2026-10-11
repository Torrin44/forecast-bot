"""Forecast one question end to end: research -> ensemble -> aggregate -> comment."""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from . import aggregate, numeric, parsing, prompts
from .config import Settings, Stack
from .llm import BudgetExceeded, CostMeter, LLMError, OpenRouter
from .metaculus import Question
from .research import build_research


class SkipQuestion(RuntimeError):
    """Not enough usable forecasts; leave the question for the next run."""


@dataclass
class ModelForecast:
    model: str
    value: Any  # float | dict[str, float] | list[float] (CDF)
    summary: str  # human-readable forecast
    reasoning: str


@dataclass
class Result:
    question: Question
    forecast: Any
    comment: str
    members: list[ModelForecast]
    cost_usd: float
    research_sources: list[str]
    research_failed: dict[str, str]
    log: dict[str, Any] = field(default_factory=dict)


def _reasoning_effort(stack: Stack) -> str:
    return "high" if stack.name == "main" else "medium"


def _parse(q: Question, text: str) -> tuple[Any, str]:
    if q.type == "binary":
        p = parsing.parse_binary(text)
        return p, f"{p:.1%}"
    if q.type in ("numeric", "discrete"):
        pct = parsing.parse_percentiles(text)
        cdf = numeric.percentiles_to_cdf(
            pct, q.range_min, q.range_max, q.zero_point, q.open_lower_bound, q.open_upper_bound, q.cdf_size
        )
        p10, p50, p90 = (pct.get(0.1), pct.get(0.5), pct.get(0.9))
        return cdf, f"P10 {p10:g} / median {p50:g} / P90 {p90:g}"
    if q.type == "multiple_choice":
        dist = parsing.parse_multiple_choice(text, q.options or [])
        return dist, ", ".join(f"{k}: {v:.0%}" for k, v in dist.items())
    raise SkipQuestion(f"unsupported type {q.type}")


async def _one_forecaster(llm: OpenRouter, stack: Stack, q: Question, prompt: str, model: str, meter: CostMeter) -> ModelForecast:
    text = await llm.chat(
        model,
        [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": prompt}],
        meter=meter,
        reasoning_effort=_reasoning_effort(stack),
        max_tokens=6000,
    )
    try:
        value, summary = _parse(q, text)
    except (parsing.ParseError, ValueError):
        # A dedicated cheap parser recovers most malformed answers
        # (parser failures were a top error source for other bots).
        fixed = await llm.chat(
            stack.parser_model,
            [{"role": "user", "content": prompts.REPAIR_PROMPT.format(instructions=prompts.repair_instructions(q), text=text[-6000:])}],
            meter=meter,
            temperature=0,
            max_tokens=500,
        )
        value, summary = _parse(q, fixed)
    return ModelForecast(model=model, value=value, summary=summary, reasoning=text)


def _final_summary(q: Question, forecast: Any) -> str:
    if q.type == "binary":
        return f"{forecast:.1%}"
    if q.type == "multiple_choice":
        return ", ".join(f"{k}: {v:.1%}" for k, v in forecast.items())
    qs = [numeric.cdf_quantile(forecast, x, q.range_min, q.range_max, q.zero_point) for x in (0.1, 0.5, 0.9)]
    return f"P10 {qs[0]:.4g} / median {qs[1]:.4g} / P90 {qs[2]:.4g}"


def _comment(q: Question, final: str, members: list[ModelForecast], research_sources: list[str]) -> str:
    lines = [
        f"**Final forecast:** {final}",
        f"Method: shared research brief ({', '.join(research_sources) or 'none'}) -> {len(members)} independent model forecasts -> "
        + ("median, clipped" if q.type == "binary" else "averaged") + ".",
        "",
        "**Ensemble members:**",
    ]
    lines += [f"- {m.model}: {m.summary}" for m in members]
    budget_per_member = max(800, 6000 // max(1, len(members)))
    for m in members:
        reasoning = m.reasoning.strip()
        if len(reasoning) > budget_per_member:
            reasoning = reasoning[: budget_per_member - 40] + "\n...[truncated]..."
        lines += ["", f"--- Reasoning ({m.model}) ---", reasoning]
    return "\n".join(lines)


async def forecast_question(llm: OpenRouter, settings: Settings, stack: Stack, q: Question) -> Result:
    meter = CostMeter(ceiling_usd=stack.max_cost_per_question_usd)
    rich = stack.name == "main"
    research = await build_research(llm, settings, stack, q, meter, rich=rich)
    prompt = prompts.prompt_for(q, research.brief)

    outcomes = await asyncio.gather(
        *[_one_forecaster(llm, stack, q, prompt, model, meter) for model in stack.forecasters],
        return_exceptions=True,
    )
    members = [o for o in outcomes if isinstance(o, ModelForecast)]
    errors = {m: f"{type(o).__name__}: {o}"[:300] for m, o in zip(stack.forecasters, outcomes) if not isinstance(o, ModelForecast)}
    if len(members) < stack.min_successful_forecasters:
        raise SkipQuestion(f"only {len(members)} usable forecasts; errors={errors}")

    if q.type == "binary":
        forecast: Any = aggregate.aggregate_binary([m.value for m in members], settings.binary_clip)
    elif q.type == "multiple_choice":
        forecast = aggregate.aggregate_multiple_choice([m.value for m in members], q.options or [], settings.mc_floor)
    else:
        forecast = aggregate.aggregate_cdfs([m.value for m in members])
        numeric.validate_cdf(forecast, q.open_lower_bound, q.open_upper_bound)

    final = _final_summary(q, forecast)
    log = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_id": q.post_id,
        "question_id": q.question_id,
        "type": q.type,
        "title": q.title,
        "stack": stack.name,
        "final": final,
        "forecast": forecast if q.type != "binary" else round(forecast, 4),
        "members": [{"model": m.model, "summary": m.summary} for m in members],
        "member_errors": errors,
        "research_ok": research.sources_ok,
        "research_failed": research.sources_failed,
        "queries": research.queries,
        "research_brief": research.brief,
        "cost_usd": round(meter.spent_usd, 4),
        "calls": meter.calls,
    }
    return Result(
        question=q,
        forecast=forecast,
        comment=_comment(q, final, members, research.sources_ok),
        members=members,
        cost_usd=meter.spent_usd,
        research_sources=research.sources_ok,
        research_failed=research.sources_failed,
        log=log,
    )


__all__ = ["forecast_question", "SkipQuestion", "Result", "BudgetExceeded", "LLMError"]
