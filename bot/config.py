"""All tunable settings live here. Every value can be overridden with an env var.

Design notes (from the Spring 2026 bot-maker survey and our own analysis):
- Simple + reliable beats clever. One shared research brief, a small ensemble of
  models from *different labs*, median/mixture aggregation, light clipping.
- Main-tournament questions are worth ~8x more prize money per question than
  MiniBench questions ($50k / ~350 vs $1k / ~60), so they get the stronger stack.
- Never submit a junk forecast (e.g. 0.5 after an error): skipping scores 0,
  a bad forecast scores negative. Failed questions are retried on the next run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# Tournament identifiers (slugs or numeric ids both work with the posts API).
FALL_2026_TOURNAMENT = 33121  # https://www.metaculus.com/tournament/fall-futureeval-2026/
MINIBENCH = "minibench"  # always points at the currently active MiniBench
BOT_TESTING_AREA = "bot-testing-area"  # unscored, safe for smoke tests
METACULUS_CUP = _env("METACULUS_CUP_ID", "")  # practice vs humans (set slug when known)


@dataclass(frozen=True)
class Stack:
    """Which models do which job for a given competition."""

    name: str
    query_model: str  # cheap model that writes search queries
    web_research_model: str  # model with native web search (":online")
    forecasters: list[str]  # one forecast per entry (duplicates = extra samples)
    parser_model: str  # cheap model used only to repair unparseable outputs
    min_successful_forecasters: int = 2
    max_cost_per_question_usd: float = 1.0


MAIN_STACK = Stack(
    name="main",
    query_model=_env("QUERY_MODEL", "anthropic/claude-haiku-5.5"),
    web_research_model=_env("WEB_RESEARCH_MODEL", "openai/gpt-6.1-sol:online"),
    forecasters=_env_list(
        "MAIN_FORECASTERS",
        [
            "anthropic/claude-sonnet-5.5",
            "openai/gpt-6.1-sol",
            "google/gemini-3.8-flash",
        ],
    ),
    parser_model=_env("PARSER_MODEL", "openai/gpt-6-luna"),
    min_successful_forecasters=2,
    max_cost_per_question_usd=_env_float("MAIN_MAX_COST_PER_QUESTION", 1.00),
)

LEAN_STACK = Stack(
    name="lean",
    query_model=_env("QUERY_MODEL", "anthropic/claude-haiku-5.5"),
    web_research_model=_env("LEAN_WEB_RESEARCH_MODEL", "openai/gpt-6-luna:online"),
    forecasters=_env_list(
        "LEAN_FORECASTERS",
        [
            "anthropic/claude-haiku-5.5",
            "openai/gpt-6-luna",
            "google/gemini-3.8-flash",
        ],
    ),
    parser_model=_env("PARSER_MODEL", "openai/gpt-6-luna"),
    min_successful_forecasters=2,
    max_cost_per_question_usd=_env_float("LEAN_MAX_COST_PER_QUESTION", 0.25),
)


@dataclass(frozen=True)
class Settings:
    metaculus_token: str = field(default_factory=lambda: _env("METACULUS_TOKEN", ""))
    openrouter_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))
    asknews_api_key: str = field(default_factory=lambda: _env("ASKNEWS_API_KEY", ""))
    asknews_client_id: str = field(default_factory=lambda: _env("ASKNEWS_CLIENT_ID", ""))
    asknews_secret: str = field(default_factory=lambda: _env("ASKNEWS_SECRET", ""))

    # Binary probabilities are clipped to [clip, 1 - clip]. Cheap insurance:
    # a confident miss costs ~100x more peer score than the clip costs on a hit.
    binary_clip: float = field(default_factory=lambda: _env_float("BINARY_CLIP", 0.02))
    # Minimum probability for every multiple-choice option before renormalising.
    mc_floor: float = field(default_factory=lambda: _env_float("MC_FLOOR", 0.01))

    # Budget guard: below this OpenRouter balance the bot drops to the lean
    # stack; below the hard floor it stops forecasting entirely.
    lean_mode_below_usd: float = field(default_factory=lambda: _env_float("LEAN_MODE_BELOW_USD", 15.0))
    stop_below_usd: float = field(default_factory=lambda: _env_float("STOP_BELOW_USD", 1.0))

    @property
    def has_asknews(self) -> bool:
        return bool(self.asknews_api_key or (self.asknews_client_id and self.asknews_secret))

    max_concurrent_questions: int = 3
    llm_timeout_s: float = 240.0
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "logs"))


STACK_FOR_MODE = {
    "tournament": MAIN_STACK,
    "minibench": LEAN_STACK,
    "test": LEAN_STACK,
    "cup": LEAN_STACK,
}

TOURNAMENTS_FOR_MODE: dict[str, list[int | str]] = {
    "tournament": [FALL_2026_TOURNAMENT],
    "minibench": [MINIBENCH],
    "test": [BOT_TESTING_AREA],
    "cup": [METACULUS_CUP] if METACULUS_CUP else [],
}
