"""OpenRouter chat client with retries, per-question cost metering and a budget probe."""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class CostMeter:
    """Tracks spend for one question so a runaway model can't drain the credits."""

    ceiling_usd: float
    spent_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def check(self, optional: bool) -> None:
        # Optional calls (extra research) stop at 60% of the ceiling so the
        # required forecasting calls always have headroom.
        limit = self.ceiling_usd * (0.6 if optional else 1.0)
        if self.spent_usd >= limit:
            raise BudgetExceeded(f"question budget used: ${self.spent_usd:.3f} of ${self.ceiling_usd:.2f}")

    def add(self, model: str, cost: float, usage: dict[str, Any]) -> None:
        self.spent_usd += cost
        self.calls.append({"model": model, "cost": round(cost, 5), **usage})


class OpenRouter:
    def __init__(self, api_key: str, timeout_s: float = 240.0, max_parallel: int = 6):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_parallel)
        self.total_spent_usd = 0.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/"),
            "X-Title": "metaculus-forecast-bot",
        }

    def remaining_budget_usd(self) -> float | None:
        """Credits left on this key (None if the key has no limit or the call fails)."""
        try:
            resp = requests.get(f"{OPENROUTER_URL}/key", headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            remaining = data.get("limit_remaining")
            return float(remaining) if remaining is not None else None
        except Exception:
            return None

    def _post_sync(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{OPENROUTER_URL}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout_s
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            raise LLMError(f"retryable {resp.status_code}: {resp.text[:300]}")
        if not resp.ok:
            raise LLMError(f"fatal {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"retryable provider error: {str(data['error'])[:300]}")
        return data

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        meter: CostMeter | None = None,
        optional: bool = False,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if meter:
            meter.check(optional)
        body: dict[str, Any] = {"model": model, "messages": messages, "usage": {"include": True}}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self._sem:
                    started = time.monotonic()
                    data = await asyncio.to_thread(self._post_sync, body)
                usage = data.get("usage") or {}
                cost = float(usage.get("cost") or 0.0)
                self.total_spent_usd += cost
                if meter:
                    meter.add(
                        model,
                        cost,
                        {
                            "in": usage.get("prompt_tokens"),
                            "out": usage.get("completion_tokens"),
                            "secs": round(time.monotonic() - started, 1),
                        },
                    )
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if not content or not str(content).strip():
                    raise LLMError("retryable: empty completion")
                return str(content)
            except (LLMError, requests.RequestException) as exc:
                last_error = exc
                if isinstance(exc, LLMError) and str(exc).startswith("fatal"):
                    break
                await asyncio.sleep((2 ** attempt) * 3 + random.random() * 2)
        raise LLMError(f"{model} failed: {last_error}")
