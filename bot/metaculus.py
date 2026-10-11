"""Thin Metaculus API client (only the endpoints the bot needs)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

API = "https://www.metaculus.com/api"
FORECAST_TYPES = "binary,multiple_choice,numeric,discrete"


@dataclass
class Question:
    post_id: int
    question_id: int
    type: str  # binary | multiple_choice | numeric | discrete
    title: str
    description: str
    resolution_criteria: str
    fine_print: str
    open_time: str | None
    close_time: str | None
    resolve_time: str | None
    options: list[str] | None = None
    unit: str | None = None
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    cdf_size: int = 201
    already_forecast: bool = False

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"

    @classmethod
    def from_post(cls, post: dict[str, Any]) -> "Question":
        q = post["question"]
        scaling = q.get("scaling") or {}
        qtype = q["type"]
        cdf_size = 201
        if qtype == "discrete" and scaling.get("inbound_outcome_count"):
            cdf_size = int(scaling["inbound_outcome_count"]) + 1
        latest = ((q.get("my_forecasts") or {}).get("latest") or {})
        return cls(
            post_id=post["id"],
            question_id=q["id"],
            type=qtype,
            title=q.get("title") or post.get("title") or "",
            description=q.get("description") or "",
            resolution_criteria=q.get("resolution_criteria") or "",
            fine_print=q.get("fine_print") or "",
            open_time=q.get("open_time"),
            close_time=q.get("scheduled_close_time"),
            resolve_time=q.get("scheduled_resolve_time"),
            options=q.get("options"),
            unit=q.get("unit") or None,
            range_min=scaling.get("range_min"),
            range_max=scaling.get("range_max"),
            zero_point=scaling.get("zero_point"),
            open_lower_bound=bool(q.get("open_lower_bound")),
            open_upper_bound=bool(q.get("open_upper_bound")),
            cdf_size=cdf_size,
            already_forecast=latest.get("forecast_values") is not None,
        )


class MetaculusClient:
    def __init__(self, token: str, session: requests.Session | None = None):
        if not token:
            raise ValueError("METACULUS_TOKEN is not set")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.session.request(method, f"{API}{path}", timeout=60, **kwargs)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code}: {resp.text[:300]}")
                if not resp.ok:
                    # 4xx other than 429 will not fix itself; surface immediately.
                    raise RuntimeError(f"Metaculus {method} {path} -> {resp.status_code}: {resp.text[:500]}")
                return resp.json() if resp.content else None
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Metaculus {method} {path} failed after retries: {last_error}")

    def open_questions(self, tournament: int | str) -> list[Question]:
        questions: list[Question] = []
        offset = 0
        while True:
            data = self._request(
                "GET",
                "/posts/",
                params={
                    "limit": 50,
                    "offset": offset,
                    "order_by": "-hotness",
                    "forecast_type": FORECAST_TYPES,
                    "tournaments": [tournament],
                    "statuses": "open",
                    "include_description": "true",
                },
            )
            results = data.get("results", [])
            for post in results:
                q = post.get("question")
                if q and q.get("status") == "open":
                    questions.append(Question.from_post(post))
            if not data.get("next") or not results:
                return questions
            offset += len(results)

    def post(self, post_id: int) -> Question:
        return Question.from_post(self._request("GET", f"/posts/{post_id}/"))

    def submit(self, question: Question, payload: dict[str, Any]) -> None:
        body = [{"question": question.question_id, "source": "api", **payload}]
        self._request("POST", "/questions/forecast/", json=body)

    def comment(self, question: Question, text: str) -> None:
        self._request(
            "POST",
            "/comments/create/",
            json={
                "text": text,
                "parent": None,
                "included_forecast": True,
                "is_private": True,  # Metaculus publishes bot notes after close
                "on_post": question.post_id,
            },
        )


def forecast_payload(question: Question, forecast: Any) -> dict[str, Any]:
    if question.type == "binary":
        return {"probability_yes": float(forecast), "probability_yes_per_category": None, "continuous_cdf": None}
    if question.type == "multiple_choice":
        return {"probability_yes": None, "probability_yes_per_category": dict(forecast), "continuous_cdf": None}
    return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": list(forecast)}
