"""End-to-end tests with fake LLM, fake research sources and a fake Metaculus API."""
import argparse
import asyncio

import pytest

import main as entry
from bot import numeric, research
from bot.config import LEAN_STACK, MAIN_STACK, Settings
from bot.forecaster import SkipQuestion, forecast_question
from bot.llm import CostMeter, LLMError, OpenRouter
from bot.metaculus import Question, forecast_payload


def binary_q(**kw):
    base = dict(
        post_id=1, question_id=11, type="binary", title="Will X happen before 2027?", description="bg",
        resolution_criteria="Resolves Yes if X.", fine_print="", open_time="2026-10-01T00:00:00Z",
        close_time="2026-10-11T12:00:00Z", resolve_time="2026-12-31T00:00:00Z",
    )
    base.update(kw)
    return Question(**base)


class FakeLLM(OpenRouter):
    """Answers by model name; costs $0.01 per call."""

    def __init__(self, answers, fail=()):
        self.answers = answers
        self.fail = set(fail)
        self.total_spent_usd = 0.0
        self.calls = []
        self.prompts = []

    async def chat(self, model, messages, meter=None, optional=False, **kw):
        if meter:
            meter.check(optional)
        await asyncio.sleep(0)  # like a real request: concurrent calls all pass the check first
        self.calls.append(model)
        self.prompts.append((model, messages[-1]["content"]))
        if meter:
            meter.add(model, 0.01, {})
        self.total_spent_usd += 0.01
        if model in self.fail:
            raise LLMError(f"{model} down")
        if model == MAIN_STACK.query_model:
            return '{"news_queries": ["x news"], "market_queries": ["x"], "facts_needed": ["latest x"]}'
        if model == MAIN_STACK.web_research_model:
            return "Fact: X is 5 as of 2026-10-10 (Source A)."
        return self.answers.get(model, "Probability: 50%")


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch):
    monkeypatch.setattr(research, "asknews_search", lambda settings, queries, archive: "- [2026-10-10] X news (src): something")
    monkeypatch.setattr(research, "market_search", lambda queries: "- Polymarket: \"Will X?\" -> Yes 30.0%")


def run(coro):
    return asyncio.run(coro)


def test_binary_median_of_three():
    m = MAIN_STACK.forecasters
    llm = FakeLLM({m[0]: "...Probability: 20%", m[1]: "Probability: 30%", m[2]: "Probability: 90%"})
    result = run(forecast_question(llm, Settings(), MAIN_STACK, binary_q()))
    assert result.forecast == pytest.approx(0.30)
    assert set(result.research_sources) == {"news", "web", "markets"}
    assert "Final forecast" in result.comment and "30.0%" in result.comment
    assert forecast_payload(binary_q(), result.forecast)["probability_yes"] == pytest.approx(0.30)


def test_unparseable_answer_is_repaired():
    m = MAIN_STACK.forecasters
    llm = FakeLLM({m[0]: "I think it's unlikely.", m[1]: "Probability: 30%", m[2]: "Probability: 50%",
                   MAIN_STACK.parser_model: "Probability: 10%"})
    result = run(forecast_question(llm, Settings(), MAIN_STACK, binary_q()))
    assert sorted(f.value for f in result.members) == pytest.approx([0.10, 0.30, 0.50])


def test_skip_when_too_few_models_succeed():
    m = MAIN_STACK.forecasters
    llm = FakeLLM({}, fail=[m[0], m[1]])
    with pytest.raises(SkipQuestion):
        run(forecast_question(llm, Settings(), MAIN_STACK, binary_q()))


def test_research_failure_does_not_block(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("asknews down")

    monkeypatch.setattr(research, "asknews_search", broken)
    llm = FakeLLM({m: "Probability: 40%" for m in MAIN_STACK.forecasters})
    result = run(forecast_question(llm, Settings(asknews_api_key="configured"), MAIN_STACK, binary_q()))
    assert result.forecast == pytest.approx(0.40)
    assert "news" in result.research_failed


def test_without_asknews_main_stack_uses_web_news_sweep(monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("AskNews used without credentials")

    monkeypatch.setattr(research, "asknews_search", must_not_be_called)
    llm = FakeLLM({m: "Probability: 40%" for m in MAIN_STACK.forecasters})
    result = run(forecast_question(llm, Settings(asknews_api_key="", asknews_client_id="", asknews_secret=""), MAIN_STACK, binary_q()))
    web_prompts = [p for m, p in llm.prompts if m == MAIN_STACK.web_research_model]
    assert len(web_prompts) == 2  # facts call + dedicated news sweep
    assert any("news researcher" in p for p in web_prompts)
    assert "RECENT NEWS (search-enabled assistant)" in result.log["research_brief"]
    assert set(result.research_sources) == {"news", "web", "markets"}


def test_without_asknews_lean_stack_folds_news_into_single_web_call(monkeypatch):
    monkeypatch.setattr(research, "asknews_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("used")))
    llm = FakeLLM({m: "Probability: 40%" for m in LEAN_STACK.forecasters})
    run(forecast_question(llm, Settings(asknews_api_key="", asknews_client_id="", asknews_secret=""), LEAN_STACK, binary_q()))
    web_prompts = [p for m, p in llm.prompts if m == LEAN_STACK.web_research_model]
    assert len(web_prompts) == 1 and "last 14 days" in web_prompts[0]


def test_numeric_question_produces_valid_cdf():
    q = binary_q(type="numeric", range_min=0.0, range_max=200.0, open_lower_bound=False, open_upper_bound=True, unit="units")
    block = "\n".join(f"Percentile {p}: {v}" for p, v in zip([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], [40, 50, 60, 66, 71, 75, 79, 84, 90, 100, 110]))
    llm = FakeLLM({m: block for m in MAIN_STACK.forecasters})
    result = run(forecast_question(llm, Settings(), MAIN_STACK, q))
    assert len(result.forecast) == 201
    numeric.validate_cdf(result.forecast, False, True)
    assert 70 < numeric.cdf_quantile(result.forecast, 0.5, 0, 200, None) < 80


def test_multiple_choice():
    q = binary_q(type="multiple_choice", options=["A", "B", "C"])
    m = MAIN_STACK.forecasters
    llm = FakeLLM({m[0]: "A: 60%\nB: 30%\nC: 10%", m[1]: "A: 50%\nB: 40%\nC: 10%", m[2]: "A: 70%\nB: 30%\nC: 0%"})
    result = run(forecast_question(llm, Settings(), MAIN_STACK, q))
    assert sum(result.forecast.values()) == pytest.approx(1.0)
    assert result.forecast["A"] > result.forecast["B"] > result.forecast["C"] > 0


def test_question_budget_skips_optional_research_but_still_forecasts():
    # $0.01 query call puts the meter above 60% of $0.016: optional web research is
    # skipped, but the required forecaster calls still fit under the hard ceiling.
    stack = MAIN_STACK.__class__(**{**MAIN_STACK.__dict__, "max_cost_per_question_usd": 0.016})
    llm = FakeLLM({m: "Probability: 40%" for m in stack.forecasters})
    result = run(forecast_question(llm, Settings(), stack, binary_q()))
    assert "web" in result.research_failed and result.forecast == pytest.approx(0.40)
    # Ceiling already exhausted after the query call: nothing usable -> skip, never a junk forecast.
    stack2 = MAIN_STACK.__class__(**{**MAIN_STACK.__dict__, "max_cost_per_question_usd": 0.009})
    llm2 = FakeLLM({m: "Probability: 40%" for m in stack2.forecasters})
    with pytest.raises(SkipQuestion):
        run(forecast_question(llm2, Settings(), stack2, binary_q()))


def test_cost_meter():
    meter = CostMeter(ceiling_usd=1.0)
    meter.add("m", 0.7, {})
    meter.check(optional=False)
    with pytest.raises(Exception):
        meter.check(optional=True)


def test_question_from_api_post():
    post = {
        "id": 555,
        "question": {
            "id": 777, "type": "discrete", "title": "How many?", "description": "d", "resolution_criteria": "rc",
            "fine_print": "fp", "open_time": "a", "scheduled_close_time": "b", "scheduled_resolve_time": "c",
            "scaling": {"range_min": -0.5, "range_max": 20.5, "zero_point": None, "inbound_outcome_count": 21},
            "open_lower_bound": False, "open_upper_bound": True, "my_forecasts": {"latest": None}, "status": "open",
        },
    }
    q = Question.from_post(post)
    assert q.cdf_size == 22 and q.question_id == 777 and not q.already_forecast


class FakeMetaculus:
    def __init__(self, questions):
        self.questions = questions
        self.submitted = []
        self.comments = []

    def open_questions(self, tournament):
        return self.questions

    def post(self, post_id):
        return next(q for q in self.questions if q.post_id == post_id)

    def submit(self, q, payload):
        self.submitted.append((q.question_id, payload))

    def comment(self, q, text):
        self.comments.append((q.post_id, text))


def _args(**kw):
    base = dict(dry_run=False, force=False, limit=0, post_id=0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_mode_posts_and_skips_already_forecast(tmp_path):
    qs = [binary_q(post_id=1, question_id=11), binary_q(post_id=2, question_id=22, already_forecast=True)]
    mc = FakeMetaculus(qs)
    llm = FakeLLM({m: "Probability: 25%" for m in LEAN_STACK.forecasters + MAIN_STACK.forecasters})
    settings = Settings(log_dir=str(tmp_path))
    done, skipped, failed = asyncio.run(entry.run_mode("tournament", _args(), settings, mc, llm, lean=False))
    assert (done, skipped, failed) == (1, 0, 0)
    assert mc.submitted == [(11, {"probability_yes": 0.25, "probability_yes_per_category": None, "continuous_cdf": None})]
    assert len(mc.comments) == 1
    assert list(tmp_path.glob("*.jsonl"))


def test_run_mode_rechecks_detail_endpoint_before_forecasting(tmp_path):
    listed = binary_q(post_id=3, question_id=33, already_forecast=False)  # list endpoint lacks my_forecasts
    mc = FakeMetaculus([listed])
    mc.post = lambda post_id: binary_q(post_id=3, question_id=33, already_forecast=True)
    llm = FakeLLM({m: "Probability: 25%" for m in MAIN_STACK.forecasters})
    done, skipped, failed = asyncio.run(entry.run_mode("tournament", _args(), Settings(log_dir=str(tmp_path)), mc, llm, lean=False))
    assert (done, skipped, failed) == (0, 0, 0)
    assert mc.submitted == [] and llm.calls == []  # no credits spent either


def test_comment_retried_after_successful_forecast(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(entry.asyncio, "sleep", lambda *_: real_sleep(0))
    mc = FakeMetaculus([binary_q()])
    attempts = []

    def flaky_comment(q, text):
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        mc.comments.append((q.post_id, text))

    mc.comment = flaky_comment
    llm = FakeLLM({m: "Probability: 25%" for m in MAIN_STACK.forecasters})
    done, _, failed = asyncio.run(entry.run_mode("tournament", _args(), Settings(log_dir=str(tmp_path)), mc, llm, lean=False))
    assert done == 1 and failed == 0 and len(mc.comments) == 1 and len(attempts) == 2


def test_run_mode_dry_run_posts_nothing(tmp_path):
    mc = FakeMetaculus([binary_q()])
    llm = FakeLLM({m: "Probability: 25%" for m in MAIN_STACK.forecasters})
    asyncio.run(entry.run_mode("tournament", _args(dry_run=True), Settings(log_dir=str(tmp_path)), mc, llm, lean=False))
    assert mc.submitted == [] and mc.comments == []
