"""Entry point.

Examples:
  python main.py --mode test --dry-run        # smoke test on the unscored testing area, nothing posted
  python main.py --mode test --force          # post to the testing area (re-forecasting allowed there)
  python main.py --mode all                   # live: Fall 2026 tournament + current MiniBench
  python main.py --mode status                # show budget and open-question counts
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from bot.config import LEAN_STACK, STACK_FOR_MODE, TOURNAMENTS_FOR_MODE, Settings  # noqa: E402
from bot.forecaster import SkipQuestion, forecast_question  # noqa: E402
from bot.llm import OpenRouter  # noqa: E402
from bot.metaculus import MetaculusClient, Question, forecast_payload  # noqa: E402


def _log(settings: Settings, record: dict) -> None:
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.log_dir) / f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


async def run_mode(mode: str, args: argparse.Namespace, settings: Settings, mc: MetaculusClient, llm: OpenRouter, lean: bool) -> tuple[int, int, int]:
    stack = LEAN_STACK if lean else STACK_FOR_MODE[mode]
    tournaments = TOURNAMENTS_FOR_MODE[mode]
    if not tournaments:
        print(f"[{mode}] no tournament configured, skipping")
        return 0, 0, 0

    questions: list[Question] = []
    for t in tournaments:
        found = mc.open_questions(t)
        print(f"[{mode}] tournament {t}: {len(found)} open questions")
        questions += found
    if args.post_id:
        questions = [mc.post(args.post_id)]
    todo = [q for q in questions if args.force or not q.already_forecast]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{mode}] {len(todo)} to forecast with the '{stack.name}' stack ({', '.join(stack.forecasters)})")

    sem = asyncio.Semaphore(settings.max_concurrent_questions)
    done = skipped = failed = 0

    async def handle(q: Question) -> None:
        nonlocal done, skipped, failed
        async with sem:
            if not args.force:
                # The list endpoint may not include our own past forecasts, so re-check
                # each question on its detail endpoint (as Metaculus' template does).
                # Forecasting twice breaks the one-forecast-per-question rule.
                try:
                    fresh = await asyncio.to_thread(mc.post, q.post_id)
                except Exception as exc:
                    failed += 1
                    print(f"  FAIL  {q.url} detail lookup :: {exc}")
                    return
                if fresh.already_forecast:
                    return
                q = fresh
            try:
                result = await forecast_question(llm, settings, stack, q)
            except SkipQuestion as exc:
                skipped += 1
                print(f"  SKIP  {q.url} {q.title[:70]} :: {exc}")
                _log(settings, {"post_id": q.post_id, "title": q.title, "mode": mode, "status": "skipped", "reason": str(exc)})
                return
            except Exception as exc:  # never let one question kill the run
                failed += 1
                print(f"  FAIL  {q.url} {q.title[:70]} :: {type(exc).__name__}: {exc}")
                _log(settings, {"post_id": q.post_id, "title": q.title, "mode": mode, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                return
            posted = False
            if not args.dry_run:
                try:
                    mc.submit(q, forecast_payload(q, result.forecast))
                except Exception as exc:
                    failed += 1
                    print(f"  FAIL  {q.url} posting :: {exc}")
                    _log(settings, {**result.log, "mode": mode, "status": "post_failed", "reason": str(exc)})
                    return
                # A comment is required for prize eligibility, and after the forecast is
                # in, later runs skip this question - so try hard to get it posted now.
                for attempt in range(3):
                    try:
                        mc.comment(q, result.comment)
                        break
                    except Exception as exc:
                        if attempt == 2:
                            print(f"  WARN  {q.url} forecast posted but comment failed :: {exc}")
                            _log(settings, {"post_id": q.post_id, "mode": mode, "status": "comment_failed", "reason": str(exc), "comment": result.comment})
                        else:
                            await asyncio.sleep(5 * (attempt + 1))
                posted = True
            done += 1
            tag = "POSTED" if posted else "DRYRUN"
            print(f"  {tag} {q.url} [{q.type}] {result.log['final']}  (${result.cost_usd:.3f}; research: {','.join(result.research_sources) or 'none'})")
            _log(settings, {**result.log, "mode": mode, "status": tag.lower()})

    await asyncio.gather(*(handle(q) for q in todo))
    return done, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "tournament", "minibench", "test", "cup", "status"])
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not post anything")
    parser.add_argument("--force", action="store_true", help="re-forecast questions already forecast (testing only)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--post-id", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    missing = [n for n, v in [("METACULUS_TOKEN", settings.metaculus_token), ("OPENROUTER_API_KEY", settings.openrouter_key)] if not v]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)} (see docs/SETUP.md)")
        return 2
    if args.force and args.mode not in ("test", "cup") and not args.dry_run:
        print("--force is only allowed in test/cup modes: tournament rules forbid re-running the bot on a question.")
        return 2

    mc = MetaculusClient(settings.metaculus_token)
    llm = OpenRouter(settings.openrouter_key, timeout_s=settings.llm_timeout_s)

    remaining = llm.remaining_budget_usd()
    print(f"OpenRouter credits remaining: {'unlimited/unknown' if remaining is None else f'${remaining:.2f}'}")
    if args.mode == "status":
        for mode in ("tournament", "minibench"):
            for t in TOURNAMENTS_FOR_MODE[mode]:
                qs = mc.open_questions(t)
                print(f"{mode} {t}: {len(qs)} open, {sum(not q.already_forecast for q in qs)} not yet forecast")
        return 0
    if remaining is not None and remaining < settings.stop_below_usd:
        print("Budget exhausted - not forecasting. Top up or request more credits.")
        return 0
    lean = remaining is not None and remaining < settings.lean_mode_below_usd
    if lean:
        print("Low budget: using the lean stack for every question.")

    modes = ["tournament", "minibench"] if args.mode == "all" else [args.mode]

    async def run_all() -> list[int]:
        totals = [0, 0, 0]
        for mode in modes:  # sequential: main tournament first, it is worth more
            d, s, f = await run_mode(mode, args, settings, mc, llm, lean)
            totals = [totals[0] + d, totals[1] + s, totals[2] + f]
        return totals

    totals = asyncio.run(run_all())
    print(f"Done: {totals[0]} forecast, {totals[1]} skipped (will retry), {totals[2]} failed. Spent ${llm.total_spent_usd:.3f} this run.")
    # Only fail the workflow (=> email alert) when everything failed, i.e. something systemic.
    return 1 if totals[2] and not totals[0] and not totals[1] else 0


if __name__ == "__main__":
    sys.exit(main())
