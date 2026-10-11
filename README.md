# forecast-bot

An autonomous forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/participate/) bot tournaments: $50k per four-month season, plus a $1k MiniBench every two weeks.

- **What it does and why:** [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)
- **Setting it up:** [docs/SETUP.md](docs/SETUP.md)

```
main.py                CLI: --mode all|tournament|minibench|test|status, --dry-run
bot/config.py          tournaments, model stacks, budgets (all env-overridable)
bot/metaculus.py       Metaculus API client
bot/research.py        web facts + news + prediction markets -> one research brief
bot/prompts.py         forecasting rules and per-question-type prompts
bot/llm.py             OpenRouter client, retries, per-question cost ceiling
bot/parsing.py         robust answer extraction
bot/numeric.py         percentiles -> Metaculus-valid CDF
bot/aggregate.py       median / mixture aggregation
bot/forecaster.py      one question end to end
tests/                 offline tests (fake LLM + fake Metaculus); run: python -m pytest
```

License: MIT.
