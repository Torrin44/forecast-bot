# How the bot works (plain English)

To win a prize, Metaculus asks every bot maker to describe their bot, and they may ask you to walk through it on a call. This page is that description. You don't need to understand the code to explain it.

## In one sentence

For each new question, the bot gathers fresh research, asks three AI models from three different companies to forecast independently, combines their answers, and posts the result with its reasoning. No human is involved.

## Step by step

1. **Finds new questions.** About every 15 minutes it asks the Metaculus API for open questions in the current tournament and the current MiniBench. It skips any it has already answered.
2. **Plans its research.** A small, cheap model (Claude Haiku) writes a few short search queries and lists the facts that would most change the forecast.
3. **Researches**, using three sources in parallel:
   - **Web facts.** A search-enabled model (GPT with web search) reports facts only, each with a source and date: current status, latest figures, upcoming scheduled events, and historical base rates.
   - **Recent news.** A second web search lists the last month's relevant news, newest first. If an AskNews key is configured, AskNews is used for news instead.
   - **Prediction markets.** Current prices from Polymarket and Manifold for markets that look related. The models are told to check that a market really matches the question before using it.
   - **Linked pages.** The bot fetches up to three pages linked in the question. These are often the source that decides the outcome, such as an official statistics page, so this gives the latest value directly. Page text is passed to the models as data only.
4. **Forecasts.** Three models get the same question and research and answer independently: Claude Sonnet (Anthropic), GPT (OpenAI) and Gemini (Google). They are told to:
   - read the resolution criteria literally, including the fine print and named sources
   - follow Metaculus conventions: assume the event hasn't happened yet unless shown otherwise, and treat "before <date>" questions as being about the future
   - start from a historical base rate, then adjust for the specific evidence
   - weigh the status quo, check how current every piece of evidence is, and avoid extreme probabilities unless the outcome is effectively settled
5. **Combines the answers:**
   - Yes/no questions: the **median** of the three probabilities, kept between 2% and 98%. A single confident mistake costs far more than this caution does.
   - Multiple-choice questions: the **average** of the three distributions, with at least 1% on every option.
   - Numeric questions: the **average** of the three probability distributions. This keeps their disagreement as extra uncertainty, because over-narrow ranges are the most common numeric mistake.
6. **Posts** the forecast and a private comment showing each model's forecast and reasoning. Metaculus makes these comments public after the question closes.

## Safeguards

- **No junk forecasts.** If fewer than two models give a usable answer, the bot submits nothing and tries again on the next run. Skipping scores zero; a careless forecast can score negative.
- **Answer repair.** If a model's final answer is badly formatted, a small model extracts it without changing the numbers.
- **Spending limits.** Each question has a cost ceiling. When the credit balance gets low, the bot switches to cheaper models, and it stops entirely before the credits run out.
- **Rule compliance.** Re-running the bot on a question is blocked in code, except in the unscored testing area.
- **Logs.** Every run saves the research, each model's answer and the costs, so the bot can be reviewed after questions close.

## Costs

Roughly $0.30–0.40 per main-tournament question and about $0.05 per MiniBench question. These are paid from the free credits Metaculus provides through OpenRouter.

## What was built with AI help

The bot was designed and written with Claude (Anthropic) as a coding assistant. It runs fully autonomously.
