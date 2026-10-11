# Setup guide

These are the steps that need **you**. Claude can't create accounts, sign in, or handle tokens in web forms. Total time is about 60–90 minutes. Do them in order: step 0 can make the rest pointless.

---

## Step 0 — Check you can be paid (5 min)

Prizes are paid by bank transfer through **Ramp**. Metaculus says it can't pay anyone whose country or currency Ramp doesn't support.

1. Open the [FutureEval resources page](https://www.metaculus.com/notebooks/38928/futureeval-resources-page/) and find the section **"How are prizes distributed?"**.
2. Follow its link to Ramp's *Supported Countries* list. Check that your country appears, for USD or your local currency.

If it isn't listed, stop here and tell Claude. We'd pick a different method.

## Step 1 — Metaculus account and bot (15 min)

1. Sign up at [metaculus.com](https://www.metaculus.com) as yourself (a human account) with an email address you read. Prize emails go there.
2. Go to **Settings → My Forecasting Bots → Create a Bot**. Pick a bot name; it will be public.
3. **Copy the bot's API token** and keep it somewhere private. This is `METACULUS_TOKEN`.
4. Fill in the **required participation form** (3 short questions), which is also where you ask for free LLM credits: <https://forms.gle/aQdYMq9Pisrf1v7d8>
   - Say it's a hobbyist, **open-source** bot (open-source bots get about double credits). Paste the GitHub repo link from step 2 if you have it yet.
   - Plan: a research-first ensemble of Claude Sonnet, GPT and Gemini at about $0.30–0.40 per main-tournament question.
5. Metaculus emails you an **OpenRouter key** with your credits. This is `OPENROUTER_API_KEY`. It can take a few days. Everything else can be set up meanwhile.

## Step 2 — GitHub: get the code online (15 min)

The bot runs on GitHub's free servers every 15 minutes, so your laptop can be off.

1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Install Git and the GitHub command-line tool. In a terminal:

   ```bash
   winget install --id Git.Git -e
   ```

   ```bash
   winget install --id GitHub.cli -e
   ```

3. Close and reopen the terminal, then sign in. This opens your browser and **you** complete the login:

   ```bash
   gh auth login
   ```

4. Tell Claude when that's done. Claude can then create a **public** repository and push the code. Public matters for two reasons: unlimited free Actions minutes, and the open-source credit bonus.

## Step 3 — Add the secrets (5 min)

In your repository on github.com go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Name | Value |
|---|---|
| `METACULUS_TOKEN` | the bot token from step 1 |
| `OPENROUTER_API_KEY` | the key Metaculus emailed you |

Secrets are encrypted and never shown in logs or to people viewing the public repo. **Never** paste a key into a code file.

Then switch the bot on. Open the **Variables** tab on the same page → **New repository variable**, name `BOT_ENABLED`, value `true`. Until this variable exists, every run is skipped quietly. Setting it to anything else later is your kill switch.

## Step 4 — Smoke test on the unscored testing area (10 min)

1. On GitHub: **Actions → Forecast → Run workflow**, then set *mode* = `test` and tick *dry_run*. When it finishes, open the run log: you should see forecasts for the test questions, and nothing posted.
2. Run it again with *mode* = `test`, *dry_run* **unticked** and *force* ticked. This time it posts to the testing area. These questions are unscored and re-forecasting there is allowed.
3. Check it worked: on Metaculus go to **Settings → My Bots → Switch to bot account**, open the [bot testing area](https://www.metaculus.com/tournament/bot-testing-area/), and you should see the bot's forecasts plus a private comment with its reasoning.

## Step 5 — Reliable scheduling with cron-job.org (15 min)

Each tournament question is open for only about 90 minutes, and GitHub's built-in schedule sometimes skips runs. A free external timer guarantees a run every 15 minutes.

1. **Create a narrowly scoped GitHub token.** Go to GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate new token.
   - Repository access: **only this repository**.
   - Permissions: **Actions: Read and write**. Nothing else.
   - Expiry: 1 year (add a calendar reminder to renew it).

   This token can only start workflows in this one repository.
2. Create a free account at [cron-job.org](https://cron-job.org) and add a cron job:
   - URL: `https://api.github.com/repos/<your-user>/<your-repo>/actions/workflows/forecast.yaml/dispatches`
   - Schedule: every 15 minutes
   - Method: `POST`
   - Request body: `{"ref":"main"}`
   - Headers:
     - `Authorization: Bearer <the token from 1>`
     - `Accept: application/vnd.github+json`
     - `X-GitHub-Api-Version: 2022-11-28`
3. Save it, then use cron-job.org's "test run". It should return HTTP **204**, and a new run should appear under GitHub Actions.

## Step 6 — Go live

Nothing else to do. Every run checks the Fall 2026 tournament and the current MiniBench, forecasts any new questions, and posts each forecast with its reasoning. Questions already forecast are skipped.

---

## Rules you must never break (disqualification risk)

- **Never** re-run the bot on a tournament or MiniBench question, and never change its forecasts by hand. The code refuses `--force` outside test mode.
- **Never** change the bot because of how it is doing on questions that are *still open*. Improvements are tested only on closed questions, the testing area, or the Metaculus Cup.
- Only one bot per person can win prizes. A second experimental bot must be labelled as secondary in your Metaculus settings.

## Seasonal checklist (January, May, September; about 1 hour)

1. **Bot survey:** fill it in when Metaculus emails it. It is required for prizes, including for the season just ended.
2. **LLM credits:** re-apply for the new season using the same form as step 1.
3. **Tournament ID:** ask Claude to update `FALL_2026_TOURNAMENT` in `bot/config.py` to the new season's ID. It's announced on the resources page.
4. **Review:** ask Claude to review the MiniBench results and logs (Actions → a run → *Artifacts*) and propose improvements to test on closed questions.
5. **Renew** the cron-job.org GitHub token if it is close to expiring.
