# RSS → AI Digest → Telegram

Free, always-online blog tracker. 5×/day it checks ~28 feeds, has Gemini triage everything against **your** interests (LLM training, distributed systems, MLOps, strong eng deep-dives), publishes a clean mobile digest page, and sends you **one** Telegram message with a tl;dr + link.

Telegram ping looks like:

> 📬 8 new posts · 2 must-reads
> ⚡ Master PyTorch attention profiling and Zerodha's high-throughput PDF pipeline...
> 🔴 Profiling in PyTorch (Part 3)
> 🔴 1.5+ million PDFs in 25 minutes
> Open digest →

The digest page groups posts into **Must read / Worth a skim / Low priority**, each with a "why it matters for you" line and 2–4 key takeaways. Works great in iPhone Safari. No new posts = no message.

## Setup checklist (one time)

- [ ] Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the **bot token**
- [ ] Send your new bot any message (bots can't DM you first)
- [ ] Message [@userinfobot](https://t.me/userinfobot) → copy your **chat id**
- [ ] Create a **public** GitHub repo (public = free Pages + unlimited Actions minutes; your secrets stay secret) and push these files
- [ ] Repo → Settings → Secrets and variables → Actions → add `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY`
- [ ] Repo → Settings → **Pages** → Source: "Deploy from a branch" → Branch: `main`, folder: `/docs` → Save
- [ ] Actions tab → enable workflows → "RSS Tracker" → **Run workflow** (first run just initializes; you get a ✅ message)
- [ ] Run it once more (or wait for the schedule) to see a real digest

## Notes

- **Model**: uses `gemini-flash-latest` — your key has quota there (older `gemini-2.0-flash` returns quota=0 for new keys). Retries on 429/5xx.
- **Schedule**: 5 runs/day at ~07:17, 11:17, 15:17, 19:17, 23:17 IST. Edit the cron in `.github/workflows/rss.yml` (cron is in UTC; IST = UTC+5:30).
- **Personalization**: edit `READER_PROFILE` in `tracker.py` as your interests evolve — this is the whole brain of the ranking.
- **Feeds**: edit `feeds.yml`. Sources without public RSS (Shopify Eng, Zomato, Perplexity, Prime Intellect, Anthropic News, Meta AI, Uber Eng, Epoch, Stanford HAI) are listed as comments — generate feeds for them at rss.app (free) and paste in.
- **Anti-spam caps**: max 5 new posts/feed/run, 25/run total.
- The bot's own commits keep the scheduled workflow from being auto-disabled after 60 days.
