# RSS/Discovery → AI Digest → Telegram

Free, always-online blog tracker with two independent pipelines, both running
on GitHub Actions only (no server, no database):

1. **RSS tracker** (`tracker.py`) - 5×/day it checks ~30 feeds, has Gemini
   triage everything against **your** interests, publishes a clean digest
   page, and sends **one** Telegram message with a tl;dr + link.
2. **Niche discovery** (`discovery.py`) - once/day it runs your interest
   queries through Exa's neural search, scrapes promising candidates with
   Firecrawl, and has Gemini judge them for how niche and knowledge-dense
   they are - actively penalizing mainstream/famous-creator content unless
   it's genuinely novel. Separate page, separate Telegram ping.

Both pipelines write into **The Library** (`docs/index.html` for RSS,
`docs/discovery/index.html` for discovery): every article/find ever surfaced,
ranked by tier then recency, with a search box and tier filter chips. No
rounded corners, earth-tone palette, monospace labels - a library catalog,
not a feed.

## Setup checklist (one time)

- [ ] Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the **bot token**
- [ ] Send your new bot any message (bots can't DM you first)
- [ ] Message [@userinfobot](https://t.me/userinfobot) → copy your **chat id**
- [ ] Create a **public** GitHub repo (public = free Pages + unlimited Actions minutes; your secrets stay secret) and push these files
- [ ] Repo → Settings → Secrets and variables → Actions → add:
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY` (both pipelines)
  - `EXA_API_KEY`, `FIRECRAWL_API_KEY` (discovery pipeline only - skip if you don't want it; the workflow just won't run usefully without them)
- [ ] Repo → Settings → **Pages** → Source: "Deploy from a branch" → Branch: `main`, folder: `/docs` → Save
- [ ] Actions tab → enable workflows → run **"RSS Tracker"** once (first run just initializes; you get a ✅ message) → run it once more to see a real digest
- [ ] Run **"Niche Discovery"** once too (first run has nothing to dedup against, so it'll judge a full batch immediately)

## Dedup: how it actually works now

Both pipelines keep a **permanent, global JSON store** keyed by a stable hash
of each item's id/link/URL (`articles.json` for RSS, `discovery_articles.json`
for discovery). Once an id is in there, it is never treated as new again -
full stop, no trimming, no expiry. This replaced an earlier per-feed design
that trimmed each feed's seen-list to the last 400 ids stored in a Python
`set` - since set iteration order is randomized per process, trimming dropped
effectively **random** ids instead of the oldest ones, which is why
high-volume feeds (Hugging Face's blog feed alone returns 800+ entries per
fetch) kept re-surfacing posts you'd already seen. That's fixed now: dedup
is exact membership in a store that never shrinks.

These stores also double as the data behind The Library pages (rank = tier,
then recency) - "seen" and "indexed" are the same store now, not two
different concerns.

## Personalizing

- **RSS triage**: edit `READER_PROFILE` in `tracker.py` - this is the whole
  brain of the must-read/worth-a-skim/low-priority ranking.
- **Discovery niche**: edit `discovery_interests.yml` - swap the `queries`
  and `guidance` whenever your focus shifts (e.g. from LLM systems to
  distributed systems or web3 in a few months). No code changes needed;
  the file is read fresh every run. Queries are phrased as descriptive
  sentences, not keywords, since Exa's neural search matches on meaning.
- **Feeds**: edit `feeds.yml`. Sources without public RSS are listed as
  comments - generate feeds for them at rss.app (free) and paste in.

## Notes

- **Model**: both pipelines use `gemini-flash-latest`. Retries on 429/5xx.
- **RSS schedule**: 5 runs/day at ~07:17, 11:17, 15:17, 19:17, 23:17 IST
  (cron in `.github/workflows/rss.yml`, UTC; IST = UTC+5:30).
- **Discovery schedule**: once/day at ~09:47 IST (`.github/workflows/discovery.yml`)
  - each run costs real Exa search + Firecrawl scrape credits, capped at
    15 new candidates/day (`MAX_NEW_CANDIDATES` in `discovery.py`).
- **Anti-spam caps**: RSS max 5 new posts/feed/run, 25/run total. Discovery
  max 15 new candidates/run, 6 Exa results/query.
- Both bots' own commits keep the scheduled workflows from being
  auto-disabled after 60 days of GitHub Actions inactivity.
