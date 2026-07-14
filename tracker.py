"""
RSS -> LLM triage -> HTML digest (GitHub Pages) -> single Telegram ping.

Each run:
  1. Fetches all feeds, finds new posts since last run.
  2. One Gemini call ranks everything (must-read / worth-a-skim / low-priority),
     writes key points per post, and a one-line tl;dr for the whole batch.
  3. Renders a mobile-friendly HTML digest into docs/digests/ (served by
     GitHub Pages) and updates docs/index.html (archive).
  4. Sends ONE Telegram message: tl;dr + must-read titles + link to the page.
  5. If nothing new: sends nothing. Silence means no news.

Env (GitHub Actions secrets):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY
Provided automatically by Actions: GITHUB_REPOSITORY (used to build the Pages URL).
Optional override: PAGES_BASE_URL (e.g. custom domain).
"""

import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import feedparser
import requests
import yaml

ROOT = Path(__file__).parent
SEEN_FILE = ROOT / "seen.json"
FEEDS_FILE = ROOT / "feeds.yml"
DOCS = ROOT / "docs"
DIGESTS = DOCS / "digests"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/126 Safari/537.36"}

MAX_NEW_PER_FEED = 5
MAX_SEEN_PER_FEED = 400
ARTICLE_CHAR_BUDGET = 3500   # per-article text sent to the LLM
MAX_ARTICLES_PER_RUN = 25    # safety cap on total batch size

READER_PROFILE = """
The reader is an early-career ML/backend engineer. Priorities, in order:
1. LLM training end-to-end: pretraining, MoE architectures, RLHF/GRPO/reasoning
   RL, tokenizers, positional encodings, scaling laws, small language models.
2. Distributed systems & ML infrastructure: distributed training (DDP/FSDP,
   parallelism strategies), inference serving (vLLM/PagedAttention), GPU
   platforms (Modal etc.), storage/consensus systems, performance engineering.
3. MLOps and applied ML for industrial/process optimization (works on an ML
   project reducing coke consumption in blast-furnace ironmaking: sensor data,
   gradient boosting, SHAP, constrained optimization).
4. High-quality engineering deep-dives from strong teams (Netflix, Jane Street,
   Zerodha, Cloudflare, Stripe, Meta, Indian startups) - architecture,
   reliability, performance war stories.
5. Career-relevant: what senior backend/infra and founding engineers should
   know; material worth writing technical Twitter threads about.

DOWNRANK: pure PR, funding/partnership announcements, product marketing,
consumer-feature launches, event recaps - unless strategically important
(e.g. a major model release or infra pricing change).
"""


# ---------------------------------------------------------------- state

def load_seen() -> dict:
    return json.loads(SEEN_FILE.read_text()) if SEEN_FILE.exists() else {}


def save_seen(seen: dict) -> None:
    for k in seen:
        seen[k] = seen[k][-MAX_SEEN_PER_FEED:]
    SEEN_FILE.write_text(json.dumps(seen, indent=1))


def entry_id(entry) -> str:
    raw = entry.get("id") or entry.get("link") or entry.get("title", "")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- fetching

def fetch_feed(url: str):
    """Fetch with a real browser UA (some feeds 403 the default one)."""
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except requests.RequestException as e:
        print(f"  fetch failed for {url}: {e}")
        return None


def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def get_entry_text(entry) -> str:
    for key in ("content", "summary_detail"):
        val = entry.get(key)
        if val:
            raw = val[0]["value"] if isinstance(val, list) else val.get("value", "")
            text = strip_html(raw)
            if len(text) > 300:
                return text[:ARTICLE_CHAR_BUDGET]
    link = entry.get("link")
    if link:
        try:
            r = requests.get(link, timeout=15, headers=UA)
            if r.ok:
                text = strip_html(r.text)
                if len(text) > 300:
                    return text[:ARTICLE_CHAR_BUDGET]
        except requests.RequestException:
            pass
    return strip_html(entry.get("summary", ""))[:ARTICLE_CHAR_BUDGET]


# ---------------------------------------------------------------- triage

def triage_llm(articles: list[dict]) -> dict | None:
    """One batched call. Returns {'tldr': str, 'items': {idx: {...}}} or None."""
    if not GEMINI_KEY:
        return None
    blocks = []
    for i, a in enumerate(articles):
        blocks.append(f"[{i}] SOURCE: {a['feed']}\nTITLE: {a['title']}\n"
                      f"TEXT: {a['text'][:ARTICLE_CHAR_BUDGET]}\n")
    prompt = f"""You are triaging new blog posts for one specific reader.

{READER_PROFILE}

Below are {len(articles)} new posts. For EACH, decide a tier:
- "must_read": directly advances the reader's priorities; novel technique,
  strong deep-dive, or strategically important news.
- "worth_a_skim": somewhat relevant or good general engineering content.
- "low_priority": PR/marketing/announcements or off-target topics.

For each post give:
- "why": one sentence on why it matters (or doesn't) FOR THIS READER.
- "points": the 2-4 most important takeaways - what's novel, the key numbers,
  the conclusion. For low_priority, 1 point is enough. Be concrete, not vague.

Also write "tldr": ONE sentence (max 25 words) summarizing what's most
important in this whole batch, leading with the top must-read.

Respond with ONLY valid JSON, no markdown fences:
{{"tldr": "...", "items": [{{"idx": 0, "tier": "must_read", "why": "...", "points": ["...", "..."]}}, ...]}}
Include every idx from 0 to {len(articles) - 1}.

POSTS:
{chr(10).join(blocks)}"""
    r = None
    for attempt in range(4):
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.2,
                                           "responseMimeType": "application/json"}},
                timeout=120,
            )
            if r.status_code in (429, 500, 502, 503):
                print(f"  Gemini {r.status_code}, retry {attempt + 1}/3...")
                time.sleep(10 * (attempt + 1))
                continue
            break
        except requests.RequestException as e:
            print(f"  Gemini request error ({e}), retry {attempt + 1}/3...")
            time.sleep(10 * (attempt + 1))
    try:
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        raw = "".join(p.get("text", "") for p in parts)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
        # take the first complete JSON object even if extra data follows
        data, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
        items = {it["idx"]: it for it in data.get("items", [])}
        return {"tldr": data.get("tldr", ""), "items": items}
    except Exception as e:  # noqa: BLE001
        print(f"  triage failed: {e}")
        return None


def fallback_points(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?]) +", text)
    out, total = [], 0
    for s in sentences:
        if total > 350:
            break
        out.append(s)
        total += len(s)
    return [" ".join(out)] if out else []


# ---------------------------------------------------------------- digest html

TIER_META = {
    "must_read": ("Must read", "#e5484d"),
    "worth_a_skim": ("Worth a skim", "#f5a524"),
    "low_priority": ("Low priority", "#8b8d98"),
}

PAGE_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 background:#0f1115;color:#e6e8ee;max-width:680px;margin:0 auto;
 padding:24px 16px 60px;line-height:1.55}
a{color:#7aa2ff;text-decoration:none}
h1{font-size:1.25rem;margin-bottom:4px}
.meta{color:#8b8d98;font-size:.85rem;margin-bottom:20px}
.tldr{background:#1a1d24;border-left:3px solid #7aa2ff;padding:12px 14px;
 border-radius:0 8px 8px 0;margin-bottom:24px;font-size:.95rem}
.card{background:#161920;border:1px solid #23262f;border-radius:12px;
 padding:16px;margin-bottom:14px}
.badge{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.04em;
 text-transform:uppercase;padding:3px 8px;border-radius:20px;color:#fff;
 margin-bottom:8px}
.src{color:#8b8d98;font-size:.8rem;margin-left:8px}
.card h2{font-size:1.02rem;line-height:1.35;margin-bottom:6px}
.why{color:#b8bcc8;font-size:.88rem;font-style:italic;margin-bottom:8px}
ul{padding-left:20px}
li{font-size:.9rem;margin-bottom:5px;color:#d4d7e0}
.section{margin:26px 0 10px;font-size:.8rem;font-weight:700;color:#8b8d98;
 text-transform:uppercase;letter-spacing:.06em}
.idx a{display:block;padding:12px 14px;background:#161920;border:1px solid #23262f;
 border-radius:10px;margin-bottom:10px}
.idx .sub{color:#8b8d98;font-size:.82rem;margin-top:2px}
"""


def render_digest(articles: list[dict], triage: dict | None, ts: dt.datetime) -> str:
    tiers = {"must_read": [], "worth_a_skim": [], "low_priority": []}
    for i, a in enumerate(articles):
        info = (triage or {}).get("items", {}).get(i, {})
        tier = info.get("tier", "worth_a_skim")
        tiers.setdefault(tier, tiers["worth_a_skim"])
        a["why"] = info.get("why", "")
        a["points"] = info.get("points") or fallback_points(a["text"])
        tiers[tier if tier in tiers else "worth_a_skim"].append(a)

    def card(a, tier):
        label, color = TIER_META[tier]
        pts = "".join(f"<li>{html.escape(p)}</li>" for p in a["points"])
        why = f'<div class="why">{html.escape(a["why"])}</div>' if a["why"] else ""
        return (f'<div class="card"><span class="badge" style="background:{color}">'
                f'{label}</span><span class="src">{html.escape(a["feed"])}</span>'
                f'<h2><a href="{html.escape(a["link"])}">{html.escape(a["title"])}</a></h2>'
                f'{why}<ul>{pts}</ul></div>')

    body = ""
    tldr = (triage or {}).get("tldr", "")
    if tldr:
        body += f'<div class="tldr">⚡ {html.escape(tldr)}</div>'
    for tier in ("must_read", "worth_a_skim", "low_priority"):
        if tiers[tier]:
            body += f'<div class="section">{TIER_META[tier][0]} · {len(tiers[tier])}</div>'
            body += "".join(card(a, tier) for a in tiers[tier])

    n = len(articles)
    when = ts.strftime("%a %d %b %Y, %H:%M IST")
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Digest · {when}</title><style>{PAGE_CSS}</style></head><body>'
            f'<h1>📰 Reading Digest</h1><div class="meta">{when} · {n} new posts · '
            f'<a href="../index.html">all digests</a></div>{body}</body></html>')


def rebuild_index() -> None:
    files = sorted(DIGESTS.glob("*.html"), reverse=True)[:100]
    items = ""
    for f in files:
        stamp = f.stem  # 2026-07-15-1147
        try:
            t = dt.datetime.strptime(stamp, "%Y-%m-%d-%H%M")
            label = t.strftime("%a %d %b %Y · %H:%M IST")
        except ValueError:
            label = stamp
        items += (f'<div class="idx"><a href="digests/{f.name}">{label}'
                  f'<div class="sub">tap to open digest</div></a></div>')
    DOCS.joinpath("index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Reading Digests</title><style>{PAGE_CSS}</style></head><body>'
        f'<h1>📰 All Digests</h1><div class="meta">newest first</div>{items}'
        f'</body></html>')


def pages_url(filename: str) -> str:
    base = os.environ.get("PAGES_BASE_URL", "").rstrip("/")
    if not base:
        repo = os.environ.get("GITHUB_REPOSITORY", "user/repo")
        owner, name = repo.split("/", 1)
        base = f"https://{owner}.github.io/{name}"
    return f"{base}/digests/{filename}"


# ---------------------------------------------------------------- telegram

def send_telegram(msg: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=20)
    if not r.ok:
        print(f"  Telegram error: {r.text}")


# ---------------------------------------------------------------- main

def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    feeds = yaml.safe_load(FEEDS_FILE.read_text())["feeds"]
    seen = load_seen()
    articles, first_run = [], set()

    for feed in feeds:
        name, url = feed["name"], feed["url"]
        print(f"Checking: {name}")
        parsed = fetch_feed(url)
        if parsed is None or (parsed.bozo and not parsed.entries):
            continue
        if url not in seen:  # first sighting: mark all seen, don't spam
            first_run.add(name)
            seen[url] = [entry_id(e) for e in parsed.entries]
            continue
        known = set(seen[url])
        for e in [e for e in parsed.entries if entry_id(e) not in known][:MAX_NEW_PER_FEED]:
            articles.append({"feed": name, "title": e.get("title", "Untitled"),
                             "link": e.get("link", ""), "entry": e})
        seen[url] = list(known | {entry_id(e) for e in parsed.entries})

    save_seen(seen)

    if first_run:
        send_telegram("✅ Now tracking: " + ", ".join(sorted(first_run)))
    if not articles:
        print("No new posts.")
        return

    articles = articles[:MAX_ARTICLES_PER_RUN]
    print(f"{len(articles)} new posts; fetching text...")
    for a in articles:
        a["text"] = get_entry_text(a.pop("entry"))
        time.sleep(0.3)

    triage = triage_llm(articles)

    # IST timestamps for filenames/labels
    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    fname = now.strftime("%Y-%m-%d-%H%M") + ".html"
    DIGESTS.mkdir(parents=True, exist_ok=True)
    DIGESTS.joinpath(fname).write_text(render_digest(articles, triage, now))
    rebuild_index()
    url = pages_url(fname)

    must = [i for i, a in enumerate(articles)
            if (triage or {}).get("items", {}).get(i, {}).get("tier") == "must_read"]
    tldr = (triage or {}).get("tldr", "")

    lines = [f"📬 <b>{len(articles)} new posts</b>"
             + (f" · <b>{len(must)} must-read{'s' if len(must) != 1 else ''}</b>" if must else "")]
    if tldr:
        lines.append(f"⚡ {html.escape(tldr)}")
    for i in must[:3]:
        lines.append(f"🔴 {html.escape(articles[i]['title'])}")
    lines.append(f'\n<a href="{url}">Open digest →</a>')
    send_telegram("\n".join(lines))
    print(f"Digest: {url}")


if __name__ == "__main__":
    main()
