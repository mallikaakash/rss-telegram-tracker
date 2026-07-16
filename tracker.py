"""
RSS -> LLM triage -> HTML digest (GitHub Pages) -> single Telegram ping.

Each run:
  1. Fetches all feeds, finds posts whose id isn't already in articles.json.
  2. One Gemini call ranks everything (must-read / worth-a-skim / low-priority),
     writes key points per post, and a one-line tl;dr for the whole batch.
  3. Renders a mobile-friendly HTML digest into docs/digests/ (served by
     GitHub Pages), updates docs/runs.html (chronological run list) and
     docs/index.html (the Library: every triaged article ever, ranked by
     tier then recency, searchable).
  4. Sends ONE Telegram message: tl;dr + must-read titles + link to the page.
  5. If nothing new: sends nothing. Silence means no news.

Dedup: articles.json is a permanent, global {article_id: record} store keyed
by a stable hash of the entry's id/link/title. Once an id is in there it is
never treated as new again - no per-feed trimming, so there's no risk of
losing track of an already-seen post (which is what caused repeats before).

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

import common

ROOT = Path(__file__).parent
FEEDS_FILE = ROOT / "feeds.yml"
ARTICLES_FILE = ROOT / "articles.json"
BOOTSTRAP_FILE = ROOT / "feeds_bootstrapped.json"
DOCS = ROOT / "docs"
DIGESTS = DOCS / "digests"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"

MAX_NEW_PER_FEED = 5
ARTICLE_CHAR_BUDGET = 3500   # per-article text sent to the LLM
MAX_ARTICLES_PER_RUN = 25    # safety cap on total batch size
LIBRARY_LIMIT = 500          # max cards rendered on the Library page

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

TIER_META = {
    "must_read": ("Must read", "must"),
    "worth_a_skim": ("Worth a skim", "skim"),
    "low_priority": ("Low priority", "low"),
}
TIER_RANK = {"must_read": 0, "worth_a_skim": 1, "low_priority": 2}


# ---------------------------------------------------------------- state

def load_articles() -> dict:
    return common.load_json(ARTICLES_FILE, {})


def save_articles(articles: dict) -> None:
    common.save_json(ARTICLES_FILE, articles)


def entry_id(entry) -> str:
    raw = entry.get("id") or entry.get("link") or entry.get("title", "")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- fetching

def fetch_feed(url: str):
    """Fetch with a real browser UA (some feeds 403 the default one)."""
    try:
        r = requests.get(url, headers=common.UA, timeout=25)
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
            r = requests.get(link, timeout=15, headers=common.UA)
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
    data = common.gemini_generate(GEMINI_KEY, GEMINI_MODEL, prompt)
    if data is None:
        return None
    items = {it["idx"]: it for it in data.get("items", [])}
    return {"tldr": data.get("tldr", ""), "items": items}


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

def card_html(a: dict) -> str:
    label, cls = TIER_META[a["tier"]]
    pts = "".join(f"<li>{html.escape(p)}</li>" for p in a.get("points") or [])
    why = f'<div class="why">{html.escape(a["why"])}</div>' if a.get("why") else ""
    stamp = f'<span class="stamp">{a["date"]}</span>' if a.get("date") else ""
    return (f'<div class="card"><span class="badge {cls}">{label}</span>'
            f'<span class="src">{html.escape(a["feed"])}</span>{stamp}'
            f'<h2><a href="{html.escape(a["link"])}">{html.escape(a["title"])}</a></h2>'
            f'{why}<ul>{pts}</ul></div>')


def render_digest(articles: list[dict], tldr: str, ts: dt.datetime) -> str:
    tiers = {"must_read": [], "worth_a_skim": [], "low_priority": []}
    for a in articles:
        tiers[a["tier"]].append(a)

    body = ""
    if tldr:
        body += f'<div class="tldr">&#9889; {html.escape(tldr)}</div>'
    for tier in ("must_read", "worth_a_skim", "low_priority"):
        if tiers[tier]:
            body += f'<div class="section">{TIER_META[tier][0]} &middot; {len(tiers[tier])}</div>'
            body += "".join(card_html(a) for a in tiers[tier])

    n = len(articles)
    when = ts.strftime("%a %d %b %Y, %H:%M IST")
    heading = (f'<h1>Reading Digest</h1><div class="meta">{when} &middot; {n} new posts &middot; '
               f'<a href="../index.html">library</a> &middot; <a href="../runs.html">all runs</a></div>')
    return common.page_shell(f"Digest &middot; {when}", heading + body)


def build_runs_page() -> None:
    files = sorted(DIGESTS.glob("*.html"), reverse=True)[:200]
    items = ""
    for f in files:
        stamp = f.stem  # 2026-07-15-1147
        try:
            t = dt.datetime.strptime(stamp, "%Y-%m-%d-%H%M")
            label = t.strftime("%a %d %b %Y &middot; %H:%M IST")
        except ValueError:
            label = stamp
        items += (f'<div class="idx"><a href="digests/{f.name}">{label}'
                  f'<div class="sub">open digest</div></a></div>')
    heading = ('<h1>All Runs</h1><div class="meta">newest first &middot; '
               '<a href="index.html">back to library</a></div>')
    DOCS.joinpath("runs.html").write_text(common.page_shell("All Runs", heading + items))


def build_library() -> None:
    store = load_articles()
    items = [dict(id=aid, **rec) for aid, rec in store.items()
             if rec.get("tier") in TIER_META]
    items.sort(key=lambda a: a.get("date") or "", reverse=True)
    items.sort(key=lambda a: TIER_RANK.get(a["tier"], 1))
    items = items[:LIBRARY_LIMIT]

    chips = '<span class="chip active" data-tier="all">All</span>'
    for tier, (label, _cls) in TIER_META.items():
        chips += f'<span class="chip" data-tier="{tier}">{label}</span>'
    toolbar = (f'<div class="toolbar"><input type="text" id="q" placeholder="search title or source...">'
               f'</div><div class="toolbar">{chips}</div>')

    cards = ""
    for a in items:
        label, cls = TIER_META[a["tier"]]
        q = html.escape((a.get("title", "") + " " + a.get("feed", "")).lower())
        pts = "".join(f"<li>{html.escape(p)}</li>" for p in a.get("points") or [])
        why = f'<div class="why">{html.escape(a["why"])}</div>' if a.get("why") else ""
        stamp = f'<span class="stamp">{a["date"]}</span>' if a.get("date") else ""
        cards += (f'<div class="card" data-tier="{a["tier"]}" data-q="{q}">'
                  f'<span class="badge {cls}">{label}</span>'
                  f'<span class="src">{html.escape(a.get("feed") or "")}</span>{stamp}'
                  f'<h2><a href="{html.escape(a.get("link") or "")}">{html.escape(a.get("title") or "")}</a></h2>'
                  f'{why}<ul>{pts}</ul></div>')

    empty = '<div class="empty" id="empty" style="display:none">No matches.</div>'
    script = """<script>
document.addEventListener('DOMContentLoaded', function(){
  var chips=[].slice.call(document.querySelectorAll('.chip'));
  var input=document.getElementById('q');
  var cards=[].slice.call(document.querySelectorAll('.card'));
  var activeTier='all';
  function apply(){
    var q=input.value.trim().toLowerCase();
    var shown=0;
    cards.forEach(function(c){
      var okTier = activeTier==='all' || c.dataset.tier===activeTier;
      var okQ = !q || c.dataset.q.indexOf(q) !== -1;
      var show = okTier && okQ;
      c.style.display = show ? '' : 'none';
      if(show) shown++;
    });
    document.getElementById('empty').style.display = shown ? 'none' : 'block';
  }
  chips.forEach(function(ch){
    ch.addEventListener('click', function(){
      chips.forEach(function(x){ x.classList.remove('active'); });
      ch.classList.add('active');
      activeTier = ch.dataset.tier;
      apply();
    });
  });
  input.addEventListener('input', apply);
});
</script>"""

    heading = (f'<h1>The Library</h1><div class="meta">{len(items)} indexed &middot; ranked by tier, '
               f'then recency &middot; <a href="runs.html">all runs</a> &middot; '
               f'<a href="discovery/index.html">discovery</a></div>')
    body = heading + toolbar + cards + empty
    DOCS.joinpath("index.html").write_text(common.page_shell("The Library", body, script))


# ---------------------------------------------------------------- main

def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    feeds = yaml.safe_load(FEEDS_FILE.read_text())["feeds"]
    articles_store = load_articles()
    bootstrapped = common.load_json(BOOTSTRAP_FILE, {})
    new_articles, first_run = [], set()

    for feed in feeds:
        name, url = feed["name"], feed["url"]
        print(f"Checking: {name}")
        parsed = fetch_feed(url)
        if parsed is None or (parsed.bozo and not parsed.entries):
            continue
        if url not in bootstrapped:  # first sighting: mark all seen, don't spam
            first_run.add(name)
            bootstrapped[url] = True
            for e in parsed.entries:
                aid = entry_id(e)
                articles_store.setdefault(aid, {
                    "tier": "bootstrap", "feed": name,
                    "title": e.get("title", ""), "link": e.get("link", ""),
                    "date": None, "why": None, "points": None,
                })
            continue
        candidates = [e for e in parsed.entries if entry_id(e) not in articles_store]
        for e in candidates[:MAX_NEW_PER_FEED]:
            new_articles.append({"feed": name, "title": e.get("title", "Untitled"),
                                 "link": e.get("link", ""), "id": entry_id(e), "entry": e})

    common.save_json(BOOTSTRAP_FILE, bootstrapped)

    if first_run:
        common.send_telegram(BOT_TOKEN, CHAT_ID, "✅ Now tracking: " + ", ".join(sorted(first_run)))
    if not new_articles:
        save_articles(articles_store)
        print("No new posts.")
        return

    new_articles = new_articles[:MAX_ARTICLES_PER_RUN]
    print(f"{len(new_articles)} new posts; fetching text...")
    for a in new_articles:
        a["text"] = get_entry_text(a.pop("entry"))
        time.sleep(0.3)

    triage = triage_llm(new_articles)

    # IST timestamps for filenames/labels
    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    today = now.strftime("%Y-%m-%d")
    for i, a in enumerate(new_articles):
        info = (triage or {}).get("items", {}).get(i, {})
        tier = info.get("tier", "worth_a_skim")
        a["tier"] = tier if tier in TIER_META else "worth_a_skim"
        a["why"] = info.get("why", "")
        a["points"] = info.get("points") or fallback_points(a["text"])
        articles_store[a["id"]] = {
            "feed": a["feed"], "title": a["title"], "link": a["link"],
            "date": today, "tier": a["tier"], "why": a["why"], "points": a["points"],
        }

    save_articles(articles_store)

    tldr = (triage or {}).get("tldr", "")
    fname = now.strftime("%Y-%m-%d-%H%M") + ".html"
    DIGESTS.mkdir(parents=True, exist_ok=True)
    DIGESTS.joinpath(fname).write_text(render_digest(new_articles, tldr, now))
    build_runs_page()
    build_library()
    url = common.pages_url(f"digests/{fname}")

    must = [a for a in new_articles if a["tier"] == "must_read"]

    lines = [f"📬 <b>{len(new_articles)} new posts</b>"
             + (f" · <b>{len(must)} must-read{'s' if len(must) != 1 else ''}</b>" if must else "")]
    if tldr:
        lines.append(f"⚡ {html.escape(tldr)}")
    for a in must[:3]:
        lines.append(f"🔴 {html.escape(a['title'])}")
    lines.append(f'\n<a href="{url}">Open digest →</a>')
    common.send_telegram(BOT_TOKEN, CHAT_ID, "\n".join(lines))
    print(f"Digest: {url}")


if __name__ == "__main__":
    main()
