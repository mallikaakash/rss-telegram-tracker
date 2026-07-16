"""
Exa search -> Firecrawl scrape -> Gemini niche-judge -> HTML page -> Telegram.

Deterministic daily job that hunts for niche, heavy-on-knowledge material
matching discovery_interests.yml - the opposite of mainstream recommendations.
Runs independently of tracker.py (separate store, separate pages, separate
Telegram ping) since it's a different kind of find.

Each run:
  1. Runs every query in discovery_interests.yml through Exa's neural search.
  2. Drops any result already in discovery_articles.json (permanent dedup
     store, same design as tracker.py's articles.json - exact membership,
     no trimming).
  3. Scrapes surviving candidates with Firecrawl for clean full-text
     (Exa's own snippet is kept as a fallback if the scrape fails).
  4. One Gemini call judges each candidate against discovery_interests.yml's
     guidance: hidden_gem / solid_find / skip.
  5. Renders docs/discovery/<ts>.html (only hidden_gem + solid_find), updates
     docs/discovery/runs.html and docs/discovery/index.html (the Discovery
     library), and sends one Telegram ping - unless nothing cleared the bar,
     in which case: silence.

Env (GitHub Actions secrets):
  EXA_API_KEY, FIRECRAWL_API_KEY, GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import datetime as dt
import hashlib
import html
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

import common

ROOT = Path(__file__).parent
INTERESTS_FILE = ROOT / "discovery_interests.yml"
STORE_FILE = ROOT / "discovery_articles.json"
DOCS = ROOT / "docs"
DISCOVERY_DOCS = DOCS / "discovery"
DIGESTS = DISCOVERY_DOCS / "digests"

EXA_KEY = os.environ.get("EXA_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_MODEL = "gemini-flash-latest"

EXA_NUM_RESULTS = 6          # per query
DISCOVERY_LOOKBACK_DAYS = 60 # freshness bias, not a hard requirement for quality
MAX_NEW_CANDIDATES = 15      # caps Firecrawl scrapes + judge cost per run
ARTICLE_CHAR_BUDGET = 3000
LIBRARY_LIMIT = 300

TIER_META = {
    "hidden_gem": ("Hidden gem", "gem"),
    "solid_find": ("Solid find", "skim"),
}
TIER_RANK = {"hidden_gem": 0, "solid_find": 1}


# ---------------------------------------------------------------- state

def load_store() -> dict:
    return common.load_json(STORE_FILE, {})


def save_store(store: dict) -> None:
    common.save_json(STORE_FILE, store)


def candidate_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except ValueError:
        return url


# ---------------------------------------------------------------- exa + firecrawl

def exa_search(query: str) -> list[dict]:
    start = (dt.datetime.utcnow() - dt.timedelta(days=DISCOVERY_LOOKBACK_DAYS))
    try:
        r = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
            json={
                "query": query,
                "type": "auto",
                "numResults": EXA_NUM_RESULTS,
                "startPublishedDate": start.strftime("%Y-%m-%dT00:00:00.000Z"),
                "contents": {"text": {"maxCharacters": 2000}},
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.RequestException as e:
        print(f"  Exa search failed for '{query[:50]}...': {e}")
        return []


def firecrawl_scrape(url: str) -> dict | None:
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": 30000},
            timeout=45,
        )
        if not r.ok:
            return None
        data = r.json()
        return data.get("data") if data.get("success") else None
    except requests.RequestException as e:
        print(f"  Firecrawl scrape failed for {url}: {e}")
        return None


# ---------------------------------------------------------------- judge

def judge_llm(candidates: list[dict], guidance: str) -> dict | None:
    if not GEMINI_KEY:
        return None
    blocks = []
    for i, c in enumerate(candidates):
        blocks.append(f"[{i}] URL: {c['url']}\nTITLE: {c['title']}\n"
                      f"TEXT: {c['text'][:ARTICLE_CHAR_BUDGET]}\n")
    prompt = f"""You are judging web pages found via search, to decide if they are worth
surfacing to a specific reader hunting for niche, high-signal material.

{guidance}

Below are {len(candidates)} candidate pages. For EACH, decide a tier:
- "hidden_gem": rare, deep, high-signal - exactly what the reader is hunting for.
- "solid_find": genuinely good and on-topic, but not rare/surprising enough to be a gem.
- "skip": mainstream, shallow, off-topic, marketing, or a duplicate of common knowledge.

For each candidate give:
- "why": one sentence on why it earned this tier (be specific about what's novel or rare,
  or why it's mainstream/shallow).
- "points": for hidden_gem/solid_find, 2-4 concrete takeaways - what's novel, the key
  numbers, the conclusion. For skip, omit or leave empty.

Respond with ONLY valid JSON, no markdown fences:
{{"items": [{{"idx": 0, "tier": "hidden_gem", "why": "...", "points": ["...", "..."]}}, ...]}}
Include every idx from 0 to {len(candidates) - 1}.

CANDIDATES:
{chr(10).join(blocks)}"""
    data = common.gemini_generate(GEMINI_KEY, GEMINI_MODEL, prompt)
    if data is None:
        return None
    return {it["idx"]: it for it in data.get("items", [])}


# ---------------------------------------------------------------- html

def card_html(c: dict) -> str:
    label, cls = TIER_META[c["tier"]]
    pts = "".join(f"<li>{html.escape(p)}</li>" for p in c.get("points") or [])
    why = f'<div class="why">{html.escape(c["why"])}</div>' if c.get("why") else ""
    stamp = f'<span class="stamp">{c["date"]}</span>' if c.get("date") else ""
    return (f'<div class="card"><span class="badge {cls}">{label}</span>'
            f'<span class="src">{html.escape(c.get("domain") or "")}</span>{stamp}'
            f'<h2><a href="{html.escape(c["url"])}">{html.escape(c["title"])}</a></h2>'
            f'{why}<ul>{pts}</ul></div>')


def render_digest(kept: list[dict], ts: dt.datetime) -> str:
    tiers = {"hidden_gem": [], "solid_find": []}
    for c in kept:
        tiers[c["tier"]].append(c)

    body = ""
    for tier in ("hidden_gem", "solid_find"):
        if tiers[tier]:
            body += f'<div class="section">{TIER_META[tier][0]} &middot; {len(tiers[tier])}</div>'
            body += "".join(card_html(c) for c in tiers[tier])

    when = ts.strftime("%a %d %b %Y, %H:%M IST")
    heading = (f'<h1>Discovery</h1><div class="meta">{when} &middot; {len(kept)} finds &middot; '
               f'<a href="../index.html">main library</a> &middot; <a href="index.html">discovery library</a>'
               f' &middot; <a href="runs.html">all discovery runs</a></div>')
    return common.page_shell(f"Discovery &middot; {when}", heading + body)


def build_runs_page() -> None:
    files = sorted(DIGESTS.glob("*.html"), reverse=True)[:200]
    items = ""
    for f in files:
        stamp = f.stem
        try:
            t = dt.datetime.strptime(stamp, "%Y-%m-%d-%H%M")
            label = t.strftime("%a %d %b %Y &middot; %H:%M IST")
        except ValueError:
            label = stamp
        items += (f'<div class="idx"><a href="digests/{f.name}">{label}'
                  f'<div class="sub">open digest</div></a></div>')
    heading = '<h1>All Discovery Runs</h1><div class="meta">newest first &middot; <a href="index.html">back to discovery library</a></div>'
    DISCOVERY_DOCS.joinpath("runs.html").write_text(common.page_shell("All Discovery Runs", heading + items))


def build_library() -> None:
    store = load_store()
    items = [dict(id=cid, **rec) for cid, rec in store.items() if rec.get("tier") in TIER_META]
    items.sort(key=lambda a: a.get("date") or "", reverse=True)
    items.sort(key=lambda a: TIER_RANK.get(a["tier"], 1))
    items = items[:LIBRARY_LIMIT]

    chips = '<span class="chip active" data-tier="all">All</span>'
    for tier, (label, _cls) in TIER_META.items():
        chips += f'<span class="chip" data-tier="{tier}">{label}</span>'
    toolbar = (f'<div class="toolbar"><input type="text" id="q" placeholder="search title or domain...">'
               f'</div><div class="toolbar">{chips}</div>')

    cards = ""
    for a in items:
        label, cls = TIER_META[a["tier"]]
        q = html.escape((a.get("title", "") + " " + (a.get("domain") or "")).lower())
        pts = "".join(f"<li>{html.escape(p)}</li>" for p in a.get("points") or [])
        why = f'<div class="why">{html.escape(a["why"])}</div>' if a.get("why") else ""
        stamp = f'<span class="stamp">{a["date"]}</span>' if a.get("date") else ""
        cards += (f'<div class="card" data-tier="{a["tier"]}" data-q="{q}">'
                  f'<span class="badge {cls}">{label}</span>'
                  f'<span class="src">{html.escape(a.get("domain") or "")}</span>{stamp}'
                  f'<h2><a href="{html.escape(a.get("url") or "")}">{html.escape(a.get("title") or "")}</a></h2>'
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

    heading = (f'<h1>Discovery Library</h1><div class="meta">{len(items)} indexed &middot; ranked by tier, '
               f'then recency &middot; <a href="../index.html">main library</a> &middot; '
               f'<a href="runs.html">all discovery runs</a></div>')
    body = heading + toolbar + cards + empty
    DISCOVERY_DOCS.joinpath("index.html").write_text(common.page_shell("Discovery Library", body, script))


# ---------------------------------------------------------------- main

def main() -> None:
    missing = [k for k, v in {
        "EXA_API_KEY": EXA_KEY, "FIRECRAWL_API_KEY": FIRECRAWL_KEY,
        "GEMINI_API_KEY": GEMINI_KEY, "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
        "TELEGRAM_CHAT_ID": CHAT_ID,
    }.items() if not v]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}")

    interests = yaml.safe_load(INTERESTS_FILE.read_text())
    store = load_store()

    candidates: dict[str, dict] = {}
    for q in interests["queries"]:
        print(f"Searching: {q[:70]}...")
        for res in exa_search(q):
            url = res.get("url")
            if not url:
                continue
            cid = candidate_id(url)
            if cid in store or cid in candidates:
                continue
            candidates[cid] = {
                "id": cid, "url": url, "title": res.get("title") or url,
                "query": q, "domain": domain_of(url),
                "exa_text": (res.get("text") or "")[:1500],
            }
        time.sleep(0.3)

    ranked = list(candidates.values())[:MAX_NEW_CANDIDATES]
    if not ranked:
        print("No new candidates.")
        return

    print(f"{len(ranked)} new candidates; scraping...")
    for c in ranked:
        data = firecrawl_scrape(c["url"]) or {}
        c["text"] = (data.get("markdown") or c["exa_text"])[:ARTICLE_CHAR_BUDGET]
        meta_title = (data.get("metadata") or {}).get("title")
        if meta_title:
            c["title"] = meta_title
        time.sleep(0.5)

    judged = judge_llm(ranked, interests.get("guidance", "")) or {}

    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    today = now.strftime("%Y-%m-%d")
    kept = []
    for i, c in enumerate(ranked):
        info = judged.get(i, {})
        tier = info.get("tier", "skip")
        c["tier"] = tier if tier in TIER_META else "skip"
        c["why"] = info.get("why", "")
        c["points"] = info.get("points") or []
        store[c["id"]] = {
            "url": c["url"], "title": c["title"], "domain": c["domain"],
            "query": c["query"], "date": today, "tier": c["tier"],
            "why": c["why"], "points": c["points"],
        }
        if c["tier"] != "skip":
            kept.append(c)

    save_store(store)

    if not kept:
        print("No hidden gems or solid finds today.")
        return

    fname = now.strftime("%Y-%m-%d-%H%M") + ".html"
    DIGESTS.mkdir(parents=True, exist_ok=True)
    DIGESTS.joinpath(fname).write_text(render_digest(kept, now))
    build_runs_page()
    build_library()
    url = common.pages_url(f"discovery/digests/{fname}")

    gems = [c for c in kept if c["tier"] == "hidden_gem"]
    lines = [f"\U0001f50e <b>{len(kept)} niche finds</b>"
             + (f" · <b>{len(gems)} hidden gem{'s' if len(gems) != 1 else ''}</b>" if gems else "")]
    for c in (gems or kept)[:3]:
        lines.append(f"\U0001f48e {html.escape(c['title'])}")
    lines.append(f'\n<a href="{url}">Open discovery digest →</a>')
    common.send_telegram(BOT_TOKEN, CHAT_ID, "\n".join(lines))
    print(f"Discovery digest: {url}")


if __name__ == "__main__":
    main()
