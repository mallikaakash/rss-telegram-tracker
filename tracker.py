"""
RSS -> LLM triage -> HTML digest (GitHub Pages) -> single Telegram ping.

Each run:
  1. Fetches all feeds plus the non-RSS pages in feeds.yml, and keeps posts
     whose id isn't already in articles.json. Posts older than 21 days are
     remembered and dropped. Firecrawl/Exa pages run at most once per IST day.
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
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

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
EXA_KEY = os.environ.get("EXA_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"

MAX_NEW_PER_FEED = 5
ARTICLE_CHAR_BUDGET = 3500   # per-article text sent to the LLM
MAX_ARTICLES_PER_RUN = 25    # safety cap on total batch size
MAX_AGE_DAYS = 21            # hard drop; still remembered so it never returns
LIBRARY_LIMIT = 500          # max cards rendered on the Library page
PAID_STAMP_FILE = ROOT / "pages_paid_on.json"
SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".xml",
            ".pdf", ".mp4", ".webp", ".ico")
REJECT_REASONS = {"stale", "off_topic", "marketing", "shallow"}

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


def ist_now() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)


def entry_published(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if not parsed:
            continue
        try:
            return dt.datetime(*parsed[:6])
        except (TypeError, ValueError):
            continue
    return None


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def slug_title(url: str) -> str:
    slug = urlsplit(url).path.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").strip() or url


def canon_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def same_host(a: str, b: str) -> bool:
    return urlsplit(a).netloc.removeprefix("www.") == urlsplit(b).netloc.removeprefix("www.")


def fair_cap(articles: list[dict], limit: int) -> list[dict]:
    """One post from each source before any source gets a second."""
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for article in articles:
        key = article["feed"]
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(article)
    picked: list[dict] = []
    while len(picked) < limit:
        progressed = False
        for key in order:
            if not buckets[key]:
                continue
            picked.append(buckets[key].pop(0))
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def remember(store: dict, aid: str, **rec) -> None:
    store.setdefault(aid, rec)


def fetch_html(url: str) -> str:
    try:
        r = requests.get(url, headers=common.UA, timeout=25)
        r.raise_for_status()
        return r.content[:500_000].decode(r.encoding or "utf-8", "replace")
    except requests.RequestException as e:
        print(f"  page fetch failed for {url}: {e}")
        return ""


def fetch_html_article(url: str) -> tuple[str, str]:
    raw = fetch_html(url)
    if not raw:
        return "", ""
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    title = strip_html(match.group(1)) if match else ""
    title = title.split(" | ")[0].strip()
    return title[:180], strip_html(raw)[:ARTICLE_CHAR_BUDGET]


def page_links(page: dict) -> list[str]:
    """Article URLs from a server-rendered index. One path segment past the prefix, same host."""
    raw = fetch_html(page["url"])
    if not raw:
        return []
    prefix = page["link_prefix"]
    extra = int(page.get("extra_segments") or 1)
    deny = page.get("deny_contains") or []
    found, seen = [], set()
    for href in re.findall(r"""href=["']([^"'#]+)["']""", raw, flags=re.I):
        if href.startswith(("mailto:", "javascript:", "tel:")) or "\\" in href:
            continue
        absolute = urljoin(page["url"], href)
        if not same_host(absolute, page["url"]):
            continue
        parts = urlsplit(absolute)
        path = parts.path
        if any(piece in path for piece in deny):
            continue
        if not path.startswith(prefix):
            continue
        segments = [s for s in path[len(prefix):].strip("/").split("/") if s]
        if len(segments) != extra:
            continue
        if segments[-1].lower().endswith(SKIP_EXT):
            continue
        canon = canon_url(absolute)
        if canon.rstrip("/") == page["url"].rstrip("/"):
            continue
        if canon not in seen:
            seen.add(canon)
            found.append(canon)
    return found


def firecrawl_scrape(url: str, formats: list[str]) -> dict | None:
    if not FIRECRAWL_KEY:
        return None
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
            json={"url": url, "formats": formats, "onlyMainContent": True, "timeout": 30000},
            timeout=45,
        )
        if not r.ok:
            print(f"  firecrawl {r.status_code} for {url}")
            return None
        data = r.json()
        return data.get("data") if data.get("success") else None
    except requests.RequestException as e:
        print(f"  firecrawl failed for {url}: {e}")
        return None


def uber_posts(index_url: str) -> list[str]:
    """Engineering posts from the JS-rendered Uber index. One Firecrawl scrape."""
    data = firecrawl_scrape(index_url, ["links", "markdown"]) or {}
    raw_links: list[str] = []
    for item in data.get("links") or []:
        if isinstance(item, str):
            raw_links.append(item)
        elif isinstance(item, dict):
            raw_links.append(item.get("url") or item.get("href") or "")
    raw_links += re.findall(r"\((https?://[^)\s]+)\)", data.get("markdown") or "")
    best: dict[str, str] = {}
    for url in raw_links:
        if not url or not same_host(url, "https://www.uber.com/"):
            continue
        path = urlsplit(url).path
        key = re.sub(r"^/[a-z]{2}(?:-[a-z]{2})?/[a-z]{2}(?=/blog/)", "", path).rstrip("/")
        if not re.fullmatch(r"/blog/engineering/[^/]+", key):
            continue
        slug = key.rsplit("/", 1)[-1].lower()
        if len(slug) < 3 or slug in {"engineering", "rss", "feed"}:
            continue
        chosen = canon_url(url)
        previous = best.get(key)
        if previous is None or ("/us/en/" in chosen and "/us/en/" not in previous):
            best[key] = chosen
    return list(best.values())


def exa_posts(domain: str, path_contains: str) -> list[dict]:
    if not EXA_KEY:
        return []
    start = dt.datetime.utcnow() - dt.timedelta(days=MAX_AGE_DAYS)
    try:
        r = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
            json={
                "query": f"Recent research blog post from {domain}",
                "type": "auto",
                "numResults": 8,
                "includeDomains": [domain],
                "startPublishedDate": start.strftime("%Y-%m-%dT00:00:00.000Z"),
                "contents": {"text": {"maxCharacters": 2000}},
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except requests.RequestException as e:
        print(f"  exa failed for {domain}: {e}")
        return []
    posts = []
    for res in results:
        url = res.get("url") or ""
        if path_contains not in urlsplit(url).path:
            continue
        published = (res.get("publishedDate") or "")[:10] or None
        posts.append({
            "url": canon_url(url),
            "title": res.get("title") or slug_title(url),
            "text": (res.get("text") or "")[:ARTICLE_CHAR_BUDGET],
            "published_on": published,
        })
    return posts


def paid_due(today: str) -> bool:
    return common.load_json(PAID_STAMP_FILE, {}).get("date") != today


# ---------------------------------------------------------------- triage

def triage_llm(articles: list[dict]) -> dict | None:
    """One batched call. Returns {'tldr': str, 'items': {idx: {...}}} or None."""
    if not GEMINI_KEY:
        return None
    blocks = []
    for i, a in enumerate(articles):
        published = a.get("published_on") or "unknown"
        blocks.append(f"[{i}] SOURCE: {a['feed']}\nTITLE: {a['title']}\n"
                      f"DATE: {published}\nTEXT: {a['text'][:ARTICLE_CHAR_BUDGET]}\n")
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
- "reject_reason": null for must_read and worth_a_skim. For low_priority, one
  of "off_topic", "marketing", or "shallow".

Also write "tldr": ONE sentence (max 25 words) summarizing what's most
important in this whole batch, leading with the top must-read.

Respond with ONLY valid JSON, no markdown fences:
{{"tldr": "...", "items": [{{"idx": 0, "tier": "must_read", "why": "...", "points": ["...", "..."], "reject_reason": null}}, ...]}}
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
    """docs/index.html is the Supabase library. A tracker run must not overwrite it."""
    return
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

def blank_record(name: str, title: str, link: str, tier: str, **extra) -> dict:
    rec = {"tier": tier, "feed": name, "title": title, "link": link,
           "date": None, "why": None, "points": None}
    rec.update(extra)
    return rec


def take_links(name: str, links: list[dict], articles_store: dict, bootstrapped: dict,
               boot_key: str, first_run: set, new_articles: list, today: str) -> None:
    """First sight marks every current link seen. Later sights triage only new ones."""
    if boot_key not in bootstrapped:
        if not links:
            print(f"  {name}: no links this run, bootstrap will retry")
            return
        first_run.add(name)
        bootstrapped[boot_key] = True
        for item in links:
            remember(articles_store, url_id(item["url"]), **blank_record(
                name, item.get("title") or slug_title(item["url"]), item["url"], "bootstrap"))
        print(f"  first sight {name}: {len(links)} marked seen")
        return
    fresh = []
    for item in links:
        aid = url_id(item["url"])
        if aid in articles_store:
            continue
        published = item.get("published_on")
        if published:
            try:
                age = (dt.datetime.utcnow().date() - dt.date.fromisoformat(published)).days
            except ValueError:
                age = 0
            if age > MAX_AGE_DAYS:
                remember(articles_store, aid, **blank_record(
                    name, item.get("title") or slug_title(item["url"]), item["url"], "reject",
                    date=today, reject_reason="stale", published_on=published))
                continue
        fresh.append(item)
    for item in fresh[:MAX_NEW_PER_FEED]:
        new_articles.append({
            "feed": name, "title": item.get("title") or slug_title(item["url"]),
            "link": item["url"], "id": url_id(item["url"]),
            "published_on": item.get("published_on"), "mode": item.get("mode") or "html",
            "text": item.get("text") or "",
        })


def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    cfg = yaml.safe_load(FEEDS_FILE.read_text())
    feeds, pages = cfg["feeds"], cfg.get("pages") or []
    articles_store = load_articles()
    bootstrapped = common.load_json(BOOTSTRAP_FILE, {})
    new_articles, first_run = [], set()
    now = ist_now()
    today = now.strftime("%Y-%m-%d")

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
                remember(articles_store, entry_id(e), **blank_record(
                    name, e.get("title", ""), e.get("link", ""), "bootstrap"))
            continue
        fresh = []
        for e in parsed.entries:
            aid = entry_id(e)
            if aid in articles_store:
                continue
            published = entry_published(e)
            if published and (dt.datetime.utcnow() - published).days > MAX_AGE_DAYS:
                remember(articles_store, aid, **blank_record(
                    name, e.get("title", ""), e.get("link", ""), "reject",
                    date=today, reject_reason="stale",
                    published_on=published.date().isoformat()))
                continue
            fresh.append(e)
        for e in fresh[:MAX_NEW_PER_FEED]:
            published = entry_published(e)
            new_articles.append({
                "feed": name, "title": e.get("title", "Untitled"),
                "link": e.get("link", ""), "id": entry_id(e), "entry": e,
                "mode": "rss", "text": "",
                "published_on": published.date().isoformat() if published else None,
            })

    run_paid = paid_due(today)
    paid_attempted = False
    for page in pages:
        name, mode = page["name"], page.get("mode", "html")
        print(f"Checking page: {name} ({mode})")
        if mode == "html":
            links = [{"url": url, "mode": "html"} for url in page_links(page)]
            take_links(name, links, articles_store, bootstrapped, page["url"],
                       first_run, new_articles, today)
        elif mode in ("firecrawl", "exa"):
            if not run_paid:
                print("  paid page check already ran today")
                continue
            if mode == "firecrawl" and not FIRECRAWL_KEY:
                print("  skipped, FIRECRAWL_API_KEY is not set")
                continue
            if mode == "exa" and not EXA_KEY:
                print("  skipped, EXA_API_KEY is not set")
                continue
            if mode == "firecrawl":
                links = [{"url": url, "mode": "firecrawl"} for url in uber_posts(page["url"])]
            else:
                links = []
                for post in exa_posts(page["domain"], page.get("path_contains") or "/blog/"):
                    post["mode"] = "exa"
                    links.append(post)
            take_links(name, links, articles_store, bootstrapped, page["url"],
                       first_run, new_articles, today)
            paid_attempted = True
        else:
            print(f"  unknown page mode: {mode}")
    if paid_attempted:
        common.save_json(PAID_STAMP_FILE, {"date": today})

    common.save_json(BOOTSTRAP_FILE, bootstrapped)

    if first_run:
        common.send_telegram(BOT_TOKEN, CHAT_ID, "✅ Now tracking: " + ", ".join(sorted(first_run)))
    if not new_articles:
        save_articles(articles_store)
        print("No new posts.")
        return

    new_articles = fair_cap(new_articles, MAX_ARTICLES_PER_RUN)
    print(f"{len(new_articles)} new posts; fetching text...")
    for a in new_articles:
        mode = a.get("mode") or "rss"
        if a.get("text"):
            a.pop("entry", None)
        elif mode == "firecrawl":
            data = firecrawl_scrape(a["link"], ["markdown"]) or {}
            title = (data.get("metadata") or {}).get("title")
            if title:
                a["title"] = title
            a["text"] = (data.get("markdown") or "")[:ARTICLE_CHAR_BUDGET]
            if len(a["text"]) < 300:
                title, text = fetch_html_article(a["link"])
                a["title"] = title or a["title"]
                a["text"] = text or a["text"]
        elif mode == "html":
            title, text = fetch_html_article(a["link"])
            if title:
                a["title"] = title
            a["text"] = text
        else:
            a["text"] = get_entry_text(a.pop("entry"))
        time.sleep(0.3)

    triage = triage_llm(new_articles)

    for i, a in enumerate(new_articles):
        info = (triage or {}).get("items", {}).get(i, {})
        tier = info.get("tier", "worth_a_skim")
        a["tier"] = tier if tier in TIER_META else "worth_a_skim"
        a["why"] = info.get("why", "")
        a["points"] = info.get("points") or fallback_points(a["text"])
        reason = info.get("reject_reason")
        a["reject_reason"] = reason if reason in REJECT_REASONS else None
        record = {
            "feed": a["feed"], "title": a["title"], "link": a["link"],
            "date": today, "tier": a["tier"], "why": a["why"], "points": a["points"],
        }
        if a.get("published_on"):
            record["published_on"] = a["published_on"]
        if a.get("reject_reason"):
            record["reject_reason"] = a["reject_reason"]
        articles_store[a["id"]] = record

    save_articles(articles_store)
    common.upsert_findings([
        {
            "id": a["id"], "pipeline": "rss", "tier": a["tier"],
            "title": a.get("title"), "url": a.get("link"), "source_name": a.get("feed"),
            "seen_on": today, "published_on": a.get("published_on"),
            "why": a.get("why"), "points": a.get("points") if isinstance(a.get("points"), list) else None,
            "reject_reason": a.get("reject_reason"),
        }
        for a in new_articles
    ])
    build_library()

    shown = [a for a in new_articles if a["tier"] != "low_priority"]
    if not shown:
        print("Nothing above low priority.")
        return

    tldr = (triage or {}).get("tldr", "")
    fname = now.strftime("%Y-%m-%d-%H%M") + ".html"
    DIGESTS.mkdir(parents=True, exist_ok=True)
    DIGESTS.joinpath(fname).write_text(render_digest(shown, tldr, now))
    build_runs_page()
    url = common.pages_url(f"digests/{fname}")

    must = [a for a in shown if a["tier"] == "must_read"]
    lines = [f"📬 <b>{len(shown)} new posts</b>"
             + (f" · <b>{len(must)} must-read{'s' if len(must) != 1 else ''}</b>" if must else "")]
    if tldr:
        lines.append(f"⚡ {html.escape(tldr)}")
    for a in must[:3]:
        lines.append(f"🔴 {html.escape(a['title'])}")
    library = common.pages_url("index.html")
    lines.append(f'\n<a href="{library}">Open library →</a>')
    lines.append(f'<a href="{url}">This run →</a>')
    common.send_telegram(BOT_TOKEN, CHAT_ID, "\n".join(lines))
    print(f"Digest: {url}")


if __name__ == "__main__":
    main()
