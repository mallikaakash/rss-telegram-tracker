"""
Shared plumbing for tracker.py (RSS digest) and discovery.py (niche finds):
JSON stores, Telegram send, Pages URL building, and the shared site CSS/page
shell so both pipelines render as one consistent-looking site.
"""

import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/126 Safari/537.36"}


# ---------------------------------------------------------------- gemini

def gemini_generate(api_key: str, model: str, prompt: str,
                    max_output_tokens: int = 8000, temperature: float = 0.2) -> dict | None:
    """One JSON-mode Gemini call with retries on 429/5xx. Returns parsed dict or None."""
    r = None
    for attempt in range(4):
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_output_tokens,
                                           "temperature": temperature,
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
        data, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
        return data
    except Exception as e:  # noqa: BLE001
        print(f"  Gemini parse failed: {e}")
        return None


# ---------------------------------------------------------------- json store

def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- telegram

def send_telegram(bot_token: str, chat_id: str, msg: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                      json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=20)
    if not r.ok:
        print(f"  Telegram error: {r.text}")


# ---------------------------------------------------------------- pages url

def pages_url(rel_path: str) -> str:
    """rel_path is relative to docs/, e.g. 'digests/foo.html' or 'discovery/index.html'."""
    base = os.environ.get("PAGES_BASE_URL", "").rstrip("/")
    if not base:
        repo = os.environ.get("GITHUB_REPOSITORY", "user/repo")
        owner, name = repo.split("/", 1)
        base = f"https://{owner}.github.io/{name}"
    return f"{base}/{rel_path.lstrip('/')}"


# ---------------------------------------------------------------- shared css
# Nerdy library aesthetic: earth palette, hairline borders, zero radius,
# monospace for labels/meta, serif for reading. No blur/glow anywhere.

SITE_CSS = """
:root{
  color-scheme:light dark;
  --bg:#f3ead6; --panel:#fbf6ea; --border:#cdb98c; --border-strong:#a8905f;
  --text:#241c10; --muted:#7a6a4c; --accent:#9c3b1a; --link:#7a3315;
  --must-bg:#9c3b1a; --must-fg:#f6ecd9;
  --skim-bg:#9c7a1f; --skim-fg:#231a06;
  --low-bg:#6b6b4c; --low-fg:#f0ead9;
  --gem-bg:#3c5c3a; --gem-fg:#f0ead9;
  --font-serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,'Times New Roman',serif;
  --font-mono:ui-monospace,'SF Mono','JetBrains Mono',Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#191309; --panel:#211a10; --border:#4a3c26; --border-strong:#6b5636;
    --text:#e9dfc4; --muted:#a3906c; --accent:#d4763f; --link:#e08a52;
    --must-bg:#7a2c12; --must-fg:#f6ecd9;
    --skim-bg:#7a5f16; --skim-fg:#f6ecd9;
    --low-bg:#4a4a36; --low-fg:#d9d2ba;
    --gem-bg:#2e4a2c; --gem-fg:#e9dfc4;
  }
}
*{box-sizing:border-box;margin:0}
body{font-family:var(--font-serif);background:var(--bg);color:var(--text);
 max-width:720px;margin:0 auto;padding:28px 18px 70px;line-height:1.6}
a{color:var(--link)}
h1{font-family:var(--font-mono);font-size:1.05rem;font-weight:700;
 letter-spacing:.02em;text-transform:uppercase;margin-bottom:6px}
.meta{font-family:var(--font-mono);color:var(--muted);font-size:.72rem;
 margin-bottom:22px;text-transform:uppercase;letter-spacing:.04em;
 border-bottom:1px solid var(--border);padding-bottom:12px}
.meta a{color:var(--muted)}
.tldr{background:var(--panel);border:1px solid var(--border);
 border-left:4px solid var(--accent);padding:14px 16px;margin-bottom:26px;
 font-size:.98rem}
.card{background:var(--panel);border:1px solid var(--border);
 box-shadow:3px 3px 0 0 var(--border);padding:16px 18px;margin:0 0 18px}
.badge{display:inline-block;font-family:var(--font-mono);font-size:.68rem;
 font-weight:700;letter-spacing:.06em;text-transform:uppercase;
 padding:2px 7px;border:1px solid rgba(0,0,0,.15);margin-bottom:9px}
.badge.must{background:var(--must-bg);color:var(--must-fg)}
.badge.skim{background:var(--skim-bg);color:var(--skim-fg)}
.badge.low{background:var(--low-bg);color:var(--low-fg)}
.badge.gem{background:var(--gem-bg);color:var(--gem-fg)}
.src{font-family:var(--font-mono);color:var(--muted);font-size:.72rem;
 margin-left:8px;text-transform:uppercase;letter-spacing:.03em}
.card h2{font-size:1.08rem;line-height:1.4;margin:2px 0 8px;font-weight:700}
.card h2 a{text-decoration:none;color:var(--text)}
.card h2 a:hover{color:var(--accent)}
.why{color:var(--muted);font-size:.88rem;font-style:italic;margin-bottom:9px;
 border-left:2px solid var(--border);padding-left:10px}
ul{padding-left:20px}
li{font-size:.92rem;margin-bottom:5px}
.section{margin:30px 0 12px;font-family:var(--font-mono);font-size:.74rem;
 font-weight:700;color:var(--muted);text-transform:uppercase;
 letter-spacing:.08em;border-bottom:1px solid var(--border);padding-bottom:6px}
.idx a{display:block;padding:12px 14px;background:var(--panel);
 border:1px solid var(--border);box-shadow:3px 3px 0 0 var(--border);
 margin-bottom:10px;text-decoration:none;color:var(--text)}
.idx .sub{font-family:var(--font-mono);color:var(--muted);font-size:.72rem;
 margin-top:3px;text-transform:uppercase;letter-spacing:.03em}
.stamp{font-family:var(--font-mono);color:var(--muted);font-size:.7rem;
 float:right;text-transform:uppercase}
.toolbar{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}
.toolbar input[type=text]{font-family:var(--font-mono);font-size:.85rem;
 background:var(--panel);color:var(--text);border:1px solid var(--border);
 padding:8px 10px;flex:1;min-width:160px}
.toolbar input[type=text]:focus{outline:2px solid var(--accent);outline-offset:-1px}
.chip{font-family:var(--font-mono);font-size:.7rem;text-transform:uppercase;
 letter-spacing:.04em;background:var(--panel);color:var(--muted);
 border:1px solid var(--border);padding:7px 11px;cursor:pointer;user-select:none}
.chip.active{background:var(--accent);color:var(--must-fg);border-color:var(--accent)}
.empty{color:var(--muted);font-size:.9rem;font-style:italic;padding:20px 0}
"""


def page_shell(title: str, body: str, extra_head: str = "") -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{SITE_CSS}</style>{extra_head}'
            f'</head><body>{body}</body></html>')
