#!/usr/bin/env python3
"""Build the free, static version of the website (no server needed).

    python build_site.py --site-url https://NAME.github.io/dream-game-finder/

Everything runs in the visitor's browser: Pyodide (Python compiled to
WebAssembly, loaded from the jsDelivr CDN) runs recommend.py, quiz.py and
server.py on a slimmed copy of the model. So the site can live on any free
static host (GitHub Pages, Cloudflare Pages) and never sleeps or costs money.

It also writes one page per well-reviewed game ("what players say" and games
like it), genre pages, sitemap.xml and robots.txt, so search engines (and
Google's AI Mode, which answers from Google's index) can find the site.

The output folder holds only public data: no review text and no player IDs
(steam.db never leaves this computer).
"""
from __future__ import annotations

import argparse
import calendar
import gzip
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata

import numpy as np
import scipy.sparse as sp

import quiz
import recommend as rec
import server

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
PY_FILES = ["recommend.py", "quiz.py", "server.py", "webapi.py"]
PYODIDE_VERSION = "314.0.7"      # pinned; released 2026-09-14
WEB_MIN_REVIEWS = 50             # the site never recommends games with fewer (server default)
TEXT_TOP_K = 64                  # strongest description/review words kept per game
CF_SCALE = 1000.0                # co-likes stored as float16(value * scale): keeps tiny values precise
FIRST_ROUND_SEEDS = 64           # round 1 is precomputed, so the quiz starts before Python has loaded
PAGE_MIN_POINTS = 3              # a game page needs at least this many "what players say" points
SIMILAR_N = 10
ADULT_PAGE_TAGS = set(rec.ADULT_TAGS)


# --------------------------------------------------------------------------- #
# slim model for the browser
# --------------------------------------------------------------------------- #
def top_k_rows(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of each row, then re-normalise rows to unit length."""
    M = M.tocsr()
    keep = np.zeros(M.nnz, bool)
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if e - s <= k:
            keep[s:e] = True
        else:
            keep[s + np.argpartition(-np.abs(M.data[s:e]), k)[:k]] = True
    M = M.copy()
    M.data[~keep] = 0
    M.eliminate_zeros()
    return rec.l2_rows(M)


def slim_platform(p: dict) -> dict:
    p = p or {}
    out = {k: 1 for k in ("windows", "mac", "steamos_linux", "linux") if p.get(k)}
    if p.get("steam_deck_compat_category"):
        out["steam_deck_compat_category"] = int(p["steam_deck_compat_category"])
    return out


CLASSIC_HEADER = re.compile(r"^https://shared\.akamai\.steamstatic\.com/store_item_assets/steam/apps/(\d+)/header\.jpg(\?t=\d+)?$")


def slim_header(url, appid):
    m = CLASSIC_HEADER.match(url or "")
    return None if (not url or (m and int(m.group(1)) == appid)) else url   # None -> the classic URL


def short(text: str, n: int = 200) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:-") + "…"


def write_web_model(model_path, meta_path, out_dir, min_reviews=WEB_MIN_REVIEWS, text_top_k=TEXT_TOP_K,
                    compact=True):
    """Write data/model.npz + data/meta.json.gz with only the games the site can
    recommend, fewer text terms and smaller number types. Returns the paths.
    (text_top_k=0, compact=False keeps full precision - used to measure what
    the slimming costs with `recommend.py eval`.)"""
    with np.load(model_path) as f:
        z = {k: f[k] for k in f.files}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n = meta["n_games"]
    rows = np.where(np.array([r or 0 for r in meta["review_count"]]) >= min_reviews)[0]

    def csr(prefix, shape):
        return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]), shape=shape)
    tag = csr("tag", (n, meta["n_tags"]))[rows].tocsr()
    text = top_k_rows(csr("text", (n, meta["n_terms"]))[rows], text_top_k) if text_top_k \
        else csr("text", (n, meta["n_terms"]))[rows].tocsr()
    cf = csr("cf", (n, n))[rows][:, rows].tocsr()
    text_hashes, text_idf, n_terms = z["text_hashes"], z["text_idf"], meta["n_terms"]
    used = np.unique(text.indices)
    if compact and len(used) < 65536:     # drop words no game uses any more -> 16-bit word numbers
        col = np.full(n_terms, -1, np.int64)
        col[used] = np.arange(len(used))
        text = sp.csr_matrix((text.data, col[text.indices], text.indptr), shape=(len(rows), len(used)))
        text_hashes, text_idf, n_terms = text_hashes[used], text_idf[used], len(used)

    def idx(a, size):                     # 16-bit indices whenever they fit
        return a.astype(np.uint16 if compact and size < 65536 else np.int32)
    real = np.float16 if compact else np.float32
    arrays = {
        "build_id": z["build_id"],
        "tag_data": tag.data.astype(real), "tag_indices": idx(tag.indices, meta["n_tags"]),
        "tag_indptr": tag.indptr.astype(np.int32),
        "text_data": text.data.astype(real), "text_indices": idx(text.indices, n_terms),
        "text_indptr": text.indptr.astype(np.int32),
        "cf_indices": idx(cf.indices, len(rows)), "cf_indptr": cf.indptr.astype(np.int32),
        "quality": z["quality"][rows].astype(np.float32), "support": z["support"][rows],
        "text_hashes": text_hashes, "text_idf": text_idf, "tag_idf": z["tag_idf"],
        "aspects": z["aspects"][rows].astype(real),
    }
    if compact:
        arrays.update(cf_data=(cf.data * CF_SCALE).astype(np.float16), cf_scale=np.float32(CF_SCALE))
    else:
        arrays.update(cf_data=cf.data.astype(np.float32))
    keep = rows.tolist()
    m2 = {k: meta[k] for k in ("built_at", "build_id", "db", "n_tags", "n_terms", "n_players",
                               "aspects", "tag_ids", "tag_names")}
    m2["n_games"] = len(keep)
    m2["n_terms"] = int(n_terms)
    for k in ("appids", "names", "review_count", "percent_positive", "release_date", "is_free", "notes"):
        m2[k] = [meta[k][i] for i in keep]
    m2["descriptions"] = [short(meta["descriptions"][i]) for i in keep]
    m2["platforms"] = [slim_platform(meta["platforms"][i]) for i in keep]
    m2["headers"] = [slim_header(meta["headers"][i], meta["appids"][i]) for i in keep]
    os.makedirs(out_dir, exist_ok=True)
    mp, jp = os.path.join(out_dir, "model.npz"), os.path.join(out_dir, "meta.json.gz")
    np.savez_compressed(mp, **arrays)
    with gzip.open(jp, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(m2, f, ensure_ascii=False, separators=(",", ":"))
    print(f"web model: {len(keep):,} of {n:,} games (>= {min_reviews} reviews), "
          f"text {text.nnz:,} entries, co-likes {cf.nnz:,}; "
          f"model.npz {os.path.getsize(mp) / 1e6:.1f} MB, meta.json.gz {os.path.getsize(jp) / 1e6:.1f} MB")
    return mp, jp


# --------------------------------------------------------------------------- #
# static pages
# --------------------------------------------------------------------------- #
def slugify(name: str, limit: int = 60) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:limit].rstrip("-")


def game_slug(appid, name=None):
    """Page folder for a game: the appid alone, so a renamed game keeps its URL."""
    return str(int(appid))


E = html.escape
PLATFORM_NAMES = (("windows", "Windows"), ("mac", "macOS"), ("steamos_linux", "Linux"))


def platform_text(p):
    out = [label for k, label in PLATFORM_NAMES if p.get(k) or (k == "steamos_linux" and p.get("linux"))]
    if p.get("steam_deck_compat_category") in (2, 3):
        out.append("Steam Deck " + ("verified" if p["steam_deck_compat_category"] == 3 else "playable"))
    return ", ".join(out)


def year_of(ts):
    return time.gmtime(ts).tm_year if ts else None


def note_points(note):
    if not note:
        return 0
    keys = ("praise", "complaints", "traits", "mixed", "mood")
    return sum(len(note.get(k) or []) for k in keys) + (2 if note.get("summary") else 0)


def page_shell(title, description, canonical, depth, body, *, image=None, jsonld=None, v="", noindex=False,
               base=None):
    up = base if base is not None else "../" * depth     # 404.html is served at any URL: absolute links
    head = [
        '<!doctype html>', '<html lang="en">', '<head>',
        '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{E(title)}</title>',
        f'<meta name="description" content="{E(description)}">',
        f'<link rel="canonical" href="{E(canonical)}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{E(title)}">',
        f'<meta property="og:description" content="{E(description)}">',
        f'<meta property="og:url" content="{E(canonical)}">',
    ]
    if noindex:
        head.append('<meta name="robots" content="noindex, follow">')
    if image:
        head.append(f'<meta property="og:image" content="{E(image)}">')
    head.append(f'<link rel="stylesheet" href="{up}style.css?v={v}">')
    if jsonld:
        head.append('<script type="application/ld+json">' +
                    json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/") + '</script>')
    head += ['</head>', '<body class="page">',
             '<header class="topbar">',
             f'<a class="brand" href="{up}"><span class="brand-mark" aria-hidden="true">'
             '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M12 2l2.6 6.3L21 9l-5 4.4L17.5 20 12 16.6 6.5 20 8 13.4 3 9l6.4-.7z" fill="currentColor"/></svg>'
             '</span><span>Dream Game Finder</span></a>',
             f'<nav class="page-nav"><a href="{up}genres/">Browse by genre</a><a href="{up}">Taste quiz</a></nav>',
             '</header>', '<main class="page-main">']
    foot = ['</main>',
            '<footer class="foot">What players say is summarised from public Steam reviews. '
            'Game data and images from Steam\'s public store, shown as is. Not affiliated with Valve. '
            f'<a href="{up}about/">About this data</a></footer>',
            '</body>', '</html>']
    return "\n".join(head) + "\n" + body + "\n" + "\n".join(foot) + "\n"


def players_html(note):
    if not note:
        return ""
    parts = []
    if note.get("summary"):
        parts.append(f'<p class="psummary">{E(note["summary"])}</p>')

    def lst(items, cls, label):
        items = [x for x in (items or []) if x]
        if not items:
            return ""
        lis = "".join(f'<li class="{cls}">{E(str(x))}</li>' for x in items)
        return f'<h3>{label}</h3><ul class="plist">{lis}</ul>'
    parts.append(lst(note.get("praise"), "good", "Players love"))
    parts.append(lst(note.get("traits"), "feel", "What stands out"))
    parts.append(lst(note.get("mood"), "feel", "How it feels"))
    parts.append(lst(note.get("complaints"), "warn", "Watch out for"))
    parts.append(lst(note.get("mixed"), "split", "Players are split on"))
    if note.get("best_for"):
        parts.append(f'<p class="bestfor"><b>Best for:</b> {E(note["best_for"])}</p>')
    if note.get("not_for"):
        parts.append(f'<p class="bestfor"><b>Not for:</b> {E(note["not_for"])}</p>')
    return '<section class="players page-players"><h2>What players say</h2>' + "".join(parts) + "</section>"


def indexable(note) -> bool:
    """Pages Google is asked to index: games with a written, game-specific
    summary. Pages built only from stock aspect labels stay browsable (and
    linked) but carry noindex and are left out of the sitemap, so the site
    doesn't look like mass-produced thin content (Google's scaled-content policy)."""
    return bool(note and note.get("summary"))


def build_pages(model, out, site_url, v, built_at, min_points=PAGE_MIN_POINTS):
    qi = quiz.index_for(model)
    fam_names = list(qi.family_names) + ["Other"]
    meta = model.meta
    notes = meta.get("notes") or [None] * len(model.appids)
    page_rows = [i for i in range(len(model.appids))
                 if note_points(notes[i]) >= min_points and not model.adult[i] and not model.is_demo[i]]
    slug_of = {i: game_slug(model.appids[i], model.names[i]) for i in page_rows}
    print(f"game pages: {len(page_rows):,} ({sum(indexable(notes[i]) for i in page_rows):,} offered to search engines)")
    opts = rec.make_opts(n=SIMILAR_N, min_reviews=WEB_MIN_REVIEWS)
    lastmod = time.strftime("%Y-%m-%d", time.gmtime(built_at))
    urls = [(site_url, lastmod), (site_url + "genres/", lastmod)]
    by_family: dict[int, list[int]] = {}
    t0 = time.time()
    for k, i in enumerate(page_rows):
        fam = int(qi.family[i]) if qi.family[i] >= 0 else len(fam_names) - 1
        by_family.setdefault(fam, []).append(i)
        name, appid = model.names[i], model.appids[i]
        slug = slug_of[i]
        url = f"{site_url}games/{slug}/"
        index_me = indexable(notes[i])
        if index_me:
            urls.append((url, lastmod))
        try:
            sim = rec.rec_like(model, [i], [], opts)["results"]
        except rec.RecError:
            sim = []
        note = notes[i]
        pct = meta["percent_positive"][i]
        yr = year_of(meta["release_date"][i])
        tags = model.top_tags(i, 8)
        image = meta["headers"][i] or f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"
        facts = [f"{pct}% positive" if pct is not None else None,
                 f"{int(model.reviews[i]):,} Steam reviews", str(yr) if yr else None,
                 platform_text(meta["platforms"][i]) or None]
        sim_items = []
        for r in sim:
            j = model.index.get(r["appid"])
            href = f"../{slug_of[j]}/" if j in slug_of else f"https://store.steampowered.com/app/{r['appid']}/"
            ext = "" if j in slug_of else ' rel="noopener" target="_blank"'
            ry = year_of(r.get("release_date"))
            sub = " · ".join(x for x in (f"{r['percent_positive']}% positive" if r.get("percent_positive") is not None else "",
                                         str(ry) if ry else "") if x)
            sim_items.append(f'<li><a href="{href}"{ext}>{E(r["name"])}</a> <span class="muted">{E(sub)}</span>'
                             f'<p class="why">{E(r.get("why") or "")}</p></li>')
        fam_label = fam_names[fam]
        fam_slug = slugify(fam_label)
        summary = (note or {}).get("summary")
        praise = ", ".join((note or {}).get("praise", [])[:2])
        sim_names = ", ".join(r["name"] for r in sim[:3])
        description = short(summary, 150) if summary else short(
            (f"Steam players praise {praise}. " if praise else "") +
            (f"Games like {name}: {sim_names} and more." if sim_names else f"What Steam players say about {name}."), 155)
        title = f"{name}: what players say & {len(sim)} games like it" if sim else f"{name}: what Steam players say"
        body = (
            f'<nav class="crumbs" aria-label="Breadcrumb"><a href="../../">Home</a> › '
            f'<a href="../../genres/{fam_slug}/">{E(fam_label)}</a> › <span>{E(name)}</span></nav>'
            f'<article class="game-page">'
            f'<img class="hero-img" src="{E(image)}" alt="{E(name)}" width="460" height="215">'
            f'<h1>{E(name)}</h1>'
            f'<p class="facts">{E(" · ".join(x for x in facts if x))}</p>'
            + ('<ul class="tags">' + "".join(f"<li>{E(t)}</li>" for t in tags) + "</ul>" if tags else "")
            + players_html(note)
            + (f'<section class="similar"><h2>Games like {E(name)}</h2><ol>{"".join(sim_items)}</ol></section>'
               if sim_items else "")
            + f'<p class="cta"><a class="primary" href="../../#like={appid}">Find more games like {E(name)}</a> '
            f'<a class="ghost" href="../../">Take the 7-round taste quiz</a> '
            f'<a class="ghost" href="https://store.steampowered.com/app/{appid}/" rel="noopener" target="_blank">Steam page ↗</a></p>'
            f'</article>')
        jsonld = {"@context": "https://schema.org", "@type": "VideoGame", "name": name, "url": url,
                  "image": image, "sameAs": f"https://store.steampowered.com/app/{appid}/",
                  "genre": tags[:5], "gamePlatform": platform_text(meta["platforms"][i]) or None}
        if yr:
            jsonld["datePublished"] = time.strftime("%Y-%m-%d", time.gmtime(meta["release_date"][i]))
        jsonld = {k: v_ for k, v_ in jsonld.items() if v_}
        write(os.path.join(out, "games", slug, "index.html"),
              page_shell(title, description, url, 2, body, image=image, jsonld=jsonld, v=v, noindex=not index_me))
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1:,} pages ({time.time() - t0:.0f}s)")

    # genre hubs: every game page is reachable from a hub
    hub_items = []
    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        label = fam_names[fam]
        fs = slugify(label)
        games = sorted(by_family[fam], key=lambda i: -model.reviews[i])
        lis = "".join(
            f'<li><a href="../../games/{slug_of[i]}/">{E(model.names[i])}</a> '
            f'<span class="muted">{meta["percent_positive"][i]}% · {int(model.reviews[i]):,} reviews</span></li>'
            for i in games)
        url = f"{site_url}genres/{fs}/"
        urls.append((url, lastmod))
        body = (f'<nav class="crumbs" aria-label="Breadcrumb"><a href="../../">Home</a> › '
                f'<a href="../">Genres</a> › <span>{E(label)}</span></nav>'
                f'<h1>{E(label)} games on Steam</h1>'
                f'<p class="lede">{len(games):,} {E(label.lower())} games with a summary of what players say, '
                f'most-reviewed first. Open one to see what players love, what they warn about and games like it.</p>'
                f'<ol class="hub-list">{lis}</ol>')
        write(os.path.join(out, "genres", fs, "index.html"),
              page_shell(f"{label} games on Steam: what players say", f"{len(games):,} {label} games on Steam with "
                         "what players love and warn about, and similar games for each.", url, 2, body, v=v))
        hub_items.append(f'<li><a href="{fs}/">{E(label)}</a> <span class="muted">{len(games):,} games</span></li>')
    body = ('<nav class="crumbs" aria-label="Breadcrumb"><a href="../">Home</a> › <span>Genres</span></nav>'
            '<h1>Browse Steam games by genre</h1>'
            '<p class="lede">Every game here has a summary of what Steam players say about it and a list of games like it.</p>'
            f'<ul class="hub-list genres">{"".join(hub_items)}</ul>')
    write(os.path.join(out, "genres", "index.html"),
          page_shell("Browse Steam games by genre - Dream Game Finder", "Steam games by genre, each with "
                     "what players say and similar games.", site_url + "genres/", 1, body, v=v))
    return urls, len(page_rows)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def contact_url(site_url):
    """Where to ask for a removal or correction: the issues page of the site's
    own GitHub repository (for a GitHub Pages address), else None."""
    m = re.match(r"^https://([a-z0-9-]+)\.github\.io/([^/]+)/", site_url, re.I)
    if m:
        return f"https://github.com/{m.group(1)}/{m.group(2)}/issues"
    m = re.match(r"^https://([a-z0-9-]+)\.github\.io/$", site_url, re.I)
    return f"https://github.com/{m.group(1)}/{m.group(1)}.github.io/issues" if m else None


def about_page(site_url, v, built_at, contact=None):
    """/about/: what data the site shows, whose it is, "as is", privacy and removals.
    Steam's API terms (s. 2) ask that Steam data is provided "as is" with
    disclaimers, and that users are told what is stored."""
    day = time.strftime("%d %B %Y", time.gmtime(built_at)).lstrip("0")
    ask = (f'<p>Are you a developer, a publisher or a reviewer and want something here removed or corrected? '
           f'<a href="{E(contact)}" rel="noopener">Open an issue</a> and it will be taken down in the next '
           f'update.</p>') if contact else ""
    body = f"""<nav class="crumbs"><a href="../">Home</a> › About this data</nav>
<h1>About this site and its data</h1>
<p class="muted">Steam data snapshot: {E(day)}.</p>

<h2>What's here</h2>
<ul class="plist">
<li><b>Game facts from Steam's public store</b>: names, release dates, developers and publishers, platforms,
tags, and review scores, as they were on the date above.</li>
<li><b>Short excerpts of store descriptions</b>, each linked to its Steam page. They belong to the game's
developer or publisher.</li>
<li><b>Cover images</b>, loaded straight from Steam's servers. They are not copied here.</li>
<li><b>"What players say"</b>: summaries written by an AI in its own words after reading many public
Steam reviews of a game, plus statistics computed from reviews (how often players mention boss fights,
bugs and so on).</li>
</ul>
<p>The site contains <b>no review text, no Steam IDs and no player names</b>.</p>

<h2>Provided as is</h2>
<p>All Steam data on this site is provided "as is", without any warranty. It may be incomplete, out of date
or wrong. Check the game's Steam page before you buy anything.</p>
<p>Dream Game Finder is not affiliated with or endorsed by Valve. Steam and the Steam logo are trademarks of
Valve Corporation. Game names, descriptions and images belong to their owners; this site claims no rights in
them and grants no licence to them.</p>

<h2>Our own content</h2>
<p>The AI-written summaries and the statistics this site computed are released under
<a href="https://creativecommons.org/publicdomain/zero/1.0/" rel="noopener">CC0 1.0</a>: use them for
anything. The site's code is MIT-licensed.</p>

<h2>Your privacy</h2>
<ul class="plist">
<li>No accounts, cookies, analytics or tracking.</li>
<li>Your quiz answers, favourites and filters are saved only in your own browser, so you can come back to
them. "Start over" clears your quiz; clearing this site's data in your browser removes everything. The
recommender runs in your browser too: nothing you answer is sent to us.</li>
<li>Cover images load from Steam's servers and the in-browser engine from the jsDelivr CDN, so those servers
see a request from your browser, as with any image or script on the web.</li>
<li>"Ask Google AI Mode" sends your wish to Google, only when you click it.</li>
<li>The site is hosted on GitHub Pages, which keeps its own logs
(<a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement" rel="noopener">GitHub's privacy statement</a>).</li>
</ul>
{ask}"""
    return page_shell("About this data - Dream Game Finder",
                      "What data Dream Game Finder shows, where it comes from, who owns it, and your privacy.",
                      f"{site_url}about/", 1, body, v=v)


def sitemap_xml(urls):
    items = "".join(f"<url><loc>{E(u)}</loc><lastmod>{d}</lastmod></url>" for u, d in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>\n')


# --------------------------------------------------------------------------- #
def static_index(src, *, site_url, build, google_verification):
    """web/index.html -> the static site's home page."""
    s = src
    cfg = json.dumps({"static": True, "build": build, "site": site_url})
    extra_head = [f'<link rel="canonical" href="{E(site_url)}">',
                  '<meta property="og:type" content="website">',
                  '<meta property="og:title" content="Dream Game Finder - find your next Steam game">',
                  f'<meta property="og:url" content="{E(site_url)}">',
                  '<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>']
    if google_verification:
        extra_head.append(f'<meta name="google-site-verification" content="{E(google_verification)}">')
    s = s.replace('<link rel="stylesheet" href="style.css">',
                  "\n  ".join(extra_head) + f'\n  <link rel="stylesheet" href="style.css?v={build}">')
    s = s.replace('<script src="app.js"></script>',
                  f'<script>window.DGF = {cfg};</script>\n  <script src="app.js?v={build}"></script>')
    s = s.replace('<a class="brand" href="/"', '<a class="brand" href="./"')
    s = s.replace("Recommendations run locally on this computer.",
                  'Recommendations run in your browser. <a href="about/">About this data</a> · '
                  '<a href="genres/">Browse games by genre</a>.')
    for needle in ('window.DGF', 'href="./"', 'genres/', 'about/', 'shown as is'):
        if needle not in s:
            raise SystemExit(f"build_site: web/index.html changed - couldn't place {needle!r}")
    return s


def _force_remove(func, path, _exc):
    """rmtree helper: git makes its object files read-only on Windows."""
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--site-url", required=True, help="public address ending in /, e.g. https://NAME.github.io/dream-game-finder/")
    p.add_argument("--out", default=os.path.join(HERE, "site"))
    p.add_argument("--model", default=os.path.join(HERE, rec.MODEL_FILE))
    p.add_argument("--meta", default=os.path.join(HERE, rec.META_FILE))
    p.add_argument("--google-verification", default=os.environ.get("GOOGLE_SITE_VERIFICATION", ""),
                   help="Search Console HTML-tag token (content=...)")
    p.add_argument("--no-pages", action="store_true", help="skip the per-game pages (quick test builds)")
    p.add_argument("--contact", default=None,
                   help="where people ask for removals (default: the GitHub Pages repo's issues page)")
    args = p.parse_args()
    site_url = args.site_url if args.site_url.endswith("/") else args.site_url + "/"
    if not re.match(r"^https://[^/]+/", site_url):
        sys.exit("--site-url must start with https://")
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        if not os.path.exists(os.path.join(out, ".dgf-site")):
            sys.exit(f"{out} exists and wasn't made by build_site.py - pick another --out")
        shutil.rmtree(out, onerror=_force_remove)
    os.makedirs(out)
    write(os.path.join(out, ".dgf-site"), "made by build_site.py; deleted and rebuilt on every build\n")
    t0 = time.time()

    data = os.path.join(out, "data")
    mp, jp = write_web_model(args.model, args.meta, data)
    server.load(mp, jp)                                   # the exact model the browser will run
    model = server.MODEL
    build = str(model.meta["build_id"])
    status = dict(server.STATUS)
    status["ai"] = {"available": False, "reason": "not on the web version"}
    seeds = [1009 + 7919 * k for k in range(FIRST_ROUND_SEEDS)]
    status["first_round_seeds"] = seeds
    status["pyodide"] = PYODIDE_VERSION
    status["data_bytes"] = {os.path.basename(x): os.path.getsize(x) for x in (mp, jp)}
    write(os.path.join(data, "status.json"), json.dumps(status, ensure_ascii=False, separators=(",", ":")))
    rounds = {str(s): quiz.next_round(model, [], s) for s in seeds}
    write(os.path.join(data, "rounds1.json"), json.dumps(rounds, ensure_ascii=False, separators=(",", ":")))

    for name in ("app.js", "style.css", "engine.js"):
        shutil.copyfile(os.path.join(WEB_DIR, name), os.path.join(out, name))
    for name in PY_FILES:
        os.makedirs(os.path.join(out, "py"), exist_ok=True)
        shutil.copyfile(os.path.join(HERE, name), os.path.join(out, "py", name))
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        write(os.path.join(out, "index.html"), static_index(f.read(), site_url=site_url, build=build,
                                                              google_verification=args.google_verification))
    urls, n_pages = ([(site_url, time.strftime("%Y-%m-%d"))], 0) if args.no_pages else \
        build_pages(model, out, site_url, build, model.meta.get("built_at") or time.time())
    built_at = model.meta.get("built_at") or time.time()
    write(os.path.join(out, "about", "index.html"),
          about_page(site_url, build, built_at, args.contact or contact_url(site_url)))
    urls.append((f"{site_url}about/", time.strftime("%Y-%m-%d", time.gmtime(built_at))))
    write(os.path.join(out, "sitemap.xml"), sitemap_xml(urls))
    # robots.txt only counts at the root of a host (a custom domain or NAME.github.io);
    # for a project site, the sitemap is submitted in Search Console instead
    write(os.path.join(out, "robots.txt"), f"User-agent: *\nAllow: /\n\nSitemap: {site_url}sitemap.xml\n")
    write(os.path.join(out, "404.html"), page_shell(
        "Page not found - Dream Game Finder", "This page doesn't exist.", site_url, 0,
        '<h1>Page not found</h1><p><a href="' + E(site_url) + '">Go to Dream Game Finder</a></p>',
        v=build, noindex=True, base=site_url))
    write(os.path.join(out, ".nojekyll"), "")
    write(os.path.join(out, ".dgf-complete"), build + "\n")       # last: the build finished
    files = sum(len(fs) for _, _, fs in os.walk(out))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(out) for x in fs)
    print(f"site: {files:,} files, {size / 1e6:.1f} MB, {n_pages:,} game pages, {len(urls):,} sitemap URLs "
          f"-> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
