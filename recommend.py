#!/usr/bin/env python3
"""
recommend.py - Steam game recommender built on the data from steam_reviews.py.

Three signals, blended:
  * content    - what a game IS: store tags (weighted by how many players
                 applied them) + description + words players use in reviews
  * co-likes   - collaborative filtering: games recommended by the same
                 players ("people who liked Hades also liked ...")
  * quality    - Bayesian-smoothed % positive, so a 100%-from-3-reviews game
                 doesn't outrank a 97%-from-300k one

Commands
--------
  python recommend.py build                         # once, after crawling
  python recommend.py like "Hades" "Hollow Knight"  # games similar to ones you love
  python recommend.py like "Stardew Valley" --gems  # ...favouring lesser-known games
  python recommend.py dream "cozy farming with a deep story, no combat"
  python recommend.py dream "..." --ai              # Claude interprets + explains
  python recommend.py user 76561198000000000        # for a reviewer in the database
  python recommend.py eval                          # measure quality (hit rate)

Filters (like/dream/user): --tags, --exclude-tags, --min-positive, --max-reviews,
--min-reviews, --free, --platform windows|mac|linux|deck, --year-from, -n, --json.

Needs numpy + scipy. --ai also needs `pip install anthropic` and Claude API
credentials (ANTHROPIC_API_KEY or `ant auth login`).
"""
from __future__ import annotations

import argparse
import calendar
import difflib
import gzip
import json
import math
import os
import re
import sqlite3
import sys
import time
import zlib
from array import array
from collections import Counter

import numpy as np
import scipy.sparse as sp

MODEL_FILE = "recsys_model.npz"
META_FILE = "recsys_meta.json"

STOPWORDS = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also get got really even much many like one game games play played playing
time still lot thing things make made way well dont doesnt didnt isnt cant wont im ive youre thats
steam review reviews recommend recommended hours hour minutes would 10 11 dlc buy bought price
want looking something please""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9']{2,}")
HASH_BITS = 22                      # 4M hashed term buckets (text features)
HASH_MASK = (1 << HASH_BITS) - 1


def tokenize(text: str) -> list[str]:
    toks = [t.replace("'", "") for t in TOKEN_RE.findall((text or "").lower())]
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    # unigrams + adjacent bigrams ("open world", "turn based", "story rich")
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def term_hash(t: str) -> int:
    return zlib.crc32(t.encode("utf-8")) & HASH_MASK


def l2_rows(m):
    m = sp.csr_matrix(m, dtype=np.float32)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return (sp.diags(1 / norms) @ m).tocsr()


def norm01(v):
    v = np.asarray(v, dtype=np.float32)
    mx = v.max() if v.size else 0
    return v / mx if mx > 0 else v


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def load_aspects(con, index):
    """games x aspects matrix of signed scores from analyze_reviews.py
    (+ praised quality / notable trait, - complaint or issue); zeros if the
    analysis hasn't been run."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "aspects" not in tables or "game_aspects" not in tables:
        return np.zeros((len(index), 0), np.float32), []
    try:
        from analyze_reviews import TOPIC
    except Exception:                    # pragma: no cover
        TOPIC = {}
    meta = [{"key": k, "kind": kind, "label": lab, "taste_label": tl, "area": area,
             "topic": TOPIC.get(k, lab.lower())}
            for k, kind, lab, tl, area in con.execute(
                "SELECT key, kind, label, taste_label, area FROM aspects ORDER BY key")]
    col = {a["key"]: j for j, a in enumerate(meta)}
    M = np.zeros((len(index), len(meta)), np.float32)
    for appid, key, score, role in con.execute("SELECT appid, aspect, score, role FROM game_aspects"):
        i, j = index.get(appid), col.get(key)
        if i is None or j is None or role in (None, "", "mixed"):
            continue                      # only aspects that clearly stand out for the game
        M[i, j] = score
    return M, meta


def _lowercase_starts(notes):
    """Short points are shown lowercase ("great soundtrack · tight controls"),
    but some written ones start with a capital. Lowercase the first word only
    when other points already start with that word in lowercase and the next
    word isn't capitalised, so names like "Black Mesa ..." stay as they are."""
    keys = ("praise", "complaints", "mood")
    word = lambda s: s.split(" ", 1)[0].rstrip(",.;:!?'\"").lower()
    notes = list(notes)
    lower_firsts = {word(s) for n in notes for k in keys for s in n[k] if s[:1].islower()}
    for n in notes:
        for k in keys:
            fixed = []
            for s in n[k]:
                words = s.split(" ", 2)
                first = words[0]
                nxt = words[1] if len(words) > 1 else ""
                if (first[:1].isupper() and first[1:2].islower() and word(s) in lower_firsts
                        and not nxt[:1].isupper()):
                    s = first.lower() + s[len(first):]
                fixed.append(s)
            n[k] = fixed


def load_player_notes(con, appids):
    """What players say about each game, aligned with appids:
    - written descriptions (game_descriptions, from reading reviews), else
    - automatic aspect profile (game_profile, from analyze_reviews.py), else None."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    written, auto = {}, {}
    if "game_descriptions" in tables:
        for appid, summ, pr, co, bf, nf, mood in con.execute(
                "SELECT appid, summary, praise, complaints, best_for, not_for, mood FROM game_descriptions"):
            written[appid] = {"src": "written", "summary": summ, "praise": json.loads(pr or "[]"),
                              "complaints": json.loads(co or "[]"), "best_for": bf, "not_for": nf,
                              "mood": json.loads(mood or "[]")}
    if "game_profile" in tables:
        cols = {r[1] for r in con.execute("PRAGMA table_info(game_profile)")}
        mixed_col = "mixed" if "mixed" in cols else "NULL"
        for appid, n, pr, tr, co, mx in con.execute(
                f"SELECT appid, n_reviews, praise, traits, complaints, {mixed_col} FROM game_profile"):
            p, t, c, x = (json.loads(v or "[]") for v in (pr, tr, co, mx))
            if p or t or c or x:
                auto[appid] = {"src": "auto", "praise": p, "traits": t, "complaints": c, "mixed": x,
                               "n_reviews": n}
    _lowercase_starts(written.values())
    out = []
    for a in appids:
        note = written.get(a)
        if note is not None and a in auto:
            note = {**note, "traits": auto[a]["traits"]}     # keep the measured feel too
        out.append(note or auto.get(a))
    return out


def load_games(con, min_reviews):
    cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    header = "header_url" if "header_url" in cols else "NULL"   # older databases
    return con.execute(f"""SELECT appid,name,short_description,tags,review_count,percent_positive,
                                  release_date,is_free,platforms,developers,{header}
                           FROM games WHERE review_count >= ? AND COALESCE(coming_soon,0)=0
                                 AND name IS NOT NULL AND name != ''
                           ORDER BY appid""", (min_reviews,)).fetchall()


def build_interactions(con, index):
    """Positive (thumbs-up) reviews -> compact arrays (user, game, weight).
    Weight grows gently with playtime at review time: a 200h recommendation
    says more than a 20-minute one."""
    users: dict[int, int] = {}
    u, g, w = array("i"), array("i"), array("f")
    q = con.execute("SELECT steamid, appid, author_playtime_at_review FROM reviews "
                    "WHERE voted_up=1 AND steamid IS NOT NULL AND steamid != ''")
    while True:
        batch = q.fetchmany(200000)
        if not batch:
            break
        for sid, appid, pt in batch:
            gi = index.get(appid)
            if gi is None:
                continue
            try:
                key = int(sid)
            except ValueError:
                continue
            ui = users.get(key)
            if ui is None:
                ui = users[key] = len(users)
            u.append(ui); g.append(gi)
            w.append(1.0 + 0.25 * math.log1p((pt or 0) / 60))
    return len(users), np.frombuffer(u, np.int32), np.frombuffer(g, np.int32), \
        np.frombuffer(w, np.float32)


def item_item(u, g, w, n_games, n_users, topk=150, shrink=2.0, min_user_items=2,
              max_user_items=500, block=512):
    """Item-item cosine similarity with shrinkage, keeping top-k neighbours.
    Computed in blocks of games so the full games x games product never
    exists in memory."""
    if len(u) == 0 or n_users == 0:
        return sp.csr_matrix((n_games, n_games), dtype=np.float32), np.zeros(n_games, np.int32)
    X = sp.csr_matrix((w, (u, g)), shape=(n_users, n_games), dtype=np.float32)
    X.sum_duplicates()
    per_user = np.diff(X.indptr)
    # users with hundreds of likes: keep a deterministic random sample (cost is k^2 per user)
    heavy = np.where(per_user > max_user_items)[0]
    if len(heavy):
        rng = np.random.default_rng(0)
        keep = np.ones(X.nnz, bool)
        for r in heavy:
            s, e = X.indptr[r], X.indptr[r + 1]
            drop = rng.choice(e - s, size=(e - s) - max_user_items, replace=False)
            keep[s + drop] = False
        X.data[~keep] = 0
        X.eliminate_zeros()
        per_user = np.diff(X.indptr)
    # users who liked a single game add nothing to co-occurrence;
    # downweight users who like many games (inverse user frequency)
    uw = np.where(per_user >= min_user_items, 1 / np.log2(1 + np.maximum(per_user, 1)), 0)
    X = (sp.diags(uw.astype(np.float32)) @ X).tocsr()
    X.eliminate_zeros()
    support = np.diff(X.tocsc().indptr).astype(np.int32)   # players per game
    diag = np.asarray(X.multiply(X).sum(axis=0)).ravel()
    XT = X.T.tocsr()                                         # games x users
    rows, cols, vals = [], [], []
    for s in range(0, n_games, block):
        e = min(s + block, n_games)
        co = (XT[s:e] @ X).tocsr()                           # block x games
        co.sum_duplicates()
        for r in range(e - s):
            i = s + r
            a, b = co.indptr[r], co.indptr[r + 1]
            if a == b:
                continue
            idx, v = co.indices[a:b], co.data[a:b]
            m = idx != i
            idx, v = idx[m], v[m]
            if not len(v):
                continue
            norm = np.sqrt(diag[i] * diag[idx])
            norm[norm == 0] = 1
            sim = (v / norm) * (v / (v + shrink))            # shrink thin evidence
            if len(sim) > topk:
                keep = np.argpartition(-sim, topk)[:topk]
                idx, sim = idx[keep], sim[keep]
            rows.append(np.full(len(idx), i, np.int32)); cols.append(idx.astype(np.int32))
            vals.append(sim.astype(np.float32))
    if not rows:
        return sp.csr_matrix((n_games, n_games), dtype=np.float32), support
    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_games, n_games))
    return S, support


def build_text(con, games, reviews_per_game, max_vocab, per_game_terms=1500):
    """TF-IDF over description + English review text per game.
    Streams reviews game by game and uses hashed term ids, so memory stays
    bounded (~per_game_terms entries per game) at full scale."""
    n = len(games)
    df = np.zeros(1 << HASH_BITS, np.int32)
    g_ids, g_cnt, indptr = array("i"), array("f"), array("q", [0])

    def flush(i, counter):
        counter.update(term_hash(t) for t in tokenize(games[i][2]) for _ in (0, 1))  # desc x2
        items = counter.most_common(per_game_terms) if len(counter) > per_game_terms \
            else counter.items()
        for h, c in items:
            g_ids.append(h); g_cnt.append(c)
        indptr.append(len(g_ids))

    q = con.execute("""SELECT appid, review FROM (
                         SELECT appid, review, ROW_NUMBER() OVER (PARTITION BY appid
                                ORDER BY votes_up DESC, LENGTH(review) DESC) rn
                         FROM reviews WHERE language='english' AND LENGTH(review) > 40)
                       WHERE rn <= ? ORDER BY appid""", (reviews_per_game,))
    gi, cur = 0, Counter()
    while True:
        batch = q.fetchmany(20000)
        if not batch:
            break
        for appid, text in batch:
            while gi < n and games[gi][0] < appid:      # games before this one are done
                flush(gi, cur); cur = Counter(); gi += 1
            if gi < n and games[gi][0] == appid:
                cur.update(term_hash(t) for t in tokenize(text[:3000]))
    while gi < n:
        flush(gi, cur); cur = Counter(); gi += 1

    ids = np.frombuffer(g_ids, np.int32)
    cnt = np.frombuffer(g_cnt, np.float32)
    ptr = np.frombuffer(indptr, np.int64)
    np.add.at(df, ids, 1)
    ok = np.where((df >= 3) & (df <= 0.4 * n))[0]
    if len(ok) > max_vocab:
        ok = ok[np.argsort(-df[ok], kind="stable")[:max_vocab]]
    hashes = np.sort(ok).astype(np.int32)
    idf = (np.log((1 + n) / (1 + df[hashes])) + 1).astype(np.float32)
    col = np.searchsorted(hashes, ids)
    col = np.minimum(col, len(hashes) - 1) if len(hashes) else col
    valid = (hashes[col] == ids) if len(hashes) else np.zeros(len(ids), bool)
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
    vals = (1 + np.log(np.maximum(cnt, 1))) * (idf[col] if len(hashes) else 0)
    T = sp.csr_matrix((vals[valid].astype(np.float32), (rows[valid], col[valid])),
                      shape=(n, len(hashes)))
    return l2_rows(T), hashes, idf


def build_tags(games, tag_ids):
    tcol = {t: i for i, t in enumerate(tag_ids)}
    rows, cols, vals = [], [], []
    for i, g in enumerate(games):
        tags = json.loads(g[3] or "[]")
        if not tags:
            continue
        mx = max(t.get("weight", 1) for t in tags) or 1
        for rank, t in enumerate(tags):
            j = tcol.get(t["tagid"])
            if j is not None:
                # blend vote weight and rank so the top tags dominate
                rows.append(i); cols.append(j)
                vals.append(0.5 * t.get("weight", 1) / mx + 0.5 / (1 + rank * 0.3))
    M = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(games), len(tag_ids)))
    dfreq = np.asarray((M > 0).sum(axis=0)).ravel()
    idf = np.log((1 + len(games)) / (1 + dfreq)).astype(np.float32) + 1
    return l2_rows(M @ sp.diags(idf)), idf


def bayes_quality(review_count, pct, m=50):
    rc = np.asarray(review_count, np.float32)
    p = np.asarray(pct, np.float32) / 100
    prior = float((p * rc).sum() / max(rc.sum(), 1))
    return (p * rc + prior * m) / (rc + m)


def cmd_build(args):
    t0 = time.time()
    con = sqlite3.connect(args.db)
    games = load_games(con, args.min_reviews)
    if not games:
        sys.exit("no games in the database - run `python steam_reviews.py games` first")
    index = {g[0]: i for i, g in enumerate(games)}
    tag_rows = con.execute("SELECT tagid, name FROM tags ORDER BY tagid").fetchall()
    tag_ids = [t for t, _ in tag_rows]
    print(f"{len(games):,} games with >= {args.min_reviews} reviews")

    tagM, tag_idf = build_tags(games, tag_ids)
    print(f"tag vectors: {tagM.nnz:,} entries")
    textM, hashes, idf = build_text(con, games, args.reviews_per_game, args.max_vocab)
    print(f"text vectors: {len(hashes):,} terms, {textM.nnz:,} entries")
    n_users, u, g, w = build_interactions(con, index)
    S, support = item_item(u, g, w, len(games), n_users, topk=args.topk)
    print(f"co-likes: {len(u):,} positive reviews from {n_users:,} players, "
          f"{S.nnz:,} similarity links")

    quality = bayes_quality([x[4] or 0 for x in games], [x[5] or 0 for x in games])
    aspectM, aspect_meta = load_aspects(con, index)
    print(f"aspects: {len(aspect_meta)} aspects, {int((np.abs(aspectM).sum(axis=1) > 0).sum()):,} games with an aspect profile")
    build_id = f"{int(time.time())}-{os.getpid()}-{len(games)}"
    np.savez_compressed(
        args.model, build_id=np.array(build_id),
        tag_data=tagM.data, tag_indices=tagM.indices, tag_indptr=tagM.indptr,
        text_data=textM.data, text_indices=textM.indices, text_indptr=textM.indptr,
        cf_data=S.data, cf_indices=S.indices, cf_indptr=S.indptr,
        quality=quality.astype(np.float32), support=support.astype(np.int32),
        text_hashes=hashes, text_idf=idf, tag_idf=tag_idf,
        aspects=aspectM,
    )
    meta = {
        "built_at": int(time.time()), "build_id": build_id, "db": args.db, "n_games": len(games),
        "n_tags": len(tag_ids), "n_terms": int(len(hashes)), "n_players": n_users,
        "appids": [x[0] for x in games], "names": [x[1] for x in games],
        "descriptions": [(x[2] or "")[:300] for x in games],
        "review_count": [x[4] or 0 for x in games], "percent_positive": [x[5] for x in games],
        "release_date": [x[6] for x in games], "is_free": [x[7] for x in games],
        "platforms": [json.loads(x[8] or "{}") for x in games],
        "developers": [json.loads(x[9] or "[]") for x in games],
        "headers": [x[10] for x in games],
        "notes": load_player_notes(con, [x[0] for x in games]),
        "aspects": aspect_meta,
        "tag_ids": tag_ids, "tag_names": [nm for _, nm in tag_rows],
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"saved {args.model} + {args.meta} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9"}
# common gamer shorthand (first word of a query)
ALIASES = {"gta": "grand theft auto", "cs": "counter strike", "csgo": "counter strike 2",
           "cs2": "counter strike 2", "rdr": "red dead redemption", "rdr2": "red dead redemption 2",
           "ff": "final fantasy", "tes": "the elder scrolls", "dbd": "dead by daylight",
           "r6": "tom clancy s rainbow six", "botw": "the legend of zelda", "dst": "don t starve together",
           "civ": "sid meier s civilization", "bg3": "baldur s gate 3", "poe": "path of exile",
           "tf2": "team fortress 2", "l4d2": "left 4 dead 2", "hk": "hollow knight"}

NEG_WORDS = {"no", "not", "without", "avoid", "hate", "zero", "non", "never", "nothing",
             "less", "dislike", "skip", "minus", "dont", "don", "doesnt", "doesn", "didnt",
             "isnt", "nobody", "wont", "cant"}
# words skipped between a negation and the thing negated
# ("I don't want any heavy combat", "not too much grinding", "no real horror")
NEG_FILLER = {"t", "a", "an", "any", "the", "real", "really", "too", "much", "many", "more",
              "so", "very", "heavy", "lots", "lot", "of", "kind", "sort", "type", "types",
              "forced", "annoying", "stupid", "boring", "intense", "extreme", "overly",
              "excessive", "constant", "big", "huge", "like", "fan", "fond", "into", "anything",
              "stuff", "want", "need", "care", "for", "be", "is", "are", "have", "has", "enjoy",
              "love", "please", "to", "of", "elements", "mechanics", "mechanic", "games", "game"}
# words that end a negation's scope ("no combat AND lots of farming")
SCOPE_END = {"and", "but", "though", "however", "yet", "while", "with", "plus", "also",
             "just", "only", "instead", "rather", "that", "which", "i", "we", "it", "where"}


class Model:
    def __init__(self, model_path, meta_path):
        try:
            # read everything and close the file (an open file can't be replaced on Windows)
            with np.load(model_path) as f:
                z = {k: f[k] for k in f.files}
            # the web build (build_site.py) ships a gzipped meta file
            opener = gzip.open if str(meta_path).endswith(".gz") else open
            with opener(meta_path, "rt", encoding="utf-8") as f:
                self.meta = m = json.load(f)
        except FileNotFoundError:
            sys.exit("model not built yet - run `python recommend.py build`")
        if "text_hashes" not in z:
            sys.exit("model was built by an older version - run `python recommend.py build` again")
        bid = str(z["build_id"]) if "build_id" in z else None
        if bid and m.get("build_id") and bid != m["build_id"]:
            raise RuntimeError("model and meta files come from different builds (a swap is in progress?)")
        n = m["n_games"]
        # the web build (build_site.py) stores some arrays as float16 / uint16
        f32 = lambda a: np.asarray(a, np.float32)
        i32 = lambda a: np.asarray(a, np.int32)
        self.tag = sp.csr_matrix((f32(z["tag_data"]), i32(z["tag_indices"]), i32(z["tag_indptr"])),
                                 shape=(n, m["n_tags"]))
        self.text = sp.csr_matrix((f32(z["text_data"]), i32(z["text_indices"]), i32(z["text_indptr"])),
                                  shape=(n, m["n_terms"]))
        cf_scale = float(z["cf_scale"]) if "cf_scale" in z else 1.0
        self.cf = sp.csr_matrix((f32(z["cf_data"]) / np.float32(cf_scale), i32(z["cf_indices"]),
                                 i32(z["cf_indptr"])), shape=(n, n))
        self.quality, self.support = f32(z["quality"]), z["support"]
        self.text_hashes, self.text_idf, self.tag_idf = z["text_hashes"], z["text_idf"], z["tag_idf"]
        self.aspects = f32(z["aspects"]) if "aspects" in z else np.zeros((n, 0), np.float32)
        self.aspect_meta = m.get("aspects", [])
        an = np.linalg.norm(self.aspects, axis=1) if self.aspects.size else np.zeros(n)
        self.has_aspects = an > 0
        # thin profiles (one weak aspect) shouldn't count as much as rich ones:
        # divide by max(norm, median norm) instead of the norm itself
        med = float(np.median(an[an > 0])) if self.has_aspects.any() else 1.0
        self.aspects_unit = (self.aspects / np.maximum(an, med)[:, None]) if self.aspects.size else self.aspects
        self.aspects_pos = np.maximum(self.aspects_unit, 0)            # praise / notable traits
        self.aspects_neg = np.maximum(-self.aspects_unit, 0).sum(axis=1) if self.aspects.size else np.zeros(n)
        self.aspects_unit = self.aspects / np.where(an > 0, an, 1)[:, None] if self.aspects.size else self.aspects
        self.appids = m["appids"]
        self.names = m["names"]
        self.index = {a: i for i, a in enumerate(self.appids)}
        self.reviews = np.array(m["review_count"], np.float32)
        self.tag_col = {self._norm(nm): i for i, nm in enumerate(m["tag_names"])}
        self._norm_names = [self._norm(x) for x in self.names]
        self._tagT = None
        self.pct_positive = np.array([80 if p is None else p for p in m["percent_positive"]], np.float32)
        self.rating_factor = rating_factor(self.pct_positive)
        self.adult = self._adult_mask()
        self.is_demo = np.array([bool(DEMO_NAME.search(nm or "")) for nm in self.names])

    def _adult_mask(self):
        """Games players tag mainly as adult content: 'Hentai' among the top 10 tags
        or 'Sexual Content' as THE top tag (by votes). 'Nudity' / 'Sexual Content'
        further down are common on mainstream mature games (Cyberpunk 2077, GTA V,
        Stellar Blade), which stay."""
        n = len(self.appids)
        out = np.zeros(n, bool)
        names = self.meta["tag_names"]
        rules = [(names.index(t), k) for t, k in (("Hentai", 10), ("Sexual Content", 1)) if t in names]
        if not rules:
            return out
        T = self.tag
        cand = np.zeros(n, bool)
        for c, _ in rules:
            cand |= self.has_tag(c)
        for i in np.where(cand)[0]:
            s, e = T.indptr[i], T.indptr[i + 1]
            idx = T.indices[s:e]
            votes = T.data[s:e] / self.tag_idf[idx]            # undo idf -> relative vote weight
            order = idx[np.argsort(-votes)]
            out[i] = any(c in order[:k] for c, k in rules)
        return out

    def tag_cos(self, i, j):
        """Store-tag similarity of two games (rows are unit length)."""
        return float(self.tag[i].multiply(self.tag[j]).sum())

    @staticmethod
    def _norm(s):
        s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
        return " ".join(ROMAN.get(w, w) for w in s.split())

    def has_tag(self, c):
        if self._tagT is None:
            self._tagT = self.tag.tocsc()
        out = np.zeros(self.tag.shape[0], bool)
        out[self._tagT.indices[self._tagT.indptr[c]:self._tagT.indptr[c + 1]]] = True
        return out

    # ---- lookup ---------------------------------------------------------- #
    def find(self, query: str, strict=False):
        """Catalogue match for a game name (or appid). Returns (index, how) where
        how is 'exact', 'prefix', 'contains' or 'fuzzy'; (None, None) if none.
        strict=True only accepts exact / prefix / very close fuzzy matches."""
        if query.strip().isdigit() and int(query) in self.index:
            return self.index[int(query)], "exact"
        q = self._norm(query)
        if not q:
            return None, None
        first, _, rest_q = q.partition(" ")
        if first in ALIASES:                  # "gta v" -> "grand theft auto 5"
            q = (ALIASES[first] + " " + rest_q).strip()

        def pick(ids):
            # prefer names closest in length to the query; popularity breaks ties
            shortest = min(len(self._norm_names[i]) for i in ids)
            near = [i for i in ids if len(self._norm_names[i]) <= shortest + 4]
            return max(near, key=lambda i: self.reviews[i])

        exact = [i for i, nm in enumerate(self._norm_names) if nm == q]
        if exact:
            return max(exact, key=lambda i: self.reviews[i]), "exact"
        prefix = [i for i, nm in enumerate(self._norm_names) if nm.startswith(q + " ")]
        if prefix:
            return pick(prefix), "prefix"
        if not strict:
            contains = [i for i, nm in enumerate(self._norm_names) if f" {q} " in f" {nm} "]
            if contains:
                return pick(contains), "contains"
        cutoff = 0.9 if strict else 0.8
        close = difflib.get_close_matches(q, self._norm_names, n=10, cutoff=cutoff)
        if close:
            best = difflib.SequenceMatcher(None, q, close[0]).ratio()
            near = {c for c in close if difflib.SequenceMatcher(None, q, c).ratio() >= best - 0.02}
            ids = [i for i, nm in enumerate(self._norm_names) if nm in near]
            return max(ids, key=lambda i: self.reviews[i]), "fuzzy"
        return None, None

    def tag_cols(self, names):
        cols, missing = [], []
        for nm in names:
            if not nm.strip():
                continue
            c = self.tag_col.get(self._norm(nm))
            (cols if c is not None else missing).append(c if c is not None else nm)
        return cols, missing

    def _tag_at(self, words, start, length):
        phrase = " ".join(words[start:start + length])
        c = self.tag_col.get(phrase)
        if c is None and phrase.endswith("s"):          # 'puzzles' -> 'Puzzle'
            c = self.tag_col.get(phrase[:-1])
        return c

    # ---- wish parsing ---------------------------------------------------- #
    def parse_wish(self, text):
        """Split a free-text wish into wanted tags (weighted), unwanted tags and
        the remaining (positive) words. Handles negations like 'no combat',
        'no real combat', 'without multiplayer or pvp', 'not too much grinding'.
        A negation covers the words after it until a scope word ('and', 'but',
        'with', ...) or punctuation; negated words are never counted as wanted."""
        unwanted, positive = set(), []
        for clause in re.split(r"[,.;:!?()\n]+", (text or "").lower()):
            words = self._norm(clause).split()
            k = 0
            while k < len(words):
                if words[k] not in NEG_WORDS:
                    positive.append(words[k]); k += 1
                    continue
                # negation covers ONE thing (a tag of up to 3 words, or a single
                # unknown word) after optional filler words, plus further things
                # joined by "or"/"nor": "no real combat or pvp", "without zombies nor horror"
                j = k + 1
                while True:
                    while j < len(words) and words[j] in NEG_FILLER:
                        j += 1
                    if j >= len(words) or words[j] in SCOPE_END or words[j] in NEG_WORDS:
                        break
                    for length in (3, 2, 1):
                        if j + length <= len(words):
                            c = self._tag_at(words, j, length)
                            if c is not None:
                                unwanted.add(c); j += length
                                break
                    else:
                        j += 1                             # unknown thing ("microtransactions")
                    if j < len(words) and words[j] in ("or", "nor"):
                        j += 1
                        continue
                    break
                k = j                                      # negated words dropped entirely
        rest = " ".join(positive)
        wanted = {}
        qtoks = set(positive) - STOPWORDS
        padded = f" {rest} "
        for nm, j in self.tag_col.items():
            if j in unwanted:
                continue
            parts = [w for w in nm.split() if w not in STOPWORDS]
            if not parts:
                continue
            if f" {nm} " in padded:
                wanted[j] = 1.0                            # full tag name present
            else:
                frac = sum(w in qtoks or w + "s" in qtoks for w in parts) / len(parts)
                if frac >= 0.5 and len(parts) <= 3:
                    wanted[j] = max(wanted.get(j, 0), 0.5 * frac)
        return wanted, unwanted, rest

    # ---- signals --------------------------------------------------------- #
    def _rows_mean(self, M, idx):
        return np.asarray(M[idx].mean(axis=0)).ravel()

    def content_from_games(self, liked, disliked=()):
        def profile(M):
            v = self._rows_mean(M, liked)
            if disliked:
                v = v - 0.5 * self._rows_mean(M, list(disliked))
            v = np.maximum(v, 0)
            nv = np.linalg.norm(v)
            return v / nv if nv else v
        tp, xp = profile(self.tag), profile(self.text)
        base = 0.6 * (self.tag @ tp) + 0.4 * (self.text @ xp)
        ap = self.aspect_profile([(i, 1.0) for i in liked] + [(i, -1.0) for i in disliked])
        return self.with_aspects(base, ap), tp

    # ---- aspects (what players say: story, difficulty, mood, problems) ----
    # Aspects nudge the content score by how well what players say matches the
    # taste: x(1 + w*(shared praise - own complaints)). Paired hold-out test
    # (4,000 players, bootstrap 95% CI): w=0.2 -> +0.1 pts [-0.3, +0.5],
    # w=0.6 -> -0.5 pts [-1.1, +0.1]. So 0.2: neutral for accuracy, while it
    # keeps games players complain about from being boosted and powers the
    # "players praise its ..." explanations.
    ASPECT_WEIGHT = 0.2

    def aspect_profile(self, weighted, dislike_weight=0.5):
        """What the player likes in what-players-say terms: weighted mean of the
        PRAISE side of liked games minus dislike_weight x the same for disliked
        games (each side averaged by its own weights, like the tag profile).
        Liked games' complaints never become "wanted". -> unit vector or None."""
        if not self.aspects.size:
            return None
        A = self.aspects.shape[1]
        pos, neg = np.zeros(A, np.float32), np.zeros(A, np.float32)
        wp = wn = 0.0
        for i, w in weighted:
            if not self.has_aspects[i]:
                continue
            if w > 0:
                pos += w * self.aspects_pos[i]; wp += w
            elif w < 0:
                neg += -w * self.aspects_pos[i]; wn += -w
        v = (pos / wp if wp else pos) - dislike_weight * (neg / wn if wn else neg)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def with_aspects(self, content, ap):
        """Mix aspect similarity into a content score for games that have an
        aspect profile; others keep their plain content score."""
        if ap is None or not self.ASPECT_WEIGHT:
            return content
        sim = self.aspects_pos @ ap                        # shared praise / feel
        penalty = 0.5 * self.aspects_neg                   # its own complaints only ever lower it
        factor = np.clip(1 + self.ASPECT_WEIGHT * (sim - penalty), 0.5, 1.5)
        c = np.asarray(content, np.float32)
        return np.where(self.has_aspects, c * factor, c)

    def shared_aspects(self, i, ap, k=2):
        """Topics (story, soundtrack...) players PRAISE in game i that match the
        taste profile - qualities only, so the wording "players praise its ..."
        is always true."""
        if ap is None or not self.has_aspects[i]:
            return []
        contrib = self.aspects_pos[i] * ap
        out = []
        for j in np.argsort(-contrib):
            if len(out) >= k or contrib[j] <= 0.02:
                break
            meta = self.aspect_meta[j]
            if meta["kind"] == "quality" and self.aspects[i, j] > 0:
                out.append(meta.get("topic") or meta["label"].lower())
        return out

    def content_from_text(self, text, tag_boost=(), tag_penalty=()):
        wanted, unwanted, rest = self.parse_wish(text)
        unwanted |= set(tag_penalty)
        n = len(self.appids)
        qv = np.zeros(len(self.text_hashes), np.float32)
        if len(self.text_hashes):
            for t, c in Counter(tokenize(rest)).items():
                h = term_hash(t)
                j = np.searchsorted(self.text_hashes, h)
                if j < len(self.text_hashes) and self.text_hashes[j] == h:
                    qv[j] = (1 + math.log(c)) * self.text_idf[j]
        tv = np.zeros(self.tag.shape[1], np.float32)
        for j, wgt in wanted.items():
            tv[j] = wgt * self.tag_idf[j]
        for j in tag_boost:
            if j not in unwanted:
                tv[j] = max(tv[j], self.tag_idf[j] * 1.5)
        qn, tn = np.linalg.norm(qv), np.linalg.norm(tv)
        matched = bool(qn or tn)
        text_s = self.text @ (qv / qn) if qn else np.zeros(n, np.float32)
        tag_s = self.tag @ (tv / tn) if tn else np.zeros(n, np.float32)
        w_tag = 0.6 if tn else 0.0
        score = norm01(w_tag * tag_s + (1 - w_tag) * text_s)
        unwanted_names = [self.meta["tag_names"][c] for c in sorted(unwanted)]
        if unwanted:
            # penalise in proportion to how strongly players tagged the game with it
            pen = np.asarray(self.tag[:, sorted(unwanted)].max(axis=1).todense()).ravel()
            score = score - 2.0 * pen
        return score, (tv / tn if tn else tv), unwanted_names, matched

    def colikes(self, liked, disliked=()):
        """Summed co-like similarity, plus how many liked games have any links."""
        s = np.asarray(self.cf[liked].sum(axis=0)).ravel()
        if disliked:
            s -= 0.5 * np.asarray(self.cf[list(disliked)].sum(axis=0)).ravel()
        linked = int(sum(self.cf.indptr[i + 1] > self.cf.indptr[i] for i in liked))
        return np.maximum(s, 0), linked

    # ---- explanations ---------------------------------------------------- #
    def top_tags(self, i, k=5):
        row = self.tag[i]
        order = np.argsort(-row.data)[:k]
        return [self.meta["tag_names"][row.indices[j]] for j in order]

    def why(self, i, tag_profile, liked):
        parts = []
        if tag_profile is not None and np.any(tag_profile):
            row = self.tag[i]
            contrib = row.data * tag_profile[row.indices]
            shared = [self.meta["tag_names"][row.indices[j]]
                      for j in np.argsort(-contrib)[:3] if contrib[j] > 0]
            if shared:
                parts.append("shares: " + ", ".join(shared))
        if liked:
            # only claim "fans of X like it" when it's one of X's strongest co-like
            # links AND the games are alike - hits like Stardew are co-liked with
            # every other hit, which isn't a reason
            claims = []
            for l in liked:
                row = self.cf[l]
                hit = np.where(row.indices == i)[0]
                if not len(hit):
                    continue
                v = row.data[hit[0]]
                if int((row.data > v).sum()) < 25 and self.tag_cos(i, l) >= 0.25:
                    claims.append((float(v), l))
            if claims:
                parts.append(f"fans of {self.names[max(claims)[1]]} like it")
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# ranking (library API - used by the CLI below and by server.py)
# --------------------------------------------------------------------------- #
class RecError(Exception):
    """A user-facing problem (nothing matched, game not found, AI unavailable)."""


DEFAULT_OPTS = dict(n=10, gems=False, tags="", exclude_tags="", min_positive=0, min_reviews=50,
                    max_reviews=0, free=False, platform=None, year_from=0)


def make_opts(**kw):
    """Query options with defaults; unknown keys are ignored."""
    from types import SimpleNamespace
    o = dict(DEFAULT_OPTS)
    o.update({k: v for k, v in kw.items() if k in DEFAULT_OPTS and v is not None})
    return SimpleNamespace(**o)


def apply_filters(model: Model, opts, exclude=(), notes=None):
    """Boolean mask of games allowed by the filters. Never mutates opts."""
    m = model.meta
    ok = np.ones(len(model.appids), bool)
    ok[list(exclude)] = False
    ok &= ~model.is_demo
    wanted = {t.strip().lower() for t in (opts.tags.split(",") if isinstance(opts.tags, str) else opts.tags or [])}
    if not wanted & {t.lower() for t in ADULT_TAGS}:     # adult games only when asked for by tag
        ok &= ~model.adult
    ok &= model.reviews >= opts.min_reviews
    if opts.max_reviews:
        ok &= model.reviews <= opts.max_reviews
    min_pos = opts.min_positive or (80 if opts.gems else 0)   # gems should still be good games
    if min_pos:
        pp = np.array([p if p is not None else 0 for p in m["percent_positive"]])
        ok &= pp >= min_pos
    if opts.free:
        ok &= np.array([bool(f) for f in m["is_free"]])
    if opts.year_from:
        ts = calendar.timegm((int(opts.year_from), 1, 1, 0, 0, 0))
        ok &= np.array([(r or 0) >= ts for r in m["release_date"]])
    if opts.platform:
        p = opts.platform

        def has(pl):
            if p == "deck":           # 2 = playable, 3 = verified
                return pl.get("steam_deck_compat_category", 0) in (2, 3)
            if p == "linux":
                return bool(pl.get("steamos_linux") or pl.get("linux"))
            if p == "mac":
                return bool(pl.get("mac") or pl.get("macos"))
            return bool(pl.get(p))
        ok &= np.array([has(pl) for pl in m["platforms"]])
    for flag, keep in ((opts.tags, True), (opts.exclude_tags, False)):
        if not flag:
            continue
        cols, missing = model.tag_cols(flag.split(",") if isinstance(flag, str) else flag)
        if missing and notes is not None:
            notes.append(f"unknown tags ignored: {', '.join(missing)}")
        for c in cols:
            ok &= model.has_tag(c) if keep else ~model.has_tag(c)
    return ok


CF_CONFIDENT = 0.05   # avg co-like similarity at which we fully trust co-likes
RATING_KNEE = 75          # % positive below which a game starts to sink. eval: 80 cost -0.8 points of
                          # hit@20 [-1.2, -0.4]; 75 cost -0.2 [-0.4, +0.05], i.e. no clear difference


def rating_factor(pct, knee=RATING_KNEE):
    """1 at `knee`% positive and above, -0.04 per point below, never under 0.35."""
    return np.clip(1 - (knee - pct) / 25, 0.35, 1.0).astype(np.float32)


# free demos / prologues / playtests / soundtracks aren't games to recommend
DEMO_NAME = re.compile(r"\b(prologue|demo|playtest|soundtrack|ost)\b", re.I)
ADULT_TAGS = ("Hentai", "Sexual Content", "Nudity")


def blend(model: Model, content, cf, opts, cf_weight=0.7, n_linked=1, rating_penalty=True):
    """score = content + co-likes + quality, with an optional hidden-gem tilt.

    The co-like weight adapts to how much co-like evidence there is: with
    strong links it reaches `cf_weight` (co-likes were the strongest single
    signal in leave-one-out tests); for niche games with few shared players it
    shrinks towards 0 and the content match decides. Co-likes are gated by
    content relevance (sqrt), so they re-order similar games instead of
    pulling in unrelated blockbusters that "everyone" also plays."""
    c = norm01(content)
    f = np.clip(np.asarray(cf, np.float32) / max(1, n_linked), 0, 1)
    conf = min(1.0, float(f.max()) / CF_CONFIDENT) if f.max() > 0 else 0.0
    if f.max() > 0:
        f = f / f.max()
    w_cf = cf_weight * conf
    score = (0.9 - w_cf) * c + w_cf * f * np.sqrt(np.maximum(c, 0)) + 0.1 * model.quality
    if rating_penalty:                        # a 60%-positive game is rarely a good tip
        score = score * model.rating_factor
    if getattr(opts, "gems", False):
        # a gem is lesser-known AND well liked: tilt against popularity and
        # weight quality much more heavily than in the default blend
        pop = np.log10(1 + model.reviews) / np.log10(1 + model.reviews.max())
        score = (score + 0.4 * model.quality) * np.clip(1.25 - 1.0 * pop, 0.05, None)
    return score


def result_row(model, i, score, tag_profile, liked, reason=None):
    return {"appid": model.appids[i], "name": model.names[i], "score": round(float(score[i]), 4),
            "percent_positive": model.meta["percent_positive"][i],
            "reviews": int(model.reviews[i]), "tags": model.top_tags(i),
            "description": model.meta["descriptions"][i],
            "release_date": model.meta["release_date"][i],
            "why": model.why(i, tag_profile, list(liked)) if reason is None else reason,
            "ai": reason is not None,
            "players": (model.meta.get("notes") or [None] * len(model.appids))[i],
            "image": (model.meta.get("headers") or [None] * len(model.appids))[i]
            or f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{model.appids[i]}/header.jpg",
            "url": f"https://store.steampowered.com/app/{model.appids[i]}/"}


def describe_filters(opts):
    active = [f"{k.replace('_', ' ')}: {getattr(opts, k)}" for k in
              ("tags", "exclude_tags", "min_positive", "max_reviews", "platform", "year_from")
              if getattr(opts, k)]
    if opts.free:
        active.append("free")
    if opts.gems:
        active.append("hidden gems (min 80% positive)")
    return active


def no_match_error(opts):
    active = describe_filters(opts)
    return RecError("No games matched" + (" these filters - " + "; ".join(active) if active else "."))


def top_rows(model, score, ok, opts, tag_profile=None, liked=()):
    score = np.where(ok, score, -np.inf)
    n = max(1, min(int(opts.n), 100))
    top = [i for i in np.argsort(-score)[:n] if np.isfinite(score[i])]
    if not top:
        raise no_match_error(opts)
    return [result_row(model, i, score, tag_profile, liked) for i in top]


INTEREST_LINK = 0.35      # store-tag similarity that puts two liked games in one interest
MAX_INTERESTS = 3


def interest_groups(model, weighted, link=INTEREST_LINK, max_groups=MAX_INTERESTS):
    """Split liked games [(index, weight)] into up to `max_groups` interests by
    store-tag similarity: strongest game first; each game joins the interest it
    is most similar to (if some member is at least `link` alike), otherwise it
    starts a new one. Averaging everything into one taste lets the biggest
    interest swallow the rest (Stardew + Civilization -> only farming games)."""
    order = sorted(weighted, key=lambda x: -x[1])
    if not order:
        return []
    rows = [i for i, _ in order]
    S = (model.tag[rows] @ model.tag[rows].T).toarray()       # tag rows are unit length
    groups, members = [], []                                  # members: positions into `order`
    for p, (i, w) in enumerate(order):
        sims = [max(S[p, q] for q in g) for g in members]
        best = int(np.argmax(sims)) if sims else -1
        if best >= 0 and sims[best] >= link:
            groups[best].append((i, w)); members[best].append(p)
        elif len(groups) < max_groups:
            groups.append([(i, w)]); members.append([p])
        elif sims[best] >= link / 2:
            groups[best].append((i, w)); members[best].append(p)
        # else: unlike every interest - it still counts in the overall score
    return groups


def interleave(ranked, weights, n, accept=None):
    """Share n slots between ranked lists in proportion to their weights
    (D'Hondt: the next slot goes to the list with the highest weight/(taken+1)).
    Returns [(item, list_no)], skipping repeats and items `accept` rejects."""
    taken, ptr, out, seen = [0] * len(ranked), [0] * len(ranked), [], set()
    while len(out) < n:
        progressed = False
        for g in sorted(range(len(ranked)), key=lambda g: -weights[g] / (taken[g] + 1)):
            while ptr[g] < len(ranked[g]):
                i = ranked[g][ptr[g]]
                ptr[g] += 1
                if i in seen or (accept is not None and not accept(i)):
                    continue
                seen.add(i)
                out.append((i, g))
                taken[g] += 1
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            break
    return out


def multi_interest_rows(model, groups, scorer, ok, opts, overall=None, accept=None):
    """One ranking per interest, merged by `interleave`. scorer(games) ->
    (score, tag_profile). A little of the overall score is mixed into each
    interest so games that fit several of them rank well everywhere."""
    n = max(1, min(int(opts.n), 100))
    lists, profiles, scores, weights = [], [], [], []
    for g in groups:
        s, tp = scorer([i for i, _ in g])
        if overall is not None:
            s = 0.75 * norm01(s) + 0.25 * norm01(overall)
        s = np.where(ok, s, -np.inf)
        lists.append([i for i in np.argsort(-s)[: n * 8] if np.isfinite(s[i])])
        profiles.append(tp)
        scores.append(s)
        weights.append(max(1e-6, sum(w for _, w in g)))
    picked = interleave(lists, weights, n, accept)
    if not picked:
        raise no_match_error(opts)
    return [result_row(model, i, scores[g], profiles[g], [j for j, _ in groups[g]]) for i, g in picked]


def resolve(model, names, strict=False, notes=None):
    out = []
    for nm in names:
        i, how = model.find(str(nm), strict=strict)
        if i is None:
            if notes is not None:
                notes.append(f"couldn't find a game called '{nm}'")
        else:
            if how != "exact" and notes is not None:
                notes.append(f"'{nm}' matched {model.names[i]}")
            if i not in out:
                out.append(i)
    return out


def search_names(model, q, limit=8):
    """Autocomplete: games whose name starts with / contains the query,
    most-reviewed first."""
    q = model._norm(q)
    if not q:
        return []
    starts, contains = [], []
    for i, nm in enumerate(model._norm_names):
        if nm.startswith(q):
            starts.append(i)
        elif f" {q}" in f" {nm}":
            contains.append(i)
    ids = sorted(starts, key=lambda i: -model.reviews[i])[:limit]
    if len(ids) < limit:
        ids += sorted(contains, key=lambda i: -model.reviews[i])[:limit - len(ids)]
    return [{"appid": model.appids[i], "name": model.names[i], "reviews": int(model.reviews[i])}
            for i in ids]


def rec_like(model: Model, liked, disliked=(), opts=None):
    """liked/disliked: catalogue indices. Returns {"results", "notes"}."""
    opts = opts or make_opts()
    if not liked:
        raise RecError("Add at least one game you like.")
    notes = []
    ok = apply_filters(model, opts, exclude=list(liked) + list(disliked), notes=notes)

    def scorer(games):
        content, tp = model.content_from_games(games, disliked)
        cf, linked = model.colikes(games, disliked)
        return blend(model, content, cf, opts, n_linked=linked), tp
    overall, tp = scorer(list(liked))
    groups = interest_groups(model, [(i, 1.0) for i in liked])
    if len(groups) == 1:
        rows = top_rows(model, overall, ok, opts, tp, liked)
    else:
        rows = multi_interest_rows(model, groups, scorer, ok, opts, overall=overall)
    return {"results": rows, "notes": notes}


def rec_dream(model: Model, text, opts=None, exclude=()):
    opts = opts or make_opts()
    if not (text or "").strip():
        raise RecError("Describe the game you're dreaming of.")
    notes = []
    content, tp, unwanted, matched = model.content_from_text(text)
    if unwanted:
        notes.append("steering away from: " + ", ".join(unwanted))
    if not matched:
        notes.append("none of those words matched a tag or anything players write in reviews - "
                     "showing generally top-rated games; try different words or AI mode")
    ok = apply_filters(model, opts, exclude=exclude, notes=notes)
    rows = top_rows(model, blend(model, content, np.zeros_like(content), opts), ok, opts, tp)
    return {"results": rows, "notes": notes}


# --------------------------------------------------------------------------- #
# AI dream mode (Claude)
# --------------------------------------------------------------------------- #
AI_MODEL = "claude-opus-5-5"


def ai_available():
    """(ok, reason): whether the anthropic SDK is installed and a credential
    source looks configured. Only a real API call can say for sure."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "pip install anthropic"
    import os
    from pathlib import Path
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
            or os.environ.get("ANTHROPIC_PROFILE") \
            or (Path.home() / ".config" / "anthropic").exists():
        return True, ""
    return False, "set ANTHROPIC_API_KEY (or run `ant auth login`) before starting"


_CLIENT = None


def claude_client():
    global _CLIENT
    try:
        import anthropic
    except ImportError:
        raise RecError("AI mode needs the Anthropic SDK: pip install anthropic")
    if _CLIENT is None:
        try:
            _CLIENT = anthropic.Anthropic()
        except Exception as e:                        # no credentials configured at all
            raise RecError(f"Claude API not configured: {e}")
    return anthropic, _CLIENT


def ask_claude(anthropic, client, system, prompt, schema_cls, effort):
    try:
        resp = client.beta.messages.parse(
            model=AI_MODEL, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema_cls, output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise RecError("Claude API credentials missing/invalid - set ANTHROPIC_API_KEY or run `ant auth login`")
    except anthropic.RateLimitError:
        raise RecError("Claude API rate limit hit - try again in a minute")
    except anthropic.APIStatusError as e:
        raise RecError(f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise RecError("couldn't reach the Claude API - check your connection")
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    return resp.parsed_output


def rec_dream_ai(model: Model, text, opts=None, exclude=()):
    """1) Claude turns the free-text wish into structured search intent,
       2) the local model retrieves candidates,
       3) Claude re-ranks the candidates and explains each pick.
    Falls back to the local ranking whenever Claude's answer can't be used."""
    from pydantic import BaseModel, Field
    opts = opts or make_opts()
    if not (text or "").strip():
        raise RecError("Describe the game you're dreaming of.")

    class Intent(BaseModel):
        summary: str = Field(description="one-sentence restatement of what the player wants")
        include_tags: list[str] = Field(description="Steam tags that fit (from the given list)")
        must_have_tags: list[str] = Field(description="tags the game MUST have; usually empty")
        exclude_tags: list[str] = Field(description="tags the player explicitly does not want")
        similar_games: list[str] = Field(description="exact Steam titles of games that match the wish")
        search_text: str = Field(description="keywords a matching game's reviews would contain")
        hidden_gems: bool = Field(description="true if they want lesser-known games")

    class Pick(BaseModel):
        appid: int
        reason: str = Field(description="1-2 sentences, specific to this player's wish")

    class Ranking(BaseModel):
        picks: list[Pick]

    anthropic, client = claude_client()
    notes = []
    tag_list = ", ".join(sorted(model.meta["tag_names"]))
    intent = ask_claude(
        anthropic, client,
        "You turn a gamer's description of their dream game into a structured Steam search. "
        "Only use tag names from this exact list: " + tag_list,
        f"Player's wish:\n{text}", Intent, "low")
    if intent is None:
        out = rec_dream(model, text, opts, exclude)
        out["notes"].insert(0, "Claude couldn't interpret this wish - using the local matcher")
        return out
    notes.append(f"understood: {intent.summary}")

    boost, _ = model.tag_cols(intent.include_tags)
    avoid, _ = model.tag_cols(intent.exclude_tags)
    must, _ = model.tag_cols(intent.must_have_tags)
    content, tp, unwanted, _ = model.content_from_text(text + ". " + intent.search_text,
                                                       boost + must, avoid)
    seeds = resolve(model, intent.similar_games, strict=True)
    if seeds:
        notes.append("seeded from Claude's examples: " + ", ".join(model.names[i] for i in seeds))
        c2, tp2 = model.content_from_games(seeds)
        content = 0.6 * norm01(content) + 0.4 * norm01(c2)
        tp = tp + tp2
    cf, linked = model.colikes(seeds) if seeds else (np.zeros_like(content), 0)
    if unwanted:
        notes.append("steering away from: " + ", ".join(unwanted))
    o = make_opts(**vars(opts))
    o.gems = bool(opts.gems or intent.hidden_gems)
    if o.gems and not opts.gems:
        notes.append("looking for lesser-known games")

    def candidates(must_cols):
        o.tags = ",".join(filter(None, [opts.tags] + [model.meta["tag_names"][c] for c in must_cols]))
        ok = apply_filters(model, o, exclude=exclude)   # seeds stay eligible: they ARE good answers
        s = np.where(ok, blend(model, content, cf, o, cf_weight=0.25, n_linked=max(1, linked)), -np.inf)
        return s, [i for i in np.argsort(-s)[:40] if np.isfinite(s[i])]

    score, cand = candidates(must)
    if must and len(cand) < 10:
        notes.append("too few games have all of: " + ", ".join(model.meta["tag_names"][c] for c in must) +
                     " - treated them as preferences instead")
        score, cand = candidates([])
    if not cand:
        raise no_match_error(o)

    lines = [json.dumps({
        "appid": model.appids[i], "name": model.names[i],
        "tags": model.top_tags(i, 10), "percent_positive": model.meta["percent_positive"][i],
        "reviews": int(model.reviews[i]), "description": model.meta["descriptions"][i]},
        ensure_ascii=False) for i in cand]
    n = max(1, min(int(opts.n), 40))
    ranking = ask_claude(
        anthropic, client,
        "You are a game recommendation expert. From the candidate list only, pick the games "
        "that best fit the player's wish, best first, and say concretely why each one fits. "
        "Respect anything the player said they do not want. Never invent games.",
        f"Player's wish:\n{text}\n\nCandidates (JSON lines):\n" + "\n".join(lines) +
        f"\n\nReturn up to {n} picks.", Ranking, "medium")

    allowed = {model.appids[i]: i for i in cand}
    picks, seen = [], set()
    for p in (ranking.picks if ranking else []):
        i = allowed.get(p.appid)
        if i is not None and i not in seen:
            seen.add(i); picks.append((i, p.reason))
    if len(picks) < n:
        if not picks:
            notes.append("Claude's ranking couldn't be used - showing the local ranking")
        for i in cand:                          # top up from the local ranking
            if len(picks) >= n:
                break
            if i not in seen:
                seen.add(i); picks.append((i, None))
    rows = [result_row(model, i, score, tp, seeds, reason) for i, reason in picks[:n]]
    return {"results": rows, "notes": notes, "ai": True}


# --------------------------------------------------------------------------- #
# CLI wrappers
# --------------------------------------------------------------------------- #
def print_rows(rows, as_json):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    for r, row in enumerate(rows, 1):
        pct = f"{row['percent_positive']}%" if row["percent_positive"] is not None else "?"
        print(f"{r:>2}. {row['name']}  ({pct} of {row['reviews']:,} reviews)")
        print(f"    {', '.join(row['tags'])}")
        if row["why"]:
            print(f"    {'' if row.get('ai') else 'why: '}{row['why']}")
        print(f"    {row['url']}")


def _opts_from_args(args):
    return make_opts(**{k: getattr(args, k) for k in DEFAULT_OPTS if hasattr(args, k)})


def _run(fn, args, *a):
    try:
        out = fn(*a, _opts_from_args(args))
    except RecError as e:
        sys.exit(str(e))
    for note in out["notes"]:
        print(f"({note})", file=sys.stderr)
    print_rows(out["results"], args.json)


def cmd_like(args):
    model = Model(args.model, args.meta)
    notes = []
    liked = resolve(model, args.games, notes=notes)
    disliked = resolve(model, args.dislike or [], notes=notes)
    for note in notes:
        print(f"({note})", file=sys.stderr)
    if not liked:
        sys.exit("none of those games were found")
    print("based on: " + ", ".join(model.names[i] for i in liked) +
          (" | avoiding: " + ", ".join(model.names[i] for i in disliked) if disliked else ""),
          file=sys.stderr)
    _run(rec_like, args, model, liked, disliked)


def cmd_dream(args):
    model = Model(args.model, args.meta)
    _run(rec_dream_ai if args.ai else rec_dream, args, model, args.text)


def cmd_user(args):
    model = Model(args.model, args.meta)
    con = sqlite3.connect(args.db)
    rows = con.execute("SELECT appid, voted_up FROM reviews WHERE steamid=?", (args.steamid,)).fetchall()
    liked = [model.index[a] for a, up in rows if up and a in model.index]
    disliked = [model.index[a] for a, up in rows if not up and a in model.index]
    if not liked:
        sys.exit("that player has no positive reviews in the database")
    print("player liked: " + ", ".join(model.names[i] for i in liked[:10]) +
          (" ..." if len(liked) > 10 else ""), file=sys.stderr)
    _run(rec_like, args, model, liked, disliked)


# --------------------------------------------------------------------------- #
# evaluation: leave-one-out hit rate on real players
# --------------------------------------------------------------------------- #
def cmd_eval(args):
    model = Model(args.model, args.meta)
    con = sqlite3.connect(args.db)
    n_users, u, g, w = build_interactions(con, model.index)
    order = np.argsort(u, kind="stable")
    u, g, w = u[order], g[order], w[order]
    starts = np.searchsorted(u, np.arange(n_users + 1))
    counts = np.diff(starts)
    cand_users = np.where(counts >= 3)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(cand_users)
    cand_users = cand_users[: args.users]
    if not len(cand_users):
        sys.exit("no players with 3+ positive reviews of catalogue games yet - crawl more reviews")
    held = {}
    drop = np.zeros(len(u), bool)
    for x in cand_users:
        pos = starts[x] + rng.integers(counts[x])
        held[x] = int(g[pos])
        drop[pos] = True
    # rebuild co-likes WITHOUT the held-out reviews (no leakage)
    model.cf, _ = item_item(u[~drop], g[~drop], w[~drop], len(model.appids), n_users)
    pop = np.log1p(model.reviews)

    class Neutral:
        gems = False
    K = args.k
    knees = [75, 70, 65]
    names = ["popularity", "content only", "co-likes only", "hybrid (default)", "hybrid w/o aspects",
             "hybrid w/o rating penalty"] + [f"hybrid, sink below {k}%" for k in knees] + ["hybrid multi-interest"]
    knee_f = [rating_factor(model.pct_positive, k) for k in knees]
    rec_hits = {nm: [] for nm in names}       # per-player 0/1, for confidence intervals
    is_niche = []
    w_aspects = model.ASPECT_WEIGHT
    for x in cand_users:
        liked = sorted({int(v) for v in g[starts[x]:starts[x + 1]]} - {held[x]})
        if not liked:
            continue
        target = held[x]
        is_niche.append(model.reviews[target] < args.niche)
        content, _ = model.content_from_games(liked)
        model.ASPECT_WEIGHT = 0
        content_plain, _ = model.content_from_games(liked)
        model.ASPECT_WEIGHT = w_aspects
        cf, linked = model.colikes(liked)
        hyb = blend(model, content, cf, Neutral, n_linked=linked)
        variants = [pop.copy(), norm01(content) + 1e-3 * model.quality,
                    norm01(cf) + 1e-3 * pop, hyb.copy(),
                    blend(model, content_plain, cf, Neutral, n_linked=linked)]
        no_pen = blend(model, content, cf, Neutral, n_linked=linked, rating_penalty=False)
        variants += [no_pen.copy()] + [no_pen * f for f in knee_f]
        for name, s in zip(names, variants):
            s[liked] = -np.inf
            rec_hits[name].append(int(target in np.argpartition(-s, K)[:K]))
        groups = interest_groups(model, [(i, 1.0) for i in liked])
        if len(groups) > 1:                    # same merge as the quiz / "games I love"
            lists, wts = [], []
            for grp in groups:
                gl = [i for i, _ in grp]
                c_g, _ = model.content_from_games(gl)
                cf_g, lk = model.colikes(gl)
                s = 0.75 * norm01(blend(model, c_g, cf_g, Neutral, n_linked=lk)) + 0.25 * norm01(hyb)
                s[liked] = -np.inf
                lists.append(list(np.argsort(-s)[:K * 2]))
                wts.append(len(grp))
            top = [i for i, _ in interleave(lists, wts, K)]
            rec_hits["hybrid multi-interest"].append(int(target in top))
        else:
            rec_hits["hybrid multi-interest"].append(rec_hits["hybrid (default)"][-1])
    H = {nm: np.array(v) for nm, v in rec_hits.items()}
    niche = np.array(is_niche, bool)
    n = len(niche)
    rng_b = np.random.default_rng(7)
    boot = rng_b.integers(0, n, size=(1000, n))          # same resamples for every variant (paired)

    def ci(v):
        if not len(v):
            return "   n/a          "
        m = v[boot].mean(axis=1) if len(v) == n else None
        lo, hi = (np.percentile(m, [2.5, 97.5]) if m is not None else (np.nan, np.nan))
        return f"{v.mean():6.1%} [{lo:5.1%}-{hi:5.1%}]"
    print(f"leave-one-out over {n:,} players (each with 3+ liked games); "
          f"hit@{K} = the hidden game is in the top {K}; [95% bootstrap interval]")
    print(f"  {'':<26} {'all':>24}   niche (<{args.niche:,} reviews, n={int(niche.sum())})")
    for name in names:
        v = H[name]
        nv = v[niche]
        nm_str = f"{nv.mean():6.1%}" if len(nv) else "   n/a"
        print(f"  {name:<26} {ci(v):>24}   {nm_str}")
    pairs = [("aspects from reviews", "hybrid (default)", "hybrid w/o aspects"),
             (f"sinking games below {RATING_KNEE}%", "hybrid (default)", "hybrid w/o rating penalty")]
    pairs += [(f"sinking games below {k}%", f"hybrid, sink below {k}%", "hybrid w/o rating penalty")
              for k in knees]
    pairs += [("multi-interest merge", "hybrid multi-interest", "hybrid (default)")]
    for label, a, b in pairs:
        d = H[a] - H[b]                                # paired: same players, same resamples
        dm = d[boot].mean(axis=1)
        lo, hi = np.percentile(dm, [2.5, 97.5])
        print(f"  effect of {label}: {d.mean():+.2%} [{lo:+.2%} to {hi:+.2%}] "
              f"({'helps' if lo > 0 else 'hurts' if hi < 0 else 'no clear difference'})")
    print(f"  (random guess ≈ {K / len(model.appids):.2%})")
    print("note: players and games come from what has been crawled, which so far "
          "favours popular games - expect co-likes/popularity to look better than they will at full scale")


# --------------------------------------------------------------------------- #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="steam.db")
    p.add_argument("--model", default=MODEL_FILE)
    p.add_argument("--meta", default=META_FILE)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the recommender from steam.db")
    b.add_argument("--min-reviews", type=int, default=20,
                   help="only include games with at least this many reviews (default 20)")
    b.add_argument("--reviews-per-game", type=int, default=150,
                   help="most-helpful English reviews per game used for text (default 150)")
    b.add_argument("--max-vocab", type=int, default=100000)
    b.add_argument("--topk", type=int, default=150, help="neighbours kept per game")

    def filters(sp_):
        sp_.add_argument("-n", type=int, default=10, help="how many results (default 10)")
        sp_.add_argument("--gems", action="store_true",
                         help="favour lesser-known games (also sets --min-positive 80 unless given)")
        sp_.add_argument("--tags", default="", help="comma list: must have all these tags")
        sp_.add_argument("--exclude-tags", default="", help="comma list: none of these tags")
        sp_.add_argument("--min-positive", type=int, default=0, help="min %% positive reviews")
        sp_.add_argument("--min-reviews", type=int, default=50,
                         help="min review count of recommended games (default 50)")
        sp_.add_argument("--max-reviews", type=int, default=0, help="max review count")
        sp_.add_argument("--free", action="store_true", help="free-to-play only")
        sp_.add_argument("--platform", choices=["windows", "mac", "linux", "deck"])
        sp_.add_argument("--year-from", type=int, default=0)
        sp_.add_argument("--json", action="store_true", help="machine-readable output")

    l = sub.add_parser("like", help="games similar to ones you love")
    l.add_argument("games", nargs="+")
    l.add_argument("--dislike", nargs="*", help="games to steer away from")
    filters(l)
    d = sub.add_parser("dream", help="describe your dream game in words")
    d.add_argument("text")
    d.add_argument("--ai", action="store_true", help="let Claude interpret the wish and explain picks")
    filters(d)
    u = sub.add_parser("user", help="recommendations for a reviewer in the database")
    u.add_argument("steamid")
    filters(u)
    e = sub.add_parser("eval", help="leave-one-out quality check on real players")
    e.add_argument("--users", type=int, default=3000)
    e.add_argument("-k", type=int, default=20)
    e.add_argument("--niche", type=int, default=20000,
                   help="games with fewer reviews than this count as niche (default 20000)")

    args = p.parse_args()
    {"build": cmd_build, "like": cmd_like, "dream": cmd_dream,
     "user": cmd_user, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
