#!/usr/bin/env python3
"""
analyze_reviews.py - turn every collected review into a detailed profile of
each game: what players praise, what they complain about, and how it feels.

    python analyze_reviews.py                      # uses steam.db + aspects.json
    python analyze_reviews.py --workers 8          # CPU processes (default: all cores)
    python analyze_reviews.py --rescore            # recompute from stored counts (seconds)
    python analyze_reviews.py --show "Hades"       # print one game's profile

How it works
------------
aspects.json is a dictionary of ~65 specific aspects (gripping story,
punishing difficulty, relaxing, grindy, dead servers, ...) with precise phrase
patterns in several languages (English, Simplified Chinese, Russian,
Portuguese, Spanish, German, French), each tested against real reviews.

For every game and aspect we count, over all its reviews:
  hits      reviews mentioning the aspect
  hits_up   ... of which were thumbs-up reviews (and not a negative mention)
  neg_hits  reviews with a clearly negative mention ("the story is boring")
  n         reviews in the languages that aspect has patterns for
and compare the game's mention rate (hits / n, smoothed) with the average rate
across games (lift).

An aspect becomes part of the game's profile when players mention it clearly
more than usual (lift) and often enough (hits):
  quality (story, soundtrack, combat...)  -> praise, or complaint when the
                                             mentions are clearly negative
  trait   (hard, relaxing, short, scary)  -> describes the feel
  issue   (bugs, grind, cheaters...)      -> complaint

Results go to SQLite tables:
  aspects(key, kind, label, taste_label, area, global_share)
  game_aspects(appid, aspect, hits, hits_up, neg_hits, n, share, lift, score, role)
  game_profile(appid, n_reviews, praise, traits, complaints, description)
Re-run any time after crawling more reviews; it reads everything each time,
split across CPU cores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# review languages we have patterns for ("latam" is Latin-American Spanish)
LANGS = ["english", "schinese", "russian", "brazilian", "spanish", "latam", "german", "french"]
LANG_ALIAS = {"latam": "spanish"}

MIN_REVIEWS = 15     # games with fewer analysed reviews get no profile
MIN_TEXT = 20        # ignore reviews shorter than this (characters)


# --------------------------------------------------------------------------- #
# lexicon
# --------------------------------------------------------------------------- #
def _compiles(p):
    try:
        re.compile(p)
        return True
    except re.error:
        return False


def load_lexicon(path):
    """aspects.json -> list of aspect dicts (raw patterns, not compiled)."""
    with open(path, encoding="utf-8") as _f:
        data = json.load(_f)
    aspects = data["aspects"] if isinstance(data, dict) else data
    out, seen = [], set()
    for a in aspects:
        pos = {LANG_ALIAS.get(l, l): [p for p in ps if _compiles(p)] for l, ps in (a.get("patterns") or {}).items()}
        neg = {LANG_ALIAS.get(l, l): [p for p in ps if _compiles(p)] for l, ps in (a.get("neg_patterns") or {}).items()}
        pos = {l: ps for l, ps in pos.items() if ps}
        neg = {l: ps for l, ps in neg.items() if ps}
        if "english" not in pos or a["key"] in seen:
            continue
        seen.add(a["key"])
        out.append({"key": a["key"], "kind": a["kind"], "label": a["label"],
                    "taste_label": a.get("taste_label", a["label"]), "area": a.get("area", ""),
                    "patterns": pos, "neg_patterns": neg, "no_gate": bool(a.get("no_gate")),
                    "langs": sorted(pos)})          # languages this aspect can be found in
    return out


try:                                   # the regex parser (public name differs by version)
    import re._parser as _sre_parse    # Python 3.11+
except ImportError:                     # pragma: no cover
    import sre_parse as _sre_parse

_FREQ_SAMPLE = ""                       # lowercased sample text used to rank candidate words


def _freq(words):
    return sum(_FREQ_SAMPLE.count(w) + 1 for w in words)


def _required(items):
    """For a parsed regex sequence, return a set of literal strings such that
    EVERY match contains at least one of them (or None if we can't tell).
    Picks the rarest such set among the parts of the sequence."""
    P = _sre_parse
    LITERAL, SUBPATTERN, BRANCH = P.LITERAL, P.SUBPATTERN, P.BRANCH
    REPEATS = (P.MAX_REPEAT, P.MIN_REPEAT)
    POSS = getattr(P, "POSSESSIVE_REPEAT", None)
    cands, run = [], []

    def flush():
        s = "".join(run).strip().lower()
        if len(s) >= 3 or (s and any(ord(ch) > 0x2e80 for ch in s) and len(s) >= 1):
            cands.append({s})
        run.clear()
    for op, av in items:
        if op is LITERAL:
            run.append(chr(av))
            continue
        flush()
        if op is SUBPATTERN:
            r = _required(av[-1])
            if r:
                cands.append(r)
        elif op is BRANCH:
            subs = [_required(b) for b in av[1]]
            if subs and all(subs):
                cands.append(set().union(*subs))
        elif op in REPEATS or (POSS is not None and op is POSS):
            lo, _hi, sub = av
            if lo >= 1:
                r = _required(sub)
                if r:
                    cands.append(r)
        # AT (\b), IN ([...]), ANY (.), ASSERT/ASSERT_NOT (lookarounds) give no required text
    flush()
    if not cands:
        return None
    return min(cands, key=_freq)


def gate_words(patterns):
    """Cheap pre-filter: for each pattern, the rarest set of literal strings
    every match must contain (computed with Python's own regex parser). A
    review containing none of them can't match and is skipped without running
    the slow regexes. Returns None (= no gate) when some pattern has no
    required text. Gates are also verified empirically by --check-gates."""
    words = set()
    for p in patterns:
        try:
            parsed = _sre_parse.parse(p, re.IGNORECASE)
        except Exception:
            return None
        r = _required(list(parsed))
        if not r:
            return None
        words |= r
    return sorted(words)


def set_frequency_sample(texts):
    """Give the gate builder a sample of review text to rank words by rarity."""
    global _FREQ_SAMPLE
    _FREQ_SAMPLE = "\n".join(t.lower() for t in texts)


def compute_gates(aspects, sample_texts=()):
    """{aspect key: {language: gate words or None}} - computed once, then sent
    to the worker processes."""
    if sample_texts:
        set_frequency_sample(sample_texts)
    return {a["key"]: {l: (None if a.get("no_gate") else
                           gate_words(a["patterns"].get(l, []) + a["neg_patterns"].get(l, [])))
                       for l in set(a["patterns"]) | set(a["neg_patterns"])}
            for a in aspects}


def compile_aspects(aspects, use_gates=True, gates=None):
    """Attach compiled regexes (one alternation per aspect and language) and
    the word gates (precomputed `gates`, or computed here)."""
    for a in aspects:
        a["pos"] = {l: re.compile("|".join(f"(?:{p})" for p in ps), re.IGNORECASE)
                    for l, ps in a["patterns"].items()}
        a["neg"] = {l: re.compile("|".join(f"(?:{p})" for p in ps), re.IGNORECASE)
                    for l, ps in a["neg_patterns"].items()}
        a["gate"] = {}
        if use_gates and not a.get("no_gate"):
            if gates is not None:
                a["gate"] = dict(gates.get(a["key"], {}))
            else:
                for l in set(a["patterns"]) | set(a["neg_patterns"]):
                    a["gate"][l] = gate_words(a["patterns"].get(l, []) + a["neg_patterns"].get(l, []))
    return aspects


# --------------------------------------------------------------------------- #
# scanning (runs in worker processes)
# --------------------------------------------------------------------------- #
# review templates: "☐ Infinite replayability" lines are unticked boxes, not opinions
_UNCHECKED_LINE = re.compile(
    r"^[ \t]*(?:[-*•][ \t]*)?(?:☐|□|⬜|🔲|▢|◻|◽|❌|✖|✗|\[ ?\]|\( ?\))[^\n]*$", re.MULTILINE)
# a negator directly governing an issue mention cancels it ("no bugs at all",
# "never had a crash") - at most 2 words in between, and never across a clause
# boundary ("never crashed on launch, but it crashes every hour" still counts)
_NEGATOR_EN = re.compile(
    r"\b(?:no|not|never|without|zero|barely|hardly|rarely|haven'?t|hasn'?t|didn'?t|don'?t|doesn'?t|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|free of|lack of)\b(?:\W+\w+){0,2}\W*$", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[,;:.!?\n()]|\b(?:but|though|although|however|yet|except|and then|because|if)\b",
                           re.IGNORECASE)


def _negated(text, start):
    prefix = text[max(0, start - 40):start]
    cut = None
    for cut in _CLAUSE_BREAK.finditer(prefix):
        pass
    if cut is not None:
        prefix = prefix[cut.end():]
    return _NEGATOR_EN.search(prefix) is not None


def _issue_hit(rx, text):
    """True if any match of an English issue pattern is not negated."""
    return any(not _negated(text, m.start()) for m in rx.finditer(text))


def _clean(text):
    return _UNCHECKED_LINE.sub(" ", text)


def _scan_chunk(job):
    """Count aspect mentions per game AND language for appids in [lo, hi].
    Returns (n[g][lang], up[g][lang], counts[g][lang] -> int32 array (3, A)
    with rows hits / hits_up / neg_hits, reviews scanned)."""
    db_path, lexicon, lo, hi = job[:4]
    gates = job[4] if len(job) > 4 else None
    # the lexicon arrives already parsed (same as the main process uses), or as a path
    raw = [dict(a) for a in lexicon] if isinstance(lexicon, list) else load_lexicon(lexicon)
    aspects = compile_aspects(raw, use_gates=gates is not False,
                              gates=gates if isinstance(gates, dict) else None)
    A = len(aspects)
    by_lang = defaultdict(list)
    for j, a in enumerate(aspects):
        for lang in set(a["pos"]) | set(a["neg"]):
            by_lang[lang].append((j, a["gate"].get(lang), a["pos"].get(lang), a["neg"].get(lang),
                                  a["kind"] == "issue" and lang == "english"))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    rows = con.execute(
        f"SELECT appid, language, review, voted_up FROM reviews WHERE appid BETWEEN ? AND ? "
        f"AND language IN ({','.join('?' * len(LANGS))}) AND LENGTH(review) >= ?",
        [lo, hi, *LANGS, MIN_TEXT])
    n = defaultdict(lambda: defaultdict(int))
    up = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(dict)
    scanned = 0
    for appid, lang, text, voted_up in rows:
        lang = LANG_ALIAS.get(lang, lang)
        scanned += 1
        n[appid][lang] += 1
        if voted_up:
            up[appid][lang] += 1
        text = _clean(text)
        low = text.lower()
        for j, gate, rx, nrx, check_neg in by_lang.get(lang, ()):
            if gate is not None and not any(w in low for w in gate):
                continue
            if rx is None:
                hit = False
            elif check_neg:
                hit = _issue_hit(rx, text)           # "no bugs", "never crashed once" don't count
            else:
                hit = rx.search(text) is not None
            nhit = nrx is not None and nrx.search(text) is not None
            if not hit and not nhit:
                continue
            c = counts[appid].get(lang)
            if c is None:
                c = counts[appid][lang] = np.zeros((3, A), np.int32)
            c[0, j] += 1
            if voted_up and not nhit:
                c[1, j] += 1
            if nhit:
                c[2, j] += 1
    con.close()
    return ({g: dict(v) for g, v in n.items()}, {g: dict(v) for g, v in up.items()},
            dict(counts), scanned)


def _sample_texts(con, n=4000):
    return [t for (t,) in con.execute(
        f"SELECT review FROM reviews WHERE language IN ({','.join('?' * len(LANGS))}) "
        f"AND LENGTH(review) >= ? ORDER BY RANDOM() LIMIT ?", [*LANGS, MIN_TEXT, n])]


def _chunks(con, n_chunks, langs):
    """Split games into contiguous appid ranges with similar review counts."""
    counts = con.execute(
        f"SELECT appid, COUNT(*) FROM reviews WHERE language IN ({','.join('?' * len(langs))}) "
        f"GROUP BY appid ORDER BY appid", langs).fetchall()
    total = sum(c for _, c in counts)
    target = max(1, total // max(1, n_chunks))
    out, lo, acc = [], None, 0
    for appid, c in counts:
        if lo is None:
            lo = appid
        acc += c
        if acc >= target:
            out.append((lo, appid)); lo, acc = None, 0
    if lo is not None:
        out.append((lo, counts[-1][0]))
    return out, total


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS aspects (
    key TEXT PRIMARY KEY, kind TEXT, label TEXT, taste_label TEXT, area TEXT,
    global_share REAL
);
CREATE TABLE IF NOT EXISTS game_aspects (
    appid INTEGER, aspect TEXT, hits INTEGER, hits_up INTEGER, neg_hits INTEGER,
    n INTEGER, share REAL, lift REAL, score REAL, role TEXT,
    expected REAL, p_value REAL,
    PRIMARY KEY (appid, aspect)
);
CREATE TABLE IF NOT EXISTS game_profile (
    appid INTEGER PRIMARY KEY, n_reviews INTEGER, praise TEXT, traits TEXT,
    complaints TEXT, mixed TEXT, description TEXT, updated_at INTEGER
);
-- raw counts at scan time, so --rescore never mixes old hits with new totals
CREATE TABLE IF NOT EXISTS scan_counts (
    appid INTEGER, language TEXT, n INTEGER, up INTEGER, PRIMARY KEY (appid, language)
);
CREATE TABLE IF NOT EXISTS scan_hits (
    appid INTEGER, language TEXT, aspect TEXT, hits INTEGER, hits_up INTEGER, neg_hits INTEGER,
    PRIMARY KEY (appid, language, aspect)
);
"""

# what a quality is "about" - used for "Weak story", "Players split on story"
TOPIC = {
    "story": "story", "characters": "characters", "writing": "writing", "lore": "lore & worldbuilding",
    "ending": "ending", "combat": "combat", "boss_fights": "boss fights", "tight_controls": "controls",
    "strategic_depth": "strategic depth", "replayability": "replayability", "exploration": "exploration",
    "creative_freedom": "building & creativity", "puzzles": "puzzles", "progression": "progression",
    "beautiful_visuals": "visuals", "soundtrack": "soundtrack", "voice_acting": "voice acting",
    "atmospheric": "atmosphere", "sound_design": "sound design", "solo_friendly": "solo play",
    "helpful_community": "community", "mod_support": "mod support", "great_value": "value for money",
    "runs_well": "performance", "steam_deck": "Steam Deck support", "active_devs": "developer support",
}
# opposite aspects: if both stand out, the clearly stronger wins, otherwise players are split
OPPOSITES = [("runs_well", "poor_performance", "performance"),
             ("too_easy", "challenging", "difficulty"), ("too_easy", "punishing", "difficulty"),
             ("short", "lots_of_content", "length"), ("great_value", "overpriced", "value for money"),
             ("helpful_community", "toxic_players", "community"), ("active_devs", "abandoned", "developer support"),
             ("tight_controls", "clunky_controls", "controls"), ("solo_friendly", "dead_servers", "player count")]

FDR_Q = 0.10           # Benjamini-Hochberg: at most ~10% of the shown aspects are chance
MIN_RATIO = 1.5        # ... and at least this many times the expected mentions
MIN_HITS = 3


def _wilson_lower(k, n, z=1.64):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def classify(kind, h, hu, nh, notable, base_up):
    """Role of one aspect for one game: '', 'praise', 'trait', 'complaint' or 'mixed'."""
    if not notable:
        return ""
    down = h - hu - nh                         # plain mentions inside thumbs-down reviews
    if kind == "issue":
        # a real problem shows up in negative reviews, not only in jokes in positive ones
        neg_mentions = h - hu
        if neg_mentions >= max(2, math.ceil(0.25 * h)) or neg_mentions / h >= (1 - base_up) + 0.15                 or h >= 12:
            return "complaint"
        return ""
    if kind == "trait":
        return "trait"
    # quality: needs positive evidence to be praise, clear negative evidence to be a complaint
    if (nh >= 3 and nh / h >= 0.4) or (h - hu >= 8 and hu / h < 0.35):
        return "complaint"
    # praise also needs explicit negative mentions ("no workshop support") to be rare
    if hu >= 3 and _wilson_lower(hu, h) >= min(base_up, 0.5) - 0.15 and nh <= max(1, 0.1 * h):
        return "praise"
    # "players split" is a claim too: it needs explicit negative mentions, or many
    # mentions inside thumbs-down reviews (those often still praise the aspect)
    if nh >= 2 or (h - hu >= 8 and hu / h < 0.5):
        return "mixed"
    return ""


def _label(a, role, topic_override=None):
    topic = topic_override or TOPIC.get(a["key"], a["label"].lower())
    if role == "complaint" and a["kind"] == "quality":
        return f"Weak {topic}"
    if role == "mixed":
        return f"Players split on {topic}"
    return a["label"]


def score_and_store(con, aspects, n, up, counts, t0):
    """n/up: appid -> {lang: reviews / thumbs-up}; counts: appid -> {lang: (3, A) array}.
    Expected mentions per game use the per-language base rate of each aspect,
    so languages with looser or stricter patterns don't skew the result."""
    from scipy.stats import poisson
    A = len(aspects)
    keys = [a["key"] for a in aspects]
    langs = sorted({l for d in n.values() for l in d})
    L = {l: k for k, l in enumerate(langs)}
    has_lang = np.zeros((len(langs), A), bool)
    for j, a in enumerate(aspects):
        for l in a["langs"]:
            if l in L:
                has_lang[L[l], j] = True
    games = [g for g, d in n.items() if sum(d.values()) >= MIN_REVIEWS]
    if not games:
        sys.exit("not enough reviews to analyse - crawl more first")

    # per-language base rates (games weighted equally, shrunk towards the pooled rate)
    acc = np.zeros((len(langs), A)); cnt = np.zeros(len(langs))
    pool_h = np.zeros(A); pool_n = np.zeros(A)
    for g in games:
        for l, nl in n[g].items():
            c = counts.get(g, {}).get(l)
            h = c[0] if c is not None else np.zeros(A)
            if nl >= 10:
                acc[L[l]] += h / nl; cnt[L[l]] += 1
            pool_h += h * has_lang[L[l]]; pool_n += nl * has_lang[L[l]]
    pooled = np.divide(pool_h, pool_n, out=np.full(A, 1e-4), where=pool_n > 0)
    K = 20.0
    rate = (acc + K * pooled) / (cnt[:, None] + K)
    rate = np.where(has_lang, np.maximum(rate, 1e-5), 0.0)

    # pass 1: expected mentions and Poisson p-values for every game x aspect
    stats = {}
    cand_p = []
    for g in games:
        nl = np.array([n[g].get(l, 0) for l in langs], np.float64)
        tot = np.zeros((3, A), np.int64)
        for l, c in counts.get(g, {}).items():
            tot += c
        expected = nl @ rate
        p = poisson.sf(tot[0] - 1, np.maximum(expected, 1e-9))         # P(X >= h); 1 when h == 0
        tested = (nl @ has_lang) > 0                                   # every pair we could observe
        cand = (tot[0] >= MIN_HITS) & (tot[0] >= MIN_RATIO * expected)
        cand_p.append(p[tested])
        stats[g] = (nl, tot, expected, p, cand)
    # Benjamini-Hochberg over the FULL family of tests (not just the ones that
    # already look interesting - choosing the family from the data would break
    # the false-discovery guarantee); the hit/ratio rules apply afterwards
    allp = np.sort(np.concatenate(cand_p)) if cand_p else np.zeros(0)
    m = len(allp)
    ok = allp <= FDR_Q * np.arange(1, m + 1) / max(1, m)
    p_thr = float(allp[np.nonzero(ok)[0].max()]) if ok.any() else 0.0
    print(f"{m:,} game x aspect tests; FDR {FDR_Q:.0%} threshold p <= {p_thr:.2g}")

    rows, profiles = [], []
    now = int(time.time())
    for g in games:
        nl, tot, expected, p, cand = stats[g]
        n_total = int(nl.sum())
        base_up = sum(up.get(g, {}).values()) / max(1, n_total)
        h, hu, nh = tot
        nj = nl @ has_lang                                     # reviews in languages with patterns
        if not h.any():
            profiles.append((g, n_total, "[]", "[]", "[]", "[]", "", now))
            continue
        ratio = (h + 1) / (expected + 1)
        notable = cand & (p <= p_thr)
        roles = {}
        for j, a in enumerate(aspects):
            if h[j] and nj[j]:
                roles[j] = classify(a["kind"], int(h[j]), int(hu[j]), int(nh[j]), bool(notable[j]), base_up)
        topic_of = {}
        idx = {k: j for j, k in enumerate(keys)}
        for x, y, topic in OPPOSITES:                           # resolve contradictions
            jx, jy = idx.get(x), idx.get(y)
            if jx is None or jy is None or not roles.get(jx) or not roles.get(jy):
                continue
            pos_x = roles[jx] != "complaint"
            pos_y = roles[jy] != "complaint"
            if pos_x == pos_y:
                continue                                        # they agree (e.g. both complaints)
            if ratio[jx] >= 2 * ratio[jy]:
                roles[jy] = ""
            elif ratio[jy] >= 2 * ratio[jx]:
                roles[jx] = ""
            else:
                roles[jx], roles[jy] = "mixed", ""
                topic_of[jx] = topic
        praise, traits, complaints, mixed = [], [], [], []
        for j, role in roles.items():
            a = aspects[j]
            strength = math.log2(max(ratio[j], 1e-6)) * min(1.0, h[j] / 10)
            score = {"praise": strength, "trait": strength, "complaint": -abs(strength)}.get(role, 0.0)
            rows.append((g, a["key"], int(h[j]), int(hu[j]), int(nh[j]), int(nj[j]),
                         float(h[j] / max(1, nj[j])), float(ratio[j]), float(score), role,
                         float(expected[j]), float(p[j])))
            label = _label(a, role, topic_of.get(j))
            {"praise": praise, "trait": traits, "complaint": complaints, "mixed": mixed}.get(
                role, []).append((strength, label))
        top = lambda xs, k: [l for _, l in sorted(xs, reverse=True)[:k]]
        pr, tr, co, mx = top(praise, 4), top(traits, 3), top(complaints, 3), top(mixed, 2)
        parts = []
        if pr:
            parts.append("Players love: " + ", ".join(pr))
        if tr:
            parts.append("Feels: " + ", ".join(tr))
        if co:
            parts.append("Watch out: " + ", ".join(co))
        if mx:
            parts.append(" / ".join(mx))
        profiles.append((g, n_total, json.dumps(pr), json.dumps(tr), json.dumps(co), json.dumps(mx),
                         " · ".join(parts), now))

    gs = np.divide(pool_h, pool_n, out=np.zeros(A), where=pool_n > 0)
    _replace_tables(con, [
        ("aspects", "INSERT INTO aspects VALUES (?,?,?,?,?,?)",
         [(a["key"], a["kind"], a["label"], a["taste_label"], a["area"], float(gs[j]))
          for j, a in enumerate(aspects)]),
        ("game_aspects", "INSERT INTO game_aspects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows),
        ("game_profile", "INSERT INTO game_profile VALUES (?,?,?,?,?,?,?,?)", profiles),
    ])
    with_profile = sum(1 for pr in profiles if pr[6])
    roles_count = defaultdict(int)
    for r in rows:
        roles_count[r[9] or "-"] += 1
    print(f"profiles for {len(games):,} games ({with_profile:,} with at least one notable aspect); "
          f"roles {dict(roles_count)} - {time.time() - t0:.0f}s total")
    return len(games), with_profile


def _table_sql():
    """CREATE statements from SCHEMA, by table name."""
    out = {}
    for stmt in SCHEMA.split(";"):
        s = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()
        if s.upper().startswith("CREATE TABLE"):
            out[s.split()[5] if s.split()[2].upper() == "IF" else s.split()[2]] = s
    return out


def _replace_tables(con, tables):
    """Replace the contents of several tables in ONE transaction: readers see
    either the old data or the new data, never empty tables. Tables whose
    columns changed (older schema) are recreated inside the same transaction."""
    ddl = _table_sql()
    old_iso = con.isolation_level
    con.isolation_level = None                       # we manage the transaction
    try:
        con.execute("BEGIN IMMEDIATE")
        for name, insert_sql, rows in tables:
            want = ddl[name]
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
            n_vals = insert_sql.count("?")
            if cols and len(cols) != n_vals:
                con.execute(f"DROP TABLE {name}")
                cols = []
            if not cols:
                con.execute(want)
            else:
                con.execute(f"DELETE FROM {name}")
            con.executemany(insert_sql, rows)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.isolation_level = old_iso


def _store_scan(con, aspects, n, up, counts):
    keys = [a["key"] for a in aspects]
    _replace_tables(con, [
        ("scan_counts", "INSERT INTO scan_counts VALUES (?,?,?,?)",
         [(g, l, int(c), int(up.get(g, {}).get(l, 0))) for g, d in n.items() for l, c in d.items()]),
        ("scan_hits", "INSERT INTO scan_hits VALUES (?,?,?,?,?,?)",
         [(g, l, keys[j], int(c[0, j]), int(c[1, j]), int(c[2, j]))
          for g, d in counts.items() for l, c in d.items() for j in np.nonzero(c[0])[0]]),
    ])


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def analyze(db_path, lexicon_path, workers=None):
    t0 = time.time()
    aspects = load_lexicon(lexicon_path)
    print(f"{len(aspects)} aspects loaded")
    workers = workers or max(1, (os.cpu_count() or 2))
    con = sqlite3.connect(db_path, timeout=60)
    chunks, total = _chunks(con, workers * 4, LANGS)     # several chunks per worker balances load
    gates = compute_gates(aspects, _sample_texts(con))
    print(f"scanning ~{total:,} reviews in {len(chunks)} chunks on {workers} processes...")
    n, up, counts = {}, {}, {}
    done = 0
    jobs = [(os.path.abspath(db_path), aspects, lo, hi, gates) for lo, hi in chunks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_scan_chunk, j) for j in jobs]):
            a, b, c, scanned = fut.result()
            n.update(a); up.update(b); counts.update(c)
            done += scanned
            print(f"  {done:,} reviews ({time.time() - t0:.0f}s)", file=sys.stderr, end="\r")
    print(f"\nscanned {done:,} reviews of {len(n):,} games in {time.time() - t0:.0f}s")
    _store_scan(con, aspects, n, up, counts)
    return score_and_store(con, aspects, n, up, counts, t0)


def rescore(db_path, lexicon_path):
    """Recompute scores/roles/profiles from the counts saved by the last scan
    (no re-scan, consistent even while a crawler adds reviews). Use after
    changing thresholds or labels; pattern changes need a full run."""
    t0 = time.time()
    con = sqlite3.connect(db_path, timeout=60)
    aspects = load_lexicon(lexicon_path)
    col = {a["key"]: j for j, a in enumerate(aspects)}
    if not con.execute("SELECT name FROM sqlite_master WHERE name='scan_counts'").fetchone():
        sys.exit("no saved scan counts - run a full analysis first")
    A = len(aspects)
    n, up, counts = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for g, l, c, u in con.execute("SELECT appid, language, n, up FROM scan_counts"):
        n[g][l], up[g][l] = c, u
    if not n:
        sys.exit("no saved scan counts - run a full analysis first")
    for g, l, key, h, hu, nh in con.execute("SELECT appid, language, aspect, hits, hits_up, neg_hits FROM scan_hits"):
        j = col.get(key)
        if j is None:
            continue
        c = counts[g].get(l)
        if c is None:
            c = counts[g][l] = np.zeros((3, A), np.int32)
        c[:, j] = (h, hu, nh)
    return score_and_store(con, aspects, dict(n), dict(up), dict(counts), t0)


def check_gates(db_path, lexicon_path, sample=60000):
    """Verify the word gates never change results: scan a sample of reviews
    with and without gates and compare counts per aspect. Aspects whose
    counts differ get "no_gate": true written into the lexicon."""
    con = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=60)
    reviews = con.execute(
        f"SELECT language, review FROM reviews WHERE language IN "
        f"({','.join('?' * len(LANGS))}) AND LENGTH(review) >= ? ORDER BY RANDOM() LIMIT ?",
        [*LANGS, MIN_TEXT, sample]).fetchall()           # fixed random sample, held in memory
    lex = load_lexicon(lexicon_path)
    for a in lex:
        a["no_gate"] = False                # re-check every aspect, ignoring earlier verdicts
    gates = compute_gates(lex, _sample_texts(con))
    aspects = compile_aspects(lex, use_gates=True, gates=gates)
    missed = defaultdict(int)
    times = {}
    for gated in (False, True):
        t0 = time.time()
        for lang, text in reviews:
            lang = LANG_ALIAS.get(lang, lang)
            low = text.lower()
            for a in aspects:
                rx, nrx = a["pos"].get(lang), a["neg"].get(lang)
                if rx is None and nrx is None:
                    continue
                gate = a["gate"].get(lang)
                if gated and gate is not None and not any(w in low for w in gate):
                    if not gated:
                        continue
                    # would the regex have matched? (only computed in the gated pass to find misses)
                    if (rx is not None and rx.search(text)) or (nrx is not None and nrx.search(text)):
                        missed[a["key"]] += 1
                    continue
                if rx is not None:
                    rx.search(text)
                if nrx is not None:
                    nrx.search(text)
        times[gated] = time.time() - t0
    bad = sorted(missed)
    t0 = time.time()                        # pure gated cost
    for lang, text in reviews:
        lang = LANG_ALIAS.get(lang, lang)
        low = text.lower()
        for a in aspects:
            rx = a["pos"].get(lang)
            if rx is None:
                continue
            g = a["gate"].get(lang)
            if g is not None and not any(w in low for w in g):
                continue
            rx.search(text)
    gated_time = time.time() - t0
    print(f"sample of {len(reviews):,} reviews: ungated {times[False]:.0f}s, gated {gated_time:.0f}s; "
          f"aspects where the gate would drop a real match: {dict(missed) or 'none'}")
    with open(lexicon_path, encoding="utf-8") as _f:
        data = json.load(_f)
    for a in (data["aspects"] if isinstance(data, dict) else data):
        a.pop("no_gate", None)
        if a["key"] in bad:
            a["no_gate"] = True
    with open(lexicon_path, "w", encoding="utf-8") as _f:
        json.dump(data, _f, ensure_ascii=False, indent=1)
    if bad:
        print(f"marked {len(bad)} aspects as no_gate in {lexicon_path}")
    return bad


def show(db_path, name):
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT g.appid, g.name, p.n_reviews, p.description FROM games g "
                      "JOIN game_profile p ON p.appid=g.appid WHERE g.name LIKE ? "
                      "ORDER BY g.review_count DESC LIMIT 1", (f"%{name}%",)).fetchone()
    if not row:
        sys.exit("no profile for that game (not enough reviews analysed?)")
    appid, nm, n, desc = row
    print(f"{nm}  ({n} reviews analysed)\n  {desc or '(nothing stands out)'}")
    for key, hits, lift, role, score in con.execute(
            "SELECT aspect, hits, lift, role, score FROM game_aspects WHERE appid=? ORDER BY ABS(score) DESC LIMIT 12",
            (appid,)):
        print(f"    {key:<26} {hits:>4} mentions  x{lift:4.1f}  {role or '-':<9} {score:+.2f}")
    if con.execute("SELECT name FROM sqlite_master WHERE name='game_descriptions'").fetchone():
        d = con.execute("SELECT summary, praise, complaints, best_for FROM game_descriptions WHERE appid=?",
                        (appid,)).fetchone()
        if d:
            print(f"  written description: {d[0]}\n    praise: {d[1]}\n    complaints: {d[2]}\n    best for: {d[3]}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="steam.db")
    p.add_argument("--lexicon", default="aspects.json")
    p.add_argument("--workers", type=int, default=0, help="processes (default: all CPU cores)")
    p.add_argument("--show", help="print the profile of a game (name)")
    p.add_argument("--rescore", action="store_true",
                   help="recompute profiles from stored counts without re-reading reviews")
    p.add_argument("--check-gates", action="store_true",
                   help="verify the fast word pre-filter gives identical counts (and disable it where not)")
    args = p.parse_args()
    if args.show:
        show(args.db, args.show)
    elif args.check_gates:
        check_gates(args.db, args.lexicon)
    elif args.rescore:
        rescore(args.db, args.lexicon)
    else:
        analyze(args.db, args.lexicon, args.workers or None)


if __name__ == "__main__":
    main()
