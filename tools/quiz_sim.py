#!/usr/bin/env python3
"""
quiz_sim.py - real Steam reviewers play the taste quiz end to end, to measure
whether "knew it / never heard of it" answers improve the recommendations and
to tune quiz.FAMILIARITY_TASTE / FAMILIARITY_GENRE / RESURFACE_UNKNOWN.

    python tools/quiz_sim.py                          # 1500 players, all variants, 6 workers
    python tools/quiz_sim.py --players 30 --verify    # smoke test + leakage / rebuild checks
    python tools/quiz_sim.py --variants base,taste.3  # a subset (base is always included)
    python tools/quiz_sim.py --variant mine=0.3,0,3   # an extra combo: TASTE,GENRE,RESURFACE

Players: reviewers with 10+ reviews of catalogue games (a seeded sample). One
random thumbs-up game with 50+ reviews (so it is recommendable) is hidden; they
need 3+ other thumbs-up games. Co-likes are rebuilt without the hidden reviews,
exactly as `recommend.py eval` does (no leakage). Then each player answers every
round from their reviews:
  * the hidden game: "never heard of it" (they haven't found it yet)
  * games they reviewed: ranked - thumbs-up first by playtime at review (ties:
    total playtime), then thumbs-down games, least played last. If every game
    they know in a round was a thumbs-down: "I don't like any of them"
  * games they never reviewed: "never heard of it"
  (the "no interest" / "interested" boxes are never used - reviews can't say)
After quiz.ROUNDS rounds, quiz.results(n=20); hit = the hidden game is in it.

Every variant plays the same players with the same seeds, so differences are
paired (same players, same bootstrap resamples, as in cmd_eval).
Caveat: reviewed != known. Real players know many games they never reviewed,
so "never heard of it" is far broader here than in real use.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import quiz  # noqa: E402
import recommend as rec  # noqa: E402

# name: (FAMILIARITY_TASTE, FAMILIARITY_GENRE, RESURFACE_UNKNOWN)
VARIANTS = {
    "base": (0.0, 0.0, 0),            # the old quiz: knowing a game is not evidence
    "taste.15": (0.15, 0.0, 0),
    "taste.3": (0.3, 0.0, 0),
    "taste.6": (0.6, 0.0, 0),
    "genre.5": (0.0, 0.5, 0),
    "resurf3": (0.0, 0.0, 3),
    "all-default": (0.3, 0.5, 3),
}
SPLIT_KNOWN = 5          # players who knew < 5 of the 35 quiz games vs >= 5 (in the base run)


# --------------------------------------------------------------------------- #
# players
# --------------------------------------------------------------------------- #
def interactions(con, index):
    """recommend.build_interactions, line for line, also returning the
    steamid -> user-row map (needed to drop the held-out reviews)."""
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
    return (len(users), np.frombuffer(u, np.int32), np.frombuffer(g, np.int32),
            np.frombuffer(w, np.float32), users)


def sample_players(con, model, n_players, seed, min_reviews=10, min_target_reviews=50, min_other_likes=3,
                   min_wellknown=0):
    """Seeded sample of reviewers with `min_reviews`+ reviews of catalogue games,
    each with one hidden thumbs-up game (50+ reviews) and 3+ other thumbs-ups.
    min_wellknown: also require that many reviews of quiz-eligible big hits /
    popular games (tiers 0-1: what the quiz shows first)."""
    index = model.index
    qi = quiz.index_for(model)
    wellknown = qi.eligible & (qi.tier >= 0) & (qi.tier <= 1)
    sid, app, up, pa, pf = array("q"), array("q"), array("b"), array("q"), array("q")
    q = con.execute("SELECT steamid, appid, voted_up, author_playtime_at_review, author_playtime_forever "
                    "FROM reviews WHERE steamid IS NOT NULL AND steamid != ''")
    while True:
        batch = q.fetchmany(200000)
        if not batch:
            break
        for s, a, v, x, y in batch:
            if a not in index:
                continue
            try:
                k = int(s)
                sid.append(k)
            except (ValueError, OverflowError):
                continue
            app.append(a); up.append(1 if v == 1 else 0); pa.append(x or 0); pf.append(y or 0)
    sid, app = np.frombuffer(sid, np.int64), np.frombuffer(app, np.int64)
    up, pa, pf = np.frombuffer(up, np.int8), np.frombuffer(pa, np.int64), np.frombuffer(pf, np.int64)
    order = np.lexsort((app, sid))
    sid, app, up, pa, pf = sid[order], app[order], up[order], pa[order], pf[order]
    first = np.ones(len(sid), bool)                       # one review per (player, game)
    first[1:] = (sid[1:] != sid[:-1]) | (app[1:] != app[:-1])
    sid, app, up, pa, pf = sid[first], app[first], up[first], pa[first], pf[first]
    users, starts, counts = np.unique(sid, return_index=True, return_counts=True)
    cand = np.where(counts >= min_reviews)[0]
    if min_wellknown:
        wk = wellknown[np.array([index[int(a)] for a in app])].astype(np.int64)
        wk_per = np.add.reduceat(wk, starts) if len(wk) else np.zeros(0, np.int64)
        cand = cand[wk_per[cand] >= min_wellknown]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    players, skipped = [], 0
    for x in cand:
        if len(players) >= n_players:
            break
        s, e = starts[x], starts[x] + counts[x]
        ups = [j for j in range(s, e) if up[j] == 1]
        targets = [j for j in ups if model.reviews[index[int(app[j])]] >= min_target_reviews]
        if len(ups) - 1 < min_other_likes or not targets:
            skipped += 1
            continue
        t = targets[int(rng.integers(len(targets)))]
        players.append({"p": len(players), "sid": int(users[x]), "target": int(app[t]),
                        "seed": int(rng.integers(2 ** 31)),
                        "reviews": {int(app[j]): (int(up[j]), int(pa[j]), int(pf[j])) for j in range(s, e)}})
    return players, {"reviews": len(sid), "reviewers": len(users), "candidates": len(cand), "skipped": skipped}


# --------------------------------------------------------------------------- #
# worker: one model per process, co-likes rebuilt once in the parent
# --------------------------------------------------------------------------- #
_M = None


def _init(model_path, meta_path, cf_path):
    global _M
    m = rec.Model(model_path, meta_path)
    m.cf = sp.load_npz(cf_path).tocsr()                   # without the held-out reviews
    quiz.index_for(m)                                     # cached on the model; does not use cf
    _M = m


def answer(shown, player):
    """One round's answers from the player's reviews (see the module doc)."""
    target, rv = player["target"], player["reviews"]
    known = [a for a in shown if a != target and a in rv]
    unknown = [a for a in shown if a not in known]
    by_play = lambda a: (-rv[a][1], -rv[a][2], a)
    ups = sorted((a for a in known if rv[a][0]), key=by_play)
    downs = sorted((a for a in known if not rv[a][0]), key=by_play)     # least played last
    if known and not ups:
        return {"shown": shown, "order": [], "unknown": unknown, "none_liked": True}
    return {"shown": shown, "order": ups + downs, "unknown": unknown}


def play(player, variants, k=20):
    m = _M
    target = player["target"]
    out = {"p": player["p"], "sid": player["sid"], "target": target, "v": {}}
    # reference: "games I love" with all their other thumbs-up games (what the quiz tries to learn)
    liked = sorted(m.index[a] for a, (u, _, _) in player["reviews"].items() if u and a != target)
    try:
        ids = [r["appid"] for r in rec.rec_like(m, liked, opts=rec.make_opts(n=k))["results"]]
        out["ref_hit"] = int(target in ids)
    except rec.RecError:
        out["ref_hit"] = 0
    opts = rec.make_opts(n=k)
    for name, (ft, fg, ru) in variants:
        quiz.FAMILIARITY_TASTE, quiz.FAMILIARITY_GENRE, quiz.RESURFACE_UNKNOWN = ft, fg, ru
        r = {"hit": 0, "rank": -1, "known": 0, "shown_target": 0, "from_quiz_hit": 0,
             "n_from_quiz": 0, "n_rows": 0, "none_rounds": 0, "err": ""}
        hist = []
        try:
            for _ in range(quiz.ROUNDS):
                rd = quiz.next_round(m, hist, player["seed"])
                ans = answer([c["appid"] for c in rd["games"]], player)
                hist.append(ans)
                r["known"] += len(ans["shown"]) - len(ans["unknown"])
                r["shown_target"] |= int(target in ans["shown"])
                r["none_rounds"] += int(bool(ans.get("none_liked")))
            rows = quiz.results(m, hist, opts)["results"]
            ids = [x["appid"] for x in rows]
            r["n_rows"] = len(rows)
            r["n_from_quiz"] = sum(1 for x in rows if x.get("from_quiz"))
            if target in ids:
                r["hit"], r["rank"] = 1, ids.index(target) + 1
                r["from_quiz_hit"] = int(bool(rows[ids.index(target)].get("from_quiz")))
        except Exception as e:                            # noqa: BLE001 - record, keep the run going
            r["err"] = f"{type(e).__name__}: {e}"[:200]
        out["v"][name] = r
    return out


def _task(args):
    return play(*args)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def verdict(lo, hi):
    return "helps" if lo > 0 else "hurts" if hi < 0 else "no clear difference"


def report(recs, variants, k, population, n_boot=1000):
    names = [v for v, _ in variants]
    n = len(recs)
    V = lambda name, key: np.array([r["v"][name][key] for r in recs], float)
    H = {nm: V(nm, "hit") for nm in names}
    known_base = V("base", "known")
    groups = [("all", np.ones(n, bool)), (f"knew <{SPLIT_KNOWN}", known_base < SPLIT_KNOWN),
              (f"knew >={SPLIT_KNOWN}", known_base >= SPLIT_KNOWN)]
    rng_b = np.random.default_rng(7)
    boots = {g: rng_b.integers(0, int(mask.sum()), size=(n_boot, int(mask.sum()))) if mask.any() else None
             for g, mask in groups}

    def ci(v, g):
        b = boots[g]
        if b is None or not len(v):
            return float("nan"), float("nan"), float("nan")
        m = v[b].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        return v.mean(), lo, hi

    L = []
    L.append(f"{n:,} players ({population}, hidden thumbs-up game with 50+ reviews), "
             f"{quiz.ROUNDS} rounds x {quiz.PER_ROUND} games; hit@{k} = hidden game in the top {k}; "
             f"[95% bootstrap interval, {n_boot} resamples, paired]")
    ref = np.array([r["ref_hit"] for r in recs], float)
    m_, lo, hi = ci(ref, "all")
    L.append(f"reference - 'games I love' fed ALL their other thumbs-up games: {m_:6.1%} [{lo:5.1%}-{hi:5.1%}]")
    L.append("")
    L.append(f"  {'variant':<13} {'T/G/R':<12} {'hit@' + str(k):>8} {'95% CI':>15}   {'vs base':>7} "
             f"{'95% CI':>17}  {'verdict':<20} {'shown':>6} {'resurf':>6} {'known':>5} {'unk':>5} {'err':>4}")
    for nm, (ft, fg, ru) in variants:
        v = H[nm]
        m_, lo, hi = ci(v, "all")
        d = v - H["base"]
        dm, dlo, dhi = ci(d, "all")
        vs = (f"{dm:+7.2%} [{dlo:+6.2%} to {dhi:+6.2%}]  {verdict(dlo, dhi):<20}" if nm != "base"
              else f"{'':>7} {'':>17}  {'':<20}")
        known = V(nm, "known").mean()
        L.append(f"  {nm:<13} {f'{ft:g}/{fg:g}/{ru}':<12} {m_:8.1%} [{lo:5.1%}-{hi:5.1%}]   {vs} "
                 f"{V(nm, 'shown_target').mean():6.1%} {V(nm, 'from_quiz_hit').mean():6.1%} "
                 f"{known:5.1f} {quiz.ROUNDS * quiz.PER_ROUND - known:5.1f} "
                 f"{sum(1 for r in recs if r['v'][nm]['err']):4d}")
    L.append("  shown = hidden game appeared in the quiz (answered 'never heard'); resurf = hits that came "
             "back as a 'new to you in the quiz' result; known / unk = average answers out of 35")
    L.append("")
    L.append(f"split by how many of the 35 quiz games they knew in the base run (paired difference vs base):")
    hdr = "".join(f"  {g + ' (n=' + format(int(mask.sum()), ',') + ')':<44}" for g, mask in groups[1:])
    L.append(f"  {'variant':<13}{hdr}")
    for nm, _ in variants:
        cells = []
        for g, mask in groups[1:]:
            if not mask.any():
                cells.append(f"  {'n/a':<44}")
                continue
            m_ = H[nm][mask].mean()
            if nm == "base":
                cells.append(f"  {m_:6.1%}{'':<38}")
                continue
            dm, dlo, dhi = ci(H[nm][mask] - H["base"][mask], g)
            cells.append(f"  {m_:6.1%} {dm:+6.2%} [{dlo:+6.2%},{dhi:+6.2%}] {verdict(dlo, dhi)[:12]:<12}")
        L.append(f"  {nm:<13}" + "".join(cells))
    # sanity numbers
    kb = known_base
    L.append("")
    L.append(f"base run: known answers per player mean {kb.mean():.1f} / median {np.median(kb):.0f} of 35 "
             f"(quartiles {np.percentile(kb, 25):.0f}-{np.percentile(kb, 75):.0f}; knew none: {(kb == 0).mean():.1%}); "
             f"'never heard' answers mean {35 - kb.mean():.1f}; 'like none' rounds {V('base', 'none_rounds').mean():.2f}"
             f"/player; results per player {V('base', 'n_rows').mean():.1f}")
    for nm, (_, _, ru) in variants:
        if ru:
            L.append(f"{nm}: 'new to you in the quiz' rows per player {V(nm, 'n_from_quiz').mean():.2f}")
    errs = [(nm, r["v"][nm]["err"]) for r in recs for nm in names if r["v"][nm]["err"]]
    if errs:
        L.append(f"errors: {len(errs)} (e.g. {errs[0][0]}: {errs[0][1]})")
    return L


# --------------------------------------------------------------------------- #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(ROOT, "steam.db"))
    ap.add_argument("--model", default=os.path.join(ROOT, rec.MODEL_FILE))
    ap.add_argument("--meta", default=os.path.join(ROOT, rec.META_FILE))
    ap.add_argument("--players", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-reviews", type=int, default=10, help="reviews of catalogue games a player needs")
    ap.add_argument("--min-wellknown", type=int, default=0,
                    help="also need this many reviews of quiz-eligible big hits / popular games (tiers 0-1). "
                         "steam.db holds only ~600 reviews per big hit, so most reviewers have none")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list of names (base always runs)")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=T,G,R",
                    help="extra combo: FAMILIARITY_TASTE,FAMILIARITY_GENRE,RESURFACE_UNKNOWN")
    ap.add_argument("--log", default=os.path.join(HERE, "quiz_sim.log"), help="table is appended here")
    ap.add_argument("--records", default="", help="write per-player results as JSON here")
    ap.add_argument("--verify", action="store_true",
                    help="also check the interactions match recommend.build_interactions and that "
                         "co-likes rebuilt without dropping anything match the stored model")
    args = ap.parse_args()

    table = dict(VARIANTS)
    for spec in args.variant:
        nm, _, vals = spec.partition("=")
        ft, fg, ru = vals.split(",")
        table[nm] = (float(ft), float(fg), int(ru))
    names = ["base"] + [v for v in args.variants.split(",") if v and v != "base"]
    names += [s.partition("=")[0] for s in args.variant if s.partition("=")[0] not in names]
    unknown = [v for v in names if v not in table]
    if unknown:
        sys.exit(f"unknown variants: {unknown} (known: {', '.join(table)})")
    variants = [(v, table[v]) for v in names]

    t0 = time.time()
    model = rec.Model(args.model, args.meta)
    con = sqlite3.connect(args.db)
    players, info = sample_players(con, model, args.players, args.seed, min_reviews=args.min_reviews,
                                   min_wellknown=args.min_wellknown)
    population = f"{args.min_reviews}+ reviews of catalogue games" + (
        f", {args.min_wellknown}+ of them of quiz big hits / popular games" if args.min_wellknown else "")
    print(f"[{time.time() - t0:5.0f}s] {info['reviews']:,} catalogue reviews from {info['reviewers']:,} reviewers; "
          f"{info['candidates']:,} with {population}; sampled {len(players):,} "
          f"(skipped {info['skipped']} without a usable hidden game)", flush=True)

    n_users, u, g, w, users = interactions(con, model.index)
    if args.verify:
        n2, u2, g2, w2 = rec.build_interactions(con, model.index)
        same = n2 == n_users and np.array_equal(u, u2) and np.array_equal(g, g2) and np.array_equal(w, w2)
        print(f"check: interactions identical to recommend.build_interactions: {same}", flush=True)
        if not same:
            sys.exit("interactions differ from recommend.build_interactions")
    order = np.argsort(u, kind="stable")                  # as in cmd_eval
    u, g, w = u[order], g[order], w[order]
    n = len(model.appids)
    key = u.astype(np.int64) * n + g
    held = np.array([users[p["sid"]] * n + model.index[p["target"]] for p in players], np.int64)
    drop = np.isin(key, held)
    if drop.sum() < len(players):
        sys.exit(f"only {int(drop.sum())} held-out reviews found for {len(players)} players")
    print(f"[{time.time() - t0:5.0f}s] {len(u):,} positive reviews, dropping {int(drop.sum())} held-out; "
          f"rebuilding co-likes...", flush=True)
    cf, _ = rec.item_item(u[~drop], g[~drop], w[~drop], n, n_users)
    if args.verify:
        full, _ = rec.item_item(u, g, w, n, n_users)
        diff = abs(full - model.cf)
        print(f"check: co-likes rebuilt from all reviews vs stored model: nnz {full.nnz:,} vs {model.cf.nnz:,}, "
              f"max |diff| {diff.max() if diff.nnz else 0:.2e}", flush=True)
        changed = abs(full - cf)
        rows = sorted({model.index[p["target"]] for p in players})
        print(f"check: dropping the held-out reviews changed {changed.nnz:,} co-like entries; "
              f"target rows changed: {sum(1 for i in rows if changed[i].nnz)}/{len(rows)}", flush=True)
        qi = quiz.index_for(model)
        before = (qi.eligible.copy(), qi.tier.copy(), qi.family.copy())
        model.cf = cf
        model._quiz_index = None
        qi2 = quiz.index_for(model)
        print("check: quiz index independent of co-likes: "
              f"{all(np.array_equal(a, b) for a, b in zip(before, (qi2.eligible, qi2.tier, qi2.family)))}",
              flush=True)
    del model
    tmp = tempfile.mkdtemp(prefix="quiz_sim_")
    cf_path = os.path.join(tmp, "cf.npz")
    sp.save_npz(cf_path, cf.tocsr(), compressed=False)
    t1 = time.time()
    print(f"[{t1 - t0:5.0f}s] co-likes rebuilt ({cf.nnz:,} links); playing {len(players):,} players x "
          f"{len(variants)} variants on {args.workers} workers: {', '.join(names)}", flush=True)

    recs = []
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init, initargs=(args.model, args.meta, cf_path)) as pool:
            tasks = [(p, variants, args.k) for p in players]
            step = max(1, len(players) // 20)
            for r in pool.imap_unordered(_task, tasks, chunksize=2):
                recs.append(r)
                if len(recs) % step == 0 or len(recs) == len(players):
                    el = time.time() - t1
                    hits = {nm: np.mean([x["v"][nm]["hit"] for x in recs]) for nm in names}
                    print(f"[{time.time() - t0:5.0f}s] {len(recs):,}/{len(players):,} players, "
                          f"eta {el / len(recs) * (len(players) - len(recs)):.0f}s; hit so far: "
                          + " ".join(f"{nm} {h:.1%}" for nm, h in hits.items()), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    recs.sort(key=lambda r: r["p"])
    play_s = time.time() - t1
    lines = report(recs, variants, args.k, population)
    lines.append(f"timing: {time.time() - t0:.0f}s total, {play_s:.0f}s playing "
                 f"({play_s * args.workers / max(1, len(recs) * len(variants)):.2f}s per player per variant "
                 f"per worker, {args.workers} workers)")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    if args.log:
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M')}  players={len(recs)} seed={args.seed} "
                    f"min_reviews={args.min_reviews} min_wellknown={args.min_wellknown} "
                    f"variants={','.join(f'{v}={t[0]:g}/{t[1]:g}/{t[2]}' for v, t in variants)}\n{out}\n")
    if args.records:
        with open(args.records, "w", encoding="utf-8") as f:
            json.dump({"variants": variants, "records": recs}, f)


if __name__ == "__main__":
    main()
