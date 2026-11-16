#!/usr/bin/env python3
"""
refresh.py - bring everything up to date with one command.

    python refresh.py                          # new reviews for crawled games -> analyse -> rebuild
    python refresh.py --catalogue              # also refresh the game catalogue first
    python refresh.py --more-games 5000 --per-game 200   # also extend coverage to more games
    python refresh.py --skip-crawl             # only re-analyse and rebuild
    python refresh.py --skip-tests             # don't run the test suite

Steps (each one stops the run if it fails, nothing half-done is kept):
  1. (optional) catalogue          steam_reviews.py games
  2. new reviews                   steam_reviews.py reviews --update [--limit-games N --max-per-game M]
  3. review analysis               analyze_reviews.py   (all CPU cores)
  4. test suite                    python -m unittest   (synthetic data, ~1 min)
  5. build into temporary files    recommend.py build
  6. sanity check the new model    (loads, answers a query and a quiz round)
  7. swap it in atomically         the running server picks it up within a minute

Safe to stop with Ctrl+C at any point; re-run to continue (the crawler resumes).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(step, cmd, log):
    t0 = time.time()
    print(f"\n=== {step}: {' '.join(cmd[1:])}", flush=True)
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {step}: {' '.join(cmd)}\n")
    log.flush()
    r = subprocess.run(cmd, cwd=HERE)
    took = time.time() - t0
    log.write(f"    exit {r.returncode} after {took:.0f}s\n")
    if r.returncode != 0:
        sys.exit(f"step '{step}' failed (exit {r.returncode}) - stopped; the current model is unchanged")
    print(f"    ok ({took:.0f}s)")


def _replace(src, dst, tries=10):
    """os.replace with retries (on Windows a reader may hold the file briefly)."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(1 + k)


def swap(new_model, new_meta, live_model, live_meta, log):
    """Put the new pair in place. Both files carry the same build id, so a
    reader that catches them mid-swap refuses the mismatched pair and keeps the
    old model. If the second replace fails, the old meta is restored."""
    backup = live_meta + ".old"
    have_old = os.path.exists(live_meta)
    if have_old:
        _replace(live_meta, backup)
    _replace(new_meta, live_meta)
    try:
        _replace(new_model, live_model)
    except Exception as e:
        if have_old:
            _replace(backup, live_meta)              # roll back to the old, matching pair
        log.write(f"    swap FAILED, old model kept: {e}\n")
        sys.exit(f"could not swap in the new model ({e}) - the current model is unchanged")
    if have_old:
        try:
            os.remove(backup)
        except OSError:
            pass


def sanity_check(model_path, meta_path):
    sys.path.insert(0, HERE)
    import recommend as rec
    import quiz
    m = rec.Model(model_path, meta_path)
    assert len(m.appids) > 1000, "suspiciously small model"
    i = int(m.reviews.argmax())
    out = rec.rec_like(m, [i], (), rec.make_opts(n=5))
    assert len(out["results"]) == 5, "like query failed"
    rd = quiz.next_round(m, [], 1)
    assert len(rd["games"]) == 5, "quiz round failed"
    return len(m.appids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue", action="store_true", help="refresh the game catalogue first")
    p.add_argument("--more-games", type=int, default=0,
                   help="also crawl reviews for the N most-reviewed games (extends coverage)")
    p.add_argument("--per-game", type=int, default=200, help="reviews per game when extending (default 200)")
    p.add_argument("--skip-crawl", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--delay", default="0.5", help="seconds between Steam requests (default 0.5)")
    args = p.parse_args()

    log = open(os.path.join(HERE, "refresh.log"), "a", encoding="utf-8")
    t0 = time.time()
    crawl = [PY, "steam_reviews.py", "--delay", args.delay]
    if args.catalogue and not args.skip_crawl:
        run("catalogue", crawl + ["games"], log)
    if not args.skip_crawl:
        run("new reviews", crawl + ["reviews", "--update", "--workers", "4"], log)
        if args.more_games:
            run("more games", crawl + ["reviews", "--limit-games", str(args.more_games),
                                       "--max-per-game", str(args.per_game), "--workers", "4"], log)
    if not args.skip_analysis:
        run("review analysis", [PY, "analyze_reviews.py"], log)
    if not args.skip_tests:
        run("tests", [PY, "-m", "unittest", "discover", "-s", "tests"], log)
    tmp_model, tmp_meta = "recsys_model.new.npz", "recsys_meta.new.json"
    run("build", [PY, "recommend.py", "--model", tmp_model, "--meta", tmp_meta, "build"], log)
    print("\n=== sanity check")
    try:
        n = sanity_check(os.path.join(HERE, tmp_model), os.path.join(HERE, tmp_meta))
    except Exception as e:                       # keep the old model
        log.write(f"    sanity check FAILED: {e}\n")
        sys.exit(f"the new model failed its sanity check ({e}) - the current model is unchanged")
    swap(os.path.join(HERE, tmp_model), os.path.join(HERE, tmp_meta),
         os.path.join(HERE, "recsys_model.npz"), os.path.join(HERE, "recsys_meta.json"), log)
    msg = f"done in {(time.time() - t0) / 60:.1f} min - model with {n:,} games is live"
    print(f"\n{msg}")
    log.write(f"    {msg}\n")


if __name__ == "__main__":
    main()
