#!/usr/bin/env python3
"""
steam_reviews.py - collect Steam games + user reviews into SQLite,
as raw data for a game recommendation system.

Uses only official, key-less Steam Web API endpoints (verified Nov 2026):
  * IStoreQueryService/Query/v1          -> every game on the store + tags,
                                            categories, platforms, review summary
  * IStoreService/GetTagList/v1          -> tag id -> tag name
  * IUserReviewsService/GetAppReviews/v1 -> the reviews themselves
    (replaces store.steampowered.com/appreviews/<appid>?json=1, which Valve
     switched off on 22 Oct 2026 - it now just 302s to the store homepage)

No third-party packages needed (stdlib only: urllib + sqlite3). Python 3.8+.

Typical use
-----------
  # 1. Catalogue of all games (~200k, a few minutes)
  python steam_reviews.py games

  # 2. Reviews for every game with >= 50 reviews, most-reviewed first
  python steam_reviews.py reviews --min-reviews 50

  # 2b. Quick sample: top 200 games, at most 2000 reviews each
  #     (later run without --max-per-game and they continue where they stopped)
  python steam_reviews.py reviews --limit-games 200 --max-per-game 2000

  # 3. Later: fetch reviews posted since the last run (+ re-fetch edited ones)
  python steam_reviews.py reviews --update --refresh-edits

  # 4. Look at what you have / dump it for ML
  python steam_reviews.py stats
  python steam_reviews.py export --out reviews.jsonl --games-out games.jsonl

Everything is resumable: stop with Ctrl+C (or even kill the process) and
re-run the same command; each game continues from its saved cursor.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import random
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

API = "https://api.steampowered.com"
USER_AGENT = "steam-review-collector/1.1 (research; python-urllib)"

# GetAppReviews enums (must be numeric - words are silently ignored by Steam)
FILTER_RECENT = 1          # newest created first; pages all the way to the end
FILTER_UPDATED = 2         # most recently updated first
PURCHASE_ALL = 1           # Steam + non-Steam (key) purchases

# Steam EResult values worth retrying (x-eresult header): Fail, Busy, Timeout,
# ServiceUnavailable, RateLimitExceeded
RETRY_ERESULTS = {"2", "10", "16", "20", "84"}

# progress.done values
IN_PROGRESS, COMPLETE, CAPPED = 0, 1, 2

log = logging.getLogger("steam")
STOP = threading.Event()   # set on Ctrl+C -> workers finish current page and exit


class SteamError(Exception):
    """Non-retryable API error (bad params, unknown app, ...)."""

    def __init__(self, msg, eresult=None):
        super().__init__(msg)
        self.eresult = eresult


# --------------------------------------------------------------------------- #
# HTTP with global rate limit + retry/backoff
# --------------------------------------------------------------------------- #
class RateLimiter:
    """At most one request every `interval` seconds, shared by all threads.
    The interval grows after 429s and slowly relaxes back afterwards."""

    def __init__(self, interval: float):
        self.base = interval
        self.interval = interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            t = max(time.monotonic(), self._next)
            self._next = t + self.interval
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self) -> None:
        with self._lock:
            self.interval = min(self.interval * 1.5 + 0.5, 30.0)
            log.warning("rate limited - slowing to 1 request / %.1fs", self.interval)

    def relax(self) -> None:
        with self._lock:
            if self.interval > self.base:
                self.interval = max(self.base, self.interval * 0.98)


def interruptible_sleep(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if STOP.is_set():
            raise KeyboardInterrupt
        time.sleep(min(0.5, max(0.0, end - time.monotonic())))


class SteamClient:
    def __init__(self, interval: float, max_retries: int = 8, key: str | None = None):
        self.rl = RateLimiter(interval)
        self.max_retries = max_retries
        self.key = key

    def call(self, method: str, input_json: dict | None = None, **params) -> dict:
        """Call a Steam Web API service method and return its 'response' object.
        Retries network errors, 429/5xx and transient EResults with backoff."""
        if input_json is not None:
            params["input_json"] = json.dumps(input_json, separators=(",", ":"))
        if self.key:
            params["key"] = self.key
        url = f"{API}/{method}/?{urllib.parse.urlencode(params)}"
        for attempt in range(self.max_retries + 1):
            if STOP.is_set():
                raise KeyboardInterrupt
            self.rl.wait()
            wait_hint = 0.0
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                           "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    eresult = r.headers.get("x-eresult")
                    body = r.read()
                if eresult and eresult != "1":
                    if eresult in RETRY_ERESULTS:
                        raise ConnectionError(f"eresult {eresult}")
                    raise SteamError(f"{method}: eresult {eresult}", eresult)
                data = json.loads(body)
                self.rl.relax()
                return data.get("response", data)
            except urllib.error.HTTPError as e:
                status = e.code
                eresult = e.headers.get("x-eresult") if e.headers else None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    e.close()
                except Exception:                 # an error without a body can't be closed
                    pass
                if status == 429:
                    self.rl.penalize()
                    if retry_after and retry_after.isdigit():
                        wait_hint = float(retry_after)
                elif status not in (500, 502, 503, 504):
                    raise SteamError(f"{method}: HTTP {status}", eresult) from None
                err = f"HTTP {status}"
            except SteamError:
                raise
            # OSError covers URLError, ConnectionError, TimeoutError, SSL errors;
            # ValueError covers JSONDecodeError / UnicodeDecodeError
            except (OSError, http.client.HTTPException, ValueError) as e:
                err = f"{type(e).__name__}: {e}"
            if attempt == self.max_retries:
                break
            backoff = max(wait_hint, min(2 ** attempt, 120) + random.uniform(0, 1))
            log.warning("%s on %s (attempt %d/%d) - retrying in %.0fs",
                        err, method, attempt + 1, self.max_retries, backoff)
            interruptible_sleep(backoff)
        raise RuntimeError(f"giving up on {method} after {self.max_retries} retries ({err})")


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    appid            INTEGER PRIMARY KEY,
    name             TEXT,
    short_description TEXT,
    developers       TEXT,     -- JSON list
    publishers       TEXT,     -- JSON list
    release_date     INTEGER,  -- unix ts
    coming_soon      INTEGER,
    is_free          INTEGER,
    review_count     INTEGER,  -- store summary (Steam purchases, off-topic filtered)
    percent_positive INTEGER,
    review_score_label TEXT,
    tags             TEXT,     -- JSON [{"tagid":..,"weight":..}] by weight desc
    categories       TEXT,     -- JSON {feature_categoryids, supported_player_categoryids, ...}
    platforms        TEXT,     -- JSON {windows, mac, linux, steam_deck_compat_category, ...}
    updated_at       INTEGER,
    header_url       TEXT      -- cover image (newer games use hashed asset paths)
);
CREATE TABLE IF NOT EXISTS tags (
    tagid INTEGER PRIMARY KEY,
    name  TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    recommendationid    INTEGER PRIMARY KEY,
    appid               INTEGER NOT NULL,
    steamid             TEXT,             -- SteamID64 as text; NULL if missing
    language            TEXT,
    review              TEXT,
    voted_up            INTEGER,
    votes_up            INTEGER,
    votes_funny         INTEGER,
    weighted_vote_score REAL,
    comment_count       INTEGER,
    steam_purchase      INTEGER,
    received_for_free   INTEGER,
    written_during_early_access INTEGER,
    primarily_steam_deck INTEGER,
    refunded            INTEGER,
    timestamp_created   INTEGER,
    timestamp_updated   INTEGER,
    author_num_reviews  INTEGER,
    author_playtime_forever INTEGER,      -- minutes
    author_playtime_at_review INTEGER,    -- minutes
    author_playtime_last_two_weeks INTEGER,
    author_last_played  INTEGER,
    developer_response  TEXT,
    raw                 TEXT              -- full JSON (only with --raw)
);
CREATE INDEX IF NOT EXISTS idx_reviews_appid   ON reviews(appid);
CREATE INDEX IF NOT EXISTS idx_reviews_steamid ON reviews(steamid);

-- per-game crawl state, so every run can resume / update
CREATE TABLE IF NOT EXISTS progress (
    appid          INTEGER PRIMARY KEY,
    done           INTEGER DEFAULT 0,  -- 0 in progress, 1 complete, 2 stopped by --max-per-game
    cursor         TEXT,               -- next cursor of the full crawl (NULL when complete)
    fetched        INTEGER DEFAULT 0,  -- reviews received by the full crawl
    total_matching INTEGER,
    newest_ts      INTEGER,            -- newest timestamp_created fully covered
    upd_cursor     TEXT,               -- state of an unfinished --update pass
    upd_since      INTEGER,
    upd_newest     INTEGER,
    edits_checked_at INTEGER,          -- last finished --refresh-edits pass
    updated_at     INTEGER
);
"""


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(games)")}
        if "header_url" not in cols:            # databases created by older versions
            self.conn.execute("ALTER TABLE games ADD COLUMN header_url TEXT")
            self.conn.commit()
        self.lock = threading.Lock()

    def query(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def write(self, fn):
        """Run fn(conn) inside one transaction, serialized across threads."""
        with self.lock, self.conn:
            return fn(self.conn)


def b(v):
    return None if v is None else int(bool(v))


def js(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# games
# --------------------------------------------------------------------------- #
def cmd_games(args, client: SteamClient, db: DB):
    tags = client.call("IStoreService/GetTagList/v1", language=args.lang).get("tags", [])
    db.write(lambda c: c.executemany("INSERT OR REPLACE INTO tags(tagid,name) VALUES (?,?)",
                                     [(t["tagid"], t["name"]) for t in tags]))
    log.info("saved %d tag names", len(tags))

    page, start, total = 1000, 0, None
    seen: set[int] = set()
    while total is None or start < total:
        q = {
            # sort=1 gives a stable order; the default order reshuffles between
            # pages and silently skips ~1/3 of all games (measured Nov 2026)
            "query": {"start": start, "count": page, "sort": 1,
                      "filters": {"type_filters": {"include_games": True}}},
            "context": {"language": args.lang, "country_code": args.country},
            "data_request": {"include_basic_info": True, "include_reviews": True,
                             "include_tag_count": 20, "include_release": True,
                             "include_platforms": True, "include_assets": True},
        }
        resp = client.call("IStoreQueryService/Query/v1", q)
        total = resp.get("metadata", {}).get("total_matching_records", 0)
        items = resp.get("store_items", [])
        if not items:
            break
        now = int(time.time())
        rows = []
        for it in items:
            if not it.get("success", 1) or "appid" not in it:
                continue
            bi = it.get("basic_info") or {}
            rs = (it.get("reviews") or {}).get("summary_filtered") or {}
            rel = it.get("release") or {}
            tg = sorted(it.get("tags") or [], key=lambda t: -t.get("weight", 0))
            assets = it.get("assets") or {}
            header = None
            if assets.get("header") and "${FILENAME}" in (assets.get("asset_url_format") or ""):
                header = ("https://shared.akamai.steamstatic.com/store_item_assets/" +
                          assets["asset_url_format"].replace("${FILENAME}", assets["header"]))
            seen.add(it["appid"])
            rows.append((
                it["appid"], it.get("name"), bi.get("short_description"),
                js([d.get("name") for d in bi.get("developers", [])]),
                js([p.get("name") for p in bi.get("publishers", [])]),
                rel.get("steam_release_date") or rel.get("original_release_date"),
                b(rel.get("is_coming_soon", it.get("is_coming_soon"))), b(it.get("is_free")),
                rs.get("review_count", 0), rs.get("percent_positive"),
                rs.get("review_score_label"), js(tg),
                js(it.get("categories") or {}), js(it.get("platforms") or {}), now, header,
            ))
        db.write(lambda c: c.executemany("""INSERT OR REPLACE INTO games
            (appid,name,short_description,developers,publishers,release_date,coming_soon,is_free,
             review_count,percent_positive,review_score_label,tags,categories,platforms,updated_at,
             header_url)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows))
        start += len(items)
        if start % 10000 < page or start >= total:
            log.info("games: %d / %d", min(start, total), total)
    log.info("done - %d distinct games saved (store reported %d)", len(seen), total or 0)


# --------------------------------------------------------------------------- #
# reviews
# --------------------------------------------------------------------------- #
REVIEW_COLS = ["recommendationid", "appid", "steamid", "language", "review", "voted_up",
               "votes_up", "votes_funny", "weighted_vote_score", "comment_count",
               "steam_purchase", "received_for_free", "written_during_early_access",
               "primarily_steam_deck", "refunded", "timestamp_created", "timestamp_updated",
               "author_num_reviews", "author_playtime_forever", "author_playtime_at_review",
               "author_playtime_last_two_weeks", "author_last_played", "developer_response",
               "raw"]
# upsert; keep previously stored raw JSON when this run doesn't use --raw
REVIEW_SQL = (f"INSERT INTO reviews ({','.join(REVIEW_COLS)}) "
              f"VALUES ({','.join('?' * len(REVIEW_COLS))}) "
              "ON CONFLICT(recommendationid) DO UPDATE SET " +
              ", ".join(f"{c}=excluded.{c}" for c in REVIEW_COLS[1:-1]) +
              ", raw=COALESCE(excluded.raw, raw)")


def review_row(appid: int, r: dict, keep_raw: bool):
    a = r.get("author") or {}
    sid = a.get("steamid")
    return (
        int(r["recommendationid"]), appid, str(sid) if sid else None, r.get("language"),
        r.get("review"), b(r.get("voted_up")), r.get("votes_up"), r.get("votes_funny"),
        float(r.get("weighted_vote_score") or 0), r.get("comment_count"),
        b(r.get("steam_purchase")), b(r.get("received_for_free")),
        b(r.get("written_during_early_access")), b(r.get("primarily_steam_deck")),
        b(r.get("refunded")), r.get("timestamp_created"), r.get("timestamp_updated"),
        a.get("num_reviews"), a.get("playtime_forever"), a.get("playtime_at_review"),
        a.get("playtime_last_two_weeks"), a.get("last_played"),
        r.get("developer_response"), js(r) if keep_raw else None,
    )


class Pager:
    """Walks GetAppReviews pages for one query, with protection against
    transient empty pages and cursor loops."""

    def __init__(self, client, base: dict, cursor: str):
        self.client, self.base, self.cursor = client, base, cursor or "*"
        self.seen = {self.cursor}
        self.total_matching = None

    def pages(self, expected_total=None, received_so_far=0, check_total=True):
        """Yield (reviews, next_cursor, is_last). Stops on STOP.
        check_total: compare against total_matching to detect transient empty
        pages (only meaningful for an unfiltered full crawl)."""
        got = received_so_far
        empty_retries = 0
        while not STOP.is_set():
            resp = self.client.call("IUserReviewsService/GetAppReviews/v1",
                                    {**self.base, "cursor": self.cursor})
            if self.total_matching is None:
                self.total_matching = resp.get("total_matching")
            expected = (expected_total or self.total_matching) if check_total else None
            reviews = resp.get("reviews") or []
            nxt = resp.get("cursor")
            if not reviews:
                # Empty page: real end, or a transient hiccup half way through?
                far_from_end = expected and got < expected * 0.99 - 100
                if far_from_end and empty_retries < 3:
                    empty_retries += 1
                    log.warning("appid %s: empty page at %d/%d, retrying",
                                self.base["appid"], got, expected)
                    interruptible_sleep(5 * empty_retries)
                    continue
                if far_from_end:
                    log.warning("appid %s: Steam stopped returning reviews at %d of %d",
                                self.base["appid"], got, expected)
                yield [], None, True
                return
            empty_retries = 0
            got += len(reviews)
            loop = not nxt or nxt in self.seen
            if loop and nxt:
                log.debug("appid %s: cursor repeated - end of list", self.base["appid"])
            yield reviews, (None if loop else nxt), loop
            if loop:
                return
            self.seen.add(nxt)
            self.cursor = nxt


def base_query(appid, args, flt):
    return {"appid": appid, "filter": flt, "languages": args.languages,
            "purchase_type": PURCHASE_ALL, "num_per_page": 100,
            "filter_offtopic_activity": not args.include_offtopic}


def full_crawl(appid, st, args, client, db) -> int:
    cap = args.max_per_game
    fetched, newest = st["fetched"], st["newest_ts"]
    pager = Pager(client, base_query(appid, args, FILTER_RECENT), st["cursor"])
    added = 0
    for reviews, nxt, last in pager.pages(st["total_matching"], fetched):
        rows = [review_row(appid, r, args.raw) for r in reviews if r.get("recommendationid")]
        for r in reviews:
            ts = r.get("timestamp_created") or 0
            newest = ts if newest is None else max(newest, ts)
        fetched += len(reviews)
        added += len(rows)
        capped = bool(cap) and fetched >= cap and not last
        state = COMPLETE if last else (CAPPED if capped else IN_PROGRESS)

        def tx(c, rows=rows, nxt=nxt, state=state):
            if rows:
                c.executemany(REVIEW_SQL, rows)
            c.execute("""INSERT INTO progress(appid,done,cursor,fetched,total_matching,newest_ts,updated_at)
                         VALUES (?,?,?,?,?,?,?)
                         ON CONFLICT(appid) DO UPDATE SET done=excluded.done, cursor=excluded.cursor,
                           fetched=excluded.fetched,
                           total_matching=COALESCE(excluded.total_matching,total_matching),
                           newest_ts=excluded.newest_ts, updated_at=excluded.updated_at""",
                      (appid, state, nxt, fetched, pager.total_matching, newest, int(time.time())))
        db.write(tx)  # a page and its progress are committed together
        if state != IN_PROGRESS:
            break
    return added


def update_new(appid, st, args, client, db) -> int:
    """Fetch reviews created after newest_ts. newest_ts only moves forward
    once the whole pass has finished, so an interrupted pass is resumed."""
    since = st["upd_since"] or (st["newest_ts"] or 0) + 1
    run_newest = max(st["upd_newest"] or 0, st["newest_ts"] or 0)
    base = base_query(appid, args, FILTER_RECENT)
    base.update(date_range_start=since, date_range_end=int(time.time()) + 86400)
    pager = Pager(client, base, st["upd_cursor"])
    added = 0
    for reviews, nxt, last in pager.pages(check_total=False):
        rows = [review_row(appid, r, args.raw) for r in reviews if r.get("recommendationid")]
        for r in reviews:
            run_newest = max(run_newest, r.get("timestamp_created") or 0)
        added += len(rows)

        def tx(c, rows=rows, nxt=nxt, last=last):
            if rows:
                c.executemany(REVIEW_SQL, rows)
            if last:
                c.execute("""UPDATE progress SET newest_ts=?, upd_cursor=NULL, upd_since=NULL,
                             upd_newest=NULL, updated_at=? WHERE appid=?""",
                          (run_newest or None, int(time.time()), appid))
            else:
                c.execute("""UPDATE progress SET upd_cursor=?, upd_since=?, upd_newest=?,
                             updated_at=? WHERE appid=?""",
                          (nxt, since, run_newest, int(time.time()), appid))
        db.write(tx)
    return added


def refresh_edits(appid, st, args, client, db) -> int:
    """Re-fetch reviews edited since the last finished check (filter=Updated,
    newest-updated first; stop once we pass the previous check time)."""
    started = int(time.time())
    last_check = st["edits_checked_at"]
    if last_check is None:
        # nothing to refresh yet: everything we have was fetched in the full crawl
        db.write(lambda c: c.execute("UPDATE progress SET edits_checked_at=? WHERE appid=?",
                                     (started, appid)))
        return 0
    pager = Pager(client, base_query(appid, args, FILTER_UPDATED), "*")
    n = 0
    for reviews, _nxt, last in pager.pages(check_total=False):
        fresh = [r for r in reviews if (r.get("timestamp_updated") or 0) >= last_check - 60]
        rows = [review_row(appid, r, args.raw) for r in fresh if r.get("recommendationid")]
        n += len(rows)
        if rows:
            db.write(lambda c, rows=rows: c.executemany(REVIEW_SQL, rows))
        if last or len(fresh) < len(reviews):
            break
    if not STOP.is_set():
        db.write(lambda c: c.execute("UPDATE progress SET edits_checked_at=? WHERE appid=?",
                                     (started, appid)))
    return n


def load_state(db, appid):
    cols = ["done", "cursor", "fetched", "total_matching", "newest_ts", "upd_cursor",
            "upd_since", "upd_newest", "edits_checked_at"]
    row = db.query(f"SELECT {','.join(cols)} FROM progress WHERE appid=?", (appid,))
    if not row:
        return dict(zip(cols, [IN_PROGRESS, "*", 0, None, None, None, None, None, None]))
    st = dict(zip(cols, row[0]))
    st["fetched"] = st["fetched"] or 0
    return st


def crawl_game(appid: int, name: str, args, client: SteamClient, db: DB) -> int:
    st = load_state(db, appid)
    cap = args.max_per_game
    added, parts = 0, []
    did_full = False

    needs_full = (st["done"] == IN_PROGRESS or
                  (st["done"] == CAPPED and (not cap or st["fetched"] < cap)))
    if needs_full:
        if st["done"] == CAPPED:
            st["done"] = IN_PROGRESS
        try:
            n = full_crawl(appid, st, args, client, db)
        except SteamError as e:
            if e.eresult == "42":   # no such app / no reviews
                db.write(lambda c: c.execute(
                    "INSERT OR REPLACE INTO progress(appid,done,fetched,updated_at) VALUES (?,?,0,?)",
                    (appid, COMPLETE, int(time.time()))))
                return 0
            raise
        added += n
        parts.append(f"+{n}")
        did_full = True

    if args.update and not did_full and not STOP.is_set():
        st = load_state(db, appid)
        if st["done"] in (COMPLETE, CAPPED):
            n = update_new(appid, st, args, client, db)
            added += n
            parts.append(f"+{n} new")
            if args.refresh_edits and not STOP.is_set():
                parts.append(f"{refresh_edits(appid, load_state(db, appid), args, client, db)} edited")

    if parts:
        log.info("%-8s %-40.40s %s", appid, name or "?", ", ".join(parts))
    return added


def run_batch(targets, args, client, db):
    """Run crawl_game over targets with a bounded queue so Ctrl+C is handled
    within ~1s and nothing is submitted after a stop."""
    total, failed, finished = 0, [], 0
    t0 = time.time()
    it = iter(targets)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = {}

        def fill():
            while len(pending) < args.workers * 2 and not STOP.is_set():
                try:
                    a, n = next(it)
                except StopIteration:
                    return
                pending[ex.submit(crawl_game, a, n, args, client, db)] = (a, n)

        fill()
        while pending:
            done, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
            for f in done:
                target = pending.pop(f)
                finished += 1
                try:
                    total += f.result()
                except KeyboardInterrupt:
                    pass
                except Exception as e:  # one bad game must not kill the run
                    log.error("appid %s failed: %s", target[0], e)
                    failed.append(target)
                if finished % 50 == 0:
                    log.info("progress: %d/%d games, %d new reviews (%.0f/s)", finished,
                             len(targets), total, total / max(1.0, time.time() - t0))
            fill()
    return total, failed


def cmd_reviews(args, client: SteamClient, db: DB):
    if args.appids:
        targets = []
        for a in (int(x) for x in args.appids.split(",") if x.strip()):
            row = db.query("SELECT name FROM games WHERE appid=?", (a,))
            targets.append((a, row[0][0] if row else None))
    else:
        if db.query("SELECT COUNT(*) FROM games")[0][0] == 0:
            sys.exit("games table is empty - run `python steam_reviews.py games` first "
                     "(or pass --appids 1145360,730,...)")
        sql = ("SELECT g.appid, g.name FROM games g LEFT JOIN progress p ON p.appid=g.appid "
               "WHERE g.review_count >= ? AND COALESCE(g.coming_soon,0)=0 ")
        if not args.update:
            sql += f"AND COALESCE(p.done,0) != {COMPLETE} "
        sql += "ORDER BY g.review_count DESC"
        if args.limit_games:
            sql += f" LIMIT {int(args.limit_games)}"
        targets = db.query(sql, (args.min_reviews,))
    log.info("%d games to process (workers=%d, max %.1f requests/s)",
             len(targets), args.workers, 1 / args.delay)

    total, failed = run_batch(targets, args, client, db)
    if failed and not STOP.is_set():
        log.info("retrying %d failed games once", len(failed))
        more, failed = run_batch(failed, args, client, db)
        total += more
    if failed:
        log.warning("%d games failed (re-run to retry): %s", len(failed),
                    ",".join(str(a) for a, _ in failed[:50]) + (" ..." if len(failed) > 50 else ""))
    log.info("finished: %d new reviews this run%s", total,
             " (interrupted - re-run the same command to resume)" if STOP.is_set() else "")


# --------------------------------------------------------------------------- #
# stats / export
# --------------------------------------------------------------------------- #
def cmd_stats(args, client, db: DB):
    q = lambda s: db.query(s)[0][0]
    print(f"games in catalogue : {q('SELECT COUNT(*) FROM games'):,}")
    print(f"games complete     : {q('SELECT COUNT(*) FROM progress WHERE done=1'):,}")
    print(f"games capped       : {q('SELECT COUNT(*) FROM progress WHERE done=2'):,}")
    print(f"games in progress  : {q('SELECT COUNT(*) FROM progress WHERE done=0'):,}")
    print(f"reviews stored     : {q('SELECT COUNT(*) FROM reviews'):,}")
    print(f"distinct reviewers : {q('SELECT COUNT(DISTINCT steamid) FROM reviews'):,}")
    print("\ntop languages:")
    for lang, c in db.query("SELECT language, COUNT(*) c FROM reviews GROUP BY language "
                            "ORDER BY c DESC LIMIT 10"):
        print(f"  {lang:<12} {c:,}")


EXPORT_COLS = ["recommendationid", "appid", "steamid", "language", "voted_up", "votes_up",
               "votes_funny", "weighted_vote_score", "steam_purchase", "received_for_free",
               "refunded", "written_during_early_access", "primarily_steam_deck",
               "author_num_reviews", "author_playtime_forever", "author_playtime_at_review",
               "timestamp_created", "timestamp_updated", "review"]


def cmd_export(args, client, db: DB):
    # keyset pagination in short read transactions (doesn't pin the WAL while a
    # crawler is running, and never loads everything into memory)
    n, last = 0, -1
    sel = (f"SELECT {','.join('r.' + c for c in EXPORT_COLS)} FROM reviews r "
           "WHERE r.recommendationid > ? ORDER BY r.recommendationid LIMIT 5000")
    with open(args.out, "w", encoding="utf-8") as f:
        while True:
            batch = db.query(sel, (last,))
            if not batch:
                break
            for row in batch:
                f.write(js(dict(zip(EXPORT_COLS, row))) + "\n")
            last = batch[-1][0]
            n += len(batch)
    print(f"wrote {n:,} reviews to {args.out}")

    if args.games_out:
        tagname = dict(db.query("SELECT tagid, name FROM tags"))
        rows = db.query("SELECT appid,name,short_description,developers,publishers,release_date,"
                        "is_free,review_count,percent_positive,tags,categories,platforms FROM games")
        with open(args.games_out, "w", encoding="utf-8") as f:
            for (appid, name, desc, dev, pub, rel, free, rc, pp, tags, cats, plat) in rows:
                tg = json.loads(tags or "[]")
                f.write(js({"appid": appid, "name": name, "description": desc,
                            "developers": json.loads(dev or "[]"),
                            "publishers": json.loads(pub or "[]"),
                            "release_date": rel, "is_free": free, "review_count": rc,
                            "percent_positive": pp,
                            "tags": [tagname.get(t["tagid"], t["tagid"]) for t in tg],
                            "tag_weights": [t.get("weight") for t in tg],
                            "categories": json.loads(cats or "{}"),
                            "platforms": json.loads(plat or "{}")}) + "\n")
        print(f"wrote {len(rows):,} games to {args.games_out}")


# --------------------------------------------------------------------------- #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="steam.db", help="SQLite file (default steam.db)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="min seconds between requests, shared by all workers (default 1.0)")
    p.add_argument("--key", default=None, help="optional Steam Web API key (not required)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("games", help="download the catalogue of all games + tags")
    g.add_argument("--lang", default="english")
    g.add_argument("--country", default="US")

    r = sub.add_parser("reviews", help="download reviews (resumable)")
    r.add_argument("--min-reviews", type=int, default=10,
                   help="skip games with fewer store reviews (default 10)")
    r.add_argument("--limit-games", type=int, default=0, help="only the N most-reviewed games")
    r.add_argument("--max-per-game", type=int, default=0,
                   help="pause each game after N reviews (0 = all); a later run with a "
                        "bigger value or 0 continues from where it paused")
    r.add_argument("--appids", default="", help="comma list of appids instead of the catalogue")
    r.add_argument("--languages", default="all",
                   help="comma list of Steam language codes, e.g. english,tchinese (default all)")
    r.add_argument("--include-offtopic", action="store_true",
                   help="also keep reviews Steam flagged as off-topic review bombs")
    r.add_argument("--update", action="store_true",
                   help="also fetch reviews created since the last run for already-crawled games")
    r.add_argument("--refresh-edits", action="store_true",
                   help="with --update: also re-fetch reviews edited since the last check")
    r.add_argument("--workers", type=int, default=2,
                   help="games in parallel (default 2); total speed is still capped by --delay")
    r.add_argument("--raw", action="store_true", help="also store each review's full JSON")

    sub.add_parser("stats", help="print what is in the database")
    e = sub.add_parser("export", help="export to JSON Lines")
    e.add_argument("--out", default="reviews.jsonl")
    e.add_argument("--games-out", default="", help="also write games (with tag names) here")

    args = p.parse_args()
    if args.cmd == "reviews":
        args.languages = [x.strip() for x in args.languages.split(",") if x.strip()]
        args.workers = max(1, args.workers)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    def on_sigint(*_):
        if STOP.is_set():
            log.warning("forced exit")
            import os
            os._exit(130)   # progress is committed per page, so this is safe
        log.warning("Ctrl+C - finishing current pages, progress is saved (press again to force)")
        STOP.set()
    signal.signal(signal.SIGINT, on_sigint)

    db = DB(args.db)
    client = SteamClient(args.delay, key=args.key)
    try:
        {"games": cmd_games, "reviews": cmd_reviews,
         "stats": cmd_stats, "export": cmd_export}[args.cmd](args, client, db)
    except KeyboardInterrupt:
        log.info("stopped; re-run the same command to resume")
    finally:
        db.conn.close()


if __name__ == "__main__":
    main()
