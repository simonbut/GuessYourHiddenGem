#!/usr/bin/env python3
"""
server.py - web UI for the Steam dream-game finder.

    python server.py                 # then open http://127.0.0.1:8000
    python server.py --port 9000
    python server.py --host 0.0.0.0  # let other devices on your network use it

Needs the model from `python recommend.py build`. Standard library only
(plus numpy/scipy for the recommender). AI mode works when the anthropic SDK
is installed and ANTHROPIC_API_KEY (or an `ant auth login` profile) is set.

API (JSON):
  GET  /api/status                 model stats, tag list, starter games, AI availability
  GET  /api/search?q=hollow        name autocomplete
  GET  /api/games?ids=1145360,730  names for appids
  POST /api/like   {"liked":[appid,...], "disliked":[...], "options":{...}}
  POST /api/dream  {"text":"...", "ai":false, "exclude":[appid,...], "options":{...}}
  POST /api/quiz/next    {"history":[{"shown":[appid x5],"order":[...],"unknown":[...],
                          "no_interest":[...],"interested":[...]},...], "seed":123, "extra":{...}}
  POST /api/quiz/result  {"history":[...], "options":{...}, "extra":{...}}
       extra (optional): {"favorites":[appids they added], "more":[...], "less":[...]}
       - "More like this" / "Not for me" on earlier results
POSTs must send Content-Type: application/json and X-Requested-With: dream-game-finder.
options: n, gems, tags[], exclude_tags[], min_positive, min_reviews,
         max_reviews, free, platform (windows|mac|linux|deck), year_from
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import ipaddress
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import quiz
import recommend as rec

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
STATIC = {                       # whitelist: url path -> (file, content type)
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY = 64 * 1024
PLATFORMS = {"windows", "mac", "linux", "deck"}

ALLOWED_HOSTS: set[str] = set()     # filled in main(); blocks DNS-rebinding attacks
MODEL: rec.Model | None = None
STATUS: dict = {}
AI_LOCK = threading.Semaphore(4)     # don't fire unlimited concurrent Claude calls


class BadRequest(Exception):
    pass


_local = threading.local()


def M():
    """The model for this request: one snapshot per request, so an automatic
    reload in the middle of a request can never mix two models."""
    return getattr(_local, "model", None) or MODEL


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def _int(v, lo, hi, default):
    if v in (None, ""):
        return default
    try:
        v = int(v)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest(f"expected a number, got {v!r}")
    return max(lo, min(hi, v))


def _tags(v):
    if not v:
        return ""
    if isinstance(v, str):
        v = v.split(",")
    if not isinstance(v, list):
        raise BadRequest("tags must be a list")
    return ",".join(str(t)[:60] for t in v[:20] if str(t).strip())


def parse_options(o) -> object:
    if o is None:
        o = {}
    if not isinstance(o, dict):
        raise BadRequest("options must be an object")
    platform = o.get("platform") or None
    if platform is not None and (not isinstance(platform, str) or platform not in PLATFORMS):
        raise BadRequest(f"platform must be one of {sorted(PLATFORMS)}")
    year = _int(o.get("year_from"), 0, 2100, 0)
    return rec.make_opts(
        n=_int(o.get("n"), 1, 50, 12),
        gems=bool(o.get("gems")),
        tags=_tags(o.get("tags")),
        exclude_tags=_tags(o.get("exclude_tags")),
        min_positive=_int(o.get("min_positive"), 0, 100, 0),
        min_reviews=_int(o.get("min_reviews"), 0, 10_000_000, 50),
        max_reviews=_int(o.get("max_reviews"), 0, 100_000_000, 0),
        free=bool(o.get("free")),
        platform=platform,
        year_from=year if year >= 1970 else 0,
    )


def appids_to_idx(ids, field):
    if ids is None:
        return []
    if not isinstance(ids, list) or len(ids) > 25:
        raise BadRequest(f"{field} must be a list of up to 25 appids")
    out = []
    for a in ids:
        try:
            i = M().index.get(int(a))
        except (TypeError, ValueError, OverflowError):
            raise BadRequest(f"bad appid in {field}: {a!r}")
        if i is not None and i not in out:
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #
def api_status(_q):
    return STATUS


def api_search(q):
    term = (q.get("q") or [""])[0][:80]
    return {"results": rec.search_names(M(), term, limit=8)}


def api_games(q):
    raw = (q.get("ids") or [""])[0].split(",")[:25]
    out = []
    for a in raw:
        try:
            i = M().index.get(int(a))
        except (ValueError, OverflowError):
            continue
        if i is not None:
            out.append({"appid": M().appids[i], "name": M().names[i]})
    return {"results": out}


def api_like(body):
    liked = appids_to_idx(body.get("liked"), "liked")
    disliked = [i for i in appids_to_idx(body.get("disliked"), "disliked") if i not in liked]
    if not liked:
        raise rec.RecError("Add at least one game you like.")
    out = rec.rec_like(M(), liked, disliked, parse_options(body.get("options")))
    out["based_on"] = [M().names[i] for i in liked]
    return out


def api_dream(body):
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise rec.RecError("Describe the game you're dreaming of.")
    text = text[:1000]
    opts = parse_options(body.get("options"))
    exclude = appids_to_idx(body.get("exclude"), "exclude")
    if body.get("ai"):
        ok, reason = rec.ai_available()
        if not ok:
            raise rec.RecError(f"AI mode isn't set up on this server: {reason}")
        if not AI_LOCK.acquire(timeout=60):
            raise rec.RecError("AI mode is busy - try again in a moment")
        try:
            return rec.rec_dream_ai(M(), text, opts, exclude=exclude)
        finally:
            AI_LOCK.release()
    return rec.rec_dream(M(), text, opts, exclude=exclude)


def _seed(v):
    try:
        return int(v) % (2 ** 32)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest("seed must be a number")


def api_quiz_next(body):
    return quiz.next_round(M(), body.get("history") or [], _seed(body.get("seed", 0)), extra=body.get("extra"))


def api_quiz_result(body):
    out = quiz.results(M(), body.get("history") or [], parse_options(body.get("options")), extra=body.get("extra"))
    picks = out["profile"]["picks"]
    out["based_on"] = [p["name"] for p in picks]
    return out


GET_ROUTES = {"/api/status": api_status, "/api/search": api_search, "/api/games": api_games}
POST_ROUTES = {"/api/like": api_like, "/api/dream": api_dream,
               "/api/quiz/next": api_quiz_next, "/api/quiz/result": api_quiz_result}


def _reject_constant(c):
    raise ValueError(f"{c} is not allowed")


def host_ok(host: str) -> bool:
    """Accept the configured loopback names, or any literal IP address (DNS
    rebinding needs a domain name, so IP literals are safe)."""
    if not host:
        return False
    h = host.strip().lower()
    if h in ALLOWED_HOSTS:
        return True
    if h.startswith("["):                       # [ipv6]:port
        name = h[1:h.find("]")] if "]" in h else h
    else:
        name = h.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "DreamGameFinder/1.0"

    def log_message(self, fmt, *args):          # quieter, one line per request
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, status, body: bytes, ctype, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _dispatch(self, fn, arg):
        _local.model = MODEL                       # snapshot for this request
        try:
            self._json(200, fn(arg))
        except BadRequest as e:
            self._json(400, {"error": str(e)})
        except rec.RecError as e:
            self._json(422, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "Something went wrong on the server - see its console."})

    def _host_guard(self):
        if not host_ok(self.headers.get("Host", "")):
            self._json(421, {"error": "unexpected Host header"})
            return False
        return True

    def do_GET(self):
        if not self._host_guard():
            return
        url = urlparse(self.path)
        if url.path in GET_ROUTES:
            return self._dispatch(GET_ROUTES[url.path], parse_qs(url.query))
        if url.path in STATIC:
            name, ctype = STATIC[url.path]
            try:
                with open(os.path.join(WEB_DIR, name), "rb") as f:
                    return self._send(200, f.read(), ctype)
            except FileNotFoundError:
                return self._json(404, {"error": f"missing web/{name}"})
        if url.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon", cache=True)
        self._json(404, {"error": "not found"})

    def do_HEAD(self):
        if not host_ok(self.headers.get("Host", "")):
            self.send_response(421)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        url = urlparse(self.path)
        ok = url.path in STATIC or url.path in GET_ROUTES
        self.send_response(200 if ok else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        # read the (size-capped) body first: answering before the client has
        # finished sending makes Windows reset the connection, and the browser
        # would show a network error instead of our message
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True               # don't try to read a huge/unknown body
            return self._json(413, {"error": "request too large"})
        raw = self.rfile.read(length) if length else b""
        url = urlparse(self.path)
        fn = POST_ROUTES.get(url.path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        if not self._host_guard():
            return
        # Cross-site protection: another site's page can't send this content type
        # or the custom header without a CORS preflight, which is never approved;
        # and when a browser sends an Origin it has to be this server.
        if self.headers.get_content_type() != "application/json":
            return self._json(415, {"error": "send JSON (Content-Type: application/json)"})
        if self.headers.get("X-Requested-With") != "dream-game-finder":
            return self._json(403, {"error": "missing X-Requested-With header"})
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
            return self._json(403, {"error": "cross-origin request refused"})
        try:
            body = json.loads(raw or b"{}", parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "invalid JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "expected a JSON object"})
        self._dispatch(fn, body)


# --------------------------------------------------------------------------- #
def load(model_path, meta_path):
    global MODEL, STATUS
    t0 = time.time()
    m = rec.Model(model_path, meta_path)
    m.has_tag(0)                                   # build the tag index once, before threads
    quiz.index_for(m)                              # ... and the quiz index
    tag_counts = np.diff(m.tag.tocsc().indptr)
    order = np.argsort(-tag_counts)
    tags = [m.meta["tag_names"][j] for j in order if tag_counts[j] > 0]
    starters = [i for i in np.argsort(-m.reviews)[:24]]
    ai_ok, ai_reason = rec.ai_available()
    MODEL = m
    STATUS = {
        "games": len(m.appids), "players": m.meta.get("n_players", 0),
        "built_at": m.meta.get("built_at"), "tags": tags,
        "starters": [{"appid": m.appids[i], "name": m.names[i]} for i in starters],
        "ai": {"available": ai_ok, "reason": ai_reason},
        "quiz": {"rounds": quiz.ROUNDS, "max_rounds": quiz.MAX_ROUNDS, "per_round": quiz.PER_ROUND},
    }
    print(f"model loaded: {len(m.appids):,} games, {STATUS['players']:,} players "
          f"({time.time() - t0:.1f}s); AI mode {'on' if ai_ok else 'off - ' + ai_reason}")


def model_stamp(model_path, meta_path):
    try:
        return (os.path.getmtime(model_path), os.path.getmtime(meta_path))
    except OSError:
        return None


def watch_model(model_path, meta_path, every=30, last=None):
    """Reload the model when refresh.py (or recommend.py build) replaces it.
    The new model is loaded fully in the background and then swapped in; if
    loading fails, the current model keeps serving."""
    while True:
        time.sleep(every)
        now = model_stamp(model_path, meta_path)
        if now is None or now == last or time.time() - max(now) < 5:   # still being written
            continue
        try:
            load(model_path, meta_path)
            last = now
        except BaseException as e:                 # incl. SystemExit from Model: keep the old model
            print(f"model reload failed, keeping the current one: {e!r}")
            last = now


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Web UI for the Steam dream-game finder")
    p.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default 127.0.0.1 = this computer only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--open", action="store_true", help="open the site in your browser once it's ready")
    p.add_argument("--allow-host", action="append", default=[],
                   help="extra host name to accept, e.g. mypc.local (IP addresses always work)")
    p.add_argument("--model", default=os.path.join(HERE, rec.MODEL_FILE))
    p.add_argument("--meta", default=os.path.join(HERE, rec.META_FILE))
    args = p.parse_args()
    stamp = model_stamp(args.model, args.meta)      # taken BEFORE loading, so a swap during
    load(args.model, args.meta)                    # startup is still picked up
    ALLOWED_HOSTS.update({f"localhost:{args.port}", f"127.0.0.1:{args.port}", f"[::1]:{args.port}"})
    if args.host not in ("0.0.0.0", "", "::", "127.0.0.1", "localhost"):
        ALLOWED_HOSTS.add(f"{args.host.lower()}:{args.port}")
    for h in args.allow_host:
        ALLOWED_HOSTS.add(h.lower() if ":" in h else f"{h.lower()}:{args.port}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=watch_model, args=(args.model, args.meta, 30, stamp), daemon=True).start()
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print(f"open http://{shown}:{args.port}  (Ctrl+C to stop)")
    if args.open:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(f"http://{shown}:{args.port}/",)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
