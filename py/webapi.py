"""Bridge for the static web build (build_site.py).

In the browser, Pyodide runs this file next to recommend.py, quiz.py and
server.py. The page's worker (web/engine.mjs) calls handle() with the same
paths and JSON bodies the local server takes, and gets back the same answers,
so the website behaves the same with or without a server.
"""
from __future__ import annotations

import json
import traceback
from urllib.parse import parse_qs, urlparse

import recommend as rec
import server


def load(model_path: str, meta_path: str) -> None:
    server.load(model_path, meta_path)


def _out(status: int, data) -> str:
    return json.dumps({"status": status, "data": data}, ensure_ascii=False)


def handle(method: str, path: str, body_json: str = "") -> str:
    """Answer one API call; returns JSON text {"status": http status, "data": ...}."""
    if server.MODEL is None:
        return _out(503, {"error": "the recommender is still loading"})
    url = urlparse(path or "")
    if method == "GET":
        fn, arg = server.GET_ROUTES.get(url.path), parse_qs(url.query)
    else:
        fn = server.POST_ROUTES.get(url.path)
        if len(body_json or "") > server.MAX_BODY:
            return _out(413, {"error": "request too large"})
        try:
            arg = json.loads(body_json or "{}", parse_constant=server._reject_constant)
        except ValueError:
            return _out(400, {"error": "invalid JSON"})
        if not isinstance(arg, dict):
            return _out(400, {"error": "expected a JSON object"})
    if fn is None:
        return _out(404, {"error": "not found"})
    server._local.model = server.MODEL
    try:
        return _out(200, fn(arg))
    except server.BadRequest as e:
        return _out(400, {"error": str(e)})
    except rec.RecError as e:
        return _out(422, {"error": str(e)})
    except Exception:                                   # pragma: no cover - shown in the console
        traceback.print_exc()
        return _out(500, {"error": "Something went wrong in the recommender - try again."})
