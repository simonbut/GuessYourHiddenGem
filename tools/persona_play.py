#!/usr/bin/env python3
"""
persona_play.py - play the taste quiz through the running site's API, from the
command line (for testing the quiz as different kinds of players).

    python tools/persona_play.py new NAME [--seed N]       # start; prints round 1
    python tools/persona_play.py rank NAME 3,1,5 [--unknown 2] [--no-interest 4] [--interested 2]
                                     # rank the games played by their number (favourite
                                     # first); put the others in a box: never heard of it,
                                     # heard of it but no interest, interested but not
                                     # played. Prints the next round
    python tools/persona_play.py none NAME [--unknown 2] [--no-interest 1] [--interested 3]
                                     # "I don't like any of them" (boxes still sent)
    python tools/persona_play.py fav NAME APPID [APPID...] [--clear]
                                     # add favourites the rounds didn't show (they count
                                     # like ranking a game 1st), from the next request on.
                                     # --clear empties the list first (alone: no favourites)
    python tools/persona_play.py result NAME [--more 1,3] [--less 2]
                                     # taste summary + recommendations. --more / --less mark
                                     # results by their number in the last printed list
                                     # ("More like this" / "Not for me"); the marks are kept
                                     # and sent with every later request

Every game shown must be ranked or in exactly one box (except with `none`).
After the last round the results print automatically. Favourites and marks
are sent as the request's "extra" ({"favorites", "more", "less"}); a game is
in one of those lists at most (the newest choice wins). State lives in
tools/personas/NAME.json. Needs the server running (default http://127.0.0.1:8765).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "personas")
MAX_FAVS = 10          # as on the site
MAX_FEEDBACK = 50      # the server's limit per list
EXTRA_KEYS = ("favorites", "more", "less")


def call(base, path, body=None):
    if body is None:
        req = urllib.request.Request(base + path)
    else:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "X-Requested-With": "dream-game-finder"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        with e:
            sys.exit(f"server said {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the server at {base} ({e.reason}) - is it running?")


def post(base, path, body):
    return call(base, path, body)


def load(name):
    p = os.path.join(STATE_DIR, f"{name}.json")
    if not os.path.exists(p):
        sys.exit(f"no persona '{name}' - start with: new {name}")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    for k in EXTRA_KEYS:                   # personas saved before favourites / marks existed
        st.setdefault(k, [])
    st.setdefault("names", {})
    st.setdefault("last_results", [])
    return st


def save(name, st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def extra(st):
    return {k: list(st.get(k) or []) for k in EXTRA_KEYS}


def mark(st, appid, key, name=None):
    """Put a game in one extra list (favorites / more / less), out of the others."""
    for k in EXTRA_KEYS:
        st[k] = [a for a in st[k] if a != appid]
    st[key].append(appid)
    if name:
        st["names"][str(appid)] = name


def game_name(st, appid):
    return st["names"].get(str(appid)) or f"#{appid}"


def short(players):
    if not players:
        return ""
    bits = (players.get("praise") or [])[:2] + [f"(-) {c}" for c in (players.get("complaints") or [])[:1]]
    return " · ".join(bits)


def show_round(rd):
    print(f"ROUND {rd['round']}/{rd['rounds']}  - {rd.get('hint', '')}")
    for k, g in enumerate(rd["games"], 1):
        print(f"  {k}. {g['name']}  [{g['genre']}, {g.get('year') or '?'}, {g['reviews']:,} reviews, "
              f"{g['percent_positive']}% positive]")
        print(f"     tags: {', '.join(g.get('tags') or [])}")
        s = short(g.get("players"))
        if s:
            print(f"     players say: {s}")
    print("Next: rank NAME <favourite-first numbers> [--unknown N,..] [--no-interest N,..] [--interested N,..]"
          "   or   none NAME   (fav NAME APPID.. adds favourites we didn't show)")


def show_extra(st):
    for key, label in (("favorites", "your favourites"), ("more", "more like"), ("less", "not for me")):
        if st[key]:
            print(f"  {label}: " + ", ".join(game_name(st, a) for a in st[key]))


def show_result(base, st, name):
    res = post(base, "/api/quiz/result", {"history": st["history"], "options": {}, "extra": extra(st)})
    p = res["profile"]
    print("TASTE SUMMARY")
    fam = p.get("familiarity")
    if fam:
        print(f"  familiarity: {fam if isinstance(fam, str) else json.dumps(fam, ensure_ascii=False)}")
    for key, title in (("styles", "styles"), ("loves", "loves"), ("praised", "players praise favourites for"),
                       ("avoids", "less into"), ("radar", "not on your radar")):
        items = p.get(key) or []
        if not items:
            continue
        print(f"  {title}:")
        for it in items:
            label = it.get("name") or it.get("label") or it.get("game")
            more = it.get("because") or it.get("text") or it.get("points") or ""
            if isinstance(more, list):
                more = "; ".join(map(str, more))
            print(f"    - {label} - {more}")
    if p.get("moods"):
        print("  feel: " + ", ".join(m if isinstance(m, str) else (m.get("label") or m.get("name") or str(m))
                                     for m in p["moods"]))
    print("  based on favourites: " + ", ".join(res.get("based_on") or []))
    if any(st[k] for k in EXTRA_KEYS):
        print("YOUR FEEDBACK")
        show_extra(st)
    print("RECOMMENDATIONS")
    shown = res["results"][:15]
    for k, r in enumerate(shown, 1):
        print(f"  {k:2}. {r['name']}  [{r['reviews']:,} reviews, {r['percent_positive']}%] "
              + ("(new to you) " if r.get("from_quiz") else "")
              + f"tags: {', '.join(r.get('tags') or [])}")
        print(f"      why: {r.get('why', '')}")
    # what --more / --less numbers refer to next time
    st["last_results"] = [{"appid": r["appid"], "name": r["name"]} for r in shown]
    save(name, st)
    if shown:
        print("Mark some: result NAME --more N,.. --less N,..   (numbers from this list)")


def cmd_fav(a, st):
    ids = []
    for x in a.args:
        try:
            v = int(x)
        except ValueError:
            sys.exit(f"not an appid: {x!r}")
        if v <= 0:
            sys.exit(f"not an appid: {x!r}")
        if v not in ids:
            ids.append(v)
    if a.clear:
        st["favorites"] = []
    known = {}
    for i in range(0, len(ids), 25):                  # /api/games takes up to 25 at once
        q = urllib.parse.quote(",".join(map(str, ids[i:i + 25])))
        known.update({g["appid"]: g["name"] for g in call(a.base, f"/api/games?ids={q}")["results"]})
    in_round = {g["appid"] for g in (st.get("current") or {}).get("games", [])}
    for appid in ids:
        if appid not in known:
            print(f"  skipped {appid}: not a game we know")
        elif appid in st["favorites"]:
            print(f"  {known[appid]} is a favourite already")
        elif appid in in_round:
            print(f"  skipped {known[appid]}: it's in this round - rank it 1st instead")
        elif len(st["favorites"]) >= MAX_FAVS:
            sys.exit(f"at most {MAX_FAVS} favourites (use --clear to start over)")
        else:
            mark(st, appid, "favorites", known[appid])
    save(a.name, st)
    print("favourites: " + (", ".join(game_name(st, x) for x in st["favorites"]) or "(none)"))
    if st.get("current"):
        print("they count from the next round")
    else:
        print(f"see what changed: result {a.name}")


def apply_marks(a, st):
    if not (a.more or a.less):                 # plain `result NAME`: nothing to mark
        return
    last = st.get("last_results") or []
    if not last:
        sys.exit(f"no result list printed yet - run: result {a.name}")

    def nums(s, opt):
        try:
            out = [int(x) for x in s.replace(" ", "").split(",") if x]
        except ValueError:
            sys.exit(f"bad number list for {opt}: {s!r}")
        for x in out:
            if not 1 <= x <= len(last):
                sys.exit(f"{opt}: result {x} is not in the last list (1-{len(last)})")
        return out

    more, less = nums(a.more, "--more"), nums(a.less, "--less")
    if set(more) & set(less):
        sys.exit("a result can't be both --more and --less")
    for key, picked in (("more", more), ("less", less)):
        for k in picked:
            g = last[k - 1]
            if g["appid"] not in st[key] and len(st[key]) >= MAX_FEEDBACK:
                sys.exit(f"at most {MAX_FEEDBACK} games per list")
            mark(st, g["appid"], key, g["name"])
    if more or less:
        print("marked: " + "; ".join(f"{label} {', '.join(last[k - 1]['name'] for k in picked)}"
                                     for label, picked in (("more like", more), ("not for me", less)) if picked))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["new", "rank", "none", "fav", "result"])
    ap.add_argument("name")
    ap.add_argument("args", nargs="*", help="rank: the order (e.g. 3,1,5); fav: appids")
    ap.add_argument("--unknown", default="", help="never heard of it (game numbers, e.g. 2,4)")
    ap.add_argument("--no-interest", dest="no_interest", default="", help="heard of it, no interest")
    ap.add_argument("--interested", default="", help="interested, haven't played")
    ap.add_argument("--clear", action="store_true", help="fav: empty the favourites first")
    ap.add_argument("--more", default="", help="result: 'More like this' (numbers in the last result list)")
    ap.add_argument("--less", default="", help="result: 'Not for me' (numbers in the last result list)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    if not a.name.replace("_", "").replace("-", "").isalnum():
        sys.exit("NAME: letters, digits, - and _ only")
    if a.cmd not in ("rank", "none", "fav") and a.args:
        sys.exit(f"unexpected arguments for {a.cmd}: {' '.join(a.args)}")
    if a.cmd != "result" and (a.more or a.less):
        sys.exit("--more / --less go with: result NAME")

    if a.cmd == "new":
        st = {"history": [], "seed": a.seed, "favorites": [], "more": [], "less": [], "names": {},
              "last_results": []}
        rd = post(a.base, "/api/quiz/next", {"history": [], "seed": a.seed, "extra": extra(st)})
        st["current"] = rd
        save(a.name, st)
        return show_round(rd)

    st = load(a.name)
    if a.cmd == "fav":
        if not a.args and not a.clear:
            sys.exit("fav NAME APPID [APPID...]  (or --clear)")
        return cmd_fav(a, st)
    if a.cmd == "result":
        apply_marks(a, st)
        return show_result(a.base, st, a.name)
    rd = st.get("current")
    if not rd:
        sys.exit("quiz finished - use: result NAME (or new NAME to restart)")
    shown = [g["appid"] for g in rd["games"]]

    def nums(s):
        try:
            out = [int(x) for x in s.replace(" ", "").split(",") if x]
        except ValueError:
            sys.exit(f"bad number list: {s!r}")
        for x in out:
            if not 1 <= x <= len(shown):
                sys.exit(f"game number {x} is not on this round (1-{len(shown)})")
        return out

    order = nums(",".join(a.args))
    boxes = {"unknown": nums(a.unknown), "no_interest": nums(a.no_interest), "interested": nums(a.interested)}
    placed = order + [k for b in boxes.values() for k in b]
    if len(set(placed)) != len(placed):
        sys.exit("a game number appears twice (each game goes in one place: ranked or one box)")
    if a.cmd == "rank":
        missing = set(range(1, len(shown) + 1)) - set(placed)
        if missing:
            sys.exit(f"rank every game or put it in a box; missing: {sorted(missing)}")
        entry = {"shown": shown, "order": [shown[k - 1] for k in order]}
    else:
        entry = {"shown": shown, "order": []}
    entry.update({key: [shown[k - 1] for k in b] for key, b in boxes.items()})
    if a.cmd == "none":
        entry["none_liked"] = True
    st["history"].append(entry)
    names = {g["appid"]: g["name"] for g in rd["games"]}
    print("recorded: " + (" > ".join(names[x] for x in entry["order"]) or "(none liked)")
          + "".join(f" | {label}: {', '.join(names[x] for x in entry[key])}"
                    for key, label in (("no_interest", "no interest"), ("unknown", "never heard of"),
                                       ("interested", "interested")) if entry[key]))
    if len(st["history"]) >= rd["rounds"]:
        st["current"] = None
        save(a.name, st)
        print()
        return show_result(a.base, st, a.name)
    nxt = post(a.base, "/api/quiz/next", {"history": st["history"], "seed": st["seed"], "extra": extra(st)})
    st["current"] = nxt
    save(a.name, st)
    print()
    show_round(nxt)


if __name__ == "__main__":
    main()
