"""
Test suite for the Steam dream-game finder.

    python -m unittest discover -s tests -v

Most tests use a small synthetic database (300 games in 12 genre clusters,
1,500 players, reviews with known phrases) so they run offline in under a
minute. A few "real data" smoke tests run only when steam.db and the built
model exist next to the code.
"""
import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import analyze_reviews  # noqa: E402
import quiz  # noqa: E402
import recommend as rec  # noqa: E402
import steam_reviews  # noqa: E402

CLUSTERS = [  # (genre tag, words players use, an aspect phrase from aspects.json)
    ("Horror", "terrifying monsters dark corridors", "genuinely scary"),
    ("Puzzle", "logic riddles brain teasers", "clever puzzles"),
    ("RPG", "quests party dialogue", "great story"),
    ("Shooter", "guns aim headshots", "full of cheaters"),
    ("Farming Sim", "crops harvest farm animals", "so relaxing"),
    ("Racing", "cars drifting tracks", "great soundtrack"),
    ("Strategy", "armies empire turns", "addictive"),
    ("Platformer", "jumps levels precise", "tight controls"),
    ("Roguelike", "runs permadeath builds", "high replayability"),
    ("Story Rich", "narrative choices characters", "memorable characters"),
    ("Survival", "hunger base crafting", "fun with friends"),
    ("Fighting", "combos fighters arcade", "satisfying combat"),
]
GAMES_PER = 25


def gname(ci, k):
    """Unique, series-free name for game k of cluster ci."""
    return f"Title{ci:02d}x{k:02d} {CLUSTERS[ci][0]}"


FILLERS = ["Indie", "Singleplayer", "Co-op", "Difficult", "Cozy"]


def make_db(path, n_users=1500, seed=1):
    """Synthetic steam.db with the real schema."""
    rng = random.Random(seed)
    db = steam_reviews.DB(path)
    con = db.conn
    tag_names = [c[0] for c in CLUSTERS] + FILLERS
    con.executemany("INSERT INTO tags(tagid, name) VALUES (?,?)", [(i + 1, t) for i, t in enumerate(tag_names)])
    games = []
    for ci, (tag, words, _) in enumerate(CLUSTERS):
        for k in range(GAMES_PER):
            appid = 1000 + ci * 100 + k
            i = ci * GAMES_PER + k
            tags = [{"tagid": ci + 1, "weight": 1000}, {"tagid": len(CLUSTERS) + 1 + (k % len(FILLERS)), "weight": 300}]
            games.append((appid, gname(ci, k), f"A {tag.lower()} game about {words}.", json.dumps(["Dev"]),
                          json.dumps(["Pub"]), 1500000000 + i * 86400, 0, int(k % 7 == 0),
                          200000 - i * 600, 80 + (k % 18), "Very Positive", json.dumps(tags), "{}",
                          json.dumps({"windows": True, "mac": k % 2 == 0, "steamos_linux": k % 3 == 0,
                                      "steam_deck_compat_category": 3 if k % 4 == 0 else 0}),
                          int(time.time()), None))
    con.executemany("""INSERT INTO games (appid,name,short_description,developers,publishers,release_date,
        coming_soon,is_free,review_count,percent_positive,review_score_label,tags,categories,platforms,
        updated_at,header_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", games)
    rows, rid = [], 1
    for u in range(n_users):
        ci = u % len(CLUSTERS)
        tag, words, phrase = CLUSTERS[ci]
        liked = rng.sample(range(GAMES_PER), 4)
        for k in liked:
            appid = 1000 + ci * 100 + k
            text = f"This is a {tag.lower()} gem: {words}. It is {phrase}, honestly worth it."
            rows.append((rid, appid, str(76561190000000000 + u), "english", text, 1, rng.randint(0, 50), 0, 0.5,
                         0, 1, 0, 0, 0, 0, 1700000000 + rid, 1700000000 + rid, 5, 600, 300, 0, 0, None, None))
            rid += 1
        other = (ci + 1 + u % 5) % len(CLUSTERS)
        appid = 1000 + other * 100 + rng.randrange(GAMES_PER)
        rows.append((rid, appid, str(76561190000000000 + u), "english",
                     "Not for me at all, boring and slow, refunded.", 0, 0, 0, 0.5, 0, 1, 0, 0, 0, 0,
                     1700000000 + rid, 1700000000 + rid, 5, 30, 20, 0, 0, None, None))
        rid += 1
    con.executemany(f"INSERT INTO reviews VALUES ({','.join('?' * 24)})", rows)
    con.commit()
    db.conn.close()


def build_model(tmp, db_path):
    args = argparse.Namespace(db=db_path, model=os.path.join(tmp, "m.npz"), meta=os.path.join(tmp, "m.json"),
                              min_reviews=20, reviews_per_game=150, max_vocab=100000, topk=150)
    rec.cmd_build(args)
    return rec.Model(args.model, args.meta)


def _read(path):
    with open(path) as f:
        return f.read()


class Synthetic(unittest.TestCase):
    """Shared synthetic database, analysis and model for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dgf_test_")
        cls.db = os.path.join(cls.tmp, "steam.db")
        make_db(cls.db)
        cls.lexicon = os.path.join(cls.tmp, "aspects.json")
        shutil.copy(os.path.join(ROOT, "aspects.json"), cls.lexicon)
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            analyze_reviews.analyze(cls.db, cls.lexicon, workers=2)
            cls.model = build_model(cls.tmp, cls.db)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def idx(self, name):
        i, _ = self.model.find(name)
        self.assertIsNotNone(i, name)
        return i

    def cluster_of(self, i):
        return (self.model.appids[i] - 1000) // 100


# --------------------------------------------------------------------------- #
class TestAnalysis(Synthetic):
    def roles(self, cluster):
        con = sqlite3.connect(self.db)
        appids = [1000 + cluster * 100 + k for k in range(GAMES_PER)]
        q = f"SELECT aspect, role, COUNT(*) FROM game_aspects WHERE appid IN ({','.join('?' * len(appids))}) " \
            f"AND role != '' GROUP BY aspect, role"
        return {(a, r): n for a, r, n in con.execute(q, appids)}

    def test_known_phrases_become_the_right_roles(self):
        self.assertGreater(self.roles(0).get(("scary", "trait"), 0), 15)            # Horror
        self.assertGreater(self.roles(1).get(("puzzles", "praise"), 0), 15)         # Puzzle
        self.assertGreater(self.roles(3).get(("cheaters", "complaint"), 0), 15)     # Shooter
        self.assertGreater(self.roles(4).get(("relaxing", "trait"), 0), 15)         # Farming

    def test_aspects_dont_leak_to_other_clusters(self):
        self.assertEqual(self.roles(2).get(("scary", "trait"), 0), 0)               # RPG isn't scary

    def test_unticked_template_lines_are_ignored(self):
        t = "Graphics:\n☐ Infinite replayability\n☑ Great story\n[ ] Full of cheaters"
        cleaned = analyze_reviews._clean(t)
        self.assertNotIn("replayability", cleaned)
        self.assertNotIn("cheaters", cleaned)
        self.assertIn("Great story", cleaned)

    def test_negated_issue_mentions_are_not_counted(self):
        neg = analyze_reviews._NEGATOR_EN
        text = "Honestly there are no bugs at all"
        start = text.index("bugs")
        self.assertIsNotNone(neg.search(text, max(0, start - 40), start))
        text2 = "So many bugs everywhere"
        self.assertIsNone(neg.search(text2, 0, text2.index("bugs")))

    def test_gates_never_drop_a_match(self):
        with mock.patch("sys.stdout"):
            bad = analyze_reviews.check_gates(self.db, self.lexicon, sample=3000)
        self.assertEqual(bad, [])

    def test_rescore_is_consistent_with_full_run(self):
        con = sqlite3.connect(self.db)
        before = sorted(con.execute("SELECT appid, aspect, role FROM game_aspects"))
        with mock.patch("sys.stdout"):
            analyze_reviews.rescore(self.db, self.lexicon)
        after = sorted(sqlite3.connect(self.db).execute("SELECT appid, aspect, role FROM game_aspects"))
        self.assertEqual(before, after)


# --------------------------------------------------------------------------- #
class TestRecommender(Synthetic):
    def test_like_returns_same_cluster(self):
        liked = [self.idx(gname(0, 3))]
        out = rec.rec_like(self.model, liked, (), rec.make_opts(n=10, min_reviews=0))
        clusters = [self.cluster_of(self.model.index[r["appid"]]) for r in out["results"]]
        self.assertGreaterEqual(clusters.count(0), 8, clusters)
        self.assertNotIn(self.model.appids[liked[0]], [r["appid"] for r in out["results"]])

    def test_two_interests_both_get_results(self):
        # 3 farming games + 1 strategy game: averaging them would give only farming
        liked = [self.idx(gname(4, k)) for k in (1, 2, 3)] + [self.idx(gname(6, 1))]
        groups = rec.interest_groups(self.model, [(i, 1.0) for i in liked])
        self.assertEqual(sorted(len(g) for g in groups), [1, 3])
        out = rec.rec_like(self.model, liked, (), rec.make_opts(n=12, min_reviews=0))
        clusters = [self.cluster_of(self.model.index[r["appid"]]) for r in out["results"]]
        self.assertGreaterEqual(clusters.count(4), 6, clusters)       # the bigger interest leads
        self.assertGreaterEqual(clusters.count(6), 2, clusters)       # the other one is there too
        self.assertEqual(clusters[0], 4, clusters)

    def test_interleave_shares_slots_by_weight(self):
        out = rec.interleave([[1, 2, 3, 4, 5, 6], [10, 11, 12], [1, 20]], [3, 1, 1], 6)
        self.assertEqual([i for i, _ in out][:2], [1, 2])
        self.assertEqual(len({i for i, _ in out}), 6)                  # no repeats
        self.assertEqual([sum(g == k for _, g in out) for k in range(3)], [4, 1, 1])   # 3:1:1 of 6 slots
        rejected = rec.interleave([[1, 2, 3]], [1], 3, accept=lambda i: i != 2)
        self.assertEqual([i for i, _ in rejected], [1, 3])

    def test_adult_and_demo_filtered_unless_asked(self):
        m = self.model
        a, d = self.idx(gname(1, 5)), self.idx(gname(1, 6))
        saved = m.adult.copy(), m.is_demo.copy()
        try:
            m.adult[a] = True
            m.is_demo[d] = True
            ok = rec.apply_filters(m, rec.make_opts(min_reviews=0))
            self.assertFalse(ok[a])
            self.assertFalse(ok[d])
            self.assertTrue(rec.apply_filters(m, rec.make_opts(min_reviews=0, tags="Hentai"))[a] or
                            "Hentai" not in m.meta["tag_names"])
        finally:
            m.adult[:], m.is_demo[:] = saved

    def test_dream_respects_negation(self):
        out = rec.rec_dream(self.model, "clever logic riddles, no horror", rec.make_opts(n=10, min_reviews=0))
        self.assertTrue(any("Horror" in n for n in out["notes"]), out["notes"])
        clusters = [self.cluster_of(self.model.index[r["appid"]]) for r in out["results"]]
        self.assertNotIn(0, clusters)
        self.assertGreaterEqual(clusters.count(1), 5, clusters)

    def test_filters(self):
        liked = [self.idx(gname(2, 1))]
        for opts, check in [
            (rec.make_opts(n=20, min_reviews=0, platform="mac"),
             lambda i: self.model.meta["platforms"][i].get("mac")),
            (rec.make_opts(n=20, min_reviews=0, free=True), lambda i: self.model.meta["is_free"][i]),
            (rec.make_opts(n=20, min_reviews=0, min_positive=95),
             lambda i: self.model.meta["percent_positive"][i] >= 95),
            (rec.make_opts(n=20, min_reviews=0, platform="linux"),
             lambda i: self.model.meta["platforms"][i].get("steamos_linux")),
        ]:
            out = rec.rec_like(self.model, liked, (), opts)
            for r in out["results"]:
                self.assertTrue(check(self.model.index[r["appid"]]), (opts, r["name"]))

    def test_impossible_filters_raise_a_clear_error(self):
        with self.assertRaises(rec.RecError):
            rec.rec_like(self.model, [self.idx(gname(2, 1))], (),
                         rec.make_opts(n=5, min_reviews=0, tags="Horror,Racing"))

    def test_find(self):
        self.assertEqual(self.model.find(gname(1, 7))[1], "exact")
        self.assertIsNotNone(self.model.find(gname(1, 7).lower())[0])
        self.assertIsNone(self.model.find("Completely Unknown Title Xyz", strict=True)[0])

    def test_negation_parser(self):
        T = self.model.meta["tag_names"]
        w, u, _ = self.model.parse_wish("cozy farm game, I don't want any horror or racing")
        self.assertEqual({T[j] for j in u}, {"Horror", "Racing"})

    def test_aspect_profile_ignores_liked_games_complaints(self):
        shooter = self.idx(gname(3, 2))
        ap = self.model.aspect_profile([(shooter, 1.0)])
        if ap is not None:
            j = [a["key"] for a in self.model.aspect_meta].index("cheaters")
            self.assertLessEqual(ap[j], 0)          # cheaters is never "wanted"

    def test_results_explain_with_praise_only(self):
        out = quiz.results(self.model, [{"shown": [self.model.appids[self.idx(gname(1, 0))]],
                                         "order": [self.model.appids[self.idx(gname(1, 0))]],
                                         "unknown": []}], rec.make_opts(n=5, min_reviews=0))
        for r in out["results"]:
            if "players praise its" in r["why"]:
                self.assertNotIn("cheater", r["why"])


# --------------------------------------------------------------------------- #
class TestQuiz(Synthetic):
    def play(self, seed, answer):
        hist = []
        for k in range(quiz.ROUNDS):
            rd = quiz.next_round(self.model, hist, seed)
            hist.append(answer(k, [g["appid"] for g in rd["games"]], rd))
        return hist

    def test_round_invariants(self):
        qi = quiz.index_for(self.model)
        for seed in range(6):
            seen = set()
            hist = []
            for k in range(quiz.ROUNDS):
                rd = quiz.next_round(self.model, hist, seed)
                ids = [g["appid"] for g in rd["games"]]
                self.assertEqual(len(ids), 5)
                self.assertEqual(len(set(ids)), 5)
                self.assertFalse(seen & set(ids), "a game was shown twice")
                seen |= set(ids)
                keys = [qi.franchise[self.model.index[a]] for a in ids]
                for a in range(5):
                    for b in range(a + 1, 5):
                        self.assertFalse(quiz.same_series(keys[a], keys[b]))
                hist.append({"shown": ids, "order": ids[:2], "unknown": ids[2:]})

    def test_same_seed_same_round(self):
        a = quiz.next_round(self.model, [], 42)
        b = quiz.next_round(self.model, [], 42)
        self.assertEqual([g["appid"] for g in a["games"]], [g["appid"] for g in b["games"]])

    def test_results_exclude_shown_and_profile_is_specific(self):
        hist = self.play(3, lambda k, ids, rd: {"shown": ids, "order": ids, "unknown": []})
        res = quiz.results(self.model, hist, rec.make_opts(n=10, min_reviews=0))
        shown = {a for h in hist for a in h["shown"]}
        self.assertTrue(res["results"])
        self.assertFalse(shown & {r["appid"] for r in res["results"]})
        p = res["profile"]
        self.assertTrue(p["styles"] or p["loves"])
        loved = {x["label"] for x in p["loves"]}
        self.assertFalse(loved & {x["label"] for x in p["avoids"]}, "loved and avoided at once")

    def test_validation(self):
        ids = [g["appid"] for g in quiz.next_round(self.model, [], 1)["games"]]
        bad = [
            "not a list",
            [{"shown": ids, "order": [999999], "unknown": []}],                # ranked game not shown
            [{"shown": ids, "order": ids[:1], "unknown": ids[:1]}],            # ranked AND unknown
            [{"shown": ids, "order": [ids[0], ids[0]]}],                       # duplicate
            [{"shown": ids, "order": ids}] * (quiz.MAX_ROUNDS + 1),           # too many rounds
            [{"shown": [], "order": []}],                                      # empty round
            [{"shown": ids, "order": [True]}],                                 # bool is not an appid
        ]
        for h in bad:
            with self.assertRaises(quiz.QuizError, msg=str(h)[:80]):
                quiz.next_round(self.model, h, 1)

    def test_old_pick_format_still_works(self):
        ids = [g["appid"] for g in quiz.next_round(self.model, [], 1)["games"]]
        rd = quiz.next_round(self.model, [{"shown": ids, "pick": ids[0]}, {"shown": ids[:1], "pick": None}], 1)
        self.assertEqual(len(rd["games"]), 5)

    def test_none_liked_and_all_unknown(self):
        res = quiz.results(self.model, self.play(5, lambda k, ids, rd: {"shown": ids, "order": [], "unknown": [],
                                                                         "none_liked": True}),
                           rec.make_opts(n=5, min_reviews=0))
        self.assertTrue(res["results"])
        res = quiz.results(self.model, self.play(6, lambda k, ids, rd: {"shown": ids, "order": [], "unknown": ids}),
                           rec.make_opts(n=5, min_reviews=0))
        self.assertTrue(res["results"])

    # -- the three boxes and knowing / not knowing as evidence --------------- #
    def ids_of(self, ci, ks):
        return [self.model.appids[self.idx(gname(ci, k))] for k in ks]

    def state(self, hist):
        return quiz.State(quiz.index_for(self.model), quiz.parse_history(self.model, hist))

    def test_boxes_score_and_validate(self):
        ix = lambda a: self.model.index[a]
        ids = self.ids_of(2, range(5))
        st = self.state([{"shown": ids, "order": ids[:2], "unknown": ids[2:3],
                          "no_interest": ids[3:4], "interested": ids[4:]}])
        self.assertEqual(st.score[ix(ids[3])], quiz.NO_INTEREST_SCORE)
        self.assertEqual(st.score[ix(ids[4])], quiz.INTERESTED_SCORE)
        self.assertNotIn(ix(ids[2]), st.score)                    # never heard of it: no like score
        self.assertEqual(st.unknown, {ix(ids[2])})
        self.assertEqual(st.interested, {ix(ids[4])})
        self.assertEqual(st.known_by_tier.sum(), 4)               # boxed "heard of it" games are known
        # an opinion box says it directly; ranked and never-heard games carry knew-it evidence
        self.assertEqual(set(st.familiar), {ix(a) for a in ids[:3]})
        self.assertLess(st.familiar[ix(ids[2])], 0)
        self.assertGreater(st.familiar[ix(ids[0])], 0)
        for bad in ({"shown": ids, "order": ids[:1], "no_interest": ids[:1]},      # two places
                    {"shown": ids, "unknown": ids[:1], "interested": ids[:1]},
                    {"shown": ids[:4], "interested": [ids[4]]},                     # not shown
                    {"shown": ids, "no_interest": "x"},                             # not a list
                    {"shown": ids, "interested": [ids[0], ids[0]]}):                # duplicate
            with self.assertRaises(quiz.QuizError, msg=str(bad)[:80]):
                quiz.parse_history(self.model, [bad])

    def test_none_liked_spares_boxed_games(self):
        ix = lambda a: self.model.index[a]
        ids = self.ids_of(3, range(5))
        st = self.state([{"shown": ids, "order": [], "unknown": ids[:1], "interested": ids[1:2],
                          "no_interest": ids[2:3], "none_liked": True}])
        self.assertNotIn(ix(ids[0]), st.score)
        self.assertEqual(st.score[ix(ids[1])], quiz.INTERESTED_SCORE)
        self.assertEqual(st.score[ix(ids[2])], quiz.NO_INTEREST_SCORE)
        self.assertEqual(st.score[ix(ids[3])], -0.5)
        self.assertEqual(st.score[ix(ids[4])], -0.5)

    def test_not_knowing_counts_more_for_someone_who_knows_most(self):
        ix = lambda a: self.model.index[a]
        ids = self.ids_of(5, range(5))                         # one tier (the synthetic catalogue is small)
        expert = self.state([{"shown": ids, "order": ids[:4], "unknown": ids[4:]}])
        newbie = self.state([{"shown": ids, "order": ids[:1], "unknown": ids[1:]}])
        e_exp, e_new = expert.familiar[ix(ids[4])], newbie.familiar[ix(ids[4])]
        self.assertLess(e_exp, e_new)                          # an expert not knowing it says more
        self.assertLess(e_new, 0)
        self.assertGreater(newbie.familiar[ix(ids[0])], expert.familiar[ix(ids[0])])  # and knowing it, less
        for st in (expert, newbie):
            for e in st.familiar.values():
                self.assertTrue(-1 <= e <= 1)

    def test_never_heard_games_can_come_back_as_results(self):
        a = self.ids_of(6, range(0, 5))
        b = self.ids_of(6, range(5, 10))
        hist = [{"shown": a, "order": a[:3], "unknown": a[3:]}, {"shown": b, "order": b[:2], "unknown": b[2:]}]
        known = set(a[:3] + b[:2])
        never = set(a[3:] + b[2:])
        for cap in (0, 1, 3):
            with mock.patch.object(quiz, "RESURFACE_UNKNOWN", cap):
                res = quiz.results(self.model, hist, rec.make_opts(n=15, min_reviews=0))
            got = {r["appid"] for r in res["results"]}
            self.assertFalse(got & known, "a game they know came back")
            back = [r for r in res["results"] if r.get("from_quiz")]
            self.assertEqual({r["appid"] for r in back}, got & never)
            self.assertLessEqual(len(back), cap)
            for r in back:
                self.assertTrue(r["why"].startswith("New to you"))
            if cap:
                self.assertTrue(back, "a well-matching never-heard game should come back")
            self.assertEqual(len(res["results"]), 15)

    def test_radar_lists_genres_they_dont_know(self):
        qi = quiz.index_for(self.model)
        horror, racing = self.ids_of(0, range(5)), self.ids_of(5, range(5))
        res = quiz.results(self.model, [{"shown": horror, "order": horror, "unknown": []},
                                        {"shown": racing, "order": [], "unknown": racing}],
                           rec.make_opts(n=5, min_reviews=0))
        labels = [x["label"] for x in res["profile"]["radar"]]
        self.assertIn(qi.family_names[qi.family[self.idx(gname(5, 0))]], labels)
        self.assertNotIn(qi.family_names[qi.family[self.idx(gname(0, 0))]], labels)
        # someone who knows (almost) nothing gets no radar: everything would be on it
        res = quiz.results(self.model, [{"shown": racing, "order": racing[:1], "unknown": racing[1:]}],
                           rec.make_opts(n=5, min_reviews=0))
        self.assertEqual(res["profile"]["radar"], [])

    def test_old_behaviour_when_weights_are_zero(self):
        hist = self.play(7, lambda k, ids, rd: {"shown": ids, "order": ids[:2], "unknown": ids[2:]})
        with mock.patch.multiple(quiz, FAMILIARITY_TASTE=0, FAMILIARITY_GENRE=0, RESURFACE_UNKNOWN=0):
            res = quiz.results(self.model, hist, rec.make_opts(n=10, min_reviews=0))
        shown = {a for h in hist for a in h["shown"]}
        self.assertFalse(shown & {r["appid"] for r in res["results"]})

    def test_rejected_games_never_pull_the_taste_profile(self):
        ix = lambda a: self.model.index[a]
        liked, rejected, never = self.ids_of(7, range(5)), self.ids_of(0, range(5)), self.ids_of(8, range(5))
        hist = [{"shown": liked, "order": liked, "unknown": []},
                {"shown": rejected, "order": [], "unknown": [], "none_liked": True},
                {"shown": never, "order": never[:1], "unknown": never[1:]}]
        st = self.state(hist)
        fam, scale = quiz.taste_familiarity(st)
        used = {i for i, _ in fam}
        self.assertFalse(used & {ix(a) for a in rejected + liked[:2] + liked[3:]}, "a rated game counted as 'knew it'")
        self.assertTrue({ix(a) for a in never[1:]} <= used)
        self.assertGreaterEqual(scale, sum(s for s in st.score.values() if s > 0))
        # one interest's profile is scaled by all liked games, so a small interest isn't swamped
        _, _, _, tp_small = quiz.taste_scores(self.model, st, only={ix(never[0])})
        with mock.patch.object(quiz, "FAMILIARITY_TASTE", 0):
            _, _, _, tp_plain = quiz.taste_scores(self.model, st, only={ix(never[0])})
        self.assertGreater(float(tp_small @ tp_plain), 0.9)

    def test_radar_never_names_a_genre_they_like(self):
        qi = quiz.index_for(self.model)
        x = self.ids_of(9, range(8))
        other = self.ids_of(10, range(12))
        hist = [{"shown": [x[0]] + other[:4], "order": [other[0], x[0]] + other[1:4], "unknown": []},  # x0 2nd
                {"shown": x[1:5] + other[4:5], "order": other[4:5], "unknown": x[1:5]},       # never heard of 4
                {"shown": x[5:8] + other[5:7], "order": other[5:7], "unknown": x[5:7],        # 2 more, and
                 "no_interest": [x[7]]}]                                                      # one turned down
        st = self.state(hist)
        f = qi.family[self.idx(gname(9, 0))]
        self.assertLessEqual(st.fam_score[f], 0)              # net score of that genre is not positive...
        self.assertLessEqual(st.fam_familiar[f], -0.5)        # ...and its games were mostly new to them...
        labels = [r["label"] for r in quiz.taste_details(self.model, st)["radar"]]
        self.assertNotIn(qi.family_names[f], labels)          # ...but they liked a game of it

    # -- favourites they add, and feedback on results ------------------------- #
    def test_added_favourites_count_as_first_place(self):
        ix = lambda a: self.model.index[a]
        fav = self.ids_of(4, [0])[0]                       # a farming game, never shown
        st = quiz.State(quiz.index_for(self.model), [], quiz.parse_extra(self.model, {"favorites": [fav]}))
        self.assertEqual(st.score[ix(fav)], quiz.FAVORITE_SCORE)
        self.assertIn(ix(fav), st.favs)
        self.assertIn(ix(fav), st.shown)                   # never shown in a round, never recommended
        self.assertEqual(st.known_by_tier.sum(), 0)        # not part of the knowledge-depth measure
        rd = quiz.next_round(self.model, [], 3, extra={"favorites": [fav]})
        qi = quiz.index_for(self.model)
        self.assertNotIn(fav, [g["appid"] for g in rd["games"]])
        self.assertIn(qi.family[ix(fav)], [qi.family[ix(g["appid"])] for g in rd["games"]])   # followed up at once
        self.assertNotIn("Five big hits", rd["hint"])
        hist = self.play(4, lambda k, ids, rd: {"shown": ids, "order": ids[:1], "unknown": ids[1:]})
        res = quiz.results(self.model, hist, rec.make_opts(n=10, min_reviews=0), extra={"favorites": [fav]})
        self.assertNotIn(fav, {r["appid"] for r in res["results"]})
        self.assertIn(fav, [p["appid"] for p in res["profile"]["picks"]])
        # only favourites, no rounds yet: results still work
        self.assertTrue(quiz.results(self.model, [], rec.make_opts(n=5, min_reviews=0),
                                     extra={"favorites": [fav]})["results"])

    def test_feedback_on_results_changes_them(self):
        hist = self.play(5, lambda k, ids, rd: {"shown": ids, "order": ids[:2], "unknown": ids[2:]})
        opts = rec.make_opts(n=12, min_reviews=0)
        before = quiz.results(self.model, hist, opts)["results"]
        ix = lambda a: self.model.index[a]
        qi = quiz.index_for(self.model)
        fam = lambda a: qi.family[ix(a)]
        sim = lambda a, b: float((self.model.tag[ix(a)] @ self.model.tag[ix(b)].T).toarray()[0, 0])
        more = before[0]["appid"]
        less = min((r["appid"] for r in before[1:]), key=lambda a: sim(a, more))     # the least alike
        extra = {"more": [more], "less": [less]}
        after = quiz.results(self.model, hist, opts, extra=extra)
        with mock.patch.multiple(quiz, MORE_LIKE_SCORE=0.0, NOT_FOR_ME_SCORE=0.0):
            plain = quiz.results(self.model, hist, opts, extra=extra)      # the same games left out, no weight
        got = [r["appid"] for r in after["results"]]
        self.assertNotIn(more, got)                         # both are answered now
        self.assertNotIn(less, got)
        self.assertNotEqual(got, [r["appid"] for r in plain["results"]], "the feedback had no effect")
        # results move toward the "more like this" game and away from the "not for me" one
        near = lambda rows, a: sum(sim(r["appid"], a) for r in rows) / len(rows)
        self.assertGreater(near(after["results"], more), near(plain["results"], more))
        self.assertLess(near(after["results"], less), near(plain["results"], less))
        self.assertNotIn(more, [p["appid"] for p in after["profile"]["picks"]])   # not a game they played
        st = self.state(hist)
        st_fb = quiz.State(qi, quiz.parse_history(self.model, hist), quiz.parse_extra(self.model, extra))
        self.assertLess(st_fb.fam_score[fam(less)], st.fam_score[fam(less)])     # its genre is toned down
        # the taste summary describes the games they ranked, not the "more like this" one
        for item in after["profile"]["styles"] + after["profile"]["loves"]:
            self.assertNotIn(self.model.names[ix(more)], item["because"])
        for bad in ("x", {"more": [more], "less": [more]}, {"favorites": list(range(1, quiz.MAX_EXTRA + 2))},
                    {"less": [True]}):
            with self.assertRaises(quiz.QuizError, msg=str(bad)[:60]):
                quiz.results(self.model, hist, opts, extra=bad)
        # unknown appids (e.g. after a data rebuild) are dropped, not an error
        self.assertTrue(quiz.results(self.model, hist, opts, extra={"more": [999999999]})["results"])

    def test_newest_answer_wins_over_a_round(self):
        ix = lambda a: self.model.index[a]
        a = self.ids_of(1, range(5))
        hist = [{"shown": a, "order": a, "unknown": []}]                  # a[4] ranked 5th: -1
        st = quiz.State(quiz.index_for(self.model), quiz.parse_history(self.model, hist),
                        quiz.parse_extra(self.model, {"favorites": [a[4]], "less": [a[0]]}))
        self.assertEqual(st.score[ix(a[4])], quiz.FAVORITE_SCORE)        # a favourite now, not -1 + 1 = 0
        self.assertIn(ix(a[4]), st.favs)
        self.assertEqual(st.score[ix(a[0])], quiz.NOT_FOR_ME_SCORE)      # changed their mind about the 1st
        b = self.ids_of(2, range(5))
        st = quiz.State(quiz.index_for(self.model),
                        quiz.parse_history(self.model, [{"shown": b, "order": b[:1], "unknown": b[1:]}]),
                        quiz.parse_extra(self.model, {"favorites": [b[3]]}))
        self.assertNotIn(ix(b[3]), st.unknown)                            # they know it after all
        self.assertNotIn(ix(b[3]), st.familiar)

    def test_taste_slot_follows_newest_favourite(self):
        qi = quiz.index_for(self.model)
        ids = lambda ci, ks: [self.model.appids[self.idx(gname(ci, k))] for k in ks]
        h1, h2, farm = ids(0, range(0, 5)), ids(0, range(5, 10)), ids(4, range(0, 5))
        hist = [{"shown": h1, "order": h1[:2], "unknown": h1[2:]},       # horror 1st, twice
                {"shown": h2, "order": h2[:2], "unknown": h2[2:]},
                {"shown": farm, "order": farm[:2], "unknown": farm[2:]}]  # then a farming game 1st
        st = quiz.State(qi, quiz.parse_history(self.model, hist))
        horror, farming = qi.family[self.idx(gname(0, 0))], qi.family[self.idx(gname(4, 0))]
        self.assertEqual(st.top_families[0], horror)                     # horror is stronger overall...
        self.assertEqual(quiz._exploit_families(qi, st)[0], farming)     # ...but the new favourite comes first

    def test_series_keys_and_editions(self):
        generic = quiz._generic_first_words(["Dragon Age", "Dragon Quest", "Dragon's Dogma", "Dragon Ball",
                                             "Dragon Saga", "Dragon Fin", "Dragon Up", "Hades"])
        self.assertIn("dragon", generic)
        self.assertNotEqual(quiz._franchise("Dragon Age: Inquisition", generic),
                            quiz._franchise("Dragon Quest XI", generic))
        self.assertEqual(quiz._franchise("Dark Souls III"), quiz._franchise("Dark Souls Remastered"))
        self.assertTrue(quiz._is_edition_or_dlc("mass effect", ["mass effect legendary edition"]))
        self.assertTrue(quiz._is_edition_or_dlc("ori and the blind forest definitive edition",
                                                ["ori and the blind forest"]))
        self.assertFalse(quiz._is_edition_or_dlc("hades", ["hades 2"]))
        self.assertFalse(quiz._is_edition_or_dlc("hades 2", ["hades"]))

    def test_rank_scores(self):
        self.assertEqual(quiz.rank_scores(5), [1.0, 0.5, 0.0, -0.5, -1.0])
        self.assertEqual(quiz.rank_scores(2), [1.0, 0.5])         # a partial ranking is not "hate"
        self.assertLess(quiz.rank_scores(1)[0], quiz.FAV_THRESHOLD)


# --------------------------------------------------------------------------- #
class TestServer(Synthetic):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import server
        cls.server = server
        with mock.patch("sys.stdout"):
            server.load(os.path.join(cls.tmp, "m.npz"), os.path.join(cls.tmp, "m.json"))
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        server.ALLOWED_HOSTS.update({f"127.0.0.1:{cls.port}", f"localhost:{cls.port}"})
        server.Handler.log_message = lambda *a, **k: None
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        super().tearDownClass()

    def req(self, path, body=None, headers=None, method=None):
        h = {"Content-Type": "application/json", "X-Requested-With": "dream-game-finder"} if body is not None else {}
        h.update(headers or {})
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_status_and_quiz_flow(self):
        code, st = self.req("/api/status")
        self.assertEqual(code, 200)
        self.assertGreater(st["games"], 100)
        code, rd = self.req("/api/quiz/next", {"history": [], "seed": 7})
        self.assertEqual(code, 200)
        ids = [g["appid"] for g in rd["games"]]
        code, res = self.req("/api/quiz/result", {"history": [{"shown": ids, "order": ids, "unknown": []}],
                                                  "options": {"min_reviews": 0}})
        self.assertEqual(code, 200)
        self.assertTrue(res["results"])

    def test_like_and_validation(self):
        appid = 1000 + 2 * 100 + 1
        code, out = self.req("/api/like", {"liked": [appid], "options": {"n": 5, "min_reviews": 0}})
        self.assertEqual(code, 200)
        self.assertEqual(len(out["results"]), 5)
        self.assertEqual(self.req("/api/like", {"liked": ["x"]})[0], 400)
        self.assertEqual(self.req("/api/like", {"liked": [appid], "options": {"platform": "ps5"}})[0], 400)
        self.assertEqual(self.req("/api/like", b"{bad json")[0], 400)

    def test_cross_site_protection(self):
        appid = 1000 + 2 * 100 + 1
        body = {"liked": [appid]}
        self.assertEqual(self.req("/api/like", body, {"X-Requested-With": ""})[0], 403)
        self.assertEqual(self.req("/api/like", body, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.req("/api/like", body, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("/api/status", headers={"Host": "attacker.example"})[0], 421)


# --------------------------------------------------------------------------- #
class TestWebBuild(Synthetic):
    """The static web version: slim model, the in-browser API bridge, pages."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import build_site
        import server
        import webapi
        cls.bs, cls.server, cls.webapi = build_site, server, webapi
        cls.web = os.path.join(cls.tmp, "web")
        with mock.patch("sys.stdout"):
            mp, jp = build_site.write_web_model(os.path.join(cls.tmp, "m.npz"), os.path.join(cls.tmp, "m.json"),
                                                cls.web, min_reviews=0)
            webapi.load(mp, jp)
        cls.slim = server.MODEL

    def call(self, method, path, body=None):
        out = json.loads(self.webapi.handle(method, path, "" if body is None else
                                            (body if isinstance(body, str) else json.dumps(body))))
        return out["status"], out["data"]

    def test_slim_model_recommends_like_the_full_one(self):
        opts = rec.make_opts(n=10, min_reviews=0)
        same = []
        for ci in (0, 4, 8):
            i = self.idx(gname(ci, 3))
            full = [r["appid"] for r in rec.rec_like(self.model, [i], [], opts)["results"]]
            j = self.slim.index[self.model.appids[i]]
            slim = [r["appid"] for r in rec.rec_like(self.slim, [j], [], opts)["results"]]
            same.append(len(set(full) & set(slim)) / len(full))
        self.assertTrue(same)
        self.assertGreaterEqual(min(same), 0.7, same)
        # stored compactly, loaded as the usual types
        self.assertEqual(self.slim.cf.dtype, np.float32)
        self.assertEqual(self.slim.tag.indices.dtype, np.int32)

    def test_bridge_answers_like_the_server(self):
        code, st = self.call("GET", "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(st["games"], len(self.slim.appids))
        code, rd = self.call("POST", "/api/quiz/next", {"history": [], "seed": 7})
        self.assertEqual(code, 200)
        self.assertEqual(rd, quiz.next_round(self.slim, [], 7))      # what round 1 is precomputed from
        ids = [g["appid"] for g in rd["games"]]
        code, res = self.call("POST", "/api/quiz/result", {"history": [{"shown": ids, "order": ids, "unknown": []}],
                                                           "options": {"min_reviews": 0}})
        self.assertEqual(code, 200)
        self.assertTrue(res["results"])
        code, found = self.call("GET", "/api/search?q=Title01x0")
        self.assertEqual(code, 200)
        self.assertTrue(found["results"])

    def test_bridge_errors(self):
        self.assertEqual(self.call("POST", "/api/like", "{bad json")[0], 400)
        self.assertEqual(self.call("POST", "/api/like", "[1, 2]")[0], 400)
        self.assertEqual(self.call("POST", "/api/like", {"liked": ["x"]})[0], 400)
        self.assertEqual(self.call("POST", "/api/like", {"liked": []})[0], 422)
        self.assertEqual(self.call("POST", "/api/nope", {})[0], 404)
        self.assertEqual(self.call("POST", "/api/like", "x" * (self.server.MAX_BODY + 1))[0], 413)

    def test_static_home_page(self):
        with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as f:
            page = self.bs.static_index(f.read(), site_url="https://a.example/x/", build="b1",
                                        google_verification="tok<en")
        self.assertIn('window.DGF = {"static": true, "build": "b1", "site": "https://a.example/x/"}', page)
        self.assertIn('<link rel="canonical" href="https://a.example/x/">', page)
        self.assertIn('name="google-site-verification" content="tok&lt;en"', page)
        self.assertNotIn('href="/"', page)
        self.assertIn('href="genres/"', page)
        self.assertIn('href="about/"', page)
        self.assertIn("shown as is", page)

    def test_about_page_and_contact(self):
        bs = self.bs
        self.assertEqual(bs.contact_url("https://Name.github.io/my-site/"), "https://github.com/Name/my-site/issues")
        self.assertEqual(bs.contact_url("https://name.github.io/"), "https://github.com/name/name.github.io/issues")
        self.assertIsNone(bs.contact_url("https://example.com/"))
        page = bs.about_page("https://name.github.io/my-site/", "b1", 1700000000, bs.contact_url("https://name.github.io/my-site/"))
        for must in ('"as is"', "not affiliated", "CC0", "no review text, no Steam IDs", "14 November 2023",
                     'href="https://github.com/name/my-site/issues"', '<link rel="canonical" href="https://name.github.io/my-site/about/">',
                     'href="../style.css?v=b1"'):
            self.assertIn(must, page)
        self.assertNotIn("Open an issue", bs.about_page("https://example.com/", "b1", 1700000000, None))
        # every generated page links to it
        self.assertIn('href="../../about/"', bs.page_shell("t", "d", "https://x/", 2, "<p>x</p>"))
        # the published site carries its own data notice (its repo's code licence isn't the data's)
        note = bs.data_notice("https://name.github.io/my-site/", 1700000000, "https://github.com/name/my-site/issues")
        for must in ("covers the code only", "CC0", "No review text, Steam IDs or player names", "14 November 2023",
                     "not affiliated", "https://github.com/name/my-site/issues", "https://name.github.io/my-site/about/"):
            self.assertIn(must, note)

    def test_slugs(self):
        self.assertEqual(self.bs.slugify("Sid Meier's Civilization® VI"), "sid-meier-s-civilization-vi")
        self.assertEqual(self.bs.game_slug(42, "Pokémon Café"), "42")      # renames keep the URL

    def test_game_pages_and_sitemap(self):
        out = os.path.join(self.tmp, "pages")
        notes = self.slim.meta["notes"]
        i = next(k for k, n in enumerate(notes) if n)                 # give one game a written summary
        saved = notes[i]
        notes[i] = dict(saved, src="written", summary="A <b>tense</b> & dark game.")
        try:
            with mock.patch("sys.stdout"):
                urls, n = self.bs.build_pages(self.slim, out, "https://a.example/x/", "b1", 1700000000, min_points=1)
        finally:
            notes[i] = saved
        self.assertGreater(n, 1)
        self.assertEqual(len(urls), len(set(u for u, _ in urls)))      # no duplicate URLs
        page_urls = [u for u, _ in urls if "/games/" in u]
        self.assertEqual(page_urls, [f"https://a.example/x/games/{self.slim.appids[i]}/"])   # only the written one

        def page(appid):
            with open(os.path.join(out, "games", str(appid), "index.html"), encoding="utf-8") as f:
                return f.read()
        html = page(self.slim.appids[i])
        self.assertIn(f'<link rel="canonical" href="{page_urls[0]}">', html)
        self.assertIn("A &lt;b&gt;tense&lt;/b&gt; &amp; dark game.", html)    # escaped
        self.assertNotIn("noindex", html)
        self.assertIn("<h1>", html)
        self.assertIn("application/ld+json", html)
        for bad in (">None<", "nan%", "undefined"):
            self.assertNotIn(bad, html)
        others = [d for d in os.listdir(os.path.join(out, "games")) if d != str(self.slim.appids[i])]
        self.assertTrue(others)
        self.assertIn('<meta name="robots" content="noindex, follow">', page(others[0]))
        xml = self.bs.sitemap_xml(urls)
        self.assertTrue(xml.startswith('<?xml version="1.0" encoding="UTF-8"?>'))
        self.assertEqual(xml.count("<url>"), len(urls))


# --------------------------------------------------------------------------- #
class TestSafety(Synthetic):


    def test_classify_split_needs_negative_evidence(self):
        import analyze_reviews as ar
        # 9 mentions, 4 in positive reviews, none explicitly negative: no claim at all
        self.assertEqual(ar.classify("quality", 9, 4, 0, True, 0.9), "")
        # explicit negative mentions ("the story is not good") make it a real split
        self.assertEqual(ar.classify("quality", 9, 5, 2, True, 0.9), "mixed")
        # mostly thumbs-down mentions -> a complaint, mostly thumbs-up -> praise
        self.assertEqual(ar.classify("quality", 12, 2, 0, True, 0.9), "complaint")
        self.assertEqual(ar.classify("quality", 12, 11, 0, True, 0.9), "praise")
        self.assertEqual(ar.classify("quality", 12, 11, 0, False, 0.9), "")    # not notable
        # "mods!" in positive reviews but "no workshop support" twice: not praise
        self.assertEqual(ar.classify("quality", 15, 10, 2, True, 0.6), "mixed")

    def test_lowercase_starts_keeps_names(self):
        notes = [{"praise": ["great soundtrack", "tight controls", "Great boss fights", "Black Mesa seen anew",
                             "Niko is lovely", "Tight, responsive controls"],
                  "complaints": [], "mood": []}]
        rec._lowercase_starts(notes)
        self.assertEqual(notes[0]["praise"], ["great soundtrack", "tight controls", "great boss fights",
                                              "Black Mesa seen anew", "Niko is lovely",
                                              "tight, responsive controls"])

    def test_mismatched_model_pair_is_refused(self):
        with open(os.path.join(self.tmp, "m.json"), encoding="utf-8") as _f:
            meta = json.load(_f)
        meta["build_id"] = "some-other-build"
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as _f:
            json.dump(meta, _f)
        with self.assertRaises(RuntimeError):
            rec.Model(os.path.join(self.tmp, "m.npz"), bad)

    def test_swap_rolls_back_when_model_is_locked(self):
        import refresh
        d = tempfile.mkdtemp(prefix="dgf_swap_")
        try:
            paths = {k: os.path.join(d, k) for k in ("live.npz", "live.json", "new.npz", "new.json")}
            for k, v in [("live.npz", "old-model"), ("live.json", "old-meta"), ("new.npz", "new-model"),
                         ("new.json", "new-meta")]:
                with open(paths[k], "w") as f:
                    f.write(v)
            real = refresh._replace

            def flaky(src, dst, tries=10):
                if dst.endswith("live.npz"):
                    raise PermissionError("locked")
                return real(src, dst, tries=1)
            log = open(os.path.join(d, "log"), "w")
            with mock.patch.object(refresh, "_replace", side_effect=flaky):
                with self.assertRaises(SystemExit):
                    refresh.swap(paths["new.npz"], paths["new.json"], paths["live.npz"], paths["live.json"], log)
            log.close()
            self.assertEqual(_read(paths["live.json"]), "old-meta")     # rolled back
            self.assertEqual(_read(paths["live.npz"]), "old-model")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_issue_negation_is_clause_aware(self):
        import re
        rx = re.compile(r"\b(bugs?|crash(es)?)\b", re.I)
        self.assertFalse(analyze_reviews._issue_hit(rx, "Honestly there are no bugs at all"))
        self.assertFalse(analyze_reviews._issue_hit(rx, "Not a single bug in 40 hours"))
        self.assertTrue(analyze_reviews._issue_hit(rx, "Never had a crash on launch, but it crashes every hour"))
        self.assertTrue(analyze_reviews._issue_hit(rx, "So many bugs everywhere"))

    def test_replace_tables_upgrades_old_schema_atomically(self):
        d = tempfile.mkdtemp(prefix="dgf_tbl_")
        try:
            con = sqlite3.connect(os.path.join(d, "t.db"))
            con.execute("CREATE TABLE game_profile (appid INTEGER, x TEXT)")      # old, different columns
            con.execute("INSERT INTO game_profile VALUES (1, 'old')")
            con.commit()
            analyze_reviews._replace_tables(con, [
                ("game_profile", "INSERT INTO game_profile VALUES (?,?,?,?,?,?,?,?)",
                 [(2, 10, "[]", "[]", "[]", "[]", "desc", 0)])])
            self.assertEqual(con.execute("SELECT appid, description FROM game_profile").fetchall(), [(2, "desc")])
            con.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
class FakeSteam:
    """GetAppReviews with N reviews per app, newest first, 100 per page."""

    def __init__(self, n=250):
        self.n = n
        self.calls = 0

    def call(self, method, input_json=None, **params):
        self.calls += 1
        assert method.startswith("IUserReviewsService/GetAppReviews")
        q = input_json
        start = 0 if q["cursor"] == "*" else int(q["cursor"])
        lo = q.get("date_range_start")
        reviews = []
        for k in range(start, min(self.n, start + q["num_per_page"])):
            ts = 2000000000 - k
            if lo and ts < lo:
                break
            reviews.append({"recommendationid": str(q["appid"] * 10000 + k), "author": {"steamid": str(k)},
                            "language": "english", "review": f"review {k}", "voted_up": True,
                            "timestamp_created": ts, "timestamp_updated": ts})
        nxt = str(start + len(reviews)) if reviews else q["cursor"]
        return {"reviews": reviews, "cursor": nxt, "total_matching": self.n}


class TestCrawler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dgf_crawl_")
        self.db = steam_reviews.DB(os.path.join(self.tmp, "c.db"))
        steam_reviews.STOP.clear()

    def tearDown(self):
        self.db.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, **kw):
        a = argparse.Namespace(max_per_game=0, languages=["all"], include_offtopic=False, raw=False,
                               update=False, refresh_edits=False)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def count(self):
        return self.db.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]

    def test_full_crawl_cap_and_resume(self):
        fake = FakeSteam(250)
        with mock.patch("sys.stdout"), mock.patch.object(steam_reviews.log, "info"):
            steam_reviews.crawl_game(7, "x", self.args(max_per_game=150), fake, self.db)
            self.assertEqual(self.count(), 200)                      # cap stops after the page that crosses it
            st = steam_reviews.load_state(self.db, 7)
            self.assertEqual(st["done"], steam_reviews.CAPPED)
            steam_reviews.crawl_game(7, "x", self.args(), fake, self.db)      # no cap -> continue to the end
        self.assertEqual(self.count(), 250)
        self.assertEqual(steam_reviews.load_state(self.db, 7)["done"], steam_reviews.COMPLETE)

    def test_update_only_fetches_new(self):
        fake = FakeSteam(120)
        with mock.patch.object(steam_reviews.log, "info"):
            steam_reviews.crawl_game(9, "x", self.args(), fake, self.db)
            calls = fake.calls
            steam_reviews.crawl_game(9, "x", self.args(update=True), fake, self.db)
        self.assertEqual(self.count(), 120)
        self.assertLessEqual(fake.calls - calls, 2)

    def test_retry_then_success(self):
        client = steam_reviews.SteamClient(0.0, max_retries=3)
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"response": {"x": 1}}'
        ok.__enter__.return_value.headers = {"x-eresult": "1"}
        err = urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "0"}, None)
        with mock.patch("urllib.request.urlopen", side_effect=[err, ok]), \
                mock.patch.object(steam_reviews, "interruptible_sleep"), \
                mock.patch.object(steam_reviews.log, "warning"):
            self.assertEqual(client.call("X/Y/v1", {"a": 1}), {"x": 1})

    def test_non_retryable_error(self):
        client = steam_reviews.SteamClient(0.0, max_retries=3)
        err = urllib.error.HTTPError("u", 403, "forbidden", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(steam_reviews.SteamError):
                client.call("X/Y/v1", {"a": 1})


# --------------------------------------------------------------------------- #
@unittest.skipUnless(os.path.exists(os.path.join(ROOT, rec.MODEL_FILE)) and
                     os.path.exists(os.path.join(ROOT, "steam.db")), "real data not present")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = rec.Model(os.path.join(ROOT, rec.MODEL_FILE), os.path.join(ROOT, rec.META_FILE))

    def test_hades_neighbours(self):
        i, _ = self.model.find("Hades")
        out = rec.rec_like(self.model, [i], (), rec.make_opts(n=5))
        names = [r["name"] for r in out["results"]]
        self.assertIn("Hades II", names)

    def test_quiz_round_and_results(self):
        rd = quiz.next_round(self.model, [], 11)
        ids = [g["appid"] for g in rd["games"]]
        self.assertEqual(len(set(ids)), 5)
        res = quiz.results(self.model, [{"shown": ids, "order": ids[:3], "unknown": ids[3:]}], rec.make_opts(n=8))
        self.assertEqual(len(res["results"]), 8)



# --------------------------------------------------------------------------- #
class TestPublishGuard(unittest.TestCase):
    """publish_site.py must never force-push over a branch it didn't make (e.g. the code)."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        import publish_site
        cls.ps = publish_site
        cls.tmp = tempfile.mkdtemp()
        cls.remote = os.path.join(cls.tmp, "remote.git")
        git = lambda *a, cwd=None: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                                                   "-c", "init.defaultBranch=main", *a], cwd=cwd, check=True,
                                                  capture_output=True)
        git("init", "-q", "--bare", cls.remote)
        work = os.path.join(cls.tmp, "work")
        git("init", "-q", work)
        with open(os.path.join(work, "app.py"), "w") as f:
            f.write("print('my code')\n")
        git("add", "-A", cwd=work)
        git("commit", "-q", "-m", "code", cwd=work)
        git("push", "-q", cls.remote, "HEAD:refs/heads/main", cwd=work)                 # the code
        with open(os.path.join(work, publish_site.MARKER), "w") as f:
            f.write("made by build_site.py\n")
        git("add", "-A", cwd=work)
        git("commit", "-q", "-m", "site", cwd=work)
        git("push", "-q", cls.remote, "HEAD:refs/heads/gh-pages", cwd=work)             # a built site

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def state(self, branch):
        with mock.patch.object(self.ps, "public_json", side_effect=urllib.error.URLError("offline")):
            return self.ps.branch_state(self.remote, "someone", "repo", branch, signed_in=False)

    def fetched_is_site(self, branch):
        out = tempfile.mkdtemp(dir=self.tmp)
        self.ps.run(["git", "init", "-q", out], quiet=True)
        return self.ps.check_fetched(out, self.remote, branch)

    def test_missing_branch_is_free(self):
        self.assertEqual(self.state("nope"), "missing")

    def test_code_branch_is_refused(self):
        self.assertEqual(self.state("main"), "unknown")             # exists: look inside before pushing
        self.assertFalse(self.fetched_is_site("main"))              # no marker: it's not ours
        with self.assertRaises(SystemExit) as e:
            self.ps.refuse("someone", "repo", "main", same_repo=False)
        self.assertIn("--gh-pages", str(e.exception))

    def test_site_branch_can_be_replaced(self):
        self.assertTrue(self.fetched_is_site("gh-pages"))
        with mock.patch.object(self.ps, "public_json", return_value={"name": ".dgf-site"}):
            self.assertEqual(self.ps.branch_state(self.remote, "someone", "repo", "gh-pages", signed_in=False), "site")

    def test_unreachable_remote_stops(self):
        with self.assertRaises(SystemExit):
            self.ps.branch_state(os.path.join(self.tmp, "no-such.git"), "someone", "repo", "main", signed_in=False)

    def test_remote_urls(self):
        self.assertEqual(self.ps.parse_github_remote("git@github.com:a/b.git"), ("a", "b"))
        self.assertEqual(self.ps.parse_github_remote("https://github.com/a/b"), ("a", "b"))
        self.assertIsNone(self.ps.parse_github_remote("https://gitlab.com/a/b.git"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
