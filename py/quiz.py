"""
quiz.py - "rank 5 games" taste quiz, built on recommend.py.

Each round shows 5 games. The player ranks the ones they have played from
favourite to least favourite and drops the others in one of three boxes -
"heard of it, no interest", "never heard of it", "interested, haven't
played" - or says they don't like any of them. Rounds adapt:

  * round 1: five big hits from five different genres
  * after favourites appear: 1 game from their best genre (chosen by similarity
    to their taste) + 4 from genres not seen yet; with 3+ favourites from
    round 4 on: 2 taste games + 3 new ones
  * "haven't heard of it" answers measure how deep their game knowledge goes
    (big hits / popular / indies / hidden gems); rounds stay at that level and
    probe one level deeper once per round while they keep knowing those games
  * low-ranked games and "don't like any of them" rounds push those genres to
    the back of the line and tone them down in the results
  * knowing or not knowing a game is taste evidence too (see KNOW_PRIOR):
    people know the well-known games of the genres they play

Results come from a weighted taste profile (favourites and "interested" +,
low ranks and "no interest" -, known/unknown games by how telling that is)
plus co-likes, excluding the games they know. A few "never heard of it" games
that fit their taste well can come back as results - that's a discovery.

The server is stateless: the client sends the whole history each time and a
random seed, so every round is reproducible.
"""
from __future__ import annotations

import re

import numpy as np

import recommend as rec

ROUNDS = 7          # default length; the player may continue to MAX_ROUNDS
MAX_ROUNDS = 12
PER_ROUND = 5

# broad genre families, each defined by Steam tags (weights come from players)
FAMILIES = {
    "Shooter": ["FPS", "Shooter", "Third-Person Shooter", "Looter Shooter", "Battle Royale",
                "Hero Shooter", "Arena Shooter", "Extraction Shooter", "Boomer Shooter", "Vehicular Combat"],
    "RPG": ["RPG", "Action RPG", "JRPG", "CRPG", "Party-Based RPG", "MMORPG", "Souls-like"],
    "Strategy": ["Strategy", "RTS", "Turn-Based Strategy", "Grand Strategy", "4X",
                 "Tower Defense", "Auto Battler", "Turn-Based Tactics", "Wargame"],
    "Building & Management": ["City Builder", "Management", "Colony Sim", "Automation",
                              "Base Building", "Resource Management"],
    "Simulation": ["Simulation", "Space Sim", "Flight", "Automobile Sim", "Job Simulator", "Physics"],
    "Cozy & Life Sim": ["Farming Sim", "Life Sim", "Cozy", "Wholesome", "Dating Sim"],
    "Survival & Crafting": ["Survival", "Open World Survival Craft", "Crafting", "Sandbox"],
    "Horror": ["Horror", "Survival Horror", "Psychological Horror"],
    "Platformer & Metroidvania": ["Platformer", "2D Platformer", "Precision Platformer",
                                  "Metroidvania", "3D Platformer"],
    "Roguelike & Deckbuilder": ["Roguelike", "Roguelite", "Action Roguelike", "Deckbuilding",
                                "Card Game", "Roguelike Deckbuilder"],
    "Story & Adventure": ["Story Rich", "Visual Novel", "Point & Click", "Walking Simulator",
                          "Interactive Fiction", "Choices Matter"],
    "Puzzle": ["Puzzle", "Logic", "Puzzle Platformer", "Hidden Object"],
    "Racing & Sports": ["Racing", "Sports", "Driving", "Football (Soccer)", "Basketball", "Golf"],
    "Fighting & Action": ["Fighting", "Hack and Slash", "Beat 'em up", "Character Action Game",
                          "Stealth"],
    "Party & Multiplayer": ["Party Game", "Social Deduction", "MOBA", "Party", "Massively Multiplayer",
                            "Online Co-Op", "Co-op", "Multiplayer"],
    "Open World": ["Open World", "Action-Adventure"],
    "Arcade & Rhythm": ["Arcade", "Rhythm", "Bullet Hell", "Shoot 'Em Up"],   # not "Music": soundtrack fans tag it
}
# broad tags that also describe many games outside their family count half
WEAK_FAMILY_TAGS = {"Simulation", "Sandbox", "Open World", "Action-Adventure", "Strategy", "RPG",
                    "Puzzle", "Story Rich", "Co-op", "Online Co-Op", "Massively Multiplayer", "Physics",
                    "Card Game", "Crafting", "Survival", "Arcade", "Choices Matter", "Management",
                    "Bullet Hell"}     # Undertale / DELTARUNE are story RPGs with bullet-hell fights
# tags that say little about genre on their own (Monster Hunter is co-op, but an RPG)
FAINT_FAMILY_TAGS = {"Co-op", "Online Co-Op", "Multiplayer"}
FAMILY_TOP_TAGS = 10        # a game's family comes from its 10 most-voted tags

# popularity tiers by review-count rank among quiz-eligible games
TIERS = [(0, 250), (250, 1500), (1500, 6000), (6000, 20000)]
TIER_NAMES = ["big hits", "popular games", "well-known indies", "hidden gems"]
FAMILIARITY = ["You know the big hits", "You know your popular games",
               "You know your indies", "You dig deep - hidden gems are your thing"]
MIN_POSITIVE = [72, 75, 80, 82]          # quality bar per tier
MIN_REVIEWS = 150
PROBE_AFTER_FAVS = 2
LONE_GAME_SCORE = 0.25      # the only game they knew in a round
FAV_THRESHOLD = 0.75        # like-score needed to count as a favourite (ranked 1st)
# the two "haven't played" boxes either side of "never heard of it": what the
# player thinks of a game they know but haven't played (ranked 1st is +1)
NO_INTEREST_SCORE = -0.6    # "heard of it, no interest"
INTERESTED_SCORE = 0.6      # "interested, haven't played"
# Knowing a game is evidence too (a small knowledge model, as in item response
# theory): people know the well-known games of the genres they play. Each
# answer counts (knew it: 1, never heard: 0) minus p, how likely THIS player was
# to know a game that well-known - their own rate for its popularity tier with
# that answer left out, pulled toward typical rates. Not knowing a big hit when
# they know most big hits points away from its kind of game; knowing a hidden
# gem points toward it; not knowing a hidden gem says almost nothing.
KNOW_PRIOR = [0.75, 0.5, 0.3, 0.15]     # typical share of players who know a game, per tier
KNOW_PRIOR_N = 2.0                      # how many answers the typical rate is worth
FAMILIARITY_TASTE = 0.3     # weight in the taste profile (results and taste slots)
FAMILIARITY_GENRE = 0.5     # weight in genre leanings (which genres the quiz explores next)
RESURFACE_UNKNOWN = 3       # at most this many "never heard of it" games come back as results
# outside the rounds (the request's "extra"): favourites the player added
# themselves, and their feedback on recommended games
FAVORITE_SCORE = 1.0        # "add a favourite we didn't show" - counts as ranked 1st
MORE_LIKE_SCORE = 1.0       # "More like this" on a result
NOT_FOR_ME_SCORE = -1.0     # "Not for me" on a result - like ranking it last
MAX_EXTRA = 50              # per list
# never used as quiz questions (still recommendable): idle/clicker "games" often
# have inflated review counts and say little about taste
QUIZ_SKIP_TAGS = ["Clicker", "Idler", "Software", "Utilities", "Benchmark"]


class QuizError(rec.RecError):
    pass


def _franchise(name: str, generic_firsts=frozenset()) -> str:
    """Rough series key so one round never shows two games of a series
    ('Dark Souls III' / 'Dark Souls Remastered' -> 'dark souls').
    generic_firsts: first words that start many unrelated series ('dragon':
    Dragon Age / Dragon Quest / Dragon's Dogma) - those keep the second word."""
    words = [w for w in rec.Model._norm(name).split() if not w.isdigit() and len(w) > 1]
    while words and words[0] in ("the", "a", "an", "sid", "tom", "tom clancy s", "sid meier s"):
        words = words[1:]
    if words and words[0] in ("meier", "clancy") and len(words) > 2:   # "sid meier s civilization"
        words = words[2:]
    if not words:
        return rec.Model._norm(name)
    # 'doom', 'xcom', 'nioh', 'witcher' - a distinctive first word is the series;
    # very short first words need the next word too ('far cry', 'age of')
    if len(words[0]) >= 4 and words[0] not in generic_firsts:
        return words[0]
    return " ".join(words[:2])


def _generic_first_words(names, min_series=6):
    """First words followed by at least `min_series` different second words."""
    seconds = {}
    for nm in names:
        w = [x for x in rec.Model._norm(nm).split() if not x.isdigit() and len(x) > 1]
        if len(w) >= 2:
            seconds.setdefault(w[0], set()).add(w[1])
    return frozenset(f for f, s in seconds.items() if len(s) >= min_series)


EDITION_WORDS = {"edition", "legendary", "definitive", "remastered", "remaster", "complete", "goty",
                 "game", "of", "the", "year", "enhanced", "director", "directors", "cut", "deluxe",
                 "ultimate", "hd", "collection", "anniversary", "classic", "royal", "golden", "reload",
                 "redux", "special", "gold", "premium", "standard", "bundle", "trilogy", "remake"}


def _edition_rest(rest_words):
    return bool(rest_words) and (rest_words[-1] == "edition" or all(w in EDITION_WORDS for w in rest_words))


def same_series(a: str, b: str) -> bool:
    """Prefix-aware: 'doom' ~ 'doom eternal', 'far cry' ~ 'far cry primal'."""
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def _series_used(key: str, used) -> bool:
    return any(same_series(key, u) for u in used)


def _is_edition_or_dlc(name_norm: str, picked_norms) -> bool:
    """'ori and the blind forest definitive edition' for a pick of 'ori and the
    blind forest' -> True; numbered sequels ('hades 2') -> False. Also the other
    way round: 'mass effect' when 'mass effect legendary edition' was picked."""
    for pn in picked_norms:
        if name_norm.startswith(pn + " "):
            rest = name_norm[len(pn) + 1:].split()
            if not rest or not rest[0].isdigit():
                return True
        elif pn.startswith(name_norm + " ") and _edition_rest(pn[len(name_norm) + 1:].split()):
            return True
    return False


class QuizIndex:
    """Per-game genre family, popularity tier and eligibility (built once)."""

    def __init__(self, model: rec.Model):
        self.m = model
        names = model.meta["tag_names"]
        col = {n: i for i, n in enumerate(names)}
        self.family_names = list(FAMILIES)
        self.family, self.purity = self._families(model, col)
        pp = np.array([p if p is not None else 0 for p in model.meta["percent_positive"]])
        order = np.argsort(-model.reviews, kind="stable")
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        self.rank = rank
        self.tier = np.full(len(order), -1)
        for t, (lo, hi) in enumerate(TIERS):
            self.tier[(rank >= lo) & (rank < hi)] = t
        ok_quality = np.zeros(len(order), bool)
        for t in range(len(TIERS)):
            ok_quality |= (self.tier == t) & (pp >= MIN_POSITIVE[t])
        skip = np.zeros(len(order), bool)
        for t in QUIZ_SKIP_TAGS:
            if t in col:
                # skip only when the tag is prominent for the game (in its top 5)
                for i in np.where(model.has_tag(col[t]))[0]:
                    if t in model.top_tags(i, 5):
                        skip[i] = True
        self.eligible = (ok_quality & (self.family >= 0) & (model.reviews >= MIN_REVIEWS) & ~skip
                         & ~model.adult & ~model.is_demo)
        generic = _generic_first_words(model.names)
        self.franchise = [_franchise(n, generic) for n in model.names]
        self.pp = pp
        # weight for "representative" picks inside a family/tier
        self.weight = (model.quality ** 3) * np.log10(10 + model.reviews)
        # how many players a genre has (well-known games) - big genres come up early
        known = self.eligible & (self.tier >= 0) & (self.tier <= 1)
        self.family_pop = np.array([np.log10(10 + model.reviews[known & (self.family == f)]).sum()
                                    for f in range(len(FAMILIES))], np.float64) + 1.0

    @staticmethod
    def _families(model, col):
        """Genre family of each game from its most-voted tags. Players' vote order
        decides, not the idf-weighted values (those inflate rare tags: 'Space Sim'
        made Kerbal Space Program a racing game). Broad tags count half."""
        fam_of_col = {}
        for f, tags in enumerate(FAMILIES.values()):
            for t in tags:
                if t in col:
                    w = 0.4 if t in FAINT_FAMILY_TAGS else 0.5 if t in WEAK_FAMILY_TAGS else 1.0
                    fam_of_col.setdefault(col[t], []).append((f, w, t in FAINT_FAMILY_TAGS))
        T, idf = model.tag, model.tag_idf
        n = T.shape[0]
        F = np.zeros((len(FAMILIES), n), np.float32)
        for i in range(n):
            s, e = T.indptr[i], T.indptr[i + 1]
            if s == e:
                continue
            idx = T.indices[s:e]
            votes = T.data[s:e] / idf[idx]
            top = np.argsort(-votes)[:FAMILY_TOP_TAGS]
            mx = float(votes[top[0]]) or 1.0
            faint = {}                               # co-op / multiplayer: the strongest one counts, once
            for j in top:
                for f, w, is_faint in fam_of_col.get(int(idx[j]), ()):
                    if is_faint:
                        faint[f] = max(faint.get(f, 0.0), w * float(votes[j]) / mx)
                    else:
                        F[f, i] += w * float(votes[j]) / mx
            for f, v in faint.items():
                F[f, i] += v
        tot = F.sum(axis=0)
        # purity: how much of a game's genre signal is its main genre (Civilization ~1,
        # World of Tanks split between shooter and strategy) - typical games teach more
        purity = np.divide(F.max(axis=0), tot, out=np.zeros(n, np.float32), where=tot > 0)
        return np.where(F.max(axis=0) > 0, F.argmax(axis=0), -1), purity

    def card(self, i):
        m = self.m
        rd = m.meta["release_date"][i]
        return {"appid": m.appids[i], "name": m.names[i],
                "genre": self.family_names[self.family[i]] if self.family[i] >= 0 else "",
                "tier": int(self.tier[i]),
                "year": int(np.datetime64(int(rd), "s").astype("datetime64[Y]").astype(int) + 1970) if rd else None,
                "reviews": int(m.reviews[i]), "percent_positive": m.meta["percent_positive"][i],
                "tags": m.top_tags(i, 3),
                "description": m.meta["descriptions"][i],
                "image": (m.meta.get("headers") or [None] * len(m.appids))[i]
                or f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{m.appids[i]}/header.jpg",
                "players": (m.meta.get("notes") or [None] * len(m.appids))[i],
                "url": f"https://store.steampowered.com/app/{m.appids[i]}/"}


_INDEX: dict[int, QuizIndex] = {}


_INDEX_LOCK = __import__("threading").Lock()


def index_for(model: rec.Model) -> QuizIndex:
    """The quiz index lives on the model object itself, so it disappears with
    the model (no stale cache across reloads, no id() reuse)."""
    qi = getattr(model, "_quiz_index", None)
    if qi is None:
        with _INDEX_LOCK:
            qi = getattr(model, "_quiz_index", None)
            if qi is None:
                qi = QuizIndex(model)
                model._quiz_index = qi
    return qi


# --------------------------------------------------------------------------- #
# history -> state
# --------------------------------------------------------------------------- #
class Round:
    __slots__ = ("shown", "order", "unknown", "none_liked", "legacy", "no_interest", "interested")

    def __init__(self, shown, order, unknown, none_liked, legacy=False, no_interest=(), interested=()):
        self.shown, self.order, self.unknown, self.none_liked = shown, order, unknown, none_liked
        self.legacy = legacy          # old "pick one" round: says nothing about which games they know
        self.no_interest, self.interested = list(no_interest), list(interested)

    @property
    def answered(self):
        """False for a round sent with nothing placed (says nothing about the games)."""
        return bool(self.order or self.unknown or self.none_liked or self.no_interest or self.interested)


def _appid_list(v, what, limit=PER_ROUND):
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > limit:
        raise QuizError(f"invalid {what} in quiz history")
    out = []
    for a in v:
        if isinstance(a, bool):
            raise QuizError(f"invalid appid in {what}")
        try:
            a = int(a)
        except (TypeError, ValueError, OverflowError):
            raise QuizError(f"invalid appid in {what}")
        if a in out:
            raise QuizError(f"duplicate game in {what}")
        out.append(a)
    return out


def parse_history(model, history):
    """history: list of rounds, each
         {"shown": [appid x5],
          "order": [appids ranked favourite -> least favourite],   # known games
          "unknown": [appids the player hasn't heard of],
          "no_interest": [appids],     # optional: heard of it, no interest
          "interested": [appids],      # optional: interested, haven't played
          "none_liked": true}          # optional: "I don't like any of them"
    Each game is in at most one of order / unknown / no_interest / interested.
    Older saved quizzes with {"pick": appid|null} are still accepted.
    Appids no longer in the catalogue (after a rebuild) are dropped."""
    if not isinstance(history, list) or len(history) > MAX_ROUNDS:
        raise QuizError("invalid quiz history")
    rounds = []
    for r in history:
        if not isinstance(r, dict):
            raise QuizError("invalid quiz round")
        shown_ids = _appid_list(r.get("shown"), "shown")
        if not shown_ids:
            raise QuizError("invalid quiz round")
        none_liked = bool(r.get("none_liked"))
        legacy = "pick" in r and "order" not in r
        noint_ids, want_ids = [], []
        if legacy:                                     # old format
            pick = r.get("pick")
            order_ids = _appid_list([] if pick is None else [pick], "pick")
            unknown_ids = []
            none_liked = pick is None
        else:
            order_ids = _appid_list(r.get("order"), "order")
            unknown_ids = _appid_list(r.get("unknown"), "unknown")
            noint_ids = _appid_list(r.get("no_interest"), "no_interest")
            want_ids = _appid_list(r.get("interested"), "interested")
        placed = order_ids + unknown_ids + noint_ids + want_ids
        if len(placed) != len(set(placed)):
            raise QuizError("a game can only be in one place (ranked, or one of the boxes)")
        if not set(placed) <= set(shown_ids):
            raise QuizError("ranked/boxed games must be among the games shown in that round")
        to_idx = lambda ids: [model.index[a] for a in ids if a in model.index]
        rounds.append(Round(to_idx(shown_ids), to_idx(order_ids), to_idx(unknown_ids), none_liked, legacy,
                            to_idx(noint_ids), to_idx(want_ids)))
    return rounds


class Extra:
    """What the player told us outside the rounds: favourites they added
    themselves, and "More like this" / "Not for me" on recommended games.
    Request format: {"favorites": [appids], "more": [appids], "less": [appids]}."""
    __slots__ = ("favorites", "more", "less")

    def __init__(self, favorites=(), more=(), less=()):
        self.favorites, self.more, self.less = list(favorites), list(more), list(less)

    def scored(self):
        return ([(i, FAVORITE_SCORE) for i in self.favorites] + [(i, MORE_LIKE_SCORE) for i in self.more]
                + [(i, NOT_FOR_ME_SCORE) for i in self.less])


def parse_extra(model, extra) -> Extra:
    if extra is None:
        return Extra()
    if not isinstance(extra, dict):
        raise QuizError("invalid extra")
    lists = {k: _appid_list(extra.get(k), k, MAX_EXTRA) for k in ("favorites", "more", "less")}
    every = [a for v in lists.values() for a in v]
    if len(every) != len(set(every)):
        raise QuizError("a game can't be a favourite, 'more like this' and 'not for me' at once")
    to_idx = lambda ids: [model.index[a] for a in ids if a in model.index]     # dropped after a rebuild
    return Extra(*(to_idx(lists[k]) for k in ("favorites", "more", "less")))


def rank_scores(n):
    """Like-scores for n ranked games, favourite first, on the fixed 5-slot
    scale: 1st +1, 2nd +0.5, 3rd 0, 4th -0.5, 5th -1. A partial ranking says
    "top / a bit lower", never "love / hate" (2 known games -> +1, +0.5).
    A single known game is weak evidence (it had to be ranked to continue)."""
    if n == 1:
        return [LONE_GAME_SCORE]
    return [1 - 2 * r / (PER_ROUND - 1) for r in range(n)]


class State:
    def __init__(self, qi: QuizIndex, rounds, extra: Extra | None = None):
        self.rounds = rounds
        extra = extra or Extra()
        F = len(qi.family_names)
        T = len(TIERS)
        self.shown = {i for r in rounds for i in r.shown}
        self.score = {}                       # game -> accumulated like score (-1 .. +1 per round)
        self.fam_score = np.zeros(F)
        self.fam_shown = np.zeros(F)
        self.known_by_tier = np.zeros(T)
        self.shown_by_tier = np.zeros(T)
        last_fav_round = np.full(F, -1)
        self.none_rounds = 0
        self.unknown = set()                  # "never heard of it"
        self.interested = set()               # "interested, haven't played"

        # an extra answer is the newest word on that game: it replaces what a round said
        # (ranked 5th, then added as a favourite -> a favourite, not -1 + 1 = 0)
        overridden = {i for i, _ in extra.scored()}

        def add(i, s, k, final=False):
            if i in overridden and not final:
                return
            self.score[i] = self.score.get(i, 0) + s
            if qi.family[i] >= 0:
                self.fam_score[qi.family[i]] += s
                if s > 0:
                    last_fav_round[qi.family[i]] = k

        for k, r in enumerate(rounds):
            self.unknown.update(r.unknown)
            self.interested.update(r.interested)
            for i in r.shown:
                if qi.family[i] >= 0:
                    self.fam_shown[qi.family[i]] += 1
                t = int(qi.tier[i])
                if not (0 <= t < T):
                    continue
                if r.legacy:                       # old rounds: only the pick is known for sure
                    if i in r.order:
                        self.shown_by_tier[t] += 1
                        self.known_by_tier[t] += 1
                elif r.answered:
                    self.shown_by_tier[t] += 1
                    if i not in r.unknown:         # ranked, boxed as heard of, or "know them, like none"
                        self.known_by_tier[t] += 1
            for i in r.no_interest:                # what they think of games they know but haven't played
                add(i, NO_INTEREST_SCORE, k)
            for i in r.interested:
                add(i, INTERESTED_SCORE, k)
            if r.none_liked:
                self.none_rounds += 1
                boxed = set(r.unknown) | set(r.no_interest) | set(r.interested)
                for i in r.shown:
                    if i not in boxed:             # can't dislike a game you never heard of
                        add(i, -0.5, k)
                continue
            # an old "pick your favourite of 5" answer is a clear favourite
            for i, s in zip(r.order, [1.0] if r.legacy else rank_scores(len(r.order))):
                add(i, s, k)
        # favourites they added and feedback on results: the newest answers. These
        # games are known now, so they are never shown in a round or recommended.
        self.added, self.more, self.less = set(extra.favorites), set(extra.more), set(extra.less)
        for i, s in extra.scored():
            add(i, s, len(rounds), final=True)
            self.shown.add(i)
        self.unknown -= overridden            # they know it now
        self.interested -= overridden
        # knowing / not knowing as evidence (see KNOW_PRIOR): for games ranked or
        # never heard of - a "no interest" / "interested" answer already says it directly.
        # All of it shapes genre leanings; the taste profile uses only the games they
        # gave no opinion of (taste_familiarity)
        self.familiar = {}
        for r in rounds:
            if r.legacy or not r.answered:
                continue
            opinion = set(r.no_interest) | set(r.interested)
            for i in r.shown:
                t = int(qi.tier[i])
                if i in opinion or i in overridden or not (0 <= t < T):
                    continue
                y = 0.0 if i in r.unknown else 1.0
                p = ((self.known_by_tier[t] - y + KNOW_PRIOR_N * KNOW_PRIOR[t])
                     / (self.shown_by_tier[t] - 1 + KNOW_PRIOR_N))
                self.familiar[i] = self.familiar.get(i, 0.0) + y - p
        self.fam_familiar = np.zeros(F)
        for i, e in self.familiar.items():
            if qi.family[i] >= 0:
                self.fam_familiar[qi.family[i]] += e
        # leaning per genre for exploring: what they liked plus which genres they know
        self.genre_lean = self.fam_score + FAMILIARITY_GENRE * self.fam_familiar
        # favourites: positively scored games, best first (later rounds break ties)
        self.favs = sorted((i for i, s in self.score.items() if s >= FAV_THRESHOLD),
                           key=lambda i: -self.score[i])
        self.dislikes = [i for i, s in self.score.items() if s < 0]
        order = sorted(range(F), key=lambda f: (-self.fam_score[f], -last_fav_round[f]))
        self.top_families = [f for f in order if self.fam_score[f] > 0]
        self.fav_franchises = {qi.franchise[i] for i in self.favs}
        # how deep does their game knowledge go? deepest tier where they know at
        # least half of what we showed (with a little evidence)
        rate = np.divide(self.known_by_tier, self.shown_by_tier,
                         out=np.zeros(T), where=self.shown_by_tier > 0)
        base = 0
        for t in range(1, T):
            # smoothed rate on 3+ answers: knowing 2 of 2 indies isn't yet "knows indies"
            smoothed = (self.known_by_tier[t] + 1) / (self.shown_by_tier[t] + 2)
            if self.shown_by_tier[t] >= 3 and smoothed >= 0.6:
                base = t
        self.base_tier = base
        self.know_rate = rate
        self.knows_base = self.shown_by_tier[base] > 0 and rate[base] >= 0.5
        self.last_none = bool(rounds) and rounds[-1].none_liked
        nxt = min(T - 1, base + 1)
        # probe one tier deeper ("do you know this indie?"): once there is some
        # evidence, at most once per round, and stop when they mostly don't know it
        deeper_ok = self.shown_by_tier[nxt] < 3 or rate[nxt] >= 0.34
        self.can_probe = (len(self.favs) >= PROBE_AFTER_FAVS and not self.last_none
                          and base < T - 1 and deeper_ok)
        # knew every deeper game so far: probe twice a round to find their level sooner
        self.eager_probe = (self.can_probe and self.shown_by_tier[nxt] >= 2
                            and self.known_by_tier[nxt] == self.shown_by_tier[nxt])


# --------------------------------------------------------------------------- #
# taste profile
# --------------------------------------------------------------------------- #
def taste_familiarity(st):
    """Knew-it / never-heard evidence for the taste profile (see KNOW_PRIOR), and
    the mass it is divided by. Only games they gave no opinion of count: a rank,
    a box or a rejection already says how they feel about a game they know (and
    'knowing' a game they turned down mustn't pull toward it). The scale is ALL
    their liked games, not one interest's, so a small interest isn't swamped,
    and at least the evidence's own size, so it stays a bounded nudge."""
    if not FAMILIARITY_TASTE:
        return [], 1.0
    fam = [(i, e) for i, e in getattr(st, "familiar", {}).items() if e and not st.score.get(i, 0)]
    liked = sum(s for s in st.score.values() if s > 0)
    return fam, max(liked, sum(abs(e) for _, e in fam), 1e-9)


def taste_scores(model, st, dislike_weight=0.5, only=None):
    """Content similarity to a weighted taste profile (favourites positive,
    low-ranked games negative) + co-likes of favourites, blended.
    only: restrict the liked side to these games (one interest)."""
    pos = [(i, s) for i, s in st.score.items() if s > 0 and (only is None or i in only)]
    neg = [(i, -s) for i, s in st.score.items() if s < 0]
    fam, fam_scale = taste_familiarity(st)

    def profile(M):
        v = np.zeros(M.shape[1], np.float32)
        if pos:
            idx, w = zip(*pos)
            v += np.asarray(M[list(idx)].T @ np.array(w, np.float32)).ravel() / sum(w)
        if neg:
            idx, w = zip(*neg)
            v -= dislike_weight * np.asarray(M[list(idx)].T @ np.array(w, np.float32)).ravel() / sum(w)
        if fam and pos:
            idx, e = zip(*fam)
            v += FAMILIARITY_TASTE * np.asarray(M[list(idx)].T @ np.array(e, np.float32)).ravel() / fam_scale
        v = np.maximum(v, 0)
        n = np.linalg.norm(v)
        return v / n if n else v
    tp, xp = profile(model.tag), profile(model.text)
    content = 0.6 * (model.tag @ tp) + 0.4 * (model.text @ xp)
    # what players say (story, difficulty, mood, problems) refines the match
    ap = model.aspect_profile([(i, s) for i, s in pos] + [(i, -dislike_weight * w) for i, w in neg])
    content = model.with_aspects(content, ap)
    cf = np.zeros(len(model.appids), np.float32)
    linked = 0
    for i, s in pos:
        row = model.cf[i]
        if row.nnz:
            linked += 1
            cf[row.indices] += s * row.data
    return content, cf, linked, tp


# store tags too broad to say anything about taste
GENERIC_TAGS = {"Singleplayer", "Multiplayer", "Indie", "Action", "Adventure", "Casual", "Simulation",
                "Strategy", "RPG", "2D", "3D", "Free to Play", "Early Access", "Great Soundtrack",
                "Colorful", "Controller", "Replay Value", "Classic", "Masterpiece", "Addictive",
                "Beautiful", "Family Friendly", "First-Person", "Third Person", "Realistic", "Stylized",
                "Fantasy", "Sci-fi", "Cute", "Funny", "Atmospheric", "Story Rich", "Difficult",
                "Immersive", "Character Customization", "Mature", "Violent", "Gore", "Nudity",
                "Sexual Content", "Moddable", "Co-op", "Online Co-Op", "PvP", "PvE", "Massively Multiplayer",
                "Exploration", "Open World", "Sandbox", "Survival", "Horror", "Puzzle", "Platformer",
                "Combat",
                "Arcade", "Retro", "Old School", "Comedy", "Dark", "Relaxing",
                "Short", "Female Protagonist", "Multiple Endings", "Choices Matter", "Cartoony",
                "Pixel Graphics", "Psychological", "Physics", "Building", "Crafting", "Management",
                # format / meta tags - not a taste
                "Remake", "Remaster", "Episodic", "Level Editor", "eSports", "VR", "Demo",
                "Sequel", "Free to Play", "Massively Multiplayer", "Inventory Management",
                "Steam Workshop", "Steam Achievements", "Full controller support", "Soundtrack",
                "Software", "Utilities", "Benchmark", "Kickstarter", "Crowdfunded", "Cult Classic",
                "Great for Beginners", "Experimental", "Instrumental Music", "Electronic Music",
                "Rock Music", "Movie", "Documentary", "Episodic"}
# mood words that describe problems rather than a feel someone goes for
# genre-level tags: a style only when 3+ favourites share them (an FPS fan's FPS)
BROAD_STYLE_TAGS = {"Shooter", "FPS", "Anime", "Horror", "Puzzle", "Platformer", "Strategy", "RPG",
                    "Survival", "Open World", "Sandbox", "Crafting", "Management", "Building"}
NEGATIVE_MOODS = {"grindy", "frustrating", "janky", "stressful", "overwhelming", "repetitive",
                  "maddening", "buggy", "tedious", "punishing", "rage-inducing", "boring", "clunky",
                  "toxic", "unfinished", "sweaty", "annoying", "unfair", "broken", "laggy", "shallow",
                  "confusing", "slow", "grim"}


def _because(model, games, k=2):
    return [model.names[i] for i in games[:k]]


def taste_details(model, st):
    """A specific picture of the player's taste, each point with the games it
    comes from:
      styles  - specific Steam tags (sub-genres) of their favourites
      loves   - what players say about their favourites (aspects from reviews)
      moods   - how their favourites feel (from the written descriptions)
      praised - the best-supported praise of each top favourite
      avoids  - aspects / styles of games they ranked low or turned down
      radar   - genres whose well-known games they had never heard of
    """
    pos = sorted(((i, s) for i, s in st.score.items() if s > 0), key=lambda x: -x[1])
    neg = sorted(((i, -s) for i, s in st.score.items() if s < 0), key=lambda x: -x[1])
    # styles and loves describe favourites (ranked 1st); mild likes only if there are too few
    # "More like this" games steer the results but aren't games they played - the
    # summary describes the games they ranked / added (those only if nothing else)
    played = [(i, s) for i, s in pos if i not in getattr(st, "more", ())] or pos
    favs = [(i, s) for i, s in played if s >= FAV_THRESHOLD]
    core = favs if len(favs) >= 2 else played
    names = model.meta["tag_names"]
    out = {"styles": [], "loves": [], "moods": [], "praised": [], "avoids": [], "radar": []}

    # styles: tags weighted by like-score x players' votes x sqrt(rarity) - rare
    # enough to be telling (Metroidvania > Platformer), not so much that an incidental
    # tag (Time Travel, Bullet Hell) wins; a tag must be in the game's top 8
    def tag_votes(games):
        votes, src = {}, {}
        for i, w in games:
            row = model.tag[i]
            top = set(model.top_tags(i, 8))
            for c, v in zip(row.indices, row.data):
                nm = names[c]
                if nm in GENERIC_TAGS or nm not in top:
                    continue
                votes[nm] = votes.get(nm, 0) + w * v / np.sqrt(model.tag_idf[c])
                src.setdefault(nm, []).append(i)
        for nm in [n for n in votes if n in BROAD_STYLE_TAGS and len(set(src[n])) < 3]:
            del votes[nm]
        return votes, src
    votes, src = tag_votes(core)
    ranked, seen_sets = [], []
    for nm, _ in sorted(votes.items(), key=lambda x: -x[1]):
        games = frozenset(src[nm])
        if games in seen_sets and len(games) <= 2:     # 'Party Game' / 'Party' / 'Board Game' from
            continue                                   # the same two games say one thing once
        seen_sets.append(games)
        ranked.append(nm)
    shared = [nm for nm in ranked if len(set(src[nm])) >= 2]      # a pattern, not one game's tag
    single = [nm for nm in ranked if len(set(src[nm])) < 2]
    picked = shared[:6] if len(shared) >= 4 else shared + single[:6 - len(shared)]
    for nm in sorted(picked, key=lambda n: -votes[n]):
        out["styles"].append({"name": nm, "because": _because(model, src[nm])})

    # loves: what players praise in their favourites (positive games only);
    # avoids: what their low-ranked games stand out for that NO favourite has
    if model.aspects.size:
        meta = model.aspect_meta
        A = model.aspects.shape[1]
        vp, vn = np.zeros(A, np.float32), np.zeros(A, np.float32)
        for i, s in core:
            if model.has_aspects[i]:
                vp += s * model.aspects_pos[i]
        for i, w in neg:
            if model.has_aspects[i]:
                vn += w * model.aspects_pos[i]
        fav_has = np.zeros(A, bool)
        for i, s in pos:
            if s >= FAV_THRESHOLD:
                fav_has |= model.aspects[i] > 0
        cands = []                                    # (label, games) by strength
        for j in np.argsort(-vp):
            if vp[j] <= 0.15 or len(cands) >= 14:
                break
            if meta[j]["kind"] == "issue":
                continue
            games = [i for i, _ in core if model.aspects[i, j] > 0]
            if games:
                cands.append((meta[j]["taste_label"], games))
        shared = [c for c in cands if len(c[1]) >= 2]   # seen in 2+ favourites = a real pattern
        keep = shared[:6] if len(shared) >= 4 else shared + [c for c in cands if len(c[1]) < 2][:6 - len(shared)]
        for label, games in sorted(keep, key=lambda c: cands.index(c)):
            out["loves"].append({"label": label, "because": _because(model, games)})
        loved = {x["label"] for x in out["loves"]}
        for j in np.argsort(-vn):
            if vn[j] <= 0.15 or len(out["avoids"]) >= 3:
                break
            # only how a game FEELS (scary, grindy, slow...) explains a low rank; its good
            # story or soundtrack doesn't mean they dislike stories or soundtracks
            if meta[j]["kind"] != "trait" or fav_has[j] or meta[j]["taste_label"] in loved or vp[j] > 0.5 * vn[j]:
                continue
            games = [i for i, _ in neg if model.aspects[i, j] > 0]
            if len(games) >= 2:                    # one low-ranked game is too little to go on
                out["avoids"].append({"label": meta[j]["taste_label"], "because": _because(model, games)})

    # avoids from styles: tags common in low-ranked games but absent from favourites
    nvotes, nsrc = tag_votes(neg)
    for nm, _ in sorted(nvotes.items(), key=lambda x: -x[1]):
        if len(out["avoids"]) >= 4:
            break
        if nm not in votes and len(set(nsrc[nm])) >= 2 and not any(a["label"] == nm for a in out["avoids"]):
            out["avoids"].append({"label": nm, "because": _because(model, nsrc[nm])})

    # not on their radar: genres whose well-known games were new to them (and that
    # they liked nothing in) - only once they know a few games, or it's everything
    if hasattr(st, "fam_familiar") and st.known_by_tier.sum() >= 3:
        qi = index_for(model)
        for f in np.argsort(st.fam_familiar, kind="stable"):
            if st.fam_familiar[f] > -0.5 or len(out["radar"]) >= 3:
                break
            if any(s > 0 and qi.family[i] == f for i, s in st.score.items()):
                continue                      # they like (or want to play) something in it
            games = sorted((i for i in st.unknown if qi.family[i] == f), key=lambda i: qi.rank[i])
            if games:
                out["radar"].append({"label": qi.family_names[f], "because": _because(model, games)})

    # moods and praise from the written descriptions (top games) / measured traits
    notes = model.meta.get("notes") or []
    mood_w, trait_w = {}, {}
    for i, s in core:
        n = notes[i] if i < len(notes) else None
        if not n:
            continue
        for m in n.get("mood") or []:                 # written mood words (top games)
            if m.lower() not in NEGATIVE_MOODS:
                mood_w[m.lower()] = mood_w.get(m.lower(), 0) + s
        for t in n.get("traits") or []:               # measured feel (fallback)
            if not any(b in t.lower() for b in NEGATIVE_MOODS) and t.lower() not in ("short", "steep learning curve"):
                trait_w[t.lower()] = trait_w.get(t.lower(), 0) + s
        if n.get("praise") and len(out["praised"]) < 3 and s >= FAV_THRESHOLD:
            out["praised"].append({"game": model.names[i], "points": n["praise"][:2]})
    for i, w in neg:                                   # a feel their low-ranked games share counts less
        n = notes[i] if i < len(notes) else None
        for m in (n or {}).get("mood") or []:
            if m.lower() in mood_w:
                mood_w[m.lower()] -= 0.5 * w
        for t in (n or {}).get("traits") or []:
            if t.lower() in trait_w:
                trait_w[t.lower()] -= 0.5 * w
    moods = [m for m, v in sorted(mood_w.items(), key=lambda x: -x[1]) if v > 0][:5]
    if len(moods) < 3:
        moods += [t for t, v in sorted(trait_w.items(), key=lambda x: -x[1])
                  if v > 0 and t not in moods][:3 - len(moods)]
    out["moods"] = moods
    return out


def _disliked_mask(qi, st):
    """Games in genres the player scored negatively overall."""
    fams = [f for f in range(len(qi.family_names)) if st.fam_score[f] < 0]
    return np.isin(qi.family, fams) if fams else np.zeros(len(qi.family), bool)


# --------------------------------------------------------------------------- #
# choosing games
# --------------------------------------------------------------------------- #
def _pool(qi, st, family, tiers, used_franchises, extra_mask=None):
    ok = qi.eligible & (qi.family == family) & np.isin(qi.tier, tiers)
    if extra_mask is not None:
        ok &= extra_mask
    idx = np.where(ok)[0]
    return [i for i in idx if i not in st.shown and not _series_used(qi.franchise[i], used_franchises)]


def _weighted_choice(rng, items, weights, top=30):
    if not items:
        return None
    w = np.asarray(weights, float)
    order = np.argsort(-w)[:top]
    w = w[order]
    w = w / w.sum() if w.sum() > 0 else None
    return items[int(rng.choice(order, p=w))]


def _tier_sequence(t):
    """Preferred tier first, then neighbours (deeper first, then shallower)."""
    seq = [t]
    for d in (1, -1, 2, -2, 3, -3):
        if 0 <= t + d < len(TIERS):
            seq.append(t + d)
    return seq


def _pick_explore(qi, st, rng, family, tier, used):
    """A typical, well-liked game of the genre: a first look at Strategy should be
    Civilization-like, not a tank shooter that players also tag 'Wargame'."""
    for t in _tier_sequence(tier):
        pool = _pool(qi, st, family, [t], used)
        if pool:
            return _weighted_choice(rng, pool, qi.weight[pool] * qi.purity[pool] ** 2)
    return None


def _pick_exploit(qi, st, rng, family, tier, used, sim):
    for t in _tier_sequence(tier):
        pool = _pool(qi, st, family, [t], used)
        if pool:
            s = sim[pool]
            # mostly similarity, a little quality; sample among the best few
            return _weighted_choice(rng, pool, (s - s.min() + 1e-6) ** 2 * qi.weight[pool], top=6)
    return None


def _exploit_families(qi, st):
    """Genres for the taste slots: the newest favourite's genre first (a fresh
    signal - Stardew ranked 1st in round 2 shouldn't wait behind round 1's pick),
    then the strongest genres overall."""
    out = []
    for r in reversed(st.rounds):
        for i in r.order[:1]:
            f = int(qi.family[i])
            if st.score.get(i, 0) >= FAV_THRESHOLD and f >= 0 and st.fam_score[f] > 0:
                out.append(f)
        if out:
            break
    return out + [f for f in st.top_families if f not in out]


def next_round(model: rec.Model, history, seed=0, adaptive=True, extra=None):
    qi = index_for(model)
    rounds = parse_history(model, history)
    k = len(rounds)
    if k >= MAX_ROUNDS:
        raise QuizError("that's the most rounds we can do - see your results")
    st = State(qi, rounds, parse_extra(model, extra))
    if not adaptive:            # baseline for evaluation: ignore the answers
        st.favs, st.top_families, st.last_none, st.base_tier, st.can_probe = [], [], False, 0, False
        st.fam_score[:] = 0
        st.genre_lean[:] = 0
        st.score, st.familiar = {}, {}
    rng = np.random.default_rng([abs(int(seed)) % (2 ** 32), k])
    # no game from a series already shown (Elden Ring -> Nightreign, Portal -> Portal 2)
    used: set[str] = ({qi.franchise[i] for i in st.shown} | set(st.fav_franchises)) if adaptive else set()
    games, hint_parts = [], []

    # slots that follow the player's taste
    n_exploit = 0
    if st.favs:
        n_exploit = 2 if (len(st.favs) >= 3 and k >= 3) else 1
    probe_tier = min(len(TIERS) - 1, st.base_tier + 1)
    probe_left = (2 if st.eager_probe else 1) if st.can_probe else 0

    if n_exploit and st.top_families:
        content, cf, linked, _ = taste_scores(model, st)
        sim = rec.blend(model, content, cf, rec.make_opts(), n_linked=linked)
        fams = (_exploit_families(qi, st) * 2)[:n_exploit]
        for f in fams:
            # a taste slot is the best place for the (single) deeper probe:
            # a lesser-known game in a genre they already like
            probing = probe_left > 0
            i = _pick_exploit(qi, st, rng, f, probe_tier if probing else st.base_tier, used, sim)
            if i is not None:
                if probing and qi.tier[i] > st.base_tier:
                    probe_left -= 1
                games.append(i); used.add(qi.franchise[i]); st.shown.add(i)
        if games:
            fam_ids = list(dict.fromkeys(int(qi.family[i]) for i in games))

            def because(f):            # their best-ranked game in that genre (played ones first)
                cands = [i for i, s in st.score.items() if s > 0 and qi.family[i] == f]
                return max(cands, key=lambda i: (i not in st.interested, st.score[i])) if cands else st.favs[0]
            if len(games) == 1:
                b = because(fam_ids[0])
                hint_parts.append(f"1 more {qi.family_names[fam_ids[0]]} game " +
                                  (f"(you'd like to play {model.names[b]})" if b in st.interested
                                   else f"(you added {model.names[b]} as a favourite)" if b in st.added
                                   else f"(you wanted more like {model.names[b]})" if b in st.more
                                   else f"(you ranked {model.names[b]} high)"))
            else:
                why = "; ".join(f"{qi.family_names[f]} like {model.names[because(f)]}" for f in fam_ids)
                hint_parts.append(f"{len(games)} for your taste ({why})")

    n_taste = len(games)

    # exploration: genres not seen yet first, then least-shown; genres they
    # ranked low or turned down go last
    taken = {int(qi.family[i]) for i in games}
    F = len(qi.family_names)
    # random order weighted by genre size (Efraimidis-Spirakis keys): Shooter, RPG and
    # Strategy come up early more often than Racing & Sports, but any genre can
    jitter = -(rng.random(F) ** (1.0 / (qi.family_pop / qi.family_pop.max())))

    def explore_key(f):
        # liked genres not in this round's taste slots stay in rotation (a third
        # liked genre would otherwise never come back)
        disliked = st.genre_lean[f] < 0      # ranked low, or its well-known games were new to them
        passed_over = st.fam_shown[f] >= 2 and st.fam_score[f] <= 0 and len(st.favs) > 0
        # a genre they told us they like (an added favourite) that no round has shown yet: first
        told_us = st.fam_score[f] > 0 and st.fam_shown[f] == 0
        return (disliked, passed_over, not told_us, st.fam_shown[f], jitter[f])
    families = sorted((f for f in range(F) if f not in taken), key=explore_key)

    n_probe = probe_left
    for f in families:
        if len(games) >= PER_ROUND:
            break
        probing = n_probe > 0
        tier = min(len(TIERS) - 1, st.base_tier + 1) if probing else st.base_tier
        i = _pick_explore(qi, st, rng, f, tier, used)
        if i is None:
            continue
        if probing and qi.tier[i] > st.base_tier:
            n_probe -= 1
        games.append(i); used.add(qi.franchise[i]); st.shown.add(i)

    if len(games) < PER_ROUND:                 # rare: fill from anything eligible, other series first
        for strict in (True, False):           # last resort allows a second game of a series
            pool = [i for i in np.where(qi.eligible)[0]
                    if i not in st.shown and i not in games and (not strict or not _series_used(qi.franchise[i], used))]
            while len(games) < PER_ROUND and pool:
                i = _weighted_choice(rng, pool, qi.weight[pool], top=200)
                pool.remove(i); games.append(i); used.add(qi.franchise[i])
                if strict:                     # keep the pool free of series now in use
                    pool = [j for j in pool if not _series_used(qi.franchise[j], used)]
    if len(games) < PER_ROUND:                 # the catalogue is nearly exhausted
        pool = [i for i in np.where(model.reviews >= MIN_REVIEWS)[0] if i not in st.shown and i not in games]
        while len(games) < PER_ROUND and pool:
            i = pool.pop(int(rng.integers(len(pool))))
            games.append(i)

    # hint
    if k == 0 and not n_taste:
        hint = "Five big hits from five different genres."
    else:
        n_new = len(games) - n_taste
        parts = list(hint_parts)
        if st.last_none:
            parts.append(f"{n_new} game{'s' if n_new != 1 else ''} from other genres")
        else:
            parts.append(f"{n_new} new direction{'s' if n_new != 1 else ''}")
        hint = ", ".join(parts) + "."
        if st.can_probe and probe_tier >= 2:
            hint += f" One is from the {TIER_NAMES[probe_tier]} - know it?"

    order = rng.permutation(len(games))
    return {"round": k + 1, "rounds": ROUNDS, "max_rounds": MAX_ROUNDS,
            "games": [qi.card(games[j]) for j in order], "hint": hint}


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def results(model: rec.Model, history, opts=None, extra=None):
    """extra: favourites they added and their "More like this" / "Not for me"
    feedback on earlier results (see Extra) - "Update results" sends it all."""
    qi = index_for(model)
    rounds = parse_history(model, history)
    ex = parse_extra(model, extra)
    if not rounds and not (ex.favorites or ex.more):
        raise QuizError("play at least one round first")
    st = State(qi, rounds, ex)
    opts = opts or rec.make_opts()
    notes = []
    exclude = sorted(st.shown)
    want = max(1, min(int(opts.n), 100))
    wide = rec.make_opts(**vars(opts))
    wide.n = min(300, want * 4)                  # extra rows to survive de-duplication
    disliked = _disliked_mask(qi, st)
    if not any(s > 0 for s in st.score.values()):
        # nothing appealed yet: widely loved games, away from the genres they turned down
        notes.append("Nothing grabbed you yet - here are widely loved games from other genres. "
                     "Try a few more rounds, or 'Describe my dream game'!")
        score = qi.weight * (1 + (qi.rank < 1500))
        score = score * np.where(disliked, 0.3, 1.0)
        if opts.gems:
            pop = np.log10(1 + model.reviews) / np.log10(1 + model.reviews.max())
            score = score * np.clip(1.25 - 1.0 * pop, 0.05, None)
        # only well-reviewed games (the quiz quality bar) - fall back to all if filters leave none
        ok = rec.apply_filters(model, opts, exclude=exclude, notes=notes)
        rows = rec.top_rows(model, score, ok & qi.eligible if (ok & qi.eligible).any() else ok, wide)
    else:
        # games they know are left out; "never heard of it" ones may come back (a discovery)
        back = (st.unknown - set(st.score)) if RESURFACE_UNKNOWN > 0 else set()   # never rated anywhere
        ok = rec.apply_filters(model, opts, exclude=sorted(st.shown - back), notes=notes)
        damp = np.where(disliked, 0.8, 1.0)            # gentle: a few rounds are weak evidence

        def scorer(games):
            c, f, lk, gtp = taste_scores(model, st, only=set(games))
            return rec.blend(model, c, f, opts, n_linked=lk) * damp, gtp
        pos = [(i, s) for i, s in st.score.items() if s > 0]
        groups = rec.interest_groups(model, [(i, s * s) for i, s in pos])
        overall = scorer([i for i, _ in pos])[0] if len(groups) > 1 else None
        shown_norms = [rec.Model._norm(model.names[p]) for p in st.shown]
        rated = [(qi.franchise[j], model.meta["release_date"][j] or 0) for j in st.score]
        used_ = []
        n_back = [0]

        def accept(i):
            """one game per series; no edition of a game already shown; no older
            entry of a series they rated (Xenoverse 1 after Xenoverse 2); only a
            few games from the quiz they'd never heard of"""
            if i in back and n_back[0] >= RESURFACE_UNKNOWN:
                return False
            key = qi.franchise[i]
            if _is_edition_or_dlc(rec.Model._norm(model.names[i]), shown_norms) or _series_used(key, used_):
                return False
            rd = model.meta["release_date"][i] or 0
            if any(same_series(key, k) and rd < d for k, d in rated):
                return False
            used_.append(key)
            n_back[0] += i in back
            return True
        n_opts = rec.make_opts(**vars(opts))
        n_opts.n = want
        rows = rec.multi_interest_rows(model, groups, scorer, ok, n_opts, overall=overall, accept=accept)
        ap = model.aspect_profile(pos)
        for r in rows:
            shared = model.shared_aspects(model.index[r["appid"]], ap)
            if shared:
                r["why"] = (r["why"] + "; " if r["why"] else "") + "players praise its " + " and ".join(shared)
            if model.index[r["appid"]] in back:
                r["from_quiz"] = True
                r["why"] = "New to you in the quiz, and it fits your taste" + ("; " + r["why"] if r["why"] else "")
    # one game per series, no editions/DLC of a favourite, sequels welcome (once)
    fav_norms = [rec.Model._norm(model.names[p]) for p in st.favs]
    kept, used = [], []
    for r in rows:
        i = model.index[r["appid"]]
        key = qi.franchise[i]
        if _is_edition_or_dlc(rec.Model._norm(model.names[i]), fav_norms) or _series_used(key, used):
            continue
        used.append(key)
        kept.append(r)
        if len(kept) >= want:
            break
    rows = kept
    for r in rows:
        i = model.index[r["appid"]]
        r["genre"] = qi.family_names[qi.family[i]] if qi.family[i] >= 0 else ""
    T = len(TIERS)
    known = int(st.known_by_tier.sum())
    seen = int(st.shown_by_tier.sum())
    # "Your favourites": ranked 1st or added by them ("More like this" games only if there's nothing else)
    picks = ([i for i in st.favs if i not in st.more]
             or sorted((i for i, s in st.score.items() if s > 0 and i not in st.more),
                       key=lambda i: (i in st.interested, -st.score[i]))
             or st.favs)
    profile = {
        "picks": [qi.card(i) for i in picks[:8]],
        "genres": [{"name": qi.family_names[f], "score": round(float(st.fam_score[f]), 2)}
                   for f in st.top_families],
        "familiarity": (FAMILIARITY[st.base_tier] if st.knows_base else
                        "You're new to most of these games - great, lots to discover") if seen else "",
        "known": known, "seen": seen,
        **taste_details(model, st),
        "rounds": len(rounds), "none_rounds": st.none_rounds,
    }
    return {"results": rows, "notes": notes, "profile": profile}
