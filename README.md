# Steam dream-game finder

| file | what it does |
|---|---|
| `steam_reviews.py` | Collects the Steam catalogue (~200k games) and player reviews into `steam.db` (SQLite) using Steam's official, key-less APIs. Resumable, rate-limited. |
| `analyze_reviews.py` + `aspects.json` | Reads every review and works out what players praise, complain about and how each game feels (65 aspects, 7 languages). |
| `recommend.py` | Builds the recommender from `steam.db` and answers queries (CLI + library). |
| `quiz.py` | The adaptive "sort 5 games" taste quiz. |
| `server.py` + `web/` | The website: a local web server and the page. |
| `build_site.py` + `webapi.py` + `web/engine.js` | The free online version: a static site where the recommender runs in the visitor's browser, plus a page per game and a sitemap for Google. |
| `publish_site.py` | Puts that static site on GitHub Pages (free). |
| `refresh.py` | One command to update everything (crawl → analyse → test → rebuild → swap in). |
| `tests/` | Automated tests (`python -m unittest discover -s tests`). |

Requirements: Python 3.10+, `numpy`, `scipy`. Optional AI mode: `pip install anthropic` plus Claude API credentials (`ANTHROPIC_API_KEY`, or `ant auth login`).

## 1. Collect data

```bash
python steam_reviews.py games                                   # catalogue, ~3 min
python steam_reviews.py reviews --limit-games 3000 --max-per-game 300   # good starter set
python steam_reviews.py reviews --min-reviews 50                # everything (weeks)
```

Stop with Ctrl+C at any time and re-run the same command to continue.

## 2. Analyse reviews and build the recommender

```bash
python analyze_reviews.py          # what players say about each game (uses all CPU cores)
python recommend.py build
```

Or do everything at once, including fresh reviews, tests and a safe swap of the model:

```bash
python refresh.py                  # new reviews for crawled games -> analyse -> test -> rebuild
python refresh.py --catalogue --more-games 12000 --per-game 200    # also new games, more coverage
```

The running website picks up a rebuilt model by itself within a minute. If the new model fails its checks, the old one stays.

## 3. Use the website

On Windows, double-click **`start.bat`**. It installs numpy and scipy if they're missing, builds the model if needed, starts the server and opens your browser. Keep its window open while you use the site; close the window to stop.

Or start it by hand:

```bash
python server.py --open
```

The site is at http://127.0.0.1:8000.

**Quick quiz (main flow):** for 7 rounds, drag the games you've played into the ranking slots from favourite to least favourite (mouse, touch or pen). Drop each game you haven't played into one of three boxes under the ranking: **Heard of it, no interest** (left), **Never heard of it** (middle) or **Interested, haven't played** (right). Then press **Next round**. You can drag games between slots and boxes, back up to the top, or out again; the cards show where a game will land while you drag. **I don't like any of them** rejects the whole round.

Without a mouse: `1`–`5` put that game in the next place, `Shift`+`1`–`5` mark it never heard of. On a selected card, `Enter` ranks it, `N` means heard of it, no interest, `H` never heard of it and `I` interested. `Backspace` undoes, `Enter` (with no card selected) goes to the next round, `0` means don't like any, `Esc` cancels a drag.

How your answers are used:
* **Rankings:** 1st counts as +1, 2nd +0.5, 3rd 0, 4th -0.5 and 5th -1. If you've played only 2 of the games, the 2nd is still a mild like, not a dislike. A game that's the only one you've played in a round counts only a little. Your *favourites* (the taste summary is built from them) are the games you ranked 1st.
* **Heard of it, no interest:** counts as a clear dislike (-0.6) of that kind of game.
* **Interested, haven't played:** counts as a like (+0.6). It isn't a favourite, because you haven't played it, and it isn't recommended back to you, because you know it already.
* **Never heard of it:** two things.
  * It shows how well you know big hits, popular games, indies and hidden gems. Rounds stay at your level, and one game per round goes a level deeper while you keep recognising those.
  * It's also taste evidence, because people know the well-known games of the genres they play. Each answer is compared with how likely *you* were to know a game that well-known. Never having heard of a big hit, when you know most big hits, points away from that kind of game. Knowing a hidden gem points toward it. Not knowing a hidden gem says almost nothing. Genres whose well-known games were new to you are explored later and listed as **Not on your radar**.
  * A game you'd never heard of that fits your taste well can come back in your results, at most 3, marked **New to you**: that's a discovery.
* **Next rounds:** 1–2 games from your best genres (the genre of your newest 1st place comes first, so a new favourite is followed up right away), plus new genres. A genre is introduced with a typical game of it (Civilization for Strategy), big genres tend to come up early, and no series is shown twice.
* **Low ranks and "don't like any":** those genres move to the back of the line and are toned down in your results.
* **Your own favourites:** under the quiz, **Got favourites we didn't show?** lets you search for and add up to 10 games. Each counts as ranked 1st, so it becomes one of your favourites. Added during the quiz, it shapes the next rounds (its genre is followed up right away). Added on the results page, it counts when you press **Update results**.

At the end you get recommendations plus a specific taste summary. Every point names the games it comes from, and points shared by two or more of your favourites come first:
* **your styles:** precise sub-genres such as Metroidvania or Souls-like;
* **what you love:** from what players say, e.g. great boss fights or story-driven games;
* **the feel you go for;**
* **what players praise in your favourites;**
* **what you're less into;**
* **what's not on your radar:** genres whose well-known games you'd never heard of.

From there you can play 3 more rounds or fine-tune the results.

**Tune your results:** on each recommended game, **More like this** and **Not for me** mark it. Click again to unmark. Nothing changes until you press **Update results**, either in the bar that appears at the bottom or the button above the results. Then everything is recalculated:
* a "More like this" game counts like a game you ranked 1st;
* a "Not for me" game counts like one you ranked last, and its genre is toned down;
* both games drop out of the list, since you've answered them.

Your marks are listed under **Your feedback**, where you can take any of them back. They also carry over if you play more rounds.

Result cards show what players say: a written summary for the most-reviewed games, then ✓ praise, mood, ⚠ complaints, "players split on …" and who it's best for.

The other tabs:
* pick games you love (with autocomplete);
* describe your dream game in words: **Ask Google AI Mode** opens Google's AI Mode (free, no sign-in) in a new tab with your wish, and asks it to use this site's game pages as a source. You can also still search our own catalogue with the text;
* turn on hidden gems, filter by platform (including Steam Deck), rating, release year or tags;
* use "More like this" and "Not for me" on any result;
* share the URL: it holds the whole search, including filters.

AI mode is optional and needs a Claude API key, which is billed per use (a Claude.ai Pro/Max subscription doesn't include API access). To turn it on, set `ANTHROPIC_API_KEY` before starting the server. Everything else works without a key.

To use the site from your phone on the same Wi-Fi, run `python server.py --host 0.0.0.0` and open `http://<your PC's IP>:8000`. Anyone on that network can then use the site, including AI mode on your Claude credits, so do this only on a network you trust.

The server only accepts requests from its own page, which stops other websites from using it. It needs only Python's standard library, plus numpy and scipy.

## 4. Put it online for free

The online version needs no server, so it costs nothing to host and never sleeps. `build_site.py` writes a static site; in the visitor's browser, [Pyodide](https://pyodide.org) (Python compiled to WebAssembly, from the jsDelivr CDN) runs the same `recommend.py`, `quiz.py` and `server.py` on a slimmed model:
* only games with 50+ reviews (the site never recommends fewer), the 64 strongest words per game, and compact number types: 14 MB instead of 74 MB. In the eval it scores 16.8% top-20 hits, against 17.1% for the same games at full size;
* round 1 of the quiz is precomputed, so it starts at once while Python and the data load in the background. That's about 35 MB on the first visit and is cached after that;
* nothing the visitor does is sent anywhere, except the wish they choose to send to Google;
* `steam.db`, review text and player IDs never leave your computer.

It also writes:
* a page for each of about 5,000 well-reviewed games, with what players say and 10 games like it (`games/APPID/`, so a renamed game keeps its address);
* genre pages that link to all of them;
* `sitemap.xml` and `robots.txt`.

Only the roughly 900 pages with a written summary are offered to Google, through the sitemap. The pages built only from stock review labels are marked `noindex`, so the site doesn't look like mass-produced thin content. They are still there for visitors. Google indexes the offered pages, so its AI Mode can use them.

```bash
gh auth login                          # once: sign in to GitHub (free account)
python publish_site.py                 # build, push, turn on GitHub Pages
```

The site goes to `https://YOURNAME.github.io/dream-game-finder/`. Free GitHub Pages needs a public repository, and it only ever holds the built site. Each publish replaces it with one fresh commit, and commits use your GitHub no-reply address.

To tell Google about the site:
1. In [Google Search Console](https://search.google.com/search-console), add a **URL prefix** property for the site address.
2. Choose **HTML tag** verification and copy the `content="…"` token.
3. Run `python publish_site.py --google-verification TOKEN`, then press **Verify**.
4. Under **Sitemaps**, submit `sitemap.xml`.

Google retired its old "ping" address for sitemaps, so Search Console is the way to submit one.

To test the static site on your computer:

```bash
python build_site.py --site-url https://example.github.io/dream-game-finder/ --out site_test/dream-game-finder
python -m http.server 8790 --directory site_test     # open http://localhost:8790/dream-game-finder/
```

## 5. Or ask from the command line

```bash
python recommend.py like "Hades" "Hollow Knight"                # more like these
python recommend.py like "Stardew Valley" --gems                # lesser-known but well liked
python recommend.py like "Celeste" --dislike "Dark Souls III" --platform deck
python recommend.py dream "cozy farming with a deep story, no combat"
python recommend.py dream "a sad, beautiful game about memory" --ai   # Claude interprets + explains
python analyze_reviews.py --show "Hollow Knight"                # what players say about a game
```

Filters: `--tags "Roguelite,Co-op"`, `--exclude-tags Horror`, `--min-positive 90`,
`--min-reviews 50`, `--max-reviews 5000`, `--free`, `--platform windows|mac|linux|deck`,
`--year-from 2020`, `-n 20`, `--json`.

## How it works

* **Content**: what a game *is*, from player-voted store tags, the description and words players use in reviews (TF-IDF).
* **Co-likes**: games recommended by the same players. This signal gets more weight the more evidence there is, and only re-orders games that are already similar in content, so blockbusters don't swamp everything.
* **Quality**: % positive with Bayesian smoothing, so 3/3 positive doesn't beat 290k/300k. Games under 75% positive sink in the ranking (a 65% game needs a much better match to show up); the eval showed a stricter 80% cut-off cost accuracy, 75% did not.
* **Several tastes at once**: liked games are grouped into up to 3 interests by how alike they are (e.g. farming sims vs. 4X strategy). Each interest gets its own ranking and a share of the results in proportion to how much you liked it, so one big interest can't crowd out the rest.
* **Not recommended**: demos, prologues, playtests and soundtracks; and games players tag mainly as adult content ("Hentai" in the top 10 tags, or "Sexual Content" as the top tag) unless you ask for those tags in the filters.
* **"Fans of X like it"** is only said when the game is one of X's 25 strongest co-like links *and* the two games are alike, never just because both are popular.
* **What players say** (`analyze_reviews.py`): 65 aspects (story, boss fights, relaxing, grindy, cheaters…) with phrase patterns in 7 languages. For each game:
  * expected mentions come from each language's base rate;
  * an aspect only counts when its mentions are statistically unusual, with the false-discovery rate held at 10% (Benjamini–Hochberg);
  * praise needs positive evidence (and few explicitly negative mentions), complaints need negative evidence, and "players split on …" needs explicit negative mentions too; otherwise nothing is claimed;
  * unticked template boxes ("☐ great story") and negated mentions ("no bugs at all") are ignored.

  These aspects explain results and refine the taste summary. In ranking they only nudge scores, and a game's own complaints can only lower it.
* **Written descriptions** for the ~900 most-reviewed games were drafted by Claude agents from samples of real reviews. Each point had to be backed by at least 3 reviews, and the drafts were then fact-checked.
* **`--gems`**: tilts against popularity, weights quality heavily, and requires at least 80% positive.
* **`dream`**: understands tags and negations ("no combat", "without multiplayer or pvp").
* **`--ai`**:
  1. Claude turns the wish into tags, things to avoid and example games.
  2. The local model finds about 40 candidates.
  3. Claude picks the best and explains why.

  Claude only chooses from real catalogue entries. If its answer can't be used, the local ranking is shown.

## Measuring quality

```bash
python recommend.py eval
```

For each test player, the eval hides one game they liked, recommends from the rest, and checks whether the hidden game shows up in the top 20 (hit@20). It compares popularity, content, co-likes and the hybrid, with 95% bootstrap intervals. Paired comparisons show what each part adds: the aspects from reviews, the low-rating penalty and the multi-interest merge.

`tools/persona_play.py` plays the quiz through the running site from the command line, for testing it as different kinds of players (`--unknown`, `--no-interest` and `--interested` fill the three boxes).

`tools/quiz_sim.py` lets real reviewers from `steam.db` play the quiz: each one's reviewed games count as known and one liked game is hidden. It measures whether the hidden game shows up in the top 20 results, comparing settings of the knew-it / never-heard weights. With today's crawl it can't tell the settings apart. `steam.db` holds only about 14% of each sampled reviewer's reviews and very few reviews of the big hits the quiz starts with, so the simulated players "know" almost none of the quiz games. It needs fuller review histories per player, or real quiz answers, to give a verdict.

## Tests

```bash
python -m unittest discover -s tests -v
```

59 tests run in about a minute on a synthetic database: crawler paging and resume with a fake Steam API, review analysis, recommendations, filters, quiz rules, server API and security, and the static web build (slim model, the in-browser API bridge, game pages, the data page and sitemap). A few smoke tests also run on the real data when it's present.

## License and data

**Code:** MIT (see `LICENSE`). That covers the code in this repository only, not Steam's data.

**Never published:** `steam.db` and the model files (`recsys_model.npz`, `recsys_meta.json`) stay on your computer, and `.gitignore` keeps them out of Git. Don't upload them anywhere (GitHub, Kaggle, Hugging Face):
* **Review text** belongs to the people who wrote it. Steam's Subscriber Agreement licenses reviews to Valve only, for running Steam, and passes nothing on to anyone else.
* **Steam IDs** are personal data. Under the EU's GDPR (and similar laws elsewhere), any review linked to an ID is too, and removing or hashing the IDs doesn't make full review text anonymous.
* **Steam's Web API Terms of Use** let you show Steam data "to end users for their personal use via your Application", not redistribute it in bulk.

**On the website:**
* game facts from Steam's store (names, dates, developers, tags, review scores);
* short store-description excerpts linked to their Steam pages;
* cover images loaded from Steam's servers;
* the project's own AI-written summaries and review statistics.

There is no review text and there are no Steam IDs or player names. Steam data is shown "as is", as Steam's API terms ask. Every page links to `/about/`, which explains:
* where the data comes from and its snapshot date;
* who owns what;
* privacy: no cookies, accounts or tracking;
* how to ask for a removal: the website repo's GitHub issues, or `--contact` for `build_site.py`.

**Licences for the website's content:**
* The AI-written summaries and computed statistics: CC0 1.0. AI-written text probably has no copyright anyway, and a check found no copied passages from the reviews.
* Game names, store excerpts and images: the project claims no rights in them and grants no licence to them.

**Running the crawler yourself:** you're responsible for following Steam's terms.
* Keep it slow: one request a second is the default, and it backs off when Steam answers "too many requests".
* Steam's Web API Terms allow at most 100,000 calls a day.
* `api.steampowered.com/robots.txt` disallows crawling. Valve documents the reviews API (`IUserReviewsService/GetAppReviews`) as key-less; the store catalogue endpoints are undocumented.
* Check the terms again before you charge money for anything built on this data.

Dream Game Finder is not affiliated with or endorsed by Valve. Steam is a trademark of Valve Corporation. This section explains the project's choices and is not legal advice.
