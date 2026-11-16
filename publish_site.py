#!/usr/bin/env python3
"""Put the website online for free on GitHub Pages.

    gh auth login                      # once: sign in to GitHub in your browser
    python publish_site.py             # build the static site, push it, turn on Pages

The site goes to https://YOURNAME.github.io/dream-game-finder/ . Free GitHub
Pages needs a public repository; it only ever holds the built site (pages,
scripts and the slimmed model - no reviews, no player IDs). Every publish
replaces the repository's content with a single fresh commit, so it never
grows, and commits use your GitHub no-reply address, so your email address
isn't published.

Google: add --google-verification TOKEN once (the HTML-tag token Search
Console shows); it's remembered in site_config.json for later publishes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "site_config.json")
DEFAULT_REPO = "dream-game-finder"


def run(cmd, cwd=None, check=True, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{(r.stderr or r.stdout).strip()}")
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    return r


def gh_json(*args):
    return json.loads(run(["gh", "api", *args], quiet=True).stdout)


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Build the static site and publish it on GitHub Pages (free).")
    p.add_argument("--repo", help=f"repository name (default {DEFAULT_REPO})")
    p.add_argument("--google-verification", help="Search Console HTML-tag token; remembered for later publishes")
    p.add_argument("--skip-build", action="store_true", help="push the existing site/ folder as it is")
    args = p.parse_args()

    if run(["gh", "auth", "status"], check=False, quiet=True).returncode != 0:
        sys.exit("Sign in to GitHub first:  gh auth login   (then run this again)")
    user = gh_json("user")
    login, uid = user["login"], user["id"]
    cfg = load_config()
    repo = args.repo or cfg.get("repo") or DEFAULT_REPO
    if args.google_verification is not None:
        raw = args.google_verification.strip()          # the token, or the whole <meta ...> tag pasted
        m = re.search(r'content\s*=\W*([A-Za-z0-9_-]{8,})', raw)
        cfg["google_verification"] = m.group(1) if m else raw
    cfg["repo"] = repo
    host = f"{login.lower()}.github.io"
    site_url = f"https://{host}/" if repo.lower() == host else f"https://{host}/{repo}/"
    cfg["site_url"] = site_url
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    full = f"{login}/{repo}"
    if run(["gh", "repo", "view", full], check=False, quiet=True).returncode != 0:
        print(f"creating public repository {full} ...")
        run(["gh", "repo", "create", full, "--public", "--homepage", site_url,
             "--description", "Dream Game Finder - find your next Steam game (static site, runs in your browser)"])

    out = os.path.join(HERE, "site")
    if not args.skip_build:
        cmd = [sys.executable, os.path.join(HERE, "build_site.py"), "--site-url", site_url, "--out", out]
        if cfg.get("google_verification"):
            cmd += ["--google-verification", cfg["google_verification"]]
        print("building the site (a few minutes) ...")
        subprocess.run(cmd, check=True)
    if not os.path.exists(os.path.join(out, ".dgf-complete")):
        sys.exit("site/ isn't a finished build - run without --skip-build")

    # one fresh commit, pushed over whatever was there
    run(["gh", "auth", "setup-git"], quiet=True)
    git = ["git", "-c", "core.autocrlf=false", "-c", f"user.name={login}",
           "-c", f"user.email={uid}+{login}@users.noreply.github.com"]
    if not os.path.isdir(os.path.join(out, ".git")):
        run(["git", "init", "-q", "-b", "main"], cwd=out)
    run(git + ["add", "-A"], cwd=out, quiet=True)
    if run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=out, check=False, quiet=True).returncode == 0:
        run(git + ["commit", "-q", "--amend", "--reset-author", "-m", f"Publish {time.strftime('%Y-%m-%d %H:%M')}"], cwd=out)
    else:
        run(git + ["commit", "-q", "-m", f"Publish {time.strftime('%Y-%m-%d %H:%M')}"], cwd=out)
    remote = f"https://github.com/{full}.git"
    if run(["git", "remote", "get-url", "origin"], cwd=out, check=False, quiet=True).returncode != 0:
        run(["git", "remote", "add", "origin", remote], cwd=out)
    else:
        run(["git", "remote", "set-url", "origin", remote], cwd=out)
    print(f"pushing to {full} ...")
    run(["git", "push", "-q", "-f", "origin", "main"], cwd=out)

    # turn on GitHub Pages (from the main branch) if it isn't on yet
    if run(["gh", "api", f"repos/{full}/pages"], check=False, quiet=True).returncode != 0:
        run(["gh", "api", "-X", "POST", f"repos/{full}/pages", "-f", "source[branch]=main", "-f", "source[path]=/"],
            quiet=True)
        print("GitHub Pages turned on")
    # with a Google token, wait for the page that carries it (an older copy is already live)
    token = cfg.get("google_verification") or ""

    def live():
        url = f"{site_url}?check={int(time.time())}" if token else site_url + "data/status.json"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}),
                                        timeout=20) as r:
                return r.status == 200 and (not token or token in r.read().decode("utf-8", "replace"))
        except Exception:
            return False

    print(f"waiting for {site_url} to go live (the first time can take a few minutes) ...")
    for _ in range(60):
        if live():
            break
        time.sleep(10)
    else:
        print("not live yet - GitHub is still building it; check again in a few minutes")
    print(f"\nonline: {site_url}\nsitemap: {site_url}sitemap.xml")
    if token:
        print("The Google verification tag is live: press Verify in Search Console now.")
    else:
        print("next: add the site in Google Search Console (URL prefix), then run\n"
              "  python publish_site.py --google-verification TOKEN\nand submit sitemap.xml there.")


if __name__ == "__main__":
    main()
