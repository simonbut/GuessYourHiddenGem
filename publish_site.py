#!/usr/bin/env python3
"""Put the website online for free on GitHub Pages.

Two ways - pick one:

  A. Next to your code, in this repository's gh-pages branch (no GitHub CLI needed):

       python publish_site.py --gh-pages

     The built site goes to the gh-pages branch of this folder's GitHub
     repository ("origin"), pushed with your normal `git push` sign-in. Your
     code on main is never touched. The site is at
     https://YOURNAME.github.io/REPO/ . If the GitHub CLI isn't signed in,
     the script tells you the one setting to change (Settings > Pages >
     Branch: gh-pages). The repository must be public (free GitHub Pages).

  B. In a repository of its own:

       gh auth login                      # once: sign in to GitHub in your browser
       python publish_site.py             # build, create the repo, push, turn on Pages

     The site goes to https://YOURNAME.github.io/dream-game-finder/ (or --repo NAME).

Either way the published branch only ever holds the built site (pages,
scripts and the slimmed model - no reviews, no player IDs). Every publish
replaces it with a single fresh commit, so it never grows, and commits use
your GitHub no-reply address, so your email address isn't published.
The script refuses to overwrite a branch it didn't make (such as your code).

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
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "site_config.json")
DEFAULT_REPO = "dream-game-finder"
MARKER = ".dgf-site"            # build_site.py writes it into every built site


def run(cmd, cwd=None, check=True, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{(r.stderr or r.stdout).strip()}")
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    return r


def gh_json(*args):
    return json.loads(run(["gh", "api", *args], quiet=True).stdout)


def gh_signed_in() -> bool:
    try:
        return run(["gh", "auth", "status"], check=False, quiet=True).returncode == 0
    except FileNotFoundError:                       # the GitHub CLI isn't installed
        return False


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def parse_github_remote(url: str):
    """(owner, repo) from a GitHub remote URL (SSH or HTTPS), else None."""
    m = re.match(r"^(?:git@github\.com:|ssh://git@github\.com/|https://(?:[^@/]+@)?github\.com/)"
                 r"([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$", url.strip())
    return (m.group(1), m.group(2)) if m else None


def http_get(url, headers=None, timeout=20):
    """(status, body). Falls back to curl when Python can't verify HTTPS certificates
    (it happens on some Windows set-ups); curl uses the system's certificate store."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, OSError) as e:
        if "CERTIFICATE" not in str(e).upper():
            return 0, b""
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout), "-w", "\n%{http_code}"]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    try:
        r = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 5)
        body, _, code = r.stdout.rpartition(b"\n")
        return int(code or 0), body
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0, b""


def public_json(url):
    status, body = http_get(url, {"Accept": "application/vnd.github+json", "User-Agent": "publish_site.py"})
    if status != 200:
        raise urllib.error.URLError(f"HTTP {status} for {url}")
    return json.loads(body)


def user_id(login: str, signed_in: bool):
    """GitHub's numeric account id (for the no-reply address), or None."""
    try:
        return gh_json("user")["id"] if signed_in else public_json(f"https://api.github.com/users/{login}")["id"]
    except (Exception, SystemExit):
        return None


def branch_state(push_url, owner, repo, branch, signed_in):
    """'missing' (no such branch yet), 'site' (made by build_site.py) or 'unknown'
    (it exists; whether it's ours is checked by fetching it - see check_fetched)."""
    r = run(["git", "ls-remote", "--heads", push_url, branch], check=False, quiet=True)
    if r.returncode != 0:
        sys.exit(f"can't reach {push_url}:\n{(r.stderr or r.stdout).strip()}\n"
                 "Check that `git push` works from this folder (your GitHub sign-in), then run this again.")
    if not r.stdout.strip():
        return "missing"
    path = f"repos/{owner}/{repo}/contents/{MARKER}?ref={branch}"
    try:
        if signed_in:
            if run(["gh", "api", path], check=False, quiet=True).returncode == 0:
                return "site"
        else:
            public_json(f"https://api.github.com/{path}")
            return "site"
    except (urllib.error.URLError, ValueError, OSError):
        pass
    return "unknown"


def check_fetched(out, push_url, branch):
    """Fetch the branch and look for the marker: never overwrite something else."""
    run(["git", "fetch", "-q", "--depth", "1", push_url, branch], cwd=out)
    return run(["git", "cat-file", "-e", f"FETCH_HEAD:{MARKER}"], cwd=out, check=False, quiet=True).returncode == 0


def refuse(owner, repo, branch, same_repo):
    hint = ("Publish next to your code instead:  python publish_site.py --gh-pages" if not same_repo else
            "Pick another branch or repository.")
    sys.exit(f"{owner}/{repo} already has a '{branch}' branch that wasn't made by this script "
             f"(your code?). Not overwriting it.\n{hint}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Build the static site and publish it on GitHub Pages (free).")
    p.add_argument("--gh-pages", action="store_true",
                   help="publish to the gh-pages branch of this folder's own GitHub repo, next to your code "
                        "(no GitHub CLI needed); remembered for later publishes")
    p.add_argument("--repo", help=f"a repository of its own for the site (default {DEFAULT_REPO}; needs gh)")
    p.add_argument("--google-verification", help="Search Console HTML-tag token; remembered for later publishes")
    p.add_argument("--skip-build", action="store_true", help="push the existing site/ folder as it is")
    args = p.parse_args()

    cfg = load_config()
    signed_in = gh_signed_in()
    same_repo = args.gh_pages or (cfg.get("mode") == "gh-pages" and not args.repo)
    if same_repo:
        r = run(["git", "remote", "get-url", "origin"], cwd=HERE, check=False, quiet=True)
        found = parse_github_remote(r.stdout) if r.returncode == 0 else None
        if not found:
            sys.exit("--gh-pages publishes next to your code, so this folder needs a GitHub 'origin' remote "
                     "(clone your repository from GitHub, or: git remote add origin URL).")
        login, repo = found
        push_url = r.stdout.strip()                  # your own remote: your normal git sign-in pushes it
        branch = "gh-pages"
        cfg["mode"] = "gh-pages"
    else:
        if not signed_in:
            sys.exit("Sign in to GitHub first:  gh auth login   (then run this again)\n"
                     "Or publish next to your code without it:  python publish_site.py --gh-pages")
        login = gh_json("user")["login"]
        repo = args.repo or cfg.get("repo") or DEFAULT_REPO
        push_url = f"https://github.com/{login}/{repo}.git"
        branch = "main"
        cfg["mode"] = "repo"
    full = f"{login}/{repo}"
    uid = user_id(login, signed_in)
    email = f"{uid}+{login}@users.noreply.github.com" if uid else f"{login}@users.noreply.github.com"

    if args.google_verification is not None:
        raw = args.google_verification.strip()          # the token, or the whole <meta ...> tag pasted
        m = re.search(r'content\s*=\W*([A-Za-z0-9_-]{8,})', raw)
        cfg["google_verification"] = m.group(1) if m else raw
    cfg["repo"] = repo
    host = f"{login.lower()}.github.io"
    site_url = f"https://{host}/" if repo.lower() == host else f"https://{host}/{repo}/"
    cfg["site_url"] = site_url

    if not same_repo:
        if run(["gh", "repo", "view", full], check=False, quiet=True).returncode != 0:
            print(f"creating public repository {full} ...")
            run(["gh", "repo", "create", full, "--public", "--homepage", site_url,
                 "--description", "Dream Game Finder - find your next Steam game (static site, runs in your browser)"])
        run(["gh", "auth", "setup-git"], quiet=True)     # git pushes over HTTPS with the gh sign-in
    # before building: never overwrite a branch this script didn't make (e.g. your code)
    state = branch_state(push_url, login, repo, branch, signed_in)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    out = os.path.join(HERE, "site")
    if not args.skip_build:
        cmd = [sys.executable, os.path.join(HERE, "build_site.py"), "--site-url", site_url, "--out", out]
        if cfg.get("google_verification"):
            cmd += ["--google-verification", cfg["google_verification"]]
        print("building the site (a few minutes) ...")
        subprocess.run(cmd, check=True)
    if not os.path.exists(os.path.join(out, ".dgf-complete")):
        sys.exit("site/ isn't a finished build - run without --skip-build")

    # one fresh commit, pushed over the previous site
    git = ["git", "-c", "core.autocrlf=false", "-c", f"user.name={login}", "-c", f"user.email={email}"]
    if not os.path.isdir(os.path.join(out, ".git")):
        run(["git", "init", "-q", "-b", "main"], cwd=out)
    if state == "unknown" and not check_fetched(out, push_url, branch):
        refuse(login, repo, branch, same_repo)
    run(git + ["add", "-A"], cwd=out, quiet=True)
    if run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=out, check=False, quiet=True).returncode == 0:
        run(git + ["commit", "-q", "--amend", "--reset-author", "-m", f"Publish {time.strftime('%Y-%m-%d %H:%M')}"], cwd=out)
    else:
        run(git + ["commit", "-q", "-m", f"Publish {time.strftime('%Y-%m-%d %H:%M')}"], cwd=out)
    print(f"pushing the site to {full} ({branch} branch) ...")
    run(["git", "push", "-q", "-f", push_url, f"HEAD:refs/heads/{branch}"], cwd=out)

    # GitHub Pages must serve that branch
    if signed_in:
        pages = run(["gh", "api", f"repos/{full}/pages"], check=False, quiet=True)
        if pages.returncode != 0:
            run(["gh", "api", "-X", "POST", f"repos/{full}/pages", "-f", f"source[branch]={branch}",
                 "-f", "source[path]=/"], quiet=True)
            print(f"GitHub Pages turned on (from the {branch} branch)")
        elif (json.loads(pages.stdout or "{}").get("source") or {}).get("branch") != branch:
            run(["gh", "api", "-X", "PUT", f"repos/{full}/pages", "-f", f"source[branch]={branch}",
                 "-f", "source[path]=/"], quiet=True)
            print(f"GitHub Pages now serves the {branch} branch")
    else:
        print(f"\nOne setting to check on GitHub (once):\n"
              f"  https://github.com/{full}/settings/pages\n"
              f"  Build and deployment > Source: Deploy from a branch > Branch: {branch}, folder / (root) > Save\n")
    # with a Google token, wait for the page that carries it (an older copy is already live)
    token = cfg.get("google_verification") or ""

    def live():
        url = f"{site_url}?check={int(time.time())}" if token else f"{site_url}data/status.json?check={int(time.time())}"
        status, body = http_get(url, {"Cache-Control": "no-cache"})
        return status == 200 and (not token or token in body.decode("utf-8", "replace"))

    print(f"waiting for {site_url} to go live (the first time can take a few minutes) ...")
    is_live = False
    for _ in range(60):
        if live():
            is_live = True
            break
        time.sleep(10)
    else:
        print("not live yet - GitHub is still building it (or the Pages setting above isn't saved yet); "
              "check again in a few minutes")
    print(f"\n{'online' if is_live else 'address (once live)'}: {site_url}\nsitemap: {site_url}sitemap.xml")
    if token:
        print("The Google verification tag is live: press Verify in Search Console now.")
    else:
        print("next: add the site in Google Search Console (URL prefix), then run\n"
              "  python publish_site.py --google-verification TOKEN\nand submit sitemap.xml there.")


if __name__ == "__main__":
    main()
