#!/usr/bin/env python3
"""
Build the public repository from the working tree, by allow-list.

The working tree is not publishable. It tracks government CCTV frames,
model weights, the working log, the private working spec and a database
of sightings — 8,000 files, most of which have no business on GitHub. The
public repository is a deliberate subset, and it is rebuilt from source
rather than patched in place so nothing stale survives a correction.

An allow-list, not a deny-list. A deny-list has to be right about every
file that must not go; this has to be right about the ones that may, and
anything new is excluded until somebody adds it. The refusal check at the
end is belt and braces: if a forbidden path ever appears in the export,
the build stops rather than pushing.

    python tools/publish_repo.py              # build and verify only
    python tools/publish_repo.py --push       # also force-push to GitHub
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "meherzankturel/sentinel-gujarat-police"

# The branch a visitor sees. Not a matter of taste: this repository's
# default is `master`, and an earlier version of this tool pushed to
# `main` -- which succeeded, reported success, and left the page a
# reviewer opens showing the previous week's code. A push that lands
# where nobody looks is indistinguishable from no push at all, so the
# branch is read back from GitHub rather than assumed.
BRANCH = "master"

# What the public repository contains. Directories are copied whole,
# filtered by SKIP_SUFFIXES and SKIP_NAMES below.
INCLUDE_DIRS = [
    "sentinel",          # the system
    "tests",             # including the checklist as acceptance tests
    "tools",             # every measurement and build tool
    "grid",              # the local mock grid
    "web",               # the two consoles
    "deploy",            # the hosted console, as deployed
    "docs",              # design, report, plan, deck
]
INCLUDE_FILES = [
    "README.md", "SUBMISSION.md", "LICENSE", "NOTICE",
    "requirements.txt", "pytest.ini", ".gitignore",
    "live_audit.py", "models/README.md",
]

# Never published, whatever else changes.
#
#   media/       government CCTV of public roads, and the entrant's own
#                street footage with a member of the public in it
#   .run/        the working log: frames, clips, probe output
#   audit/frames measured frames cut from government footage
#   CLAUDE.md    the private working spec
#   .env         credentials
#   *.db         sightings, including plates — except the reduced demo
#                copy under deploy/, which is built with the public's
#                registrations already stripped out
#   evidence/    stills cut per read; the withheld ones are in there
FORBIDDEN_PREFIXES = ("media/", ".run/", "audit/frames", "audit/osd",
                      "models/", ".venv/", ".git/", "tasks/",
                      "web/assets/evidence/", "deploy/site/assets/evidence/",
                      "deploy/sentinel/")
FORBIDDEN_NAMES = ("CLAUDE.md", ".env", "sentinel.db")
SKIP_SUFFIXES = (".pyc", ".mp4", ".mov", ".pth", ".log", ".DS_Store")
SKIP_NAMES = ("__pycache__", ".vercel", "node_modules", "evidence")

# Build artefacts that regenerate from what is published beside them.
# docs/deck/index.html is 4MB of inlined screenshots and is rebuilt by
# tools/build_deck.py from template.html and data.json, both of which are
# published. Shipping it doubled the repository for nothing and pushed the
# upload past GitHub's patience.
SKIP_PATHS = ("docs/deck/index.html",)

# models/README.md is the one file under models/ that is published: it
# says where the public baseline weights come from.
ALLOWED_DESPITE_PREFIX = ("models/README.md",)


def wanted(rel: str) -> bool:
    if rel in SKIP_PATHS:
        return False
    if rel in ALLOWED_DESPITE_PREFIX:
        return True
    if any(rel.startswith(p) for p in FORBIDDEN_PREFIXES):
        return False
    if pathlib.Path(rel).name in FORBIDDEN_NAMES:
        return False
    if rel.endswith(SKIP_SUFFIXES):
        return False
    if any(part in SKIP_NAMES for part in pathlib.Path(rel).parts):
        return False
    # The demo database is published on purpose; every other .db is not.
    if rel.endswith(".db") and rel != "deploy/sentinel-demo.db":
        return False
    return True


def collect() -> list:
    out = []
    for name in INCLUDE_FILES:
        if (ROOT / name).exists() and wanted(name):
            out.append(name)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if wanted(rel):
                out.append(rel)
    return sorted(set(out))


def main() -> int:
    files = collect()
    stage = pathlib.Path(tempfile.mkdtemp(prefix="sentinel-publish-"))

    total = 0
    for rel in files:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        total += dst.stat().st_size

    # The refusal check. Walk what was actually written, not what was
    # intended to be written.
    written = sorted(p.relative_to(stage).as_posix()
                     for p in stage.rglob("*") if p.is_file())
    leaked = [r for r in written if not wanted(r)]
    if leaked:
        sys.exit("refusing to publish; these should not be in the export:\n  "
                 + "\n  ".join(leaked))

    print(f"staged {len(written)} files, {total/1_000_000:.1f} MB, at {stage}")
    for d in INCLUDE_DIRS:
        n = len([r for r in written if r.startswith(d + "/")])
        if n:
            print(f"  {d+'/':<12} {n}")

    if "--push" not in sys.argv:
        print("\nnot pushing (pass --push). Inspect the staged tree first.")
        return 0

    # Confirm the default branch rather than trusting the constant.
    probe = subprocess.run(
        ["gh", "api", f"repos/{REPO}", "--jq", ".default_branch"],
        capture_output=True, text=True)
    default = probe.stdout.strip()
    if probe.returncode == 0 and default and default != BRANCH:
        sys.exit(f"this repository's default branch is '{default}', not "
                 f"'{BRANCH}'. Publishing to '{BRANCH}' would leave the page "
                 f"a reviewer opens unchanged. Fix BRANCH and re-run.")

    def git(*args, **kw):
        return subprocess.run(["git", *args], cwd=stage, check=True,
                              capture_output=True, text=True, **kw)

    git("init", "-q", "-b", BRANCH)
    # A 9MB first push over HTTPS ran into HTTP 408 with the default 1MB
    # buffer. Send it in one go instead of dribbling it.
    git("config", "http.postBuffer", "524288000")
    git("add", "-A")
    # The published .gitignore carries `*.db`, which is right for the
    # working tree and wrong for this one file: the reduced demo database
    # is the hosted console's data, built with the public's registrations
    # already stripped out, and the console's tests skip without it. Added
    # explicitly so the ignore rule cannot quietly drop it -- which it did,
    # leaving a clone whose hosted-console tests all skipped.
    demo_db = stage / "deploy" / "sentinel-demo.db"
    if demo_db.exists():
        git("add", "-f", "deploy/sentinel-demo.db")
    git("-c", "user.name=meherzankturel",
        "-c", "user.email=meherzankturel@gmail.com",
        "commit", "-q", "-m",
        "Sentinel — Integrated Video Management & Analytics Platform\n\n"
        "Gujarat Police Hackathon Innovation Challenge 2026.\n"
        "Built from the working tree; no inherited history.")
    git("remote", "add", "origin", f"https://github.com/{REPO}.git")
    r = subprocess.run(["git", "push", "--force", "origin", BRANCH],
                       cwd=stage, capture_output=True, text=True)
    if r.returncode:
        sys.exit("push failed:\n" + r.stderr)
    print(f"\npushed {len(written)} files to https://github.com/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
