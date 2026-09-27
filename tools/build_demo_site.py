#!/usr/bin/env python3
"""
Build the hosted read-only console for the screening committee.

The brief invites a hosted URL with test credentials. What can be hosted is
not the whole system: the analytics need PyTorch and OpenCV, which are
~700 MB and want a GPU, and a serverless function caps at 250 MB with no
accelerator. Pretending otherwise would put a "live" badge on something
that cannot infer.

So what deploys is the *console over results already computed* — the
registry with its measured capability, the officer's search and route, the
evidence bundle, the audit chain. Every number in it was produced by the
real pipeline on this machine. The site says so on its own front page
rather than letting a reviewer assume it is inferring live.

WHAT IS DELIBERATELY LEFT BEHIND
--------------------------------
One plate in the database belongs to a member of the public: GJ05JS8590,
a car that happened to be parked in the entrant's own street footage. It
is a real registration of a real person who did not consent to appear in a
public demonstration, and a submission whose central argument is
governance-as-architecture does not get to publish it. It is dropped here.

What remains is the mock grid's synthetic plates, the entrant's own
vehicle, and camera capability measurements, which are properties of
cameras rather than of people.

    python tools/build_demo_site.py
"""
import json
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DB = ROOT / "sentinel.db"
OUT = ROOT / "deploy"
OUT_DB = OUT / "sentinel-demo.db"

# Real registrations belonging to members of the public. Never published.
WITHHOLD_PLATES = ("GJ05JS8590", "GJO5JS8590", "CJ05JS8590")

TABLES = ("camera", "watchlist_entry", "sighting", "alert", "audit_log")

# The light half of the package, vendored into the deployment so the hosted
# console can run the real endpoints instead of a copy of them. A copy is
# what it used to run, and the copy drifted: it answered searches in a
# different shape, the console died on it, and a reviewer got a header full
# of numbers above an empty map.
#
# sentinel.officer defers its vision imports for exactly this reason. These
# are the modules the deferred half still needs at import time; none of them
# touches OpenCV or PyTorch, and the build below fails if that changes.
VENDOR = ("__init__.py", "registry.py", "matching.py", "officer.py")
HEAVY = ("cv2", "torch", "torchvision", "numpy", "PIL")


def build_db() -> dict:
    if not SRC_DB.exists():
        sys.exit("sentinel.db not found — nothing to publish")
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT_DB.exists():
        OUT_DB.unlink()

    src = sqlite3.connect(SRC_DB)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(OUT_DB)

    # Schema first, verbatim, so the hosted copy and the real registry
    # cannot diverge in shape.
    for row in src.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%'"):
        dst.execute(row[0])

    stats = {}
    for table in TABLES:
        rows = [dict(r) for r in src.execute(f"SELECT * FROM {table}")]
        if table == "sighting":
            before = len(rows)
            rows = [r for r in rows
                    if (r.get("plate_read") or "") not in WITHHOLD_PLATES]
            stats["withheld_sightings"] = before - len(rows)
        if not rows:
            stats[table] = 0
            continue
        cols = list(rows[0])
        dst.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [[r[c] for c in cols] for r in rows])
        stats[table] = len(rows)

    # Which evidence stills the surviving sightings point at. Only these
    # are published: the stills are cut per read, and the reads that were
    # withheld belong to a member of the public whose car happened to be
    # parked in the entrant's own street footage. Copying the whole folder
    # would put their vehicle on a public URL while the row describing it
    # was being carefully withheld from the database -- the privacy
    # equivalent of locking the door and leaving the key in it.
    kept_evidence = {r[0] for r in dst.execute(
        "SELECT DISTINCT evidence_ref FROM sighting "
        "WHERE evidence_ref IS NOT NULL")}

    # Alerts whose sighting was withheld would point at nothing.
    dst.execute("DELETE FROM alert WHERE sighting_id NOT IN "
                "(SELECT id FROM sighting)")
    dst.commit()

    # The deployment copies this into /tmp and writes the audit chain to
    # it, so a reviewer's own searches are logged exactly as an officer's
    # would be. Nothing else is written, and the copy dies with the
    # serverless instance.
    stats["alerts_kept"] = dst.execute("SELECT COUNT(*) FROM alert").fetchone()[0]
    dst.close()
    src.close()
    stats["_evidence"] = kept_evidence
    return stats


def copy_web(keep_evidence=frozenset()):
    """
    The consoles, unchanged apart from where they look for the API.

    They are copied rather than rewritten: a hosted console that has
    drifted from the one in the demonstration video is worse than no
    hosted console.
    """
    pub = OUT / "site"
    pub.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "officer.html"):
        shutil.copy(ROOT / "web" / name, pub / name)
    shutil.copytree(ROOT / "web" / "vendor", pub / "vendor", dirs_exist_ok=True)
    assets = ROOT / "web" / "assets"
    if assets.exists():
        # Everything but the evidence stills, which are published one by
        # one against the database that survived the withholding above.
        shutil.copytree(assets, pub / "assets", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("evidence"))
    published = publish_evidence(keep_evidence, pub / "assets" / "evidence")
    return sorted(p.name for p in pub.iterdir()), published


def publish_evidence(stems, out: pathlib.Path) -> int:
    """
    Copy only the stills belonging to sightings the demo database kept.

    Written as an allow-list rather than a deny-list on purpose. A
    deny-list has to be right about every file that must not go; this has
    to be right about the handful that may, and anything new is excluded
    until somebody says otherwise.
    """
    src = ROOT / "web" / "assets" / "evidence"
    if out.exists():
        shutil.rmtree(out)
    if not stems or not src.exists():
        return 0
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in sorted(stems):
        for kind in ("vehicle", "plate"):
            f = src / f"{stem}-{kind}.jpg"
            if f.exists():
                shutil.copy(f, out / f.name)
                n += 1
    leaked = [f.name for f in out.iterdir()
              if not any(f.name.startswith(st + "-") for st in stems)]
    if leaked:
        sys.exit("evidence published that no kept sighting references: "
                 + ", ".join(leaked))
    return n


def vendor_package():
    """
    Copy the light half of sentinel/ into the deployment, then prove it
    imports without the vision stack.

    The proof is the point. It is three seconds of build time, it runs
    every time, and it is the difference between learning that the hosted
    console cannot import its own endpoints here, or learning it from a
    blank screen on the public URL.
    """
    dst = OUT / "sentinel"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in VENDOR:
        shutil.copy(ROOT / "sentinel" / name, dst / name)

    check = textwrap.dedent(f"""
        import sys, importlib.abc
        BLOCK = {HEAVY!r}
        class Deny(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError("heavy import in the hosted half: " + name)
                return None
        sys.meta_path.insert(0, Deny())
        from sentinel import officer
        for route in sorted(r.path for r in officer.router.routes):
            print(route)
    """)
    r = subprocess.run([sys.executable, "-c", check], cwd=OUT,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("vendored sentinel/ will not import in a serverless "
                 "function:\n" + r.stderr.strip())
    return [p for p in r.stdout.split() if p.startswith("/api")]


if __name__ == "__main__":
    stats = build_db()
    evidence = stats.pop("_evidence", set())
    files, published = copy_web(evidence)
    mounted = vendor_package()
    size = OUT_DB.stat().st_size / 1000
    print(f"wrote {OUT_DB.relative_to(ROOT)}  ({size:.0f} KB)")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print(f"\ncopied {len(files)} web entries: {', '.join(files)}")
    print(f"published {published} evidence stills for "
          f"{len(evidence)} sightings; every other still stayed behind")
    print(f"vendored sentinel/{{{','.join(VENDOR)}}} — imports clean without "
          f"the vision stack")
    for path in mounted:
        print(f"  mounts {path}")
    print(f"\nwithheld plates: {', '.join(WITHHOLD_PLATES)}")
