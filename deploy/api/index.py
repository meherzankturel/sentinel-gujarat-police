#!/usr/bin/env python3
"""
Sentinel — the hosted read-only console.

The brief invites "a URL to their hosted platform along with test login
credentials for the screening committee". This is that, and it is
deliberately less than the system it belongs to.

WHY THIS IS NOT THE WHOLE PLATFORM
----------------------------------
The analytics need PyTorch and OpenCV. Together they are about 700 MB and
they want a GPU; a serverless function caps at 250 MB and has none. So
nothing here infers. What is served is the console over results the real
pipeline already produced — the registry with its measured capability, the
officer's search and route, the evidence bundle, the audit chain.

Every figure a reviewer sees was computed by the real system. None of it is
computed *here*, and the banner on the page says so. A hosted demo that
implies live inference would be the one dishonest thing in a submission
that has spent its whole life avoiding exactly that.

THE LOGIN IS REAL, AND IT IS ALSO NEW
-------------------------------------
The design document states that the platform has no authentication:
the case-linked gate is real and tested, but it believes whatever officer
name it is handed. That is still true of the main codebase and is recorded
there as an outstanding gap.

This deployment adds a login because a public URL without one would publish
the console to anyone who found it. It is a signed-cookie session over a
fixed credential list, which is the right weight for a screening demo and
the wrong weight for a police deployment. In deployment it belongs behind
eGujCop or departmental AD with MFA, as the design document says.

The gate it protects is the important one and it is unchanged: no movement
query is answered without a case reference, a searching officer and an
authorising officer.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import time
from typing import Optional

from fastapi import Cookie, Depends, FastAPI, Form, HTTPException
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "site"
BUNDLED_DB = ROOT / "sentinel-demo.db"

# ------------------------------------------------------- a writable copy
#
# The deployment filesystem is read-only apart from /tmp, and sqlite wants
# to write -- not the evidence, which nothing here alters, but the audit
# chain. This console tells a reviewer that "the query is written to the
# tamper-evident log before any result is returned", and on this
# deployment that sentence used to be false: the hosted copy of the
# endpoints answered searches without writing the entry. Copying the
# database into /tmp at cold start makes it true again, because the real
# endpoints -- which do write it -- can then run here unchanged.
#
# The copy lives and dies with the serverless instance. A reviewer's own
# searches appear in the chain they produced and vanish with the
# container; the hosted console is a demonstration, not a system of
# record, and the page says so.
def _writable_db() -> pathlib.Path:
    tmp = pathlib.Path("/tmp/sentinel-demo.db")
    try:
        if (not tmp.exists()
                or tmp.stat().st_mtime < BUNDLED_DB.stat().st_mtime):
            shutil.copy2(BUNDLED_DB, tmp)
        return tmp
    except OSError:
        # No /tmp to write to. Reads still work from the bundle; the audit
        # write will fail loudly rather than being silently skipped, which
        # is the right way round for a log that is meant to be trusted.
        return BUNDLED_DB


DB = _writable_db()
os.environ["SENTINEL_DB"] = str(DB)

# The officer's endpoints are not reimplemented here. An earlier version of
# this file carried its own copy of search and route, and the copy drifted:
# it answered with a different response shape, the console died on it, and
# the screen a panel is most likely to touch showed a header full of
# numbers above an empty map. So the real module is imported and mounted.
#
# It is importable in a 250MB function because sentinel.officer defers its
# vision imports -- see the note at the top of that file. Only the modules
# that read the registry are vendored into this deployment; the hunt, which
# genuinely needs PyTorch and a GPU, is answered with a 501 below.
sys.path.insert(0, str(ROOT))
from sentinel import officer as officer_api  # noqa: E402

# Screening credentials. Fixed and published to the committee on purpose:
# the point of this deployment is that they can open it, not that it is
# secret. SENTINEL_USERS overrides in the environment.
DEFAULT_USERS = {
    "reviewer": "gujarat2026",
    "scrb": "gujarat2026",
}
USERS = json.loads(os.environ.get("SENTINEL_USERS", "null")) or DEFAULT_USERS

# Signing key for the session cookie.
#
# This used to fall back to os.urandom() when the variable was unset, which
# is correct-looking and completely wrong here: every serverless invocation
# is a fresh process with a fresh random key, so a cookie signed by one
# instance fails verification on the next. The visible symptom was the
# console loading its header and then rendering nothing at all -- the page
# authenticated, the Leaflet script request landed on a different instance,
# came back 401, and the map died at "L is not defined" before a single
# camera was drawn.
#
# A missing key is now a startup failure rather than a silent per-process
# one. An auth system that fails open some of the time is worse than one
# that refuses to boot.
SECRET = os.environ.get("SENTINEL_SECRET", "")
if not SECRET:
    raise RuntimeError(
        "SENTINEL_SECRET is not set. Session cookies are signed with it, and "
        "generating one per process means a cookie signed by one serverless "
        "instance is rejected by the next. Set it: "
        "vercel env add SENTINEL_SECRET production")
SESSION_HOURS = 12

app = FastAPI(title="Sentinel (hosted console)", docs_url=None, redoc_url=None)


@app.middleware("http")
async def strip_function_prefix(request, call_next):
    """
    Vercel mounts this file at /api/index, and the requested path only
    survives if the rewrite carries it: vercel.json sends /(.*) to
    /api/index/$1 for that reason. A rewrite to a bare /api/index discards
    it -- verified against a live deployment, where every request arrived as
    /api/index and no x-vercel header held the original. The symptom was
    /login redirecting to itself forever, because every path resolved to the
    same route.

    So this strips the mount prefix and restores what the browser asked for.
    """
    path = request.scope.get("path", "")
    for prefix in ("/api/index.py", "/api/index"):
        if path == prefix or path.startswith(prefix + "/"):
            rest = path[len(prefix):] or "/"
            request.scope["path"] = rest if rest.startswith("/") else "/" + rest
            break
    return await call_next(request)


# ----------------------------------------------------------------- session


def _sign(payload: str) -> str:
    mac = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256)
    return f"{payload}.{mac.hexdigest()}"


def _verify(token: Optional[str]) -> Optional[str]:
    """Return the username a valid unexpired cookie names, else None."""
    if not token or token.count(".") != 2:
        return None
    user, expiry, sig = token.split(".")
    payload = f"{user}.{expiry}"
    expected = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256)
    # Constant-time: a timing oracle on a demo is still a bad habit to ship.
    if not hmac.compare_digest(sig, expected.hexdigest()):
        return None
    try:
        if float(expiry) < time.time():
            return None
    except ValueError:
        return None
    return user


def require_session(session: Optional[str]) -> str:
    user = _verify(session)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


# -------------------------------------------------------------------- data


def rows(sql: str, params=()) -> list[dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


# Imported, not copied. This was the third transcription of the same four
# rules, and three copies is how a camera ends up on one grid here and
# another grid there.
from sentinel.registry import grid_of  # noqa: E402

ANALYTIC_FOR = {
    "plate-capable": "anpr", "plate-occasional": "anpr_low",
    "plate-marginal": "anpr_low", "presence-only": "presence",
    "no-vehicles-observed": "presence", "unusable": "none", "unknown": "probe",
}


# ------------------------------------------------------------------- pages


LOGIN_PAGE = """<!doctype html><meta charset="utf-8">
<title>Sentinel — sign in</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='%231A1D23'/><circle cx='16' cy='16' r='7' fill='none' stroke='%230E9E83' stroke-width='3'/><circle cx='16' cy='16' r='2' fill='%230E9E83'/></svg>">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--ink:#FAF9F6;--panel:#fff;--line:#E4E1D9;--bone:#1A1D23;--mist:#5C6672;
 --dim:#8A9098;--teal:#0E9E83;--warn:#B8780F}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--bone);font-family:Archivo,sans-serif;
 min-height:100vh;display:grid;place-items:center;padding:24px}
.box{width:100%;max-width:430px}
.mark{font-weight:700;letter-spacing:.18em;text-transform:uppercase;font-size:15px}
.mark span{color:var(--teal)}
h1{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:18px 0 8px}
p{color:var(--mist);font-size:14.5px;line-height:1.6}
form{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:22px;margin-top:22px;display:flex;flex-direction:column;gap:12px}
label{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
input{font-family:"IBM Plex Mono",monospace;font-size:15px;padding:11px 13px;
 border:1px solid var(--line);border-radius:8px;background:var(--ink);color:var(--bone)}
button{font:inherit;font-weight:600;font-size:15px;padding:12px;border:0;
 border-radius:9px;background:var(--teal);color:#fff;cursor:pointer;margin-top:4px}
.creds{margin-top:16px;padding:14px 16px;background:var(--panel);
 border:1px solid var(--line);border-left:3px solid var(--teal);
 border-radius:0 10px 10px 0;font-size:13.5px;color:var(--mist)}
.creds b{color:var(--bone);font-family:"IBM Plex Mono",monospace}
.note{margin-top:14px;padding:14px 16px;background:var(--panel);
 border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:0 10px 10px 0;font-size:13px;color:var(--mist);line-height:1.6}
.err{color:#B03A4A;font-size:13.5px}
</style>
<div class="box">
  <div class="mark">Sentinel<span>.</span></div>
  <h1>Sign in</h1>
  <p>Read-only console for the screening committee.</p>
  <form method="post" action="/login">
    <div><label for="u">Username</label></div>
    <input id="u" name="username" autocomplete="username" autofocus>
    <div><label for="p">Password</label></div>
    <input id="p" name="password" type="password" autocomplete="current-password">
    __ERROR__
    <button type="submit">Sign in</button>
  </form>
  <div class="creds">Committee credentials: <b>reviewer</b> / <b>gujarat2026</b></div>
  <div class="note"><b>What this is.</b> The console over results the real
    pipeline already produced. Nothing here runs the analytics: they need
    PyTorch, OpenCV and a GPU, and a serverless function has none of those.
    Every figure shown was measured by the real system on real footage —
    just not in this browser tab.</div>
</div>"""


@app.get("/login", response_class=HTMLResponse)
def login_page(bad: int = 0):
    err = '<div class="err">Wrong username or password.</div>' if bad else ""
    return LOGIN_PAGE.replace("__ERROR__", err)


@app.post("/login")
def login(username: str = Form(""), password: str = Form("")):
    expected = USERS.get(username.strip())
    if not expected or not hmac.compare_digest(password, expected):
        return RedirectResponse("/login?bad=1", status_code=303)
    expiry = time.time() + SESSION_HOURS * 3600
    token = _sign(f"{username.strip()}.{expiry:.0f}")
    r = RedirectResponse("/", status_code=303)
    r.set_cookie("session", token, httponly=True, samesite="lax",
                 secure=True, max_age=SESSION_HOURS * 3600)
    return r


@app.get("/logout")
def logout():
    r = RedirectResponse("/login", status_code=303)
    r.delete_cookie("session")
    return r


@app.get("/", response_class=HTMLResponse)
def registry_page(session: Optional[str] = Cookie(None)):
    if not _verify(session):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(PUBLIC / "index.html")


@app.get("/officer", response_class=HTMLResponse)
def officer_page(session: Optional[str] = Cookie(None)):
    if not _verify(session):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(PUBLIC / "officer.html")


# --------------------------------------------------------------------- api


@app.get("/api/cameras")
def cameras(session: Optional[str] = Cookie(None)):
    require_session(session)
    cams = rows("SELECT * FROM camera ORDER BY id")
    by_grid: dict = {}
    for c in cams:
        c["grid"] = grid_of(c)
        c["analytic"] = ANALYTIC_FOR.get(c.get("capability_class"), "probe")
        by_grid[c["grid"]] = by_grid.get(c["grid"], 0) + 1
    return {"cameras": cams, "count": len(cams), "by_grid": by_grid}


@app.get("/api/assignments")
def assignments(session: Optional[str] = Cookie(None)):
    require_session(session)
    cams = rows("SELECT id, capability_class FROM camera")
    a = [{"camera_id": c["id"],
          "capability": c["capability_class"],
          "analytic": ANALYTIC_FOR.get(c["capability_class"], "probe")}
         for c in cams]
    anpr = [x for x in a if x["analytic"].startswith("anpr")]
    return {"assignments": a, "summary": {
        "cameras": len(a), "running_anpr": len(anpr),
        "running_presence": len([x for x in a if x["analytic"] == "presence"]),
        "anpr_share": round(len(anpr) / len(a), 3) if a else 0,
        "note": ("Plate reading is scheduled only where a plate was measured "
                 "to be legible."),
    }}


@app.get("/api/watchlist")
def watchlist(session: Optional[str] = Cookie(None)):
    require_session(session)
    return {"entries": rows(
        "SELECT * FROM watchlist_entry ORDER BY priority, id")}


@app.get("/api/alerts")
def alerts(session: Optional[str] = Cookie(None)):
    require_session(session)
    return {"alerts": rows(
        "SELECT a.*, w.plate_number, w.reason, w.priority, w.case_ref, "
        "s.camera_id, s.plate_read, s.match_confidence, s.corrected_utc "
        "FROM alert a LEFT JOIN watchlist_entry w ON w.id = a.watchlist_entry_id "
        "LEFT JOIN sighting s ON s.id = a.sighting_id "
        "ORDER BY a.raised_at DESC LIMIT 200")}


@app.get("/api/audit")
def audit(session: Optional[str] = Cookie(None)):
    require_session(session)
    entries = rows("SELECT * FROM audit_log ORDER BY id DESC LIMIT 200")
    # Recompute the chain rather than reporting a stored verdict.
    chain = rows("SELECT * FROM audit_log ORDER BY id")
    prev, intact, broken = "", True, None
    for r in chain:
        payload = json.dumps({
            "actor": r["actor"], "action": r["action"], "case_ref": r["case_ref"],
            "authorised_by": r["authorised_by"],
            "query_params": json.loads(r["query_params"]) if r["query_params"] else None,
            "result_count": r["result_count"], "at": r["at"],
        }, sort_keys=True)
        if hashlib.sha256((prev + payload).encode()).hexdigest() != r["hash"]:
            intact, broken = False, r["id"]
            break
        prev = r["hash"]
    return {"entries": entries, "count": len(chain),
            "chain_intact": intact, "broken_at": broken}


# ------------------------------------------------------- officer's console
#
# Mounted, not reimplemented. Every endpoint below this line is the one the
# real system runs: the same authorisation gate, the same tiering, the same
# evidence bundle, the same audit chain. The only thing added is the
# session, which the real system does not have and this public URL needs.


def session_required(session: Optional[str] = Cookie(None)) -> str:
    return require_session(session)


# The hunt decodes video, detects, tracks and re-identifies. That needs
# PyTorch, OpenCV and a GPU; this is a serverless function with none of
# them. Said plainly, and said *before* the real routes are mounted so
# these answers win the match -- an honest 501 is better than a 500 from an
# import that was never going to succeed here.
HUNT_UNAVAILABLE = {
    "error": "not_available_on_hosted_demo",
    "title": "The hunt does not run here",
    "message": ("Live hunting runs detection and re-identification over "
                "video. That needs PyTorch, OpenCV and a GPU; this is a "
                "serverless function with none of them. Everything on this "
                "console was computed by the real pipeline offline."),
    "remedy": ("The hunt is shown working end to end on real government "
               "footage in the demonstration video."),
}


def _hunt_unavailable():
    return JSONResponse(status_code=501, content=HUNT_UNAVAILABLE)


@app.get("/api/officer/hunt/sources")
def hunt_sources(user: str = Depends(session_required)):
    return _hunt_unavailable()


@app.post("/api/officer/hunt")
@app.post("/api/officer/hunt/designate")
def hunt_start(user: str = Depends(session_required)):
    return _hunt_unavailable()


@app.get("/api/officer/hunt/{hunt_id}")
@app.get("/api/officer/hunt/{hunt_id}/image/{name}")
def hunt_result(hunt_id: str, name: str = "", user: str = Depends(session_required)):
    return _hunt_unavailable()


app.include_router(officer_api.router,
                   dependencies=[Depends(session_required)])


# Leaflet is a public open-source library, not data. Gating it behind the
# session bought nothing and cost the whole page: the script request is a
# subresource, and any hiccup authenticating it leaves the map undefined
# and the console blank. Data stays gated; libraries do not.
OPEN_STATIC_PREFIXES = ("vendor/",)


@app.get("/static/{path:path}")
def static(path: str, session: Optional[str] = Cookie(None)):
    target = (PUBLIC / path).resolve()
    if not str(target).startswith(str(PUBLIC.resolve())) or not target.exists():
        raise HTTPException(status_code=404)
    if not path.startswith(OPEN_STATIC_PREFIXES) and not _verify(session):
        # assets/ carries a crop of government footage; it stays behind the
        # session even though it is only a car roof.
        raise HTTPException(status_code=401)
    return FileResponse(target)
