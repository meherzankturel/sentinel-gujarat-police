#!/usr/bin/env python3
"""
sentinel.registry -- the control plane.

This is deliberately not a list of cameras. It is the record of what each
camera can be trusted to do, and it is what decides which analytics run
where.

The organisers' own guidance says each connected client receives its own
copy of the stream, and to open only the cameras being actively
processed. You physically cannot hold 30 cameras open at once, let alone
80,000. Something has to choose. This is that something -- and because it
chooses on measured capability, plate reading is scheduled only where a
plate is legible, while weak cameras run cheap presence detection and act
as corroboration.

SQLite here for sprint speed. Every column maps one-to-one onto the
Postgres + PostGIS schema in the design document; nothing about the model
depends on the engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "sentinel.db"


def db_path() -> Path:
    """
    Where the registry lives, resolved per call rather than at import.

    The hosted console runs the same code against a reduced copy of the
    database with the public's registrations stripped out, and a serverless
    function only learns where that copy is once it has started. Binding
    the path to a default argument at import time made the environment
    variable arrive too late to matter.
    """
    return Path(os.environ.get("SENTINEL_DB") or DB_PATH)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS camera (
    id                  TEXT PRIMARY KEY,
    catalogue_id        TEXT,
    department          TEXT NOT NULL,
    location_name       TEXT,
    lat                 REAL,
    lon                 REAL,

    rtsp_url            TEXT,
    hls_url             TEXT,
    whep_url            TEXT,

    width               INTEGER,
    height              INTEGER,
    codec               TEXT,
    declared_fps        REAL,
    measured_fps        REAL,          -- from PTS, never CAP_PROP_FPS
    fps_gap             REAL,

    pts_gap_count       INTEGER DEFAULT 0,
    pts_gap_max_ms      REAL DEFAULT 0,
    pts_backwards       INTEGER DEFAULT 0,

    -- what the camera can actually do, measured
    capability_class    TEXT,          -- plate-capable | plate-marginal
                                       -- | presence-only | unusable | unknown
    plate_px_measured   INTEGER,
    capability_basis    TEXT,          -- direct localisation vs derived
    capability_note     TEXT,

    -- how much its clock can be trusted
    time_confidence_ms  REAL,
    time_confidence_note TEXT,

    -- condition
    brightness_mean     REAL,
    night_usable        INTEGER,
    sharpness_mean      REAL,
    health_score        REAL,

    governance_class    TEXT NOT NULL DEFAULT 'open',   -- open|restricted|sensitive
    last_seen_live      TEXT,
    last_audited        TEXT
);

CREATE TABLE IF NOT EXISTS watchlist_entry (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type         TEXT NOT NULL,          -- vehicle | person
    plate_number        TEXT,
    plate_normalised    TEXT,
    make                TEXT,
    model               TEXT,
    colour              TEXT,
    reason              TEXT NOT NULL,          -- stolen|wanted|missing|blacklisted
    priority            INTEGER NOT NULL DEFAULT 3,
    case_ref            TEXT,
    valid_from          TEXT,
    valid_to            TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_watchlist_plate ON watchlist_entry(plate_normalised);

CREATE TABLE IF NOT EXISTS sighting (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id           TEXT NOT NULL REFERENCES camera(id),
    pts_ms              REAL,
    wallclock_utc       TEXT,
    corrected_utc       TEXT,                   -- after clock-drift correction
    entity_type         TEXT NOT NULL DEFAULT 'vehicle',
    plate_read          TEXT,
    plate_normalised    TEXT,
    plate_confidence    REAL,
    plate_px            INTEGER,
    bbox                TEXT,
    frame_ref           TEXT,                   -- source file and arrival PTS
    -- The stem of the evidence stills held for this sighting: the vehicle
    -- as the camera saw it and the plate crop the OCR actually read. Only
    -- set where real footage was processed and the pixels were kept, so a
    -- row without it means "no image retained", never "no image shown".
    evidence_ref        TEXT,
    match_id            INTEGER REFERENCES watchlist_entry(id),
    match_confidence    REAL,
    match_tier          TEXT,                   -- never a boolean
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_sighting_plate  ON sighting(plate_normalised);
CREATE INDEX IF NOT EXISTS ix_sighting_camera ON sighting(camera_id, corrected_utc);

CREATE TABLE IF NOT EXISTS alert (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_entry_id  INTEGER REFERENCES watchlist_entry(id),
    sighting_id         INTEGER REFERENCES sighting(id),
    tier                TEXT NOT NULL,          -- confirmed|probable|corroborating
    raised_at           TEXT DEFAULT CURRENT_TIMESTAMP,
    acknowledged_by     TEXT,
    acknowledged_at     TEXT
);

-- Person-safety alerts (sentinel.threat layer 1: panic dispersal).
--
-- Deliberately a separate table from `alert`. That one hangs off a
-- watchlist entry and a sighting -- somebody was being looked for, and was
-- found. A dispersal alert has neither: nobody is on a list, nothing is
-- identified, and the row is a statement about a crowd's motion field
-- rather than about a person. Forcing it into the vehicle alert table
-- would mean inventing a watchlist entry for it, and an invented row is
-- how an audit trail starts lying.
--
-- `source` is not decoration. It records whether the footage that raised
-- this alert was live government feed, our own controlled rig, or a
-- rendered crowd, and the console is required to show it -- layer 1's
-- true-positive figure is measured on simulation only, so an alert whose
-- provenance is unlabelled would be a claim we cannot support.
CREATE TABLE IF NOT EXISTS dispersal_alert (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id           TEXT NOT NULL REFERENCES camera(id),
    pts_s               REAL,           -- PTS seconds, never arrival time
    raised_at           TEXT,
    source              TEXT NOT NULL,  -- government-live|own-feed|simulation
    detector            TEXT,           -- which processing node raised it
    scene_note          TEXT,           -- mandatory when source=simulation
    epicentre_x         INTEGER,
    epicentre_y         INTEGER,
    frame_w             INTEGER,
    frame_h             INTEGER,
    surge_ratio         REAL,
    coherence           REAL,
    net_drift           REAL,
    movers              INTEGER,
    confidence          REAL,
    frame_ref           TEXT,
    acknowledged_by     TEXT,
    acknowledged_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_dispersal_camera
    ON dispersal_alert(camera_id, raised_at);

-- Tamper-evident: each row carries the hash of the previous one, so a
-- deleted or edited entry breaks the chain and is detectable.
CREATE TABLE IF NOT EXISTS audit_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    actor               TEXT NOT NULL,
    action              TEXT NOT NULL,
    case_ref            TEXT,
    authorised_by       TEXT,
    query_params        TEXT,
    result_count        INTEGER,
    at                  TEXT DEFAULT CURRENT_TIMESTAMP,
    prev_hash           TEXT,
    hash                TEXT
);
"""


def normalise_plate(p: Optional[str]) -> str:
    """
    Indian plates get transcribed inconsistently: spaces, dashes, and the
    classic O/0 and I/1 confusions. Matching happens on the normal form.
    """
    if not p:
        return ""
    s = "".join(ch for ch in p.upper() if ch.isalnum())
    return s.translate(str.maketrans({"O": "0", "I": "1", "Q": "0"}))


@contextmanager
def connect(path: Optional[Path] = None):
    con = sqlite3.connect(path or db_path())
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# Columns added after the first databases were written. CREATE TABLE IF
# NOT EXISTS silently leaves an existing table alone, so a new column in
# SCHEMA above reaches a fresh database and never an old one -- which is
# the kind of difference that shows up as a missing field in production
# and nowhere in development. Added explicitly instead.
# Which grid a camera belongs to, decided by what it presents rather than
# by where a list says it lives. Three copies of this used to exist -- the
# API's, the hosted console's and the deck's -- and a camera that moved
# between grids moved in one of them at a time. It lives here because the
# registry is what knows the difference.
GOVERNMENT_HOSTS = ("cctv.corp8.cloud",)
LOCAL_HOSTS = ("127.0.0.1", "localhost")
OWN_FEED_DEPARTMENT = "Entrant"


def grid_of(cam: dict) -> str:
    """Which grid a camera belongs to, or that we cannot tell."""
    url = (cam.get("hls_url") or cam.get("rtsp_url") or "")
    if any(h in url for h in LOCAL_HOSTS):
        return "local-mock"
    if any(h in url for h in GOVERNMENT_HOSTS):
        return "government"
    if (cam.get("department") or "") == OWN_FEED_DEPARTMENT:
        return "own-feed"
    return "unclassified"


ADDED_COLUMNS = (
    ("sighting", "evidence_ref", "TEXT"),
    # What a camera can do about *people*, measured the same way as what it
    # can do about vehicles. The registry has always assigned vehicle
    # analytics on measured plate width; these are the equivalent axis for
    # the person-safety side, and they are what let the control plane
    # decline face matching on a camera rather than attempt it badly.
    ("camera", "person_body_px", "REAL"),    # p90 standing height, pixels
    ("camera", "face_iod_px", "REAL"),       # derived inter-ocular distance
    ("camera", "face_class", "TEXT"),        # face-capable|marginal|unusable
    ("camera", "face_people_seen", "INTEGER"),
    ("camera", "face_measured_at", "TEXT"),
)


def _add_missing_columns(con):
    for table, column, decl in ADDED_COLUMNS:
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init(path: Optional[Path] = None):
    path = Path(path) if path else db_path()
    with connect(path) as con:
        con.executescript(SCHEMA)
        _add_missing_columns(con)
    return path


# ------------------------------------------------------------------ cameras


def upsert_camera(con, cam: dict):
    """Catalogue fields refresh on every poll; measured fields persist."""
    cols = ("id", "catalogue_id", "department", "location_name", "lat", "lon",
            "rtsp_url", "hls_url", "whep_url", "width", "height", "codec",
            "governance_class", "last_seen_live")
    vals = [cam.get(c) for c in cols]
    con.execute(
        f"INSERT INTO camera ({','.join(cols)}) VALUES ({','.join('?'*len(cols))}) "
        f"ON CONFLICT(id) DO UPDATE SET "
        + ",".join(f"{c}=excluded.{c}" for c in cols if c != "id"),
        vals)


def record_capability(con, camera_id: str, *, capability_class: str,
                      plate_px: Optional[int], basis: str, note: str = "",
                      measured_fps=None, declared_fps=None,
                      pts_gap_count=0, pts_gap_max_ms=0.0, pts_backwards=0,
                      brightness=None, night_usable=None, sharpness=None):
    gap = None
    if declared_fps and measured_fps:
        gap = round(declared_fps - measured_fps, 2)
    con.execute("""
        UPDATE camera SET
            capability_class=?, plate_px_measured=?, capability_basis=?,
            capability_note=?, measured_fps=?, declared_fps=?, fps_gap=?,
            pts_gap_count=?, pts_gap_max_ms=?, pts_backwards=?,
            brightness_mean=?, night_usable=?, sharpness_mean=?,
            last_audited=?
        WHERE id=?""",
        (capability_class, plate_px, basis, note, measured_fps, declared_fps, gap,
         pts_gap_count, pts_gap_max_ms, pts_backwards,
         brightness, night_usable, sharpness,
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), camera_id))


def record_time_confidence(con, camera_id: str, drift_ms: float, note: str = ""):
    """
    Drift between the camera's burned-in clock and true time. A camera
    whose clock is minutes out will place a vehicle at the wrong point in
    a route reconstruction, which is how a timeline quietly becomes wrong.
    """
    con.execute("UPDATE camera SET time_confidence_ms=?, time_confidence_note=? "
                "WHERE id=?", (drift_ms, note, camera_id))


# ------------------------------------------------- analytics assignment


# The survey writes five classes, not three. plate-occasional and
# no-vehicles-observed were missing here, so nine real cameras fell through
# to "probe" -- counted as unmeasured in the assignment headline when they
# had in fact been measured, and given a different ceiling by this module
# than by sentinel.matching. A vocabulary split across two writers is a
# vocabulary that will drift again, so tests/test_facecap.py asserts the
# face equivalent stays complete.
ANALYTIC_FOR = {
    "plate-capable":        "anpr",       # full plate reading
    "plate-occasional":     "anpr_low",   # reads sometimes; corroborating only
    "plate-marginal":       "anpr_low",   # read, but only ever corroborating
    "presence-only":        "presence",   # count and track; never claim a plate
    "no-vehicles-observed": "presence",   # works; simply not watching a road
    "unusable":             "none",
    "unknown":              "probe",
}


def assignments(con) -> list[dict]:
    """
    Which analytic each camera should run. This is the control plane doing
    its job: expensive plate reading is scheduled only where a plate is
    legible, so capacity goes where it can produce evidence rather than
    being spread evenly across cameras that cannot deliver it.
    """
    rows = con.execute(
        "SELECT id, department, capability_class, plate_px_measured, "
        "governance_class FROM camera ORDER BY id").fetchall()
    out = []
    for r in rows:
        cls = r["capability_class"] or "unknown"
        out.append({
            "camera_id": r["id"],
            "department": r["department"],
            "capability": cls,
            "plate_px": r["plate_px_measured"],
            "analytic": ANALYTIC_FOR.get(cls, "probe"),
            "governance": r["governance_class"],
        })
    return out


# --------------------------------------------------------------- audit log


# Reading the previous hash and writing the next entry must not interleave:
# two threads that both read the same predecessor produce a fork, and a
# forked chain reports as tampered for the rest of its life. The officer
# console runs hunts on a background worker, which made this reachable.
_AUDIT_LOCK = threading.Lock()


def audit(con, actor: str, action: str, *, case_ref=None, authorised_by=None,
          query_params=None, result_count=None):
    with _AUDIT_LOCK:
        return _audit_locked(con, actor, action, case_ref, authorised_by,
                             query_params, result_count)


def _audit_locked(con, actor, action, case_ref, authorised_by,
                  query_params, result_count):
    prev = con.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash = prev["hash"] if prev else ""
    # The timestamp must be written exactly as it was hashed. Letting SQLite
    # fill it with CURRENT_TIMESTAMP instead would make every row fail its
    # own verification -- an audit log that always reads as tampered with is
    # worse than none, because it trains people to ignore the alarm.
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = json.dumps({
        "actor": actor, "action": action, "case_ref": case_ref,
        "authorised_by": authorised_by, "query_params": query_params,
        "result_count": result_count, "at": at,
    }, sort_keys=True)
    h = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
    con.execute("INSERT INTO audit_log (actor,action,case_ref,authorised_by,"
                "query_params,result_count,at,prev_hash,hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (actor, action, case_ref, authorised_by,
                 json.dumps(query_params) if query_params else None,
                 result_count, at, prev_hash, h))
    return h


def verify_audit_chain(con) -> tuple[bool, Optional[int]]:
    """Recompute the chain. Returns (intact, first_broken_id)."""
    prev_hash = ""
    for r in con.execute("SELECT * FROM audit_log ORDER BY id").fetchall():
        payload = json.dumps({
            "actor": r["actor"], "action": r["action"], "case_ref": r["case_ref"],
            "authorised_by": r["authorised_by"],
            "query_params": json.loads(r["query_params"]) if r["query_params"] else None,
            "result_count": r["result_count"], "at": r["at"],
        }, sort_keys=True)
        if hashlib.sha256((prev_hash + payload).encode()).hexdigest() != r["hash"]:
            return False, r["id"]
        prev_hash = r["hash"]
    return True, None
