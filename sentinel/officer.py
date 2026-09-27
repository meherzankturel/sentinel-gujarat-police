#!/usr/bin/env python3
"""
sentinel.officer -- the screen an investigating officer actually uses.

Everything else in this system exists to make three questions answerable
about one vehicle: where has it been, in what order, and how much of that
would survive being challenged in court.

Three decisions shape this module.

First, a search is an intrusion. Looking up a plate reaches across eight
departments' cameras and reconstructs a citizen's movements. So the case
reference and the authorising officer are arguments to the query, not
metadata attached afterwards -- a request without them is rejected before
it reads a single row. The audit entry is written inside the same
transaction as the read, so a search that returns results and a search
that was logged are the same event.

Second, a route is an inference, not an observation. The cameras saw
points; the line between them is something we drew. Where that line
implies a speed no vehicle achieves, the honest response is to say so on
the face of the timeline rather than to smooth it away -- a cloned plate
and a camera with a wrong clock both produce exactly this signature, and
an officer is far better placed than we are to tell which.

Third, an evidence bundle has to be checkable by someone who does not
trust us. So it carries the measured capability of every camera that
contributed, the tier of every match with the reason it was demoted, the
audit trail for the case, the verdict of the audit chain, and a hash over
all of it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .matching import (APPEARANCE_FLOOR, APPEARANCE_STRONG, TIER_ORDER,
                       plate_distance)
from .registry import (connect, grid_of, normalise_plate, audit,
                       verify_audit_chain)


# ------------------------------------------------- deferred vision imports
#
# This module has two halves. Search, route and evidence read the registry
# and touch no pixels. The hunt decodes video, detects, tracks and
# re-identifies, and for that it needs OpenCV and PyTorch -- about 700MB
# between them.
#
# Importing those at module scope made the read-only half impossible to
# host. A serverless function caps at 250MB, so serving a table of
# sightings meant shipping a deep-learning stack to do it, and the hosted
# console ended up as a *reimplementation* of these endpoints instead.
# That copy promptly drifted: it returned a different response shape and
# the officer's screen died on it.
#
# So the heavy things load on first use. The read-only half imports
# clean, the hosted console calls the real functions rather than its own,
# and there is one implementation to keep correct.


class _Deferred:
    """Stands in for a heavy module or class until something touches it."""

    def __init__(self, load):
        self._load = load
        self._real = None

    def _get(self):
        if self._real is None:
            self._real = self._load()
        return self._real

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __call__(self, *a, **kw):
        return self._get()(*a, **kw)


def _import(module, attr=None):
    def load():
        import importlib
        m = importlib.import_module(module, package=__package__)
        return getattr(m, attr) if attr else m
    return load


cv2 = _Deferred(_import("cv2"))
np = _Deferred(_import("numpy"))
Detector = _Deferred(_import(".detect", "Detector"))
Hunt = _Deferred(_import(".hunt", "Hunt"))
target_from_crop = _Deferred(_import(".hunt", "target_from_crop"))
VehicleTracker = _Deferred(_import(".track", "VehicleTracker"))

WEB = Path(__file__).resolve().parent.parent / "web"

router = APIRouter(tags=["officer"])

# A plate read that differs from the query by one ordinary character, or by
# two OCR-confusable ones, is still worth showing. Beyond that the officer
# is being handed other people's vehicles, which is both useless and a
# privacy problem in its own right.
SEARCH_TOLERANCE = 1.0

# Chosen against Indian road conditions, not against what a car can do on a
# test track. Nothing on the Gandhinagar-Ahmedabad corridor sustains this
# between two cameras, so a leg above it is evidence of something other
# than a vehicle driving.
IMPOSSIBLE_KMH = 150.0

EARTH_RADIUS_KM = 6371.0088


# ------------------------------------------------------------- geometry


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two cameras.

    Deliberately not road distance. The straight line is always shorter
    than the road, so an implied speed computed from it is a lower bound on
    the speed actually required. That makes the impossible-transit flag
    conservative: anything it raises would only look worse if we routed it
    properly, so it cannot cry wolf on account of the geometry.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _parse_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    t = s.strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ audit
#
# registry.audit reads the previous entry's hash, computes over it, and
# inserts. Two threads doing that at once compute the same predecessor and
# produce a chain with a fork in it -- and a log that fails its own
# verification is worse than no log, because it teaches people to ignore the
# alarm. This module now writes from a background worker as well as from
# request threads, so it serialises its own writes.
#
# One process is the deployment. Across processes the ordering would have to
# be taken in the database, and that belongs in registry.audit rather than
# here.

_AUDIT_LOCK = threading.Lock()


def _audit(con, *args, **kwargs) -> str:
    with _AUDIT_LOCK:
        return audit(con, *args, **kwargs)


# --------------------------------------------------------- authorisation


def _require_authorisation(officer: Optional[str], case_ref: Optional[str],
                           authorised_by: Optional[str]) -> None:
    """
    Refuse the query rather than record the gap.

    Logging an unauthorised search and returning the results anyway would
    produce a perfect record of a thing that should not have happened. The
    point of case-linking is that it is a gate, not a receipt.
    """
    missing = [name for name, value in (("officer", officer),
                                        ("case_ref", case_ref),
                                        ("authorised_by", authorised_by))
               if not (value or "").strip()]
    if missing:
        raise HTTPException(status_code=400, detail={
            "error": "authorisation_required",
            "missing": missing,
            "message": ("A vehicle search reconstructs a person's movements "
                        "across departments. It is case-linked: name the "
                        "case, the searching officer and the officer "
                        "authorising it."),
        })


# ---------------------------------------------------------------- search


SIGHTING_SELECT = """
SELECT s.id, s.camera_id, s.pts_ms, s.wallclock_utc, s.corrected_utc,
       s.plate_read, s.plate_normalised, s.plate_confidence, s.plate_px,
       s.bbox, s.frame_ref, s.evidence_ref,
       s.match_id, s.match_confidence, s.match_tier,
       c.location_name, c.lat, c.lon, c.department, c.capability_class,
       c.hls_url, c.rtsp_url,
       c.plate_px_measured, c.capability_basis, c.governance_class,
       c.time_confidence_ms, c.time_confidence_note,
       w.plate_number   AS listed_plate,
       w.reason         AS listed_reason,
       w.priority       AS listed_priority,
       w.case_ref       AS listed_case_ref
FROM sighting s
JOIN camera c          ON c.id = s.camera_id
LEFT JOIN watchlist_entry w ON w.id = s.match_id
"""


def _candidate_plates(con, target: str) -> list[str]:
    """
    Which stored plates count as the one being searched for.

    The officer types the plate from the FIR. The cameras stored whatever
    the OCR made of it, and on a marginal camera that is not the same
    string. Comparing on equality here would lose the vehicle on precisely
    the cameras where losing it matters, so the comparison is the
    confusion-aware one used for watchlist matching -- same rule for
    finding a vehicle as for alerting on it.
    """
    # Scans the distinct plate values and scores each one. Correct, and
    # fine at sandbox scale; at state scale the candidate set has to be
    # narrowed before scoring -- block on the RTO series and the trailing
    # digits, which no single OCR confusion changes together -- and the
    # scoring then runs over hundreds of rows rather than millions. Noted
    # here because it belongs in the design document, not discovered later.
    seen = con.execute("SELECT DISTINCT plate_normalised FROM sighting "
                       "WHERE plate_normalised IS NOT NULL "
                       "AND plate_normalised != ''").fetchall()
    return [r[0] for r in seen
            if r[0] == target or plate_distance(target, r[0]) <= SEARCH_TOLERANCE]


def _sightings_for(con, target: str) -> list[dict]:
    plates = _candidate_plates(con, target)
    if not plates:
        return []
    q = (SIGHTING_SELECT + " WHERE s.plate_normalised IN (" +
         ",".join("?" * len(plates)) +
         ") ORDER BY COALESCE(s.corrected_utc, s.wallclock_utc), s.id")
    rows = [dict(r) for r in con.execute(q, plates).fetchall()]

    for i, s in enumerate(rows, 1):
        s["seq"] = i
        s["at"] = s["corrected_utc"] or s["wallclock_utc"]
        # Which grid the camera is on, carried on every sighting.
        #
        # The registry page has always broken the estate down by grid; this
        # screen did not, and a row reading "cam-02 - Sector 21 Crossing,
        # Gandhinagar - Police" invites a reviewer to assume a police
        # camera on a street when it is a local mock feed. The demonstration
        # is stronger for saying which is which: the mock grid is where the
        # route is exercised, and the own feed is where real pixels are.
        s["grid"] = grid_of(s)
        # The distance is shown, not just used. An officer looking at a
        # timeline needs to know which of these rows is the plate they
        # asked for and which is a degraded read we chose to include.
        #
        # A row with no plate_read at all is a third case: a camera below
        # plate capability contributed a vehicle to this track without ever
        # claiming to have read the number. It must not be allowed to
        # present as an exact match just because the attributed plate
        # string happens to equal the query.
        s["plate_claimed"] = bool(s["plate_read"])
        if not s["plate_claimed"]:
            s["read_distance"], s["exact_read"] = None, False
        else:
            s["read_distance"] = (0.0 if s["plate_normalised"] == target
                                  else round(plate_distance(target, s["plate_normalised"]), 2))
            s["exact_read"] = s["read_distance"] == 0.0
        s["tier"] = s["match_tier"] or "corroborating"
        # A camera whose clock was measured out is still usable -- the
        # sighting is corrected on the way in. What must not happen is the
        # correction being applied silently, so both times travel together.
        drift = s["time_confidence_ms"]
        s["clock_corrected"] = bool(drift and abs(drift) >= 1000)
        # Whether the correction is worth an officer's attention is a
        # different question from whether one was applied. A drift smaller
        # than the interval between cameras cannot reorder anything or move
        # an implied speed; flagging it on every row would bury the camera
        # that is seven minutes out among five that are not.
        s["clock_material"] = bool(drift and abs(drift) >= 5000)
        s["clock_drift_s"] = round(drift / 1000.0, 1) if drift is not None else None
    return rows


@router.get("/api/officer/search")
def officer_search(plate: str = Query(..., min_length=2),
                   case_ref: Optional[str] = None,
                   officer: Optional[str] = None,
                   authorised_by: Optional[str] = None):
    """
    Every sighting of a vehicle, in corrected time order, with the
    confidence of each and the measured capability of the camera that
    produced it.
    """
    _require_authorisation(officer, case_ref, authorised_by)
    target = normalise_plate(plate)
    if not target:
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_plate", "message": "No alphanumerics in that plate."})

    with connect() as con:
        rows = _sightings_for(con, target)
        listed = con.execute(
            "SELECT * FROM watchlist_entry WHERE plate_normalised = ? "
            "OR plate_number = ?", (target, plate.upper().strip())).fetchone()
        audit_hash = _audit(con, officer.strip(), "officer_search_vehicle",
                           case_ref=case_ref.strip(),
                           authorised_by=authorised_by.strip(),
                           query_params={"plate": plate, "normalised": target,
                                         "tolerance": SEARCH_TOLERANCE},
                           result_count=len(rows))

    tiers = {t: 0 for t in TIER_ORDER}
    for s in rows:
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1

    return {
        "plate": plate.upper().strip(),
        "normalised": target,
        "watchlist": dict(listed) if listed else None,
        "sightings": rows,
        "count": len(rows),
        "tiers": tiers,
        "cameras": len({s["camera_id"] for s in rows}),
        "departments": sorted({s["department"] for s in rows}),
        "audit": {"case_ref": case_ref, "officer": officer,
                  "authorised_by": authorised_by, "entry_hash": audit_hash},
    }


# ----------------------------------------------------------------- route


def _legs(rows: list[dict]) -> list[dict]:
    """
    Turn a list of points into a journey, and mark the parts of it that
    cannot have happened.

    A leg above the threshold has three ordinary explanations and the
    system cannot distinguish between them from the data alone: a second
    vehicle wearing the same plate, a camera whose clock is wrong, or a
    misread that attached someone else's car to this track. All three are
    surfaced, because each points the officer at a different next action
    and guessing on their behalf would be the wrong kind of confidence.
    """
    out = []
    for a, b in zip(rows, rows[1:]):
        if None in (a["lat"], a["lon"], b["lat"], b["lon"]):
            continue
        km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        t1, t2 = _parse_utc(a["at"]), _parse_utc(b["at"])
        secs = (t2 - t1).total_seconds() if (t1 and t2) else None

        kmh = None
        impossible = False
        if secs is not None and secs > 0:
            kmh = km / (secs / 3600.0)
            impossible = kmh > IMPOSSIBLE_KMH
        elif secs is not None and km > 0.05:
            # Two cameras 50m or more apart cannot see the same vehicle at
            # the same corrected second. 50m is the slack for GPS on the
            # camera positions themselves.
            impossible = True

        why = []
        if impossible:
            untrusted = [s["camera_id"] for s in (a, b)
                         if (s["time_confidence_note"] or "").startswith(("unreliable", "correctable"))]
            why.append("A cloned plate: two vehicles carrying this number.")
            if untrusted:
                why.append("An untrusted camera clock on "
                           + ", ".join(untrusted) + ".")
            if not (a["exact_read"] and b["exact_read"]):
                why.append("A misread or attributed plate at one end of "
                           "this leg, putting another vehicle on the track.")

        out.append({
            "from_seq": a["seq"], "to_seq": b["seq"],
            "from_camera": a["camera_id"], "to_camera": b["camera_id"],
            "from_location": a["location_name"], "to_location": b["location_name"],
            "from_at": a["at"], "to_at": b["at"],
            "from_lat": a["lat"], "from_lon": a["lon"],
            "to_lat": b["lat"], "to_lon": b["lon"],
            "km": round(km, 3),
            "seconds": None if secs is None else round(secs, 1),
            "minutes": None if secs is None else round(secs / 60.0, 1),
            "implied_kmh": None if kmh is None else round(kmh, 1),
            "impossible": impossible,
            "threshold_kmh": IMPOSSIBLE_KMH,
            "explanations": why,
        })
    return out


@router.get("/api/officer/route")
def officer_route(plate: str = Query(..., min_length=2),
                  case_ref: Optional[str] = None,
                  officer: Optional[str] = None,
                  authorised_by: Optional[str] = None):
    """
    The reconstructed journey. Distance and elapsed time between each pair
    of consecutive sightings, the speed that implies, and a flag on any leg
    no vehicle could have driven.
    """
    # This route returned a person's full movement history -- every camera,
    # location, department and corrected time -- to any caller, with no
    # credentials. Worse, the audit write below was conditional on those
    # same credentials being present, so an unauthorised read left no trace
    # at all. It is the same data the gated search returns, reachable by
    # changing one word in the URL, and it falsified the claim this module
    # is built on.
    _require_authorisation(officer, case_ref, authorised_by)
    target = normalise_plate(plate)
    if not target:
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_plate", "message": "No alphanumerics in that plate."})

    with connect() as con:
        rows = _sightings_for(con, target)
        legs = _legs(rows)
        # Reconstruction is a second read of the same footage and is logged
        # as such, even though the officer already passed the gate at
        # search -- the audit trail should read as what was done, not as
        # what was authorised once and then repeated silently.
        _audit(con, officer.strip(), "officer_reconstruct_route",
               case_ref=case_ref.strip(), authorised_by=authorised_by,
               query_params={"plate": plate, "normalised": target},
               result_count=len(rows))

    t_first = _parse_utc(rows[0]["at"]) if rows else None
    t_last = _parse_utc(rows[-1]["at"]) if rows else None
    span = ((t_last - t_first).total_seconds() / 60.0
            if (t_first and t_last) else None)

    return {
        "plate": plate.upper().strip(),
        "normalised": target,
        "sightings": rows,
        "legs": legs,
        "summary": {
            "sightings": len(rows),
            "cameras": len({s["camera_id"] for s in rows}),
            "distance_km": round(sum(l["km"] for l in legs), 2),
            "span_minutes": None if span is None else round(span, 1),
            "impossible_legs": sum(1 for l in legs if l["impossible"]),
            "threshold_kmh": IMPOSSIBLE_KMH,
        },
    }


# -------------------------------------------------------------- evidence


def _audit_verdict(con, case_entry_ids: list[int]) -> dict:
    """
    Localise damage to the audit log instead of condemning everything after
    it.

    registry.verify_audit_chain stops at the first bad row, which is the
    right answer to "is this log clean". It is the wrong answer to the
    question an evidence bundle has to survive, which is "was *this case's*
    record altered" -- one tampered row from an unrelated case in 2024
    would otherwise render every subsequent bundle unciteable.

    So the walk continues past a break, carrying the stored hash forward.
    Each row is then checked against its own recorded predecessor, which
    isolates the altered rows rather than colouring the whole tail.
    """
    intact, first_broken = verify_audit_chain(con)
    altered: list[int] = []
    prev_hash = ""
    for r in con.execute("SELECT * FROM audit_log ORDER BY id").fetchall():
        payload = json.dumps({
            "actor": r["actor"], "action": r["action"], "case_ref": r["case_ref"],
            "authorised_by": r["authorised_by"],
            "query_params": json.loads(r["query_params"]) if r["query_params"] else None,
            "result_count": r["result_count"], "at": r["at"],
        }, sort_keys=True)
        if hashlib.sha256((prev_hash + payload).encode()).hexdigest() != r["hash"]:
            altered.append(r["id"])
        prev_hash = r["hash"]

    touched = sorted(set(altered) & set(case_entry_ids))
    return {
        "intact": intact,
        "first_broken_id": first_broken,
        "altered_entries": altered,
        "case_entries_intact": not touched,
        "case_entries_altered": touched,
        "note": ("Every entry in this case verifies against its recorded "
                 "predecessor." if not touched else
                 "Entries in this case do not verify and the bundle should "
                 "not be relied on until that is explained."),
    }


def bundle_hash(body: dict) -> str:
    """
    One hash over the whole bundle, computed on a canonical serialisation
    so that reordering a dictionary cannot change it but altering a single
    timestamp, tier or plate read must.
    """
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


@router.post("/api/officer/evidence")
def officer_evidence(plate: str = Query(..., min_length=2),
                     case_ref: Optional[str] = None,
                     officer: Optional[str] = None,
                     authorised_by: Optional[str] = None):
    """
    A court-ready bundle: what was seen, by which cameras, how far those
    cameras were measured to be trustworthy, and who looked.

    The camera block is the part that distinguishes this from an export.
    A defence solicitor asking "how do you know it was that vehicle" gets
    the measured plate width and the basis of the measurement for the
    specific camera, rather than a resolution figure from a datasheet.
    """
    _require_authorisation(officer, case_ref, authorised_by)
    target = normalise_plate(plate)

    with connect() as con:
        rows = _sightings_for(con, target)
        if not rows:
            raise HTTPException(status_code=404, detail={
                "error": "no_sightings",
                "message": f"No sighting of {plate.upper()} to bundle."})

        cam_ids = sorted({s["camera_id"] for s in rows})
        cameras = [dict(r) for r in con.execute(
            "SELECT id, department, location_name, lat, lon, width, height, "
            "codec, declared_fps, measured_fps, capability_class, "
            "plate_px_measured, capability_basis, capability_note, "
            "time_confidence_ms, time_confidence_note, governance_class, "
            "last_audited FROM camera WHERE id IN (" +
            ",".join("?" * len(cam_ids)) + ") ORDER BY id", cam_ids).fetchall()]

        listed = con.execute(
            "SELECT * FROM watchlist_entry WHERE plate_normalised = ?",
            (target,)).fetchone()
        trail = [dict(r) for r in con.execute(
            "SELECT id, actor, action, case_ref, authorised_by, query_params, "
            "result_count, at, hash FROM audit_log WHERE case_ref = ? "
            "ORDER BY id", (case_ref.strip(),)).fetchall()]
        chain = _audit_verdict(con, [r["id"] for r in trail])

        body = {
            "bundle_type": "sentinel.vehicle_movement.v1",
            "generated_at": _now(),
            "case_ref": case_ref.strip(),
            "prepared_by": officer.strip(),
            "authorised_by": authorised_by.strip(),
            "subject": {
                "plate_as_searched": plate.upper().strip(),
                "plate_normalised": target,
                "watchlist_entry": dict(listed) if listed else None,
            },
            "sightings": rows,
            "route": {"legs": _legs(rows)},
            "cameras": cameras,
            "confidence_note": (
                "Each sighting carries the tier assigned at match time. A "
                "tier is capped by the camera's measured capability before "
                "any read confidence is considered: a camera not measured "
                "able to resolve a plate cannot produce a confirmed match, "
                "however certain the reader was."),
            "audit_trail": trail,
            # The export cannot appear in the trail it prints: its own log
            # entry commits the hash of this object, so the entry can only
            # be written once the object is final. Saying so here is better
            # than letting a reader discover the gap and wonder what else
            # is missing. The entry's hash is returned beside the bundle.
            "audit_trail_note": (
                "Every access to this case logged before this bundle was "
                "built. The export itself is written to the chain "
                "immediately afterwards and is therefore necessarily "
                "absent from the trail printed inside it; its entry hash "
                "is returned alongside this bundle."),
            "audit_chain": chain,
        }
        digest = bundle_hash(body)

        # Producing a bundle is the most consequential thing on this screen
        # -- it is the artefact that leaves the system -- so its hash goes
        # into the same chain that protects the searches behind it.
        entry = _audit(con, officer.strip(), "officer_export_evidence",
                      case_ref=case_ref.strip(),
                      authorised_by=authorised_by.strip(),
                      query_params={"plate": plate, "bundle_sha256": digest},
                      result_count=len(rows))

    return {"bundle": body,
            "integrity": {"algorithm": "sha256", "sha256": digest,
                          "covers": "the bundle object exactly as returned, "
                                    "serialised with sorted keys",
                          "audit_entry_hash": entry}}


# ------------------------------------------------------------------- hunt
#
# Everything above answers questions about footage the system has already
# looked at. This answers the one the live test case actually asks: here is
# a vehicle, go and find it, on cameras nobody has searched yet.
#
# Three decisions shape it.
#
# A hunt is not a query. It decodes video, and video arrives at the speed of
# video. So the request starts the work and returns a handle, and the
# console polls it. A synchronous endpoint would hold the connection open
# for minutes and time out in front of whoever is watching.
#
# It searches one camera at a time on a single worker. That is not a
# simplification to be removed on a bigger machine. The organisers' own
# guidance is that every connected client receives its own copy of the
# stream and that only the cameras being processed should be open; serial
# search is what the grid permits, which is precisely why the registry has
# to choose which cameras are worth opening at all.
#
# And it reads recorded windows captured from these same cameras. A
# demonstration room cannot depend on the gateway being reachable. The
# identification chain never learns where its pixels came from, so a clip
# and a live socket are the same input to it -- the code path below is the
# one that would run against RTSP, with a different frame source.

FOOTAGE = Path(__file__).resolve().parent.parent / "media" / "real"
HUNT_RUN = Path(__file__).resolve().parent.parent / ".run" / "hunt"

# The picture of the vehicle the console starts with, so a cold machine can
# demonstrate a hunt without first being asked to find a vehicle to hunt.
# It lives under web/ because .run/ is working state and is not committed.
PRESET_TARGETS = (WEB / "assets" / "designated-vehicle.jpg",
                  Path(__file__).resolve().parent.parent / ".run" / "preview" / "hunt_target.jpg")

# A bound stated in the answer rather than a timeout hidden in the server.
# The officer is told how much footage was examined, so "not found" means
# "not found in this window" and never quietly means "we stopped early".
# 150 frames is six to ten seconds of footage per camera depending on the
# camera's real delivered rate. Chosen so an officer gets an answer while
# they are still looking at the screen; the officer can raise it, and
# whatever they chose is reported back with the result.
DEFAULT_FRAMES = 150
FRAMES_CEILING = 900

# Plate reading needs roughly this many pixels across the plate. It is the
# figure the capability survey measured against, and it is what makes a
# presence-only camera presence-only.
PLATE_PX_NEEDED = 120

_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9_.\-]+\.jpg$")
# Candidate ids are generated by this module as short alphanumerics, but
# they come back through a request body and are interpolated into a
# filename. The "cand_" prefix happens to defeat traversal on its own --
# every escape attempt needs a directory literally named "cand_..". That is
# an accident of the naming scheme rather than a control, so the id is
# checked explicitly instead of relying on it.
_SAFE_CANDIDATE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


# ------------------------------------------------------------- footage

_FOOTAGE: Optional[dict] = None


def footage_catalogue() -> dict:
    """
    Which cameras this machine can hunt right now, from windows recorded off
    them. Probed once: opening six files costs nothing, opening them on
    every poll would cost something.
    """
    global _FOOTAGE
    if _FOOTAGE is not None:
        return _FOOTAGE
    found: dict = {}
    for path in sorted(FOOTAGE.glob("*.mp4")):
        camera_id = path.stem.split("_")[0]
        # Two windows exist for some cameras. The daylight one is the window
        # the capability survey measured against, so it is the one whose
        # measured numbers the tiers below are entitled to cite.
        if camera_id in found and not path.stem.endswith("_day"):
            continue
        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        found[camera_id] = {
            "camera_id": camera_id, "path": str(path), "file": path.name,
            "frames": frames, "fps": round(fps, 2),
            "seconds": round(frames / fps, 1) if fps else None,
            "window": "daylight" if path.stem.endswith("_day") else "recorded",
        }
    _FOOTAGE = found
    return found


def _camera_rows(ids: list[str]) -> dict:
    if not ids:
        return {}
    with connect() as con:
        return {r["id"]: dict(r) for r in con.execute(
            "SELECT id, department, location_name, lat, lon, capability_class, "
            "plate_px_measured, capability_basis, capability_note, "
            "time_confidence_ms, governance_class FROM camera WHERE id IN ("
            + ",".join("?" * len(ids)) + ")", ids).fetchall()}


def tier_ceiling(cam: dict) -> dict:
    """
    The best tier this camera is permitted to produce, and why it is that
    and no better -- computed from the registry's measurements, before any
    footage is opened.

    Stating it in advance is the point. An officer choosing cameras should
    know that a match on this one can only ever corroborate, rather than
    discovering it afterwards from a demoted result and assuming the system
    lost confidence in the vehicle.
    """
    capability = cam.get("capability_class") or "unknown"
    px = cam.get("plate_px_measured")
    drift = cam.get("time_confidence_ms")
    px_txt = f"{px}px of number plate" if px is not None else "no measurable plate"

    if capability == "presence-only":
        return {"tier": "corroborating", "reason": (
            f"measured at {px_txt} against the ~{PLATE_PX_NEEDED}px a read "
            f"needs, so the registry classes it presence-only; a match here "
            f"corroborates and cannot promote")}
    if drift is not None and abs(drift) > 30_000:
        return {"tier": "corroborating", "reason": (
            f"clock measured {drift / 1000:+.0f}s out, past the 30s a "
            f"sighting order can absorb; anything seen here is capped")}
    if capability in ("plate-capable", "plate-marginal"):
        return {"tier": "probable", "reason": (
            f"measured at {px_txt} and classed {capability}, so a strong "
            f"appearance match is not capped by capability here")}
    return {"tier": "probable", "reason": (
        f"classed {capability} at {px_txt} — no plate to corroborate with, "
        f"but nothing here caps a strong appearance match either")}


@router.get("/api/officer/hunt/sources")
def hunt_sources():
    """
    The cameras a hunt can be run against on this machine, each with what
    the registry measured about it and the best tier it could contribute.

    No movement data crosses this endpoint -- it is the registry describing
    itself -- so it is not case-linked. Starting a hunt is.
    """
    cat = footage_catalogue()
    rows = _camera_rows(sorted(cat))
    out = []
    for cid, clip in sorted(cat.items()):
        cam = rows.get(cid, {})
        # The filename is worth showing -- it is the provenance of the
        # footage. Its path on this machine is not the panel's business.
        out.append({**{k: v for k, v in clip.items() if k != "path"},
                    "location_name": cam.get("location_name", cid),
                    "department": cam.get("department"),
                    "lat": cam.get("lat"), "lon": cam.get("lon"),
                    "capability_class": cam.get("capability_class"),
                    "plate_px_measured": cam.get("plate_px_measured"),
                    "capability_basis": cam.get("capability_basis"),
                    "time_confidence_ms": cam.get("time_confidence_ms"),
                    "governance_class": cam.get("governance_class"),
                    "ceiling": tier_ceiling(cam)})
    return {
        "cameras": out,
        "count": len(out),
        "default_frames": DEFAULT_FRAMES,
        "frames_ceiling": FRAMES_CEILING,
        "note": ("Recorded windows captured from these cameras. The chain "
                 "that identifies a vehicle is given frames and is not told "
                 "whether they arrived from a file or a socket, so this is "
                 "the same code path the live grid runs."),
    }


# ----------------------------------------------------------------- jobs


@dataclass
class _Job:
    """One hunt, or one designation pass, and everything the console shows."""
    id: str
    kind: str                     # "hunt" | "designate"
    case_ref: str
    officer: str
    authorised_by: str
    cameras: list
    max_frames: int
    dir: Path
    created: str
    plate: Optional[str] = None
    vehicle_class: Optional[str] = None
    designation: str = ""
    state: str = "queued"         # queued | running | done | failed
    stage: str = "Queued behind the camera currently open."
    error: Optional[str] = None
    started: Optional[float] = None
    finished: Optional[float] = None
    audit_start: Optional[str] = None
    audit_done: Optional[str] = None
    cams: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    sightings: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    legs: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=dict)


_JOBS: "dict[str, _Job]" = {}
_JOBS_LOCK = threading.Lock()
_JOB_QUEUE: "queue.Queue[_Job]" = queue.Queue()
_WORKER: Optional[threading.Thread] = None
_JOB_LIMIT = 24


def _ensure_worker() -> None:
    global _WORKER
    with _JOBS_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(target=_worker_loop, name="sentinel-hunt",
                                   daemon=True)
        _WORKER.start()


def _worker_loop() -> None:
    while True:
        job = _JOB_QUEUE.get()
        try:
            job.started = time.time()
            job.state = "running"
            (_run_designate if job.kind == "designate" else _run_hunt)(job)
            job.state = "done"
        except Exception as exc:                      # noqa: BLE001
            # A failed hunt is reported as failed. Returning an empty result
            # set would tell the officer the vehicle was not on those
            # cameras, which is a different and much worse claim.
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.stage = "The search stopped before it finished."
        finally:
            job.finished = time.time()
            _JOB_QUEUE.task_done()
            _close_job(job)


def _close_job(job: _Job) -> None:
    """Write the outcome to the chain, so the log records what was done."""
    try:
        with connect() as con:
            job.audit_done = _audit(
                con, job.officer,
                "officer_hunt_complete" if job.kind == "hunt"
                else "officer_designate_complete",
                case_ref=job.case_ref, authorised_by=job.authorised_by,
                query_params={"hunt_id": job.id, "cameras": job.cameras,
                              "frames_per_camera": job.max_frames,
                              "outcome": job.state},
                result_count=len(job.sightings) if job.kind == "hunt"
                else len(job.candidates))
    except Exception:                                 # noqa: BLE001
        pass


def _remember(job: _Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        if len(_JOBS) > _JOB_LIMIT:
            for old in sorted(_JOBS.values(), key=lambda j: j.created)[:-_JOB_LIMIT]:
                _JOBS.pop(old.id, None)


def _job_or_404(hunt_id: str) -> _Job:
    job = _JOBS.get(hunt_id)
    if job is None:
        raise HTTPException(status_code=404, detail={
            "error": "no_such_hunt",
            "message": "That hunt is not on this server. Hunts do not survive "
                       "a restart; start a new one."})
    return job


# --------------------------------------------------------------- running


def _frames(path: str, limit: int, job: _Job, camera_id: str):
    """
    Frames out of a recorded window, counting as it goes so the console can
    show the search advancing rather than a spinner that means nothing.
    """
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        while n < limit:
            ok, img = cap.read()
            if not ok:
                return
            n += 1
            job.cams[camera_id]["frames_done"] = n
            yield img
    finally:
        cap.release()


def _write_jpg(path: Path, img) -> None:
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _attach_pictures(job: _Job, camera_id: str, clip: dict, hits: list) -> None:
    """
    Pull back the frame each candidate was seen on and keep two pictures of
    it: the vehicle, and the whole frame with the vehicle boxed.

    This is not decoration. The system's posture is that it proposes and a
    human disposes, and an officer cannot dispose of a row of numbers. The
    crop next to the designated vehicle is how the proposal gets checked.
    """
    if not hits:
        return
    cap = cv2.VideoCapture(clip["path"])
    fps = clip.get("fps") or 25.0
    try:
        for h in hits:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, h.frame - 1))
            ok, img = cap.read()
            if not ok or img is None:
                continue
            key = f"{camera_id}_{h.frame}"
            # Position in the recorded window, from the decoder's own frame
            # position rather than from when we happened to process it.
            job.offsets[key] = round(
                (cap.get(cv2.CAP_PROP_POS_MSEC) or (h.frame / fps) * 1000.0) / 1000.0, 2)
            if h.box:
                x1, y1, x2, y2 = h.box
                crop = img[y1:y2, x1:x2]
                if crop.size:
                    _write_jpg(job.dir / f"crop_{key}.jpg", crop)
                ctx = img.copy()
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (196, 224, 87), 3)
                scale = 900.0 / ctx.shape[1]
                if scale < 1:
                    ctx = cv2.resize(ctx, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                _write_jpg(job.dir / f"ctx_{key}.jpg", ctx)
    finally:
        cap.release()


def _run_hunt(job: _Job) -> None:
    crop = cv2.imread(str(job.dir / "target.jpg"))
    if crop is None:
        raise RuntimeError("the designated picture could not be read back")
    target = target_from_crop(crop, plate=job.plate or None,
                              vehicle_class=job.vehicle_class or None)
    if target.appearance is None:
        raise RuntimeError(
            "no appearance fingerprint could be taken from that picture — it "
            "is too small or too degraded to identify a vehicle from")

    hunt = Hunt(target)
    # Hunt loads the columns it needs to make a decision. The console needs
    # more than that -- which department owns the camera, and under what
    # governance class -- so the registry row is read again here rather than
    # widening what the matcher is handed.
    job.meta = _camera_rows(job.cameras)
    cat = footage_catalogue()

    for cid in job.cameras:
        clip = cat[cid]
        cam = job.meta.get(cid) or {}
        job.cams[cid]["state"] = "searching"
        job.stage = (f"Searching {cid} — {cam.get('location_name', cid)}. "
                     f"One camera is open at a time.")
        started = time.time()
        hits = hunt.search_stream(cid, _frames(clip["path"], job.max_frames, job, cid),
                                  camera_name=cam.get("location_name", ""),
                                  max_frames=job.max_frames)
        _attach_pictures(job, cid, clip, hits)
        job.cams[cid].update({"state": "done", "sightings": len(hits),
                              "seconds": round(time.time() - started, 1)})
        # Published after every camera, so the timeline fills in as the
        # search advances instead of appearing all at once at the end.
        job.sightings = [_present_hit(job, h, i)
                         for i, h in enumerate(hunt.hits, 1)]
        job.legs = _hunt_legs(job, hunt.route())
        job.summary = _hunt_summary(job, hunt)

    job.stage = "Search complete."
    job.summary = _hunt_summary(job, hunt)


def _run_designate(job: _Job) -> None:
    """
    Offer the vehicles a camera actually saw, so a vehicle can be designated
    without anybody first having a photograph of it.

    In a live incident the officer rarely has a clean picture; what they have
    is "the car that came through this junction around then". This turns that
    into a designation.
    """
    cid = job.cameras[0]
    clip = footage_catalogue()[cid]
    cam = (_camera_rows([cid]) or {}).get(cid, {})
    job.meta = {cid: cam}
    job.cams[cid]["state"] = "searching"
    job.stage = f"Reading {cid} — {cam.get('location_name', cid)}."

    tracker = VehicleTracker()
    best: dict = {}
    for n, img in enumerate(_frames(clip["path"], job.max_frames, job, cid), 1):
        if n % 3:
            continue
        for t in tracker.update(Detector.vehicles(img, min_score=0.6), n):
            if t.last_seen != n or t.frames < 2 or t.is_static():
                continue
            # Below this a crop carries too little of the vehicle to
            # fingerprint, and offering it would invite a designation that
            # cannot match anything.
            if t.width < 90:
                continue
            prev = best.get(t.track_id)
            if prev is not None and t.width <= prev["width"]:
                continue
            x1, y1, x2, y2 = (max(0, int(v)) for v in t.box)
            piece = img[y1:y2, x1:x2]
            if piece.size == 0:
                continue
            name = f"cand_{cid}_{t.track_id}.jpg"
            _write_jpg(job.dir / name, piece)
            best[t.track_id] = {
                "id": str(t.track_id), "camera_id": cid,
                "candidate": f"{cid}_{t.track_id}",
                "camera_name": cam.get("location_name", cid),
                "vehicle_class": t.label, "width": t.width, "frame": n,
                "offset_s": round(n / (clip.get("fps") or 25.0), 1),
                "image": f"/api/officer/hunt/{job.id}/image/{name}",
            }
            job.candidates = sorted(best.values(), key=lambda c: -c["width"])[:12]

    job.cams[cid].update({"state": "done", "sightings": len(job.candidates)})
    job.stage = (f"{len(job.candidates)} vehicle(s) offered for designation."
                 if job.candidates else
                 "No vehicle large enough to designate in this window.")


# ----------------------------------------------------------- presentation


def _hunt_reasons(hit, cam: dict) -> tuple:
    """
    Why this match got this tier, in the terms the decision was actually
    made in.

    The reason is the deliverable. A tier without it is an opinion, and an
    officer asked to act on an opinion has no way to weigh it -- or to tell
    a court why they did.
    """
    capability = cam.get("capability_class") or "unknown"
    px = cam.get("plate_px_measured")
    drift = cam.get("time_confidence_ms")
    px_txt = f"{px}px of number plate" if px is not None else "no measurable plate"
    strong = hit.score >= APPEARANCE_STRONG

    reasons = [{"code": "appearance", "text": (
        f"Appearance {hit.score:.2f} against the designated vehicle. " + (
            f"At or above {APPEARANCE_STRONG:.2f}, where measured "
            f"same-vehicle and different-vehicle scores separate on this grid."
            if strong else
            f"Above the {APPEARANCE_FLOOR:.2f} floor worth an officer's "
            f"attention, below the {APPEARANCE_STRONG:.2f} mark, so it is "
            f"offered as corroboration and nothing more."))}]

    capped = None
    if capability == "presence-only":
        capped = "camera measured presence-only"
        reasons.append({"code": "capability", "text": (
            f"{hit.camera_id} was measured at {px_txt} against the "
            f"~{PLATE_PX_NEEDED}px a read needs, so the registry classes it "
            f"presence-only. A presence-only camera cannot promote a match, "
            f"however convincing the pixels look.")})
    elif capability in ("plate-capable", "plate-marginal"):
        reasons.append({"code": "capability", "text": (
            f"{hit.camera_id} was measured at {px_txt} and is classed "
            f"{capability}, so capability does not cap this match.")})
    else:
        reasons.append({"code": "capability", "text": (
            f"{hit.camera_id} is classed {capability} at {px_txt}. No plate "
            f"is available here to corroborate the appearance.")})

    if drift is not None and abs(drift) > 30_000:
        capped = capped or f"camera clock {drift / 1000:+.0f}s out"
        reasons.append({"code": "clock", "text": (
            f"{hit.camera_id} was measured {drift / 1000:+.0f}s off true "
            f"time, past the 30s that a sighting order can absorb. Capped "
            f"until the clock is explained.")})

    reasons.append({"code": "ceiling", "text": (
        "Identified on appearance alone, which never reaches confirmed. A "
        "white hatchback resembles thousands of others, and getting somebody "
        "stopped because their car looks like another one is the failure "
        "this design exists to avoid.")})

    head = f"appearance {hit.score:.2f}"
    if capped:
        head += f"; {capped}, so capped at {hit.tier}"
    elif strong:
        head += f", above the {APPEARANCE_STRONG:.2f} mark — probable"
    else:
        head += f", below the {APPEARANCE_STRONG:.2f} mark — corroborating"
    return head, reasons


def _present_hit(job: _Job, hit, seq: int) -> dict:
    cam = job.meta.get(hit.camera_id) or {}
    head, reasons = _hunt_reasons(hit, cam)
    key = f"{hit.camera_id}_{hit.frame}"
    base = f"/api/officer/hunt/{job.id}/image"
    return {
        "seq": seq,
        "camera_id": hit.camera_id,
        "camera_name": hit.camera_name or cam.get("location_name") or hit.camera_id,
        "department": cam.get("department"),
        "capability_class": cam.get("capability_class"),
        "plate_px_measured": cam.get("plate_px_measured"),
        "lat": hit.lat, "lon": hit.lon,
        "frame": hit.frame,
        "offset_s": job.offsets.get(key),
        "at_utc": hit.at_utc,
        "tier": hit.tier,
        "score": hit.score,
        "basis": hit.basis,
        "vehicle_class": hit.vehicle_class,
        "box": list(hit.box) if hit.box else None,
        "why": hit.why,
        "reason": head,
        "reasons": reasons,
        "crop_url": f"{base}/crop_{key}.jpg" if (job.dir / f"crop_{key}.jpg").exists() else None,
        "context_url": f"{base}/ctx_{key}.jpg" if (job.dir / f"ctx_{key}.jpg").exists() else None,
    }


def _hunt_legs(job: _Job, raw: list) -> list:
    """
    The journey, with the cameras named and placed so the console can draw
    it, and with the basis of the timing stated rather than implied.
    """
    out = []
    for leg in raw:
        a = job.meta.get(leg["from"]) or {}
        b = job.meta.get(leg["to"]) or {}
        out.append({**leg,
                    "from_name": a.get("location_name", leg["from"]),
                    "to_name": b.get("location_name", leg["to"]),
                    "from_lat": a.get("lat"), "from_lon": a.get("lon"),
                    "to_lat": b.get("lat"), "to_lon": b.get("lon"),
                    "same_camera": leg["from"] == leg["to"],
                    "basis": ("elapsed frames within the searched windows at "
                              "25 fps, not wall-clock between two live feeds")})
    return out


def _hunt_summary(job: _Job, hunt) -> dict:
    s = hunt.summary()
    seen = {h.camera_id for h in hunt.hits}
    clear = [c for c in job.cameras
             if c not in seen and job.cams[c]["state"] == "done"]
    return {**s,
            "cameras_searched": sum(1 for c in job.cameras
                                    if job.cams[c]["state"] == "done"),
            "cameras_clear": clear,
            "frames_examined": sum(job.cams[c]["frames_done"] for c in job.cameras),
            # Stated because it is the number that decides whether an officer
            # can trust this at all. A hunt that finds the vehicle everywhere
            # has found nothing.
            "clear_note": ("Searched and clear: nothing on these cameras "
                           "resembled the designated vehicle closely enough "
                           "to be worth an officer's attention.")}


def _snapshot(job: _Job) -> dict:
    now = time.time()
    cams = []
    for cid in job.cameras:
        p = job.cams[cid]
        cam = job.meta.get(cid) or {}
        cams.append({"camera_id": cid,
                     "location_name": cam.get("location_name", cid),
                     "department": cam.get("department"),
                     "capability_class": cam.get("capability_class"),
                     "plate_px_measured": cam.get("plate_px_measured"),
                     "lat": cam.get("lat"), "lon": cam.get("lon"),
                     "ceiling": tier_ceiling(cam), **p})
    return {
        "hunt_id": job.id,
        "kind": job.kind,
        "state": job.state,
        "stage": job.stage,
        "error": job.error,
        "created": job.created,
        "elapsed_s": round((job.finished or now) - (job.started or now), 1),
        "authorisation": {"case_ref": job.case_ref, "officer": job.officer,
                          "authorised_by": job.authorised_by,
                          "entry_hash": job.audit_start,
                          "completion_hash": job.audit_done},
        "target": {"image": f"/api/officer/hunt/{job.id}/image/target.jpg",
                   "plate": job.plate, "vehicle_class": job.vehicle_class,
                   "designation": job.designation},
        "bound": {"frames_per_camera": job.max_frames,
                  "note": (f"Each camera is searched over {job.max_frames} "
                           f"frames of its recorded window. 'Not found' means "
                           f"not found in that window.")},
        "cameras": cams,
        "sightings": job.sightings,
        "candidates": job.candidates,
        "legs": job.legs,
        "summary": job.summary,
    }


# ------------------------------------------------------------- endpoints


class HuntRequest(BaseModel):
    """
    The designation and the authorisation, in one object.

    The picture arrives base64-encoded in the body rather than as a file
    upload. A multipart parser is a dependency, and a submission that fails
    to start because a transitive package is missing on the evening of the
    demonstration is a self-inflicted wound.
    """
    case_ref: Optional[str] = None
    officer: Optional[str] = None
    authorised_by: Optional[str] = None
    cameras: list = Field(default_factory=list)
    plate: Optional[str] = None
    vehicle_class: Optional[str] = None
    max_frames: int = DEFAULT_FRAMES
    image_b64: Optional[str] = None
    from_hunt: Optional[str] = None
    candidate: Optional[str] = None
    preset: bool = False


class DesignateRequest(BaseModel):
    camera: str
    case_ref: Optional[str] = None
    officer: Optional[str] = None
    authorised_by: Optional[str] = None
    max_frames: int = DEFAULT_FRAMES


def _decode_image(b64: str) -> bytes:
    raw = b64.split(",", 1)[-1].strip()
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:                                  # noqa: BLE001
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_image",
            "message": "That designation could not be decoded as an image."})
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_image",
            "message": "That file is not an image this system can read."})
    return data


def _resolve_target(req: HuntRequest) -> bytes:
    """Where the designated picture comes from, in order of precedence."""
    if req.image_b64:
        return _decode_image(req.image_b64)
    if req.from_hunt and req.candidate:
        if not _SAFE_CANDIDATE.match(req.candidate):
            raise HTTPException(status_code=400, detail={
                "error": "bad_candidate",
                "message": "That is not a candidate identifier."})
        src = _job_or_404(req.from_hunt).dir / f"cand_{req.candidate}.jpg"
        if not src.exists():
            raise HTTPException(status_code=404, detail={
                "error": "no_such_candidate",
                "message": "That vehicle is no longer on this server."})
        return src.read_bytes()
    if req.preset:
        for p in PRESET_TARGETS:
            if p.exists():
                return p.read_bytes()
    raise HTTPException(status_code=422, detail={
        "error": "designation_required",
        "message": (
            "A hunt identifies a vehicle by appearance, so it needs a picture "
            "of the vehicle. On this grid the median vehicle yields 27px of "
            "number plate against the ~120px a read needs, which is why "
            "identity is carried by appearance and the plate corroborates it "
            "rather than the other way round."),
        "remedy": ("Designate the vehicle from a photograph, or pick it out "
                   "of a camera that saw it. A registration on its own is "
                   "searched against sightings already recorded — use "
                   "Search for that."),
    })


def _validate_cameras(ids: list) -> list:
    cat = footage_catalogue()
    chosen = [c for c in dict.fromkeys(ids) if c]
    unknown = [c for c in chosen if c not in cat]
    if not chosen or unknown:
        raise HTTPException(status_code=400, detail={
            "error": "no_footage",
            "message": ("A hunt runs against footage held on this machine. "
                        "Pick from the cameras that have a recorded window."),
            "unavailable": unknown,
            "available": sorted(cat),
        })
    return chosen


def _new_job(kind: str, req, cameras: list, frames: int) -> _Job:
    job = _Job(id=uuid.uuid4().hex[:12], kind=kind,
               case_ref=(req.case_ref or "").strip(),
               officer=(req.officer or "").strip(),
               authorised_by=(req.authorised_by or "").strip(),
               cameras=cameras,
               max_frames=max(30, min(FRAMES_CEILING, int(frames))),
               dir=HUNT_RUN / uuid.uuid4().hex[:12],
               created=_now())
    job.dir.mkdir(parents=True, exist_ok=True)
    cat = footage_catalogue()
    job.cams = {c: {"state": "queued", "frames_done": 0,
                    "frames_total": min(job.max_frames, cat[c]["frames"]),
                    "seconds_of_footage": round(
                        min(job.max_frames, cat[c]["frames"]) /
                        (cat[c]["fps"] or 25.0), 1),
                    "window": cat[c]["window"], "sightings": 0}
                for c in cameras}
    return job


@router.post("/api/officer/hunt")
def officer_hunt(req: HuntRequest):
    """
    Hunt a designated vehicle across a chosen set of cameras.

    Returns as soon as the work is queued and the query is on the audit
    chain. The console polls; nothing here blocks on video.
    """
    _require_authorisation(req.officer, req.case_ref, req.authorised_by)
    cameras = _validate_cameras(req.cameras)
    picture = _resolve_target(req)

    job = _new_job("hunt", req, cameras, req.max_frames)
    job.plate = (req.plate or "").strip().upper() or None
    job.vehicle_class = (req.vehicle_class or "").strip() or None
    job.designation = ("uploaded photograph" if req.image_b64 else
                       f"picked out of {req.from_hunt}" if req.from_hunt else
                       "the vehicle designated in the proven run")
    (job.dir / "target.jpg").write_bytes(picture)

    # Logged before a single frame is opened. A hunt reaches across
    # departments and reconstructs a citizen's movements; the entry is the
    # gate's receipt, and it is written whether or not anything is found.
    with connect() as con:
        job.audit_start = _audit(
            con, job.officer, "officer_hunt_vehicle", case_ref=job.case_ref,
            authorised_by=job.authorised_by,
            query_params={"hunt_id": job.id, "cameras": cameras,
                          "plate": job.plate, "designation": job.designation,
                          "frames_per_camera": job.max_frames})

    _remember(job)
    _ensure_worker()
    _JOB_QUEUE.put(job)
    return _snapshot(job)


@router.post("/api/officer/hunt/designate")
def officer_designate(req: DesignateRequest):
    """
    Read one camera and offer the vehicles it saw, so the officer can point
    at one. Case-linked: this opens footage and looks at vehicles, which is
    the same intrusion as any other read of the grid.
    """
    _require_authorisation(req.officer, req.case_ref, req.authorised_by)
    cameras = _validate_cameras([req.camera])
    job = _new_job("designate", req, cameras, req.max_frames)
    job.designation = f"vehicles seen on {req.camera}"

    with connect() as con:
        job.audit_start = _audit(
            con, job.officer, "officer_designate_vehicle",
            case_ref=job.case_ref, authorised_by=job.authorised_by,
            query_params={"hunt_id": job.id, "camera": req.camera,
                          "frames": job.max_frames})

    _remember(job)
    _ensure_worker()
    _JOB_QUEUE.put(job)
    return _snapshot(job)


@router.get("/api/officer/hunt/{hunt_id}")
def officer_hunt_status(hunt_id: str):
    """Everything found so far, and how far through the footage that is."""
    return _snapshot(_job_or_404(hunt_id))


@router.get("/api/officer/hunt/{hunt_id}/image/{name}", include_in_schema=False)
def officer_hunt_image(hunt_id: str, name: str):
    job = _job_or_404(hunt_id)
    # The name is built by this module, but it arrives from a URL, and a
    # path that leaves the job directory is a file-disclosure bug however it
    # got there.
    if not _SAFE_IMAGE.match(name):
        raise HTTPException(status_code=400, detail="Not an image of this hunt.")
    path = job.dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="No such frame.")
    return FileResponse(path, media_type="image/jpeg")



# ----------------------------------------------------------------- page


@router.get("/officer", include_in_schema=False)
def officer_console():
    return FileResponse(WEB / "officer.html")
