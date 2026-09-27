#!/usr/bin/env python3
"""
sentinel.api -- service layer over the registry.

Every read that touches footage or sightings is written to the audit log
with the officer and the authorising officer, because in this system
governance is a property of the architecture rather than a policy
document stapled to it.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .registry import (ANALYTIC_FOR, assignments, audit, connect, init,
                       normalise_plate, verify_audit_chain)
from .threat import layer_plan

WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Sentinel", docs_url="/api/docs")
init()


def rows(cur):
    return [dict(r) for r in cur.fetchall()]


# Three sources are federated here, and they must never be totalled into one
# number without saying so. A registry that reports 41 cameras when 30 of
# them are the government's is padding its own headline, which is the exact
# failure this project claims to have designed against.
#
# This inferred from the stream URL and defaulted to "government", which is
# the padding direction. The own-feed cameras are local files and carry no
# URL at all, so all three of Meherzan's clips were being counted as
# government cameras and the console reported 33 where there are 30.
#
# Classification is now positive: a camera must present evidence of which
# grid it belongs to. Anything that presents none is "unclassified" and is
# reported as such, because inventing a home for it is how the count drifted
# in the first place.
# Defined in sentinel.registry, which is what knows the difference.
# Re-exported here because callers have always found it at this name.
from .registry import (GOVERNMENT_HOSTS, LOCAL_HOSTS,  # noqa: E402,F401
                       OWN_FEED_DEPARTMENT, grid_of)


@app.get("/api/cameras")
def cameras():
    """
    The registry. Note what each row carries beyond identity: measured
    plate pixels, the basis for that measurement, clock drift, and the
    analytic the camera has therefore been assigned.
    """
    with connect() as con:
        cams = rows(con.execute("SELECT * FROM camera ORDER BY id"))
        assign = {a["camera_id"]: a for a in assignments(con)}
    for c in cams:
        c["analytic"] = assign.get(c["id"], {}).get("analytic", "probe")
        c["grid"] = grid_of(c)
    by_grid: dict = {}
    for c in cams:
        by_grid[c["grid"]] = by_grid.get(c["grid"], 0) + 1
    return {"cameras": cams, "count": len(cams), "by_grid": by_grid}


@app.get("/api/assignments")
def analytics_assignments():
    """
    Which analytic runs where, and what that saves.

    Running plate reading on every camera is the naive design. Here it is
    scheduled only where a plate is legible; the rest run cheap presence
    detection. The saving is the case for the whole architecture, so it is
    reported rather than asserted.
    """
    with connect() as con:
        a = assignments(con)
    anpr = [x for x in a if x["analytic"].startswith("anpr")]
    return {
        "assignments": a,
        "summary": {
            "cameras": len(a),
            "running_anpr": len(anpr),
            "running_presence": len([x for x in a if x["analytic"] == "presence"]),
            "anpr_share": round(len(anpr) / len(a), 3) if a else 0,
            "note": ("Plate reading is scheduled only where a plate was "
                     "measured to be legible. The remainder run presence "
                     "detection and serve as corroboration."),
        },
    }


@app.get("/api/watchlist")
def watchlist():
    with connect() as con:
        return {"entries": rows(con.execute(
            "SELECT * FROM watchlist_entry ORDER BY priority, id"))}


@app.get("/api/sightings")
def sightings(plate: Optional[str] = None, camera_id: Optional[str] = None,
              limit: int = Query(200, le=2000),
              actor: Optional[str] = None, case_ref: Optional[str] = None,
              authorised_by: Optional[str] = None):
    """
    Movement data is case-linked wherever it is served.

    This endpoint previously defaulted the actor to "system" and treated the
    case reference as optional, which was an unauthorised route to exactly
    the data the officer console gates. A control that one endpoint enforces
    and another quietly bypasses is not a control.
    """
    if plate and not (actor and case_ref and authorised_by):
        raise HTTPException(
            status_code=400,
            detail="A plate search requires actor, case_ref and authorised_by.")
    q = ("SELECT s.*, c.location_name, c.lat, c.lon, c.department, "
         "c.capability_class, c.time_confidence_ms "
         "FROM sighting s JOIN camera c ON c.id = s.camera_id WHERE 1=1")
    p = []
    if plate:
        q += " AND s.plate_normalised = ?"
        p.append(normalise_plate(plate))
    if camera_id:
        q += " AND s.camera_id = ?"
        p.append(camera_id)
    q += " ORDER BY COALESCE(s.corrected_utc, s.wallclock_utc) LIMIT ?"
    p.append(limit)

    with connect() as con:
        out = rows(con.execute(q, p))
        # A search against footage is itself an event worth recording.
        audit(con, actor, "search_sightings", case_ref=case_ref,
              authorised_by=authorised_by,
              query_params={"plate": plate, "camera_id": camera_id},
              result_count=len(out))
    return {"sightings": out, "count": len(out)}


@app.get("/api/alerts")
def alerts():
    with connect() as con:
        return {"alerts": rows(con.execute(
            "SELECT a.*, w.plate_number, w.reason, w.priority, w.case_ref, "
            "s.camera_id, s.plate_read, s.match_confidence, s.corrected_utc "
            "FROM alert a "
            "LEFT JOIN watchlist_entry w ON w.id = a.watchlist_entry_id "
            "LEFT JOIN sighting s ON s.id = a.sighting_id "
            "ORDER BY a.raised_at DESC LIMIT 200"))}


# ===================================================================
# Person safety -- sentinel.threat layer 1 (panic dispersal)
# ===================================================================
#
# Layer 1 runs on the cameras that can do nothing else: 22 of the 30
# government cameras cannot read a plate, and a crowd running away from a
# point is a third of the frame moving at once, which survives 854x480 and
# night and compression. So this is the analytic that makes the weak half
# of the grid worth opening at all, and until now it existed only as a
# command-line tool, which means it existed only as a clip.

# A closed set. An alert whose footage provenance is not one of these is
# refused rather than stored with a blank, because the console's honesty
# depends on this column and a NULL would render as "live" by default.
DISPERSAL_SOURCES = ("government-live", "own-feed", "simulation")

# The measured epicentre error, as a fraction of the frame diagonal. The
# divergence peak lands within roughly this of the true origin on rendered
# crowds, so the API publishes a radius and the console draws a region. A
# pinpoint marker would claim a precision that was measured not to exist.
EPICENTRE_ERROR_FRAC = 0.15

# Carried in the payload rather than written into the HTML, so the surface
# cannot drift away from what was actually measured.
DISPERSAL_VALIDATION = {
    "false_positives": ("366 s of real government footage (cam09, cam05, "
                        "cam01; day and night, traffic and pedestrians, no "
                        "dispersal present): zero alerts. Of 6995 scored "
                        "frames only 9 cleared all three gates at once, "
                        "never twice in a row."),
    "true_positives": ("Measured on rendered crowds only -- 1.8 s to alert. "
                       "There is no real footage of a crowd dispersing on "
                       "this grid and there will not be, so read this as "
                       "'the detector responds to the thing it is aimed "
                       "at', not as a recall estimate."),
    "epicentre": (f"A region, not a point: the marked origin lands within "
                  f"~{int(EPICENTRE_ERROR_FRAC * 100)}% of the frame "
                  f"diagonal of the true one. It tells an operator which "
                  f"part of the frame to look at."),
    "layers_2_and_3": ("Threat posture and weapon-in-hand are implemented "
                       "and gated, but not deployed and not validated: no "
                       "footage of an attack exists on this grid. Their "
                       "scale gates are shown because the refusal is the "
                       "useful output; no alert from either layer is "
                       "served."),
}


def _dispersal_layer1(con) -> List[dict]:
    """
    Which cameras may run layer 1, decided by threat.layer_plan rather than
    by a rule invented here. Two writers with two rules is how the analytic
    vocabulary drifted once already.
    """
    cams = rows(con.execute(
        "SELECT id, department, location_name, lat, lon, capability_class, "
        "plate_px_measured, night_usable, width, height "
        "FROM camera ORDER BY id"))
    counts = {r["camera_id"]: r["n"] for r in rows(con.execute(
        "SELECT camera_id, COUNT(*) AS n FROM dispersal_alert "
        "GROUP BY camera_id"))}
    out = []
    for c in cams:
        # is_ptz is deliberately not passed: no static column records it,
        # and the detector measures pan per frame-pair at runtime and drops
        # the pair while a unit is panning. A column would be a guess; the
        # runtime check is a measurement.
        plan = layer_plan(plate_px=c["plate_px_measured"],
                          night=(c["night_usable"] == 0))
        out.append({
            "camera_id": c["id"],
            "department": c["department"],
            "location_name": c["location_name"],
            "lat": c["lat"], "lon": c["lon"],
            "capability_class": c["capability_class"] or "unknown",
            "plate_px": c["plate_px_measured"],
            "eligible": bool(plan["dispersal"]["enabled"]),
            "why": plan["dispersal"]["why"],
            # Reported so the console can show why the other two layers are
            # refused on this camera. Neither is deployed; see
            # DISPERSAL_VALIDATION.
            "posture_gate": plan["posture"],
            "weapon_gate": plan["weapon"],
            "alerts": counts.get(c["id"], 0),
        })
    return out


def _dispersal_alert_row(r: dict) -> dict:
    """Add the epicentre region, so no caller has to know the error term."""
    w, h = r.get("frame_w") or 0, r.get("frame_h") or 0
    radius = int(round(EPICENTRE_ERROR_FRAC * math.hypot(w, h))) if w and h else None
    r["epicentre_region_px"] = radius
    r["epicentre_basis"] = DISPERSAL_VALIDATION["epicentre"]
    r["simulated"] = r.get("source") == "simulation"
    return r


@app.get("/api/dispersal")
def dispersal():
    """
    The operator surface for layer 1: which cameras it is allowed on, what
    it has raised, and what is actually known about how well it works.
    """
    with connect() as con:
        cams = _dispersal_layer1(con)
        alerts = [_dispersal_alert_row(r) for r in rows(con.execute(
            "SELECT d.*, c.department, c.location_name, c.lat, c.lon, "
            "c.capability_class, c.plate_px_measured "
            "FROM dispersal_alert d JOIN camera c ON c.id = d.camera_id "
            "ORDER BY d.raised_at DESC, d.id DESC LIMIT 200"))]
    eligible = [c for c in cams if c["eligible"]]
    no_plate = [c for c in eligible
                if c["capability_class"] in ("presence-only", "unusable",
                                             "no-vehicles-observed", "unknown")]
    return {
        "layer": 1,
        "analytic": "panic-dispersal",
        "cameras": cams,
        "alerts": alerts,
        "summary": {
            "cameras": len(cams),
            "eligible": len(eligible),
            "eligible_without_plate_capability": len(no_plate),
            "alerts": len(alerts),
            "simulated_alerts": len([a for a in alerts if a["simulated"]]),
            "live_alerts": len([a for a in alerts if not a["simulated"]]),
            "unacknowledged": len([a for a in alerts
                                   if not a.get("acknowledged_by")]),
            "note": ("Dense optical flow, no model and no weights, so it "
                     "runs on the cameras that cannot read a plate or "
                     "resolve a face -- which is most of them."),
        },
        "validation": DISPERSAL_VALIDATION,
    }


class DispersalReport(BaseModel):
    """One raised dispersal, as the detector saw it."""
    camera_id: str = Field(..., min_length=1)
    detector: str = Field(..., min_length=2,
                          description="processing node that raised it")
    source: str = Field(..., description="|".join(DISPERSAL_SOURCES))
    pts_s: float = Field(..., description="PTS seconds, never arrival time")
    epicentre: List[int] = Field(..., min_length=2, max_length=2)
    frame_w: int = Field(..., gt=0)
    frame_h: int = Field(..., gt=0)
    surge_ratio: float
    coherence: float
    net_drift: float
    movers: int = 0
    confidence: float
    scene_note: Optional[str] = None
    frame_ref: Optional[str] = None


@app.post("/api/dispersal/alerts", status_code=201)
def raise_dispersal_alert(rep: DispersalReport):
    """
    Record a dispersal alert.

    Audited, but NOT gated on a case reference and an authorising officer,
    and the difference from /api/officer/search is not an oversight.

    That gate exists because reconstructing one named person's movements
    across departments is an intrusion on that person: there is a subject,
    the search is a deliberate act by an officer, and so it can be attached
    to a case and to whoever authorised it. A dispersal alert has none of
    that shape. Nobody is on a list, nobody has been identified, no stored
    footage is retrieved, and the alert is raised by a detector watching a
    velocity field rather than requested by a person -- there is no subject
    to protect and no one to authorise. Worse, requiring a case reference
    would mean the control room could not be told that a crowd is running
    until a case existed for the thing it was running from, which inverts
    the order of events.

    It is still written to the tamper-evident chain, because this alert can
    put officers on a street. What goes into the chain is the detector that
    raised it, the provenance of the footage, and later the operator who
    acknowledged it -- so an alert that turned out to be nothing can be
    traced back to the thing that raised it, and a control room cannot
    quietly acquire a habit of ignoring them.
    """
    if rep.source not in DISPERSAL_SOURCES:
        raise HTTPException(status_code=400, detail={
            "error": "unknown_source",
            "allowed": list(DISPERSAL_SOURCES),
            "message": ("Footage provenance is mandatory. Layer 1's "
                        "true-positive rate is measured on simulation only, "
                        "so an alert that cannot say where its pixels came "
                        "from cannot be shown to an operator."),
        })
    if rep.source == "simulation" and not (rep.scene_note or "").strip():
        # Enforced at the boundary rather than in the template. A rendered
        # crowd presented as a real detection is the one failure that would
        # matter more than the feature.
        raise HTTPException(status_code=400, detail={
            "error": "scene_note_required",
            "message": ("A simulated alert must carry the note that says "
                        "what was rendered, because the console labels it "
                        "from this field."),
        })

    with connect() as con:
        cam = con.execute("SELECT id, plate_px_measured, night_usable "
                          "FROM camera WHERE id=?", (rep.camera_id,)).fetchone()
        if cam is None:
            raise HTTPException(status_code=404,
                                detail=f"no camera {rep.camera_id} in the registry")
        # The registry is the control plane, so it also refuses alerts from
        # a camera it never assigned the analytic to. Layer 1 permits every
        # camera today; the check is here so that stops being an assumption
        # the moment layer_plan changes.
        plan = layer_plan(plate_px=cam["plate_px_measured"],
                          night=(cam["night_usable"] == 0))
        if not plan["dispersal"]["enabled"]:
            raise HTTPException(status_code=409, detail={
                "error": "layer_not_assigned",
                "camera_id": rep.camera_id,
                "why": plan["dispersal"]["why"],
            })

        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = con.execute(
            "INSERT INTO dispersal_alert (camera_id, pts_s, raised_at, source, "
            "detector, scene_note, epicentre_x, epicentre_y, frame_w, frame_h, "
            "surge_ratio, coherence, net_drift, movers, confidence, frame_ref) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rep.camera_id, rep.pts_s, at, rep.source, rep.detector,
             rep.scene_note, rep.epicentre[0], rep.epicentre[1],
             rep.frame_w, rep.frame_h, rep.surge_ratio, rep.coherence,
             rep.net_drift, rep.movers, rep.confidence, rep.frame_ref))
        alert_id = int(cur.lastrowid)
        audit(con, rep.detector, "dispersal_alert",
              query_params={"camera_id": rep.camera_id, "source": rep.source,
                            "pts_s": rep.pts_s, "confidence": rep.confidence,
                            "alert_id": alert_id},
              result_count=1)
        row = dict(con.execute(
            "SELECT * FROM dispersal_alert WHERE id=?", (alert_id,)).fetchone())
    return {"alert": _dispersal_alert_row(row)}


@app.post("/api/dispersal/alerts/{alert_id}/acknowledge")
def acknowledge_dispersal_alert(alert_id: int, officer: str = Query(..., min_length=2)):
    """
    An operator has looked at the camera. Named, because the alert budget is
    only meaningful if someone is on the hook for each one -- and because
    "nobody looked" and "someone looked and it was nothing" are different
    facts about a control room.
    """
    with connect() as con:
        row = con.execute("SELECT * FROM dispersal_alert WHERE id=?",
                          (alert_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"no alert {alert_id}")
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        con.execute("UPDATE dispersal_alert SET acknowledged_by=?, "
                    "acknowledged_at=? WHERE id=?", (officer, at, alert_id))
        audit(con, officer, "acknowledge_dispersal",
              query_params={"alert_id": alert_id, "camera_id": row["camera_id"]},
              result_count=1)
        out = dict(con.execute("SELECT * FROM dispersal_alert WHERE id=?",
                               (alert_id,)).fetchone())
    return {"alert": _dispersal_alert_row(out)}


@app.get("/api/audit")
def audit_log():
    with connect() as con:
        entries = rows(con.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"))
        intact, broken = verify_audit_chain(con)
    return {"entries": entries, "chain_intact": intact, "first_broken_id": broken}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")

from .officer import router as officer_router  # noqa: E402
app.include_router(officer_router)
