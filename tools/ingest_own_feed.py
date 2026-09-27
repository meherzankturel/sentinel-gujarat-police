#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
The own-feed demonstration, built from footage rather than described.

Three clips of one black Hyundai Alcazar, registration GJ19BE3669, shot by
the entrant at three points on a Surat street on 25 September 2026. The
vehicle is on the watchlist as stolen under FIR-0231/2026. This tool puts
the three recordings into the registry as cameras, runs the real pipeline
over them, and writes whatever the pipeline actually found.

The competition excludes mock-ups and simulations, so every row this writes
has to trace back to a detection. Three rules follow from that, and they
are the whole reason this file is not simply tools/seed_demo_sightings.py
pointed at different coordinates:

  1. Nothing is written for a clip the pipeline found nothing on. A blank
     leg in the route is the honest output, and it is reported as one.
  2. The plate stored is the plate the OCR returned, not the plate on the
     FIR. Writing the clean string would launder a degraded read into an
     exact match, and the officer's console would then show certainty the
     camera never had.
  3. No sighting is attributed to a vehicle whose plate was not read. On
     the government grid a presence-only camera can corroborate a track
     that plate-capable cameras already established. Here there is no such
     track to corroborate, so attributing one of the hundreds of other
     cars on this street to GJ19BE3669 would be inventing evidence.

Capability is measured, not assigned. The same vehicle-scale method
tools/survey_grid.py runs against the government cameras runs here, so the
own feed is graded on the government grid's yardstick and its class means
the same thing. The tier a read is then allowed to claim comes out of
sentinel.matching against that measured class -- exactly as in
tools/seed_demo_sightings.py -- so it is the system's judgement, not a
label typed into this file.

    ./.venv/bin/python tools/ingest_own_feed.py           # uses the cache
    ./.venv/bin/python tools/ingest_own_feed.py --fresh   # re-reads video
"""
import json
import re
import statistics
import subprocess
from datetime import datetime, timedelta, timezone

import cv2

from sentinel.capability import CameraMotion, PLATE_TO_VEHICLE
from sentinel.detect import Detector
from sentinel.matching import match_plate
from sentinel.officer import _sightings_for
from sentinel.pipeline import CameraWorker
from sentinel.registry import (ANALYTIC_FOR, audit, connect, init,
                               normalise_plate, record_capability,
                               record_time_confidence, upsert_camera)

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEDIA = ROOT / "media" / "own"
CACHE = ROOT / "audit" / "own_feed.json"

# Where the evidence stills are written. Two per read: the vehicle as the
# camera saw it, and the plate crop the OCR actually worked from. Both are
# cut from the frame the read came from rather than from arrival, because
# an officer looking at a row is asking "show me the vehicle" and arrival
# is often the moment it first appeared at the edge of the scene.
#
# Crops, not frames. A crop of the vehicle that was read contains that
# vehicle and nothing else, so nobody uninvolved is published by accident
# -- which a full frame from this footage would do, since one of these
# recordings has a member of the public's car parked in it with the
# registration plainly legible.
EVIDENCE = ROOT / "web" / "assets" / "evidence"
VEHICLE_MAX_W = 460
PLATE_H = 72

PLATE = "GJ19BE3669"
CASE = "FIR-0231/2026"

# Same thresholds as tools/survey_grid.py, deliberately not re-tuned for
# our own footage. A yardstick that moves when you point it at your own
# work measures nothing.
PLATE_READ_PX, PLATE_MARGINAL_PX = 120, 80
SURVEY_FRAMES = 40

CLIPS = [
    ("own-A", "IMG_2044.MOV"),
    ("own-B", "IMG_2045.MOV"),
    ("own-C", "IMG_2046.MOV"),
]

# The clips carry their own position and start time in the QuickTime
# container, written by the recording device. That is a real measurement
# with a real error bar -- consumer GNSS, tens of metres -- and it is
# better evidence than a coordinate typed in from memory, so it is used in
# preference to one. It is still not a survey, and the location name says
# so on every screen it reaches.
GPS_NOTE = "position from the recording's own GPS, not surveyed"
DEPARTMENT = "Entrant"

# The recorder's clock is a phone's, kept by the network. We have not
# measured it against an external reference the way tools/measure_clocks.py
# measures a camera's burned-in clock, and saying "0 ms drift" would be
# claiming a measurement we did not make. Left unmeasured, which
# sentinel.matching reads as "no grounds to demote" rather than as "good".
CLOCK_NOTE = ("unmeasured: recorder clock (mobile device, network time); "
              "container creation_time taken at face value, never checked "
              "against an external reference")


# ------------------------------------------------------------- container


def ffprobe(path: pathlib.Path) -> dict:
    """Start time, position, codec and declared rate, from the container."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    tags = {**meta["format"].get("tags", {}), **video.get("tags", {})}

    created = tags.get("creation_time")
    if not created:
        raise SystemExit(f"{path.name}: no creation_time in the container. "
                         "Every sighting needs a real time; refusing to invent one.")
    iso = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    if not iso:
        raise SystemExit(f"{path.name}: no GPS in the container. "
                         "Refusing to place a camera at a guessed coordinate.")
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", iso)
    if not m:
        raise SystemExit(f"{path.name}: cannot parse ISO6709 {iso!r}")

    num, den = (video.get("r_frame_rate") or "0/1").split("/")
    return {
        "started_utc": created.replace(".000000Z", "Z"),
        "lat": round(float(m.group(1)), 6),
        "lon": round(float(m.group(2)), 6),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "codec": video.get("codec_name"),
        "declared_fps": round(float(num) / float(den), 2) if float(den) else None,
        "duration_s": round(float(meta["format"]["duration"]), 2),
    }


# ------------------------------------------------------------ capability


def survey(path: pathlib.Path) -> dict:
    """
    What this camera can deliver, by tools/survey_grid.py's method.

    Vehicle widths from the detector, scaled to plate width, and the
    headline number is the share of passing vehicles whose plate clears the
    legibility threshold. One deviation from survey_grid, stated because it
    changes the sample: it samples 40 frames from a 20-second live window,
    where these are whole recordings, so the same 40 samples are spread
    across the entire clip instead of the first four seconds of it.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // SURVEY_FRAMES)

    widths, people, cammot, n, used, bright = [], 0, CameraMotion(), 0, 0, []
    while used < SURVEY_FRAMES:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panning = cammot.update(gray)
        if n % step:
            continue
        used += 1
        bright.append(float(gray.mean()))
        if panning:
            continue
        for d in Detector.detect(img, ("car", "truck", "bus", "motorcycle",
                                       "person"), 0.5):
            if d.label == "person":
                people += 1
            else:
                widths.append(d.width)
    cap.release()

    row = {"frames_analysed": used, "sample_step": step,
           "vehicles": len(widths), "people": people,
           "pan_share": round(cammot.pan_share(), 3), "is_ptz": cammot.is_ptz(),
           "brightness": round(statistics.fmean(bright), 1) if bright else None}
    if widths:
        ws = sorted(widths)
        plate = [w * PLATE_TO_VEHICLE for w in ws]
        row["veh_w_p90"] = ws[int(len(ws) * 0.9)]
        row["plate_p50"] = round(plate[len(plate) // 2], 1)
        row["plate_p90"] = round(plate[int(len(plate) * 0.9)], 1)
        row["plate_max"] = round(plate[-1], 1)
        row["anpr_yield"] = round(
            sum(1 for p in plate if p >= PLATE_READ_PX) / len(plate), 4)
        row["marginal_yield"] = round(
            sum(1 for p in plate if p >= PLATE_MARGINAL_PX) / len(plate), 4)
        y = row["anpr_yield"]
        row["capability"] = ("plate-capable" if y >= 0.15 else
                             "plate-occasional" if y >= 0.02 else
                             "presence-only")
    else:
        row["capability"] = "no-vehicles-observed"
    return row


# -------------------------------------------------------------- pipeline


def frames_with_pts(path: pathlib.Path, pts_ms: list):
    """
    Frames, each with the container timestamp of that frame recorded.

    Timing is taken from the decoder's presentation timestamp and never
    from how long the loop took to get here, which is the same rule the
    live clients follow. pts_ms[i] is the timestamp of the frame
    CameraWorker counts as frame i+1.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, img = cap.read()
            if not ok or img is None:
                return
            pts_ms.append(float(pos))
            yield img
    finally:
        cap.release()


def write_evidence(path: pathlib.Path, camera_id: str, wanted: dict,
                   pts: list) -> dict:
    """
    Cut the vehicle and its plate out of the frames the reads came from.

    The file is walked rather than seeked. Frame-accurate seeking is not
    reliable across containers, and a still that is one frame out is a
    still of a different moment -- which is exactly the sort of quiet
    inaccuracy this project keeps insisting it will not ship.
    """
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    stems = {}
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while wanted:
            ok, img = cap.read()
            if not ok or img is None:
                break
            n += 1
            s = wanted.pop(n, None)
            if s is None:
                continue

            at_ms = pts[n - 1] if n - 1 < len(pts) else 0.0
            stem = f"{camera_id.lower()}-{int(round(at_ms))}"

            h, w = img.shape[:2]
            x1, y1, x2, y2 = s.best_look_box
            px, py = int((x2 - x1) * 0.06), int((y2 - y1) * 0.08)
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(w, x2 + px), min(h, y2 + py)
            car = img[y1:y2, x1:x2]
            if car.size:
                if car.shape[1] > VEHICLE_MAX_W:
                    scale = VEHICLE_MAX_W / car.shape[1]
                    car = cv2.resize(car, (VEHICLE_MAX_W,
                                           int(car.shape[0] * scale)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(EVIDENCE / f"{stem}-vehicle.jpg"), car,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])

            crop = s.best_look_crop
            if crop is not None and crop.size:
                scale = PLATE_H / crop.shape[0]
                plate = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                          PLATE_H),
                                   interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(EVIDENCE / f"{stem}-plate.jpg"), plate,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

            stems[s.track_id] = stem
    finally:
        cap.release()
    return stems


def run_pipeline(camera_id: str, path: pathlib.Path, capability: str) -> dict:
    """
    Watch the clip exactly as a camera worker watches a live feed.

    Whether plate reading is attempted at all is the registry's decision,
    not this tool's: the measured class selects an analytic, and only the
    plate-reading analytics open the OCR. On a camera measured too coarse
    to read a plate an attempted read does not fail quietly, it returns a
    confident wrong registration.
    """
    analytic = ANALYTIC_FOR.get(capability, "probe")
    attempt = analytic in ("anpr", "anpr_low")

    pts: list = []
    worker = CameraWorker(camera_id, capability=capability,
                          attempt_plates=attempt,
                          log=lambda m: print("    " + m, flush=True))
    found = worker.process(frames_with_pts(path, pts))

    measured_fps = None
    if len(pts) > 1 and pts[-1] > pts[0]:
        measured_fps = round((len(pts) - 1) / ((pts[-1] - pts[0]) / 1000.0), 2)

    # Cut the stills before the reads are reduced to JSON: the crop the OCR
    # read is an array on the sighting and does not survive that reduction.
    stems = write_evidence(
        path, camera_id,
        {s.best_look_frame: s for s in found
         if s.plate_text and s.best_look_frame and s.best_look_box},
        pts)

    reads = []
    for s in found:
        if not s.plate_text:
            continue
        # The track's first frame: the moment this camera first had the
        # vehicle. The read itself is assembled from the best evidence the
        # whole track offered, so there is no single frame to point at, and
        # arrival is the instant a route reconstruction actually wants.
        i = max(0, min(s.first_frame - 1, len(pts) - 1))
        reads.append({
            "track_id": s.track_id, "vehicle_type": s.vehicle_type,
            "frames": s.frames, "first_frame": s.first_frame,
            "pts_ms": round(pts[i], 1),
            "plate_text": s.plate_text, "plate_confidence": s.plate_confidence,
            "plate_px": s.plate_px, "plate_tier": s.plate_tier,
            "read_method": s.read_method,
            "best_vehicle_px": s.best_vehicle_px,
            # The frame the read came from, and the stills cut from it.
            # Arrival above answers "when"; this answers "show me".
            "evidence_frame": s.best_look_frame,
            "evidence_pts_ms": (round(pts[s.best_look_frame - 1], 1)
                                if s.best_look_frame
                                and s.best_look_frame - 1 < len(pts) else None),
            "evidence_ref": stems.get(s.track_id),
        })
    reads.sort(key=lambda r: r["pts_ms"])
    return {"analytic": analytic, "plates_attempted": attempt,
            "frames_processed": len(pts), "tracks": len(found),
            "measured_fps": measured_fps, "reads": reads}


# One vehicle crossing one camera is one event to an officer. The tracker
# bridges 12 missing frames, which at 60 fps is a fifth of a second, so a
# car briefly hidden behind a bus comes back as a second track -- and on
# own-B it did exactly that, producing two reads of GJ19BE3669 367 ms
# apart. Writing both would put the same pass on the timeline twice and
# raise two alerts for it, which is the alert-fatigue failure this system
# is built to argue against.
#
# Five seconds: an order of magnitude more than the tracker's own dropout
# tolerance, and far less than the time a vehicle would need to leave the
# scene, turn and come back. Nothing is discarded silently -- the merged
# count is printed, because the fragmentation is a real property of the
# tracker on this footage and hiding it would be the wrong kind of tidy.
SAME_PASS_MS = 5000


def one_per_pass(reads: list) -> list:
    """Collapse track fragments of one vehicle pass into a single read."""
    out: list = []
    for r in reads:
        key = normalise_plate(r["plate_text"])
        prev = next((o for o in reversed(out)
                     if normalise_plate(o["plate_text"]) == key
                     and r["pts_ms"] - o["pts_ms"] <= SAME_PASS_MS), None)
        if prev is None:
            out.append({**r, "merged_tracks": 1})
            continue
        # Keep the better-evidenced read of the pass, but keep the earliest
        # time: that is when the camera first had the vehicle.
        prev["merged_tracks"] += 1
        if r["plate_confidence"] > prev["plate_confidence"]:
            at = prev["pts_ms"]
            prev.update({**r, "pts_ms": at,
                         "merged_tracks": prev["merged_tracks"]})
    return out


# ------------------------------------------------------------------ main


def gather(fresh: bool) -> list:
    """
    The measurements. The cache holds what the pipeline found, unedited;
    the pass-collapsing above is policy applied on the way to the database,
    so re-running it can never quietly rewrite the evidence.
    """
    rows = _gather(fresh)
    for r in rows:
        r["pipeline"]["reads"] = one_per_pass(r["pipeline"]["reads"])
    return rows


def _gather(fresh: bool) -> list:
    if CACHE.exists() and not fresh:
        cached = json.loads(CACHE.read_text())
        # The cache is written after each clip so a long run is not lost,
        # which means an interrupted run leaves a real file covering only
        # part of the footage. This tool then deletes every own-feed
        # sighting and rewrites from what it has -- so accepting a partial
        # cache would quietly drop the journey from the database while
        # printing "using cached measurements". Covered, or re-read.
        have = {r["id"] for r in cached}
        want = {camera_id for camera_id, _ in CLIPS}
        if have >= want:
            print(f"using cached measurements from {CACHE.relative_to(ROOT)} "
                  f"(--fresh to re-read the video)\n")
            return cached
        print(f"cached measurements cover only {', '.join(sorted(have))} of "
              f"{', '.join(sorted(want))} — re-reading the video rather than "
              f"writing a partial journey\n", flush=True)

    Detector._load()
    print(f"detector on {Detector.device()}\n", flush=True)
    rows = []
    for camera_id, filename in CLIPS:
        path = MEDIA / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        print(f"{camera_id}  {filename}", flush=True)
        meta = ffprobe(path)
        cap = survey(path)
        print(f"    measured {cap['capability']}: plate p90 "
              f"{cap.get('plate_p90')}px, ANPR yield "
              f"{(cap.get('anpr_yield') or 0) * 100:.1f}% of "
              f"{cap['vehicles']} vehicles", flush=True)
        run = run_pipeline(camera_id, path, cap["capability"])
        rows.append({"id": camera_id, "file": filename,
                     **meta, "survey": cap, "pipeline": run})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(rows, indent=2))
        print(flush=True)
    return rows


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def main(fresh: bool = False):
    rows = gather(fresh)
    init()

    with connect() as con:
        listed = con.execute(
            "SELECT id, plate_number FROM watchlist_entry WHERE "
            "plate_normalised=?", (normalise_plate(PLATE),)).fetchone()
        if not listed:
            raise SystemExit(f"{PLATE} is not on the watchlist; nothing to "
                             f"match against. Add it under {CASE} first.")

        # Re-runnable. This tool owns every row whose evidence is one of
        # these files, and replaces them wholesale rather than stacking a
        # second copy of the journey on the first.
        con.execute("DELETE FROM alert WHERE sighting_id IN (SELECT id FROM "
                    "sighting WHERE frame_ref LIKE 'media/own/%')")
        con.execute("DELETE FROM sighting WHERE frame_ref LIKE 'media/own/%'")

        written, hero, alerts = [], 0, 0
        for r in rows:
            cap, run = r["survey"], r["pipeline"]
            upsert_camera(con, {
                "id": r["id"], "catalogue_id": r["file"],
                "department": DEPARTMENT,
                "location_name": f"Own feed {r['id'][-1]}, Surat ({GPS_NOTE})",
                "lat": r["lat"], "lon": r["lon"],
                "width": r["width"], "height": r["height"], "codec": r["codec"],
                "governance_class": "open",
                "last_seen_live": r["started_utc"],
            })
            y = cap.get("anpr_yield")
            record_capability(
                con, r["id"],
                capability_class=cap["capability"],
                plate_px=int(cap["plate_p90"]) if cap.get("plate_p90") else None,
                basis="measured from delivered video (vehicle scale)",
                note=(f"ANPR yield {y * 100:.1f}% of passing vehicles"
                      if y is not None else "not measured"),
                measured_fps=run.get("measured_fps"),
                declared_fps=r.get("declared_fps"),
                brightness=cap.get("brightness"))
            record_time_confidence(con, r["id"], None, CLOCK_NOTE)

            started = datetime.strptime(r["started_utc"], "%Y-%m-%dT%H:%M:%SZ"
                                        ).replace(tzinfo=timezone.utc)
            trust = CLOCK_NOTE.split(":")[0]

            for read in run["reads"]:
                at = started + timedelta(milliseconds=read["pts_ms"])
                m = match_plate(con, read["plate_text"],
                                read["plate_confidence"], read["plate_tier"],
                                cap["capability"], trust)
                if m:
                    tier, conf, wl = m[0].tier, m[0].confidence, m[0].watchlist_id
                else:
                    # Not on any watchlist: other traffic on the same road.
                    # Kept, because a search that returns this vehicle and
                    # not those ones is only meaningful if those ones are
                    # in the table to be excluded.
                    tier, conf, wl = read["plate_tier"], None, None

                con.execute(
                    "INSERT INTO sighting (camera_id, pts_ms, wallclock_utc, "
                    "corrected_utc, entity_type, plate_read, plate_normalised, "
                    "plate_confidence, plate_px, frame_ref, evidence_ref, "
                    "match_id, match_confidence, match_tier) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    # Stored as the camera read it. No clock drift has been
                    # measured on this recorder, so no correction is applied
                    # and the two times are the same instant -- which is the
                    # honest representation of "uncorrected", not a claim
                    # that the clock is right.
                    (r["id"], read["pts_ms"], iso(at), iso(at), "vehicle",
                     read["plate_text"], normalise_plate(read["plate_text"]),
                     read["plate_confidence"], read["plate_px"] or None,
                     f"media/own/{r['file']}#pts={read['pts_ms'] / 1000.0:.2f}s",
                     read.get("evidence_ref"), wl, conf, tier))

                if wl == listed["id"]:
                    hero += 1
                    if tier in ("confirmed", "probable"):
                        sid = con.execute("SELECT last_insert_rowid() AS i"
                                          ).fetchone()["i"]
                        con.execute(
                            "INSERT INTO alert (watchlist_entry_id, sighting_id,"
                            " tier, raised_at) VALUES (?,?,?,?)",
                            (listed["id"], sid, tier, iso(at)))
                        alerts += 1
                written.append((r["id"], read, tier, wl == listed["id"]))

        audit(con, "tools/ingest_own_feed", "ingest_own_feed_demonstration",
              case_ref=CASE, authorised_by="Meherzan Turel (entrant)",
              query_params={"plate": PLATE,
                            "clips": [r["file"] for r in rows],
                            "cameras": [r["id"] for r in rows]},
              result_count=len(written))

        report(con, rows, written, hero, alerts)


def report(con, rows, written, hero, alerts):
    print("=" * 78)
    print("REGISTERED")
    print(f"  {'camera':<8} {'capability':<18} {'plate':>6} {'yield':>7}  "
          f"{'fps decl/meas':<14} location")
    for r in rows:
        c = r["survey"]
        y = c.get("anpr_yield")
        print(f"  {r['id']:<8} {c['capability']:<18} "
              f"{str(int(c['plate_p90']) if c.get('plate_p90') else '-'):>6} "
              f"{(f'{y * 100:.1f}%' if y is not None else '-'):>7}  "
              f"{str(r.get('declared_fps')) + '/' + str(r['pipeline'].get('measured_fps')):<14} "
              f"{r['lat']:.4f},{r['lon']:.4f}")

    print("\nWHAT THE PIPELINE FOUND")
    for r in rows:
        p = r["pipeline"]
        mine = [w for w in written if w[0] == r["id"]]
        print(f"  {r['id']}  {r['file']}  {p['frames_processed']} frames, "
              f"{p['tracks']} vehicle tracks, analytic={p['analytic']}")
        if not p["reads"]:
            print("      no plate read on this clip. Nothing written for it: "
                  "an empty leg is the honest result.")
        for cam, read, tier, is_hero in mine:
            merged = read.get("merged_tracks", 1)
            print(f"      {read['plate_text']:<12} {read['plate_px']:>4}px  "
                  f"conf {read['plate_confidence']:.2f}  read-tier "
                  f"{read['plate_tier']:<13} -> {tier:<13} "
                  f"{PLATE + ' (watchlist)' if is_hero else 'other traffic'}"
                  f"   [{read['read_method']}"
                  f"{f'; {merged} track fragments of one pass' if merged > 1 else ''}]")

    print(f"\n  {len(written)} sighting(s) written, {hero} matched to "
          f"{PLATE}, {alerts} alert(s) raised.")

    target = normalise_plate(PLATE)
    found = _sightings_for(con, target)
    print(f"\nWHAT AN OFFICER SEARCHING {PLATE} NOW SEES")
    if not found:
        print("  nothing. The pipeline did not read this plate on any clip.")
    for s in found:
        flag = ("" if s["exact_read"] else
                "  <- recovered by confusion-aware matching")
        print(f"  {s['at']}  {s['camera_id']}  "
              f"{(s['plate_read'] or 'no plate read'):<12} {s['tier']:<14} "
              f"{s['capability_class']:<18} {s['location_name']}{flag}")
    print("=" * 78)


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
