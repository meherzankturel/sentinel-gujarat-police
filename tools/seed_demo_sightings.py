#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
A worked example for the officer's console: one stolen vehicle down the
Gandhinagar-Ahmedabad corridor.

The sightings are synthetic. Their confidence tiers are not -- every tier
below is produced by running the read through sentinel.matching against the
real registry row for that camera, so what the officer sees is the system's
own judgement about cameras we actually measured, not a label typed in
here. That matters: if the registry is later re-measured and a camera loses
its plate-capable class, this demo changes with it.

The route is built to contain three things the panel should see:

  1. The same vehicle read differently by different cameras -- cam-01
     returns GJ0IAB1Z34 and is recovered by confusion-aware matching,
     demoted to probable rather than presented as a clean read.
  2. A presence-only camera contributing evidence without claiming a plate.
  3. A final leg that cannot have been driven, so the cloned-plate flag has
     something to fire on.

    ./.venv/bin/python tools/seed_demo_sightings.py
"""
from datetime import datetime, timedelta, timezone

from sentinel.matching import match_plate
from sentinel.officer import _sightings_for
from sentinel.registry import connect, init, normalise_plate, audit

PLATE = "GJ01AB1234"
CASE = "FIR-0117/2026"

# 09:11 IST on a weekday morning. Written in UTC because every stored time
# in this system is UTC; the console converts for display.
T0 = datetime(2026, 9, 9, 3, 41, 0, tzinfo=timezone.utc)

# camera, seconds after T0, read as the OCR returned it, read confidence,
# the tier that read quality alone would support, measured plate pixels
JOURNEY = [
    ("cam-02",    0, "GJ01AB1234", 0.94, "confirmed",     118),
    ("cam-01",  165, "GJ0IAB1Z34", 0.78, "confirmed",      96),
    ("cam-05",  505, "GJ01AB1234", 0.81, "confirmed",      93),
    ("cam-06",  975, "GJ01AB1234", 0.44, "corroborating",  74),
    ("cam-07", 1235, None,         0.00, "corroborating", None),
    ("cam-08", 1290, "GJ01AB1234", 0.83, "confirmed",     108),
]

# Other traffic, so that a search visibly excludes rather than returning
# everything the cameras ever saw.
BACKGROUND = [
    ("cam-03",   90, "GJ18CD7788", 0.71, "corroborating"),
    ("cam-04",  240, "GJ27EF4501", 0.66, "corroborating"),
    ("cam-06",  700, "GJ18CD7788", 0.88, "confirmed"),
    ("cam-02", 1100, "MH12XY0099", 0.90, "confirmed"),
]


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    init()
    with connect() as con:
        cams = {r["id"]: dict(r) for r in con.execute("SELECT * FROM camera")}
        missing = [c for c, *_ in JOURNEY if c not in cams]
        if missing:
            raise SystemExit(f"registry has no {missing}; run tools/build_registry.py first")

        # Re-runnable: the demo owns these rows and replaces them wholesale
        # rather than accumulating a second journey on top of the first.
        con.execute("DELETE FROM alert WHERE sighting_id IN "
                    "(SELECT id FROM sighting WHERE frame_ref LIKE 'demo/%')")
        con.execute("DELETE FROM sighting WHERE frame_ref LIKE 'demo/%'")
        con.execute("DELETE FROM watchlist_entry WHERE plate_normalised IN (?,?)",
                    (normalise_plate(PLATE), normalise_plate("GJ18CD7788")))

        con.execute(
            "INSERT INTO watchlist_entry (entity_type, plate_number, "
            "plate_normalised, make, model, colour, reason, priority, "
            "case_ref, valid_from) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("vehicle", PLATE, normalise_plate(PLATE), "Maruti Suzuki",
             "Swift VDi", "White", "stolen", 1, CASE, "2026-09-08"))
        con.execute(
            "INSERT INTO watchlist_entry (entity_type, plate_number, "
            "plate_normalised, make, model, colour, reason, priority, "
            "case_ref, valid_from) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("vehicle", "GJ18CD7788", normalise_plate("GJ18CD7788"), "Tata",
             "Nexon", "Grey", "blacklisted", 3, "SB-0042/2026", "2026-07-01"))

        norm = normalise_plate(PLATE)
        listed = con.execute("SELECT id FROM watchlist_entry WHERE "
                             "plate_normalised=?", (norm,)).fetchone()["id"]

        for cam_id, offset, read, conf, read_tier, plate_px in JOURNEY:
            cam = cams[cam_id]
            true_at = T0 + timedelta(seconds=offset)
            # The camera stamps its own clock on the frame. Drift is what we
            # measured that clock to be out by, so the wallclock we would
            # have recorded is the corrected time plus the drift -- and the
            # console shows both, because a silently corrected timeline is
            # the failure mode this whole registry exists to prevent.
            drift_s = (cam["time_confidence_ms"] or 0) / 1000.0
            wall = true_at + timedelta(seconds=drift_s)
            trust = (cam["time_confidence_note"] or "unknown:").split(":")[0]

            if read:
                m = match_plate(con, read, conf, read_tier,
                                cam["capability_class"], trust)
                m = [x for x in m if x.watchlist_id == listed]
                if not m:
                    raise SystemExit(f"{cam_id}: read {read} did not match the "
                                     f"watchlist -- the demo would be lying")
                tier, match_conf = m[0].tier, m[0].confidence
            else:
                # A presence-only camera is not allowed to claim it read a
                # plate. It contributes a vehicle of the right description at
                # a time and place consistent with the track; the plate is an
                # attribution to that track, which is exactly what the
                # corroborating tier means.
                tier, match_conf = "corroborating", 0.35

            con.execute(
                "INSERT INTO sighting (camera_id, pts_ms, wallclock_utc, "
                "corrected_utc, entity_type, plate_read, plate_normalised, "
                "plate_confidence, plate_px, bbox, frame_ref, match_id, "
                "match_confidence, match_tier) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                # Stored normalised as the camera read it, not as the
                # watchlist spells it. Writing the clean plate here would
                # quietly launder cam-01's degraded read into an exact
                # match, which is the one thing this console must never do.
                (cam_id, float(offset * 1000), iso(wall), iso(true_at), "vehicle",
                 read, normalise_plate(read) if read else normalise_plate(PLATE),
                 conf if read else None, plate_px,
                 '{"x":612,"y":388,"w":%d,"h":%d}' % (plate_px or 0, int((plate_px or 0) / 4)),
                 f"demo/{cam_id}/{iso(true_at)}.jpg", listed,
                 match_conf, tier))

            if tier in ("confirmed", "probable"):
                sid = con.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
                con.execute("INSERT INTO alert (watchlist_entry_id, sighting_id, "
                            "tier, raised_at) VALUES (?,?,?,?)",
                            (listed, sid, tier, iso(true_at)))

        for cam_id, offset, read, conf, read_tier in BACKGROUND:
            cam = cams[cam_id]
            true_at = T0 + timedelta(seconds=offset)
            wall = true_at + timedelta(seconds=(cam["time_confidence_ms"] or 0) / 1000.0)
            con.execute(
                "INSERT INTO sighting (camera_id, pts_ms, wallclock_utc, "
                "corrected_utc, entity_type, plate_read, plate_normalised, "
                "plate_confidence, plate_px, frame_ref, match_tier) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cam_id, float(offset * 1000), iso(wall), iso(true_at), "vehicle",
                 read, normalise_plate(read), conf, cam["plate_px_measured"],
                 f"demo/{cam_id}/{iso(true_at)}-bg.jpg", read_tier))

        audit(con, "tools/seed_demo_sightings", "seed_demonstration_data",
              case_ref=CASE, authorised_by="Meherzan Turel (entrant)",
              query_params={"plate": PLATE, "cameras": [c for c, *_ in JOURNEY]},
              result_count=len(JOURNEY) + len(BACKGROUND))

        # Printed through the officer search rather than a plain equality
        # query, and both counts are shown, because the gap between them is
        # the whole argument for confusion-aware matching: cam-01 is the
        # vehicle, and an exact-match search silently loses it.
        exact = con.execute("SELECT COUNT(*) FROM sighting WHERE "
                            "plate_normalised=?", (norm,)).fetchone()[0]
        found = _sightings_for(con, norm)
        for r in found:
            flag = ("" if r["exact_read"] else
                    "  <- recovered by confusion-aware matching" if r["plate_claimed"] else
                    "  <- attributed, no plate claimed")
            print(f"{r['at']}  {r['camera_id']}  "
                  f"{(r['plate_read'] or 'no plate read'):<12} "
                  f"{r['tier']:<14} {r['capability_class']:<15} "
                  f"{r['location_name']}{flag}")
        print(f"\nexact-match search: {exact} sightings   "
              f"confusion-aware search: {len(found)} sightings")


if __name__ == "__main__":
    main()
