#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
Populate the registry from the catalogue plus whatever has been measured.

The catalogue supplies identity: who owns the camera, where it is, what it
claims to be. The measurements supply capability, condition and time
confidence -- the columns that turn a list into a control plane.

    python3 tools/build_registry.py                 # use cached measurements
    python3 tools/build_registry.py --measure       # re-measure everything
"""
import argparse, json, time, urllib.request
from pathlib import Path

from sentinel.registry import (init, connect, upsert_camera, record_capability,
                               record_time_confidence, assignments, audit)

CATALOGUE = "http://127.0.0.1:8080/api/ingest"
GOV_BY_DEPT = {"Health": "sensitive", "Police": "open", "GSRTC": "open",
               "Municipal": "open", "Panchayat": "open"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CATALOGUE)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--seconds", type=int, default=50)
    args = ap.parse_args()

    init()
    cat = json.loads(urllib.request.urlopen(args.catalogue, timeout=15).read())
    caps = {r["id"]: r for r in load("audit/capability_validation.json")}
    clocks = {r["id"]: r for r in load("audit/clock_drift.json")}

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with connect() as con:
        for c in cat["cameras"]:
            upsert_camera(con, {
                "id": c["id"], "catalogue_id": c["id"],
                "department": c["department"],
                "location_name": c.get("location") or c.get("name"),
                "lat": c.get("lat"), "lon": c.get("lon"),
                "rtsp_url": c.get("rtsp_url"), "hls_url": c.get("hls_url"),
                "whep_url": c.get("whep_url"),
                "width": c.get("width"), "height": c.get("height"),
                "codec": c.get("codec"),
                "governance_class": c.get("governance_class")
                    or GOV_BY_DEPT.get(c["department"], "open"),
                "last_seen_live": now if c.get("live") else None,
            })

            if args.measure:
                from sentinel.capability import measure as measure_cap
                r = measure_cap(c["rtsp_url"], c["id"], seconds=args.seconds)
                cls, px, basis = r.capability_class(), r.plate_px(), r.measurement_basis()
                bright, night = r.brightness, (not r.night) if r.brightness else None
            else:
                m = caps.get(c["id"])
                if not m:
                    continue
                cls, px, basis = m["cls"], m["measured"], m["basis"]
                bright, night = None, None

            record_capability(con, c["id"], capability_class=cls, plate_px=px,
                              basis=basis, brightness=bright, night_usable=night)

            k = clocks.get(c["id"])
            if k and k.get("drift_s") is not None:
                record_time_confidence(con, c["id"], k["drift_s"] * 1000.0,
                                       f'{k["confidence"]}: {k["note"]}')

        audit(con, "system", "registry_build",
              query_params={"catalogue": args.catalogue},
              result_count=len(cat["cameras"]))

    with connect() as con:
        print(f"{'camera':<8} {'dept':<10} {'capability':<16} {'plate':>6} "
              f"{'clock':>10}  {'analytic':<10} governance")
        print("-" * 82)
        rows = {r["camera_id"]: r for r in assignments(con)}
        for r in con.execute("SELECT * FROM camera ORDER BY id"):
            a = rows[r["id"]]
            drift = r["time_confidence_ms"]
            ds = "-" if drift is None else f"{drift/1000:+.1f}s"
            print(f"{r['id']:<8} {r['department']:<10} "
                  f"{str(r['capability_class']):<16} "
                  f"{str(r['plate_px_measured']):>6} {ds:>10}  "
                  f"{a['analytic']:<10} {r['governance_class']}")


if __name__ == "__main__":
    main()
