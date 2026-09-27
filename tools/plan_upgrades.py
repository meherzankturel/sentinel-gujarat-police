#!/usr/bin/env python3
"""
What would have to change, camera by camera, for this grid to do the job.

The survey says three cameras in thirty read a number plate and none of
them resolve a face. That is a finding, not a plan. A department cannot
act on "no"; it can act on "cam05 is 1.3x short and a longer lens fixes
it, cam16 is 5.5x short and needs moving, and these seven are pointed at
places vehicles never go."

Reads the registry and audit/face_survey.json. Writes audit/upgrade_plan.json.

    python tools/plan_upgrades.py
    python tools/plan_upgrades.py --face-target 60    # plan to do it well,
                                                      # not merely to attempt
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sentinel.facecap import FACE_CAPABLE_IOD, FACE_MARGINAL_IOD
from sentinel.planner import plan, summarise
from sentinel.registry import connect

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACE_SURVEY = ROOT / "audit" / "face_survey.json"
OUT = ROOT / "audit" / "upgrade_plan.json"

VERDICT_LABEL = {
    "met": "already capable",
    "optics": "a lens or a different aim",
    "resite": "move it, or add one closer",
    "replace": "a new camera in a new place",
    "not-applicable": "wrong question for this camera",
    "unknown": "not measured yet",
}


def government_cameras():
    with connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT id, location_name, department, width, height,"
            " capability_class, plate_px_measured, hls_url FROM camera")]
    rows = [r for r in rows if "127.0.0.1" not in (r.get("hls_url") or "")]

    faces = {}
    if FACE_SURVEY.exists():
        for f in json.loads(FACE_SURVEY.read_text()):
            faces[f["camera_id"]] = f
    for r in rows:
        f = faces.get(r["id"])
        if f:
            r["face_iod_px"] = f.get("iod_p90")
            r["face_class"] = f.get("face_class")
            r["body_px_p90"] = f.get("body_px_p90")
    return rows


def main() -> int:
    target = FACE_MARGINAL_IOD
    if "--face-target" in sys.argv:
        target = float(sys.argv[sys.argv.index("--face-target") + 1])

    cams = government_cameras()
    shortfalls = plan(cams, face_target=target)
    counts = summarise(shortfalls)

    print(f"{len(cams)} government cameras")
    print(f"face planning target: {target:.0f}px between the eyes "
          f"({'attempt a match' if target <= FACE_MARGINAL_IOD else 'do it well'})\n")

    for cap in ("plate", "face"):
        rows = [s for s in shortfalls if s.capability == cap]
        print("=" * 72)
        print(f"{cap.upper()} READING" if cap == "plate" else "FACE MATCHING")
        print("=" * 72)
        for verdict in ("met", "optics", "resite", "replace",
                        "not-applicable", "unknown"):
            group = [s for s in rows if s.verdict == verdict]
            if not group:
                continue
            print(f"\n  {VERDICT_LABEL[verdict].upper()}  ({len(group)})")
            for s in group:
                short = "" if s.factor is None or s.factor <= 1 \
                    else f"  {s.factor:.1f}x short"
                print(f"    {s.camera_id:<7} {(s.location or '')[:38]:<38}{short}")
                if verdict in ("optics", "resite"):
                    for o in s.options[:2]:
                        print(f"        - {o.detail}")
        print()

    print("=" * 72)
    print("WHAT THIS COSTS, AS A COUNT")
    print("=" * 72)
    for cap, by in counts.items():
        parts = [f"{n} {VERDICT_LABEL[v]}" for v, n in sorted(by.items())]
        print(f"  {cap:<6} {', '.join(parts)}")

    OUT.write_text(json.dumps({
        "face_target_px": target,
        "summary": counts,
        "shortfalls": [s.as_dict() for s in shortfalls],
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
