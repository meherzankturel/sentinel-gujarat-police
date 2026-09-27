#!/usr/bin/env python3
"""
Can any camera on this grid support face recognition?

The problem statement asks for an analytics approach covering facial
recognition. This answers it the same way the plate survey answered ANPR:
by measuring the cameras rather than assuming them, and by reporting the
answer whichever way it comes out.

Writes audit/face_survey.json, alongside audit/grid_survey.json.

    python tools/survey_faces.py             # local footage (fast)
    python tools/survey_faces.py --grid      # sample all 30 from the gateway
    python tools/survey_faces.py --load      # write the result to the registry
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from sentinel.detect import Detector
from sentinel.facecap import (FACE_CAPABLE_IOD, FACE_MARGINAL_IOD,
                              measure_frames)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "audit" / "face_survey.json"
SAMPLE_AT = 9 * 3600
SECONDS = 20


def frames_of(path, limit=900):
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while n < limit:
            ok, img = cap.read()
            if not ok:
                return
            n += 1
            yield img
    finally:
        cap.release()


def local_clips():
    for p in sorted((ROOT / "media" / "real").glob("*_day.mp4")):
        yield p.stem.split("_")[0], p


def grid_clips():
    from sentinel.gateway import Gateway
    from sentinel.hls import HLSSampler
    g = Gateway(); g.login()
    s = HLSSampler(g)
    try:
        for c in g.catalogue():
            cid = c["id"]
            try:
                yield cid, s.sample(cid, SAMPLE_AT, seconds=SECONDS)
            except Exception as e:                      # noqa: BLE001
                print(f"  {cid:<7} FAILED {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
            time.sleep(0.5)
    finally:
        s.cleanup()


def load_into_registry() -> int:
    """
    Write the measured face capability onto the camera rows.

    Measuring and recording are separate commands on purpose. The survey
    is expensive and needs footage; loading is cheap and needs only the
    file the survey wrote, so the registry can be rebuilt on a machine
    that has no clips without anybody re-deriving a number by hand.
    """
    import sqlite3
    from sentinel.registry import connect, init

    if not OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} not found — run the survey first")
    rows = json.loads(OUT.read_text())

    init()
    written, absent = 0, []
    with connect() as con:
        con.row_factory = sqlite3.Row
        for r in rows:
            cur = con.execute(
                "UPDATE camera SET person_body_px = ?, face_iod_px = ?, "
                "face_class = ?, face_people_seen = ?, face_measured_at = ? "
                "WHERE id = ?",
                (r.get("body_px_p90"), r.get("iod_p90"), r.get("face_class"),
                 r.get("people_seen"), r.get("measured_at") or _today(),
                 r["camera_id"]))
            if cur.rowcount:
                written += 1
            else:
                absent.append(r["camera_id"])

    print(f"wrote face capability onto {written} camera row(s)")
    if absent:
        # Said rather than swallowed: a measurement for a camera the
        # registry does not hold is a sign the two have drifted apart.
        print(f"  ! not in the registry, so not written: {', '.join(absent)}")
    return 0


def _today() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    if "--load" in sys.argv:
        return load_into_registry()
    use_grid = "--grid" in sys.argv
    source = grid_clips() if use_grid else local_clips()

    Detector._load()
    print(f"inter-ocular thresholds: capable >= {FACE_CAPABLE_IOD:.0f}px, "
          f"attempt >= {FACE_MARGINAL_IOD:.0f}px\n", flush=True)
    print(f"{'camera':<9}{'people':>8}{'body_p90':>10}{'eyes_px':>9}  class", flush=True)

    rows = []
    for cid, clip in source:
        r = measure_frames(frames_of(clip), cid)
        rows.append(r.as_dict())
        eyes = "-" if r.iod_p90 is None else f"{r.iod_p90:.1f}"
        body = "-" if r.body_px_p90 is None else f"{r.body_px_p90:.0f}"
        print(f"{cid:<9}{r.people_seen:>8}{body:>10}{eyes:>9}  {r.face_class}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    capable = [r for r in rows if r["face_class"] == "face-capable"]
    marginal = [r for r in rows if r["face_class"] == "face-marginal"]
    best = max((r["iod_p90"] for r in rows if r["iod_p90"] is not None),
               default=None)

    print(f"\n{len(rows)} camera(s) measured")
    print(f"  face-capable  : {len(capable)}")
    print(f"  face-marginal : {len(marginal)}")
    if best is not None:
        print(f"  best eye line : {best:.1f}px "
              f"(needs {FACE_MARGINAL_IOD:.0f} to attempt, "
              f"{FACE_CAPABLE_IOD:.0f} to be worth running)")
        if best < FACE_MARGINAL_IOD:
            print(f"  short by      : {FACE_MARGINAL_IOD/best:.0f}x on the "
                  f"best camera on the grid")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
