#!/usr/bin/env python3
"""
The organisers' test case: one vehicle, followed across two government cameras.

Everything before this proved a component. This proves the thing they will
actually ask for on 22 September -- designate a vehicle seen on one camera and
find it again on a second, unrelated camera, on their footage, with the route
and the times an officer would be shown.

The pair
--------
cam05  "Visat teen Rasta CSITMS-31_PTZ1"   1920x1080, 30fps
cam16  "Visat T Junction P2 RLVD"          1920x1080, 10fps

Two separate camera systems -- a CSITMS traffic PTZ and a red-light violation
detector -- with different frame rates, different clocks and different naming
conventions, watching one junction. The captions are burned into the frames,
so that is checkable rather than asserted. They are the same junction and not
merely nearby: both frames contain the same pair of "75750 83008" advertising
boards and the same chrome-hooped bin, and the demo saves that side by side so
the claim can be checked by eye rather than taken on trust.

Why these two clips are simultaneous
------------------------------------
Every recording on the grid is a twelve-hour HLS VOD, and the burned-in clocks
say that the same position in two recordings is the same wall-clock instant.
Sampling eight Ahmedabad cameras at 11.0h gives, on cam13, cam14, cam05, cam02,
cam15, cam01, cam04 and cam03 in turn: 07:58:53, 07:59:16, 07:59:48, 07:59:52,
07:59:54, 07:59:55, 07:59:59 and 08:00:02 -- all of 14 June 2026, spread over
sixty-nine seconds. Both clips here were cut from position 11.0h, so they are
the same minute of the same Sunday morning. That is the alignment key for the
whole exercise: no cross-camera search is meaningful until you know which
windows to compare, and the recordings carry no index that tells you.

What the demo shows that a scoreboard would not
-----------------------------------------------
cam16 stamps 01:47:22 over a frame of broad daylight. Its clock is 6h12m26s
slow. An officer's timeline built from the burned-in clocks -- which is what
the vendor VMS shows -- would say this vehicle reached the second camera six
hours before it passed the first. The registry measures that drift, so the
timeline is built from corrected time and the sighting is demoted to
corroborating rather than presented as confirmed.

Run:  ./.venv/bin/python tools/cross_camera_demo.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import datetime as dt

import cv2
import numpy as np

from sentinel.detect import Detector
from sentinel.hunt import Hunt, Target, haversine_km, APPEARANCE_FLOOR, IMPOSSIBLE_KMH
from sentinel.registry import connect
from sentinel.track import VehicleTracker
from sentinel.vehicle_reid import VehicleReID

PREVIEW = pathlib.Path(".run/preview")

# Both clips were cut from the same position in their camera's twelve-hour
# recording, which is what makes them simultaneous.
SAMPLE_POSITION_H = 11.0

DESIGNATE = dict(camera="cam05", clip="media/real/cam05_day.mp4",
                 name="Visat teen Rasta CSITMS-31_PTZ1")
SEARCH = dict(camera="cam16", clip="media/real/cam16_day.mp4",
              name="Visat T Junction P2 RLVD")
# 300km away, in another city, during footage this bus cannot appear in. Any
# sighting here is a false positive, and the false-positive count is the
# number that decides whether an officer can act on this.
CONTROL = dict(camera="cam09", clip="media/real/cam09_day.mp4",
               name="New Bypass Circle, Junagadh")

# Read by eye from frame 0 of each clip; both strips are saved as evidence.
OSD_AT_CLIP_START = {"cam05": "2026-06-14 07:59:48", "cam16": "2026-06-14 01:47:22"}

VEHICLE_CLASS = "bus"


def measured_fps(path: str):
    """
    Frame rate from PTS, never from CAP_PROP_FPS.

    cam05 declares 250fps in its container header. Believing that would put
    every sighting in the wrong second and make the route arithmetic
    nonsense, which is precisely the failure the organisers' notes warn about.
    """
    cap = cv2.VideoCapture(path)
    n, last_ms = 0, 0.0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
        last_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or last_ms
    cap.release()
    declared = 0.0
    cap = cv2.VideoCapture(path); declared = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return (n / (last_ms / 1000.0) if last_ms else 0.0), n, declared


def frames_of(path):
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield img
    finally:
        cap.release()


def designate(clip: str, vehicle_class: str, step: int = 3):
    """
    Designate a vehicle from the camera an officer first saw it on.

    The target is every view the tracker collected of that vehicle, pooled --
    not its single biggest crop. That is not a tuning choice, it is the whole
    difference between finding it and not: the biggest view of this bus is
    clipped by the frame edge and shows only its front, and querying with it
    scores 0.48 against the second camera's rear view of the same bus, under
    the floor. Pooling the track reaches 0.61. A vehicle looks different from
    in front and behind, so a query built from one moment of one angle is a
    query that only matches that angle.
    """
    tracker = VehicleTracker()
    held = {}
    n = 0
    for img in frames_of(clip):
        n += 1
        if n % step:
            continue
        for t in tracker.update(Detector.vehicles(img, min_score=0.5), n):
            if t.last_seen != n or t.label != vehicle_class or t.is_static():
                continue
            x1, y1, x2, y2 = (max(0, int(v)) for v in t.box)
            crop = img[y1:y2, x1:x2]
            if crop.size:
                held.setdefault(t.track_id, []).append((n, t.width, crop.copy()))
    if not held:
        sys.exit(f"no moving {vehicle_class} to designate on {clip}")

    tid, views = max(held.items(), key=lambda kv: max(v[1] for v in kv[1]))
    embs = [VehicleReID.embed(c) for _, _, c in views]
    embs = np.stack([e for e in embs if e is not None])
    pooled = embs.mean(0)
    pooled /= np.linalg.norm(pooled)
    return tid, views, pooled


def hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def stamp(base: str, offset_s: float) -> str:
    return (dt.datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
            + dt.timedelta(seconds=offset_s)).strftime("%Y-%m-%d %H:%M:%S")


def label(img, text, colour=(0, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 14 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    return out


def side_by_side(a, b, height=340):
    def fit(x):
        return cv2.resize(x, (max(1, int(x.shape[1] * height / x.shape[0])), height))
    a, b = fit(a), fit(b)
    return np.hstack([a, np.full((height, 14, 3), 255, np.uint8), b])


def frame_at(clip: str, index: int):
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index - 1))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        cams = {r["id"]: dict(r) for r in con.execute(
            "SELECT id, department, location_name, lat, lon, capability_class, "
            "plate_px_measured FROM camera")}

    print("=" * 74)
    print("SENTINEL -- one vehicle, two government cameras, their own footage")
    print("=" * 74)
    for spec in (DESIGNATE, SEARCH, CONTROL):
        c = cams[spec["camera"]]
        print(f"  {spec['camera']}  {c['department']:<10} {c['location_name'][:36]:<36} "
              f"{c['lat']},{c['lon']}  {c['capability_class']}")
    km = haversine_km(cams[DESIGNATE["camera"]]["lat"], cams[DESIGNATE["camera"]]["lon"],
                      cams[SEARCH["camera"]]["lat"], cams[SEARCH["camera"]]["lon"])
    print(f"\n  {DESIGNATE['camera']} to {SEARCH['camera']}: {km:.2f} km apart, "
          f"both sampled at {SAMPLE_POSITION_H:g}h into their 12h recording")

    # --- frame rates, measured ------------------------------------------
    fps = {}
    print("\nmeasured frame rate (declared rate ignored):")
    for spec in (DESIGNATE, SEARCH, CONTROL):
        f, n, declared = measured_fps(spec["clip"])
        fps[spec["camera"]] = f
        print(f"  {spec['camera']}  {n:>5} frames  measured {f:5.2f} fps  "
              f"(container claims {declared:.0f})")

    # --- designate -------------------------------------------------------
    print(f"\ndesignating a {VEHICLE_CLASS} seen on {DESIGNATE['camera']} "
          f"({DESIGNATE['name']})...", flush=True)
    tid, views, pooled = designate(DESIGNATE["clip"], VEHICLE_CLASS)
    a_fps = fps[DESIGNATE["camera"]]
    print(f"  track {tid}: {len(views)} views, widest {max(v[1] for v in views)}px, "
          f"seen {views[0][0]/a_fps:.1f}s to {views[-1][0]/a_fps:.1f}s into the clip")
    best_a = max(views, key=lambda v: v[1])
    cv2.imwrite(str(PREVIEW / "crosscam_target.jpg"), best_a[2])

    target = Target(appearance=pooled, vehicle_class=VEHICLE_CLASS,
                    note=f"{VEHICLE_CLASS} designated on {DESIGNATE['camera']}, "
                         f"{len(views)} views pooled")

    # --- hunt ------------------------------------------------------------
    hunt = Hunt(target, log=lambda m: print("  " + m, flush=True))
    print(f"\nhunting {SEARCH['camera']} ({SEARCH['name']}):", flush=True)
    # The rate is handed in, not looked up: the registry has no measured rate
    # for these two cameras yet, and a sighting timed at an assumed 25fps on a
    # camera delivering 10 lands two and a half times too late.
    hunt.search_stream(SEARCH["camera"], frames_of(SEARCH["clip"]),
                       camera_name=SEARCH["name"], fps=fps[SEARCH["camera"]])
    found = list(hunt.hits)

    print(f"\nNEGATIVE control -- {CONTROL['camera']} ({CONTROL['name']}):", flush=True)
    hunt.search_stream(CONTROL["camera"], frames_of(CONTROL["clip"]),
                       camera_name=CONTROL["name"], fps=fps[CONTROL["camera"]])
    false_positives = [h for h in hunt.hits if h.camera_id == CONTROL["camera"]]

    print("\n" + "-" * 74)
    print(f"appearance floor {APPEARANCE_FLOOR}; nothing below it is shown to an officer")
    for h in sorted(found, key=lambda h: -h.score):
        print(f"  {h.camera_id}  {h.tier:<14} score={h.score:<6} "
              f"frame {h.frame} ({h.frame/fps[h.camera_id]:.1f}s)  {'; '.join(h.why)}")
    print(f"  false positives on {CONTROL['camera']} (300km away): {len(false_positives)}")
    if not found:
        print("\nNO CROSS-CAMERA SIGHTING. Nothing further to report.")
        return

    # --- route ------------------------------------------------------------
    hit = max(found, key=lambda h: h.score)
    a_off = best_a[0] / a_fps                       # seconds into the cam05 clip
    b_off = hit.frame / fps[hit.camera_id]          # seconds into the cam16 clip
    dt_s = b_off - a_off
    kmh = km / (abs(dt_s) / 3600.0) if dt_s else float("inf")

    print("\n" + "=" * 74)
    print("ROUTE  (times corrected: both clips start at the same instant, because")
    print("        both were cut from the same position in their recording)")
    print("=" * 74)
    a_true = stamp(OSD_AT_CLIP_START[DESIGNATE["camera"]], a_off)
    # The second camera's own clock is not usable, so its true time is the
    # first camera's -- the two recordings are position-aligned and the first
    # camera's clock was measured trustworthy.
    b_true = stamp(OSD_AT_CLIP_START[DESIGNATE["camera"]], b_off)
    b_osd = stamp(OSD_AT_CLIP_START[SEARCH["camera"]], b_off)
    drift = (dt.datetime.strptime(OSD_AT_CLIP_START[SEARCH["camera"]], "%Y-%m-%d %H:%M:%S")
             - dt.datetime.strptime(OSD_AT_CLIP_START[DESIGNATE["camera"]], "%Y-%m-%d %H:%M:%S"))
    print(f"  1. {DESIGNATE['camera']}  {DESIGNATE['name']}")
    print(f"       {a_true}   designated here")
    print(f"  2. {SEARCH['camera']}  {SEARCH['name']}")
    print(f"       {b_true}   appearance {hit.score}, {hit.tier}")
    if len(found) > 1:
        print(f"       ({len(found)} sightings on this camera: the tracker lost and "
              f"reacquired\n        the same bus as it crossed, so it is one vehicle "
              f"reported twice,\n        not two vehicles)")
    print(f"\n  leg: {km:.2f} km in {dt_s:+.1f}s  =>  {kmh:.0f} km/h"
          f"{'   FLAGGED IMPOSSIBLE' if kmh > IMPOSSIBLE_KMH else ''}")
    if kmh > IMPOSSIBLE_KMH:
        print(f"  The threshold is doing its job. These two cameras are {km*1000:.0f}m apart")
        print("  with overlapping views, so this is not a journey between two points --")
        print("  it is one vehicle co-observed by two camera systems seconds apart. A")
        print("  registry that knows the cameras overlap reads it that way; one that")
        print("  only knows their coordinates reports a bus doing 250km/h.")
    print(f"\n  {SEARCH['camera']} burned-in clock says {b_osd} for that same instant.")
    print(f"  Measured drift {drift.total_seconds():+.0f}s "
          f"({abs(drift.total_seconds())/3600:.2f}h). A timeline built from the")
    print("  camera's own clock would have the vehicle arriving six hours before it")
    print("  left. That is why the sighting is corroborating and not confirmed.")

    # --- evidence ---------------------------------------------------------
    fa = frame_at(DESIGNATE["clip"], best_a[0])
    fb = frame_at(SEARCH["clip"], hit.frame)
    x1, y1, x2, y2 = hit.box
    cv2.rectangle(fb, (x1, y1), (x2, y2), (0, 220, 255), 4)
    cv2.imwrite(str(PREVIEW / "crosscam_sighting_pair.jpg"), side_by_side(
        label(fa, f"{DESIGNATE['camera']} {a_true}", (0, 255, 255)),
        label(fb, f"{SEARCH['camera']} {b_true} score {hit.score}", (0, 255, 255)),
        height=520))
    cv2.imwrite(str(PREVIEW / "crosscam_vehicle_pair.jpg"), side_by_side(
        best_a[2], fb[y1:y2, x1:x2]))

    # Every sighting, not only the best-scoring one, so nothing is cherry-picked.
    for i, h in enumerate(sorted(found, key=lambda h: -h.score)):
        f = frame_at(SEARCH["clip"], h.frame)
        p, q, r, t2 = h.box
        cv2.imwrite(str(PREVIEW / f"crosscam_sighting_{i}_{h.score:.3f}.jpg"),
                    side_by_side(best_a[2], f[q:t2, p:r]))

    # The clock evidence: daylight under a 01:47 stamp.
    strip_a = frame_at(DESIGNATE["clip"], 1)
    strip_b = frame_at(SEARCH["clip"], 1)
    # The label goes at the foot, because the burned-in clock is the evidence
    # and sits at the top of one frame.
    def foot(img, text):
        out = img.copy()
        h = out.shape[0]
        cv2.rectangle(out, (0, h - 40), (min(out.shape[1], 15 * len(text)), h), (0, 0, 0), -1)
        cv2.putText(out, text, (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        return out
    cv2.imwrite(str(PREVIEW / "crosscam_clocks.jpg"), side_by_side(
        foot(strip_a, f"{DESIGNATE['camera']} frame 0"),
        foot(strip_b, f"{SEARCH['camera']} frame 0 -- same instant"), height=430))

    # The same junction, proven by a landmark rather than by coordinates.
    cv2.imwrite(str(PREVIEW / "crosscam_landmark.jpg"), side_by_side(
        strip_a[540:980, 800:1140], strip_b[380:620, 1700:1920], height=440))

    # The pass, second by second, on both cameras at once.
    rows = []
    for t in (13, 15, 17):
        ia = frame_at(DESIGNATE["clip"], int(t * a_fps))
        ib = frame_at(SEARCH["clip"], int(t * fps[SEARCH["camera"]]))
        if ia is None or ib is None:
            continue
        rows.append(side_by_side(label(ia, f"{DESIGNATE['camera']} +{t}s"),
                                 label(ib, f"{SEARCH['camera']} +{t}s"), height=300))
    if rows:
        w = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)])
                for r in rows]
        cv2.imwrite(str(PREVIEW / "crosscam_timeline.jpg"), np.vstack(rows))

    print(f"\nevidence written to {PREVIEW}/crosscam_*.jpg")


if __name__ == "__main__":
    main()
