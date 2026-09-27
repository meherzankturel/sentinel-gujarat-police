#!/usr/bin/env python3
"""
The whole chain, end to end, on real government footage.

Designate a vehicle seen on one camera, then hunt it. Two controls, because
a hunt that finds the target proves nothing on its own if it also finds the
target everywhere else:

  positive -- hunt the same camera it was designated on. It must be found.
  negative -- hunt a camera 300km away in another city, during footage the
              vehicle cannot possibly appear in. Anything found there is a
              false positive, and the false-positive rate is the number that
              decides whether an officer can trust this.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from sentinel.detect import Detector
from sentinel.track import VehicleTracker
from sentinel.hunt import Hunt, target_from_crop
from sentinel.vehicle_reid import VehicleReID

def frames_of(path, limit=360):
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        while n < limit:
            ok, img = cap.read()
            if not ok: return
            n += 1
            yield img
    finally:
        cap.release()

# --- 1. designate a vehicle actually seen on cam09 -------------------------
print("designating a vehicle from cam09 (New Bypass Circle, Junagadh)...", flush=True)
tracker = VehicleTracker(); best = None
for i, img in enumerate(frames_of("media/real/cam09_day.mp4", 300), 1):
    if i % 3: continue
    for t in tracker.update(Detector.vehicles(img, min_score=0.6), i):
        if t.last_seen != i or t.is_static(): continue
        if best is None or t.width > best[0]:
            x1,y1,x2,y2 = (max(0,int(v)) for v in t.box)
            c = img[y1:y2, x1:x2]
            if c.size: best = (t.width, c.copy(), t.label, i)
if not best:
    sys.exit("no vehicle found to designate")
w, crop, cls, frame_no = best
cv2.imwrite(".run/preview/hunt_target.jpg", crop)
print(f"  target: {cls}, {w}px wide, first seen frame {frame_no}", flush=True)
print("  saved .run/preview/hunt_target.jpg", flush=True)

target = target_from_crop(crop, vehicle_class=cls)

# --- 2. hunt -------------------------------------------------------------
hunt = Hunt(target, log=lambda m: print("  " + m, flush=True))
print("\nPOSITIVE control -- hunting cam09, where it was seen:", flush=True)
hunt.search_stream("cam09", frames_of("media/real/cam09_day.mp4", 360),
                   camera_name="New Bypass Circle, Junagadh")
pos = len(hunt.hits)

print("\nNEGATIVE control -- hunting cam01, Ahmedabad, 300km away:", flush=True)
hunt.search_stream("cam01", frames_of("media/real/cam01_day.mp4", 360),
                   camera_name="Chimanbhai Bridge, Ahmedabad")
neg = len(hunt.hits) - pos

# --- 3. what the officer would see ---------------------------------------
print("\n" + "="*66)
s = hunt.summary()
print(f"target        : {s['target']} ({cls})")
print(f"sightings     : {s['sightings']} across {s['cameras_with_hits']} camera(s)")
print(f"by tier       : {s['by_tier']}")
print(f"positive cam  : {pos} sighting(s)   <- should be >0")
print(f"negative cam  : {neg} sighting(s)   <- false positives")
print()
for h in sorted(hunt.hits, key=lambda h: -h.score)[:8]:
    flag = "FALSE POSITIVE" if h.camera_id == "cam01" else ""
    print(f"  {h.camera_id}  {h.tier:<14} score={h.score:<6} f{h.frame:<4} "
          f"{h.vehicle_class:<10} {'; '.join(h.why)[:44]} {flag}")
legs = hunt.route()
if legs:
    print(f"\nroute legs: {len(legs)}, impossible: {sum(1 for l in legs if l['impossible'])}")
    for l in legs[:3]:
        print(f"  {l['from']} -> {l['to']}  {l['km']}km in {l['seconds']}s = "
              f"{l['kmh']}km/h {'IMPOSSIBLE' if l['impossible'] else ''}")
