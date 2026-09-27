#!/usr/bin/env python3
"""
Measure every camera on the government grid.

Their catalogue gives us an id and a name. Everything else -- resolution,
real frame rate, whether a plate is legible -- has to be measured from the
video, which is exactly what this does. One camera at a time, a short
sample each, so we stay inside the organisers' instruction to pace load.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, time, cv2
from sentinel.gateway import Gateway
from sentinel.hls import HLSSampler
from sentinel.capability import measure_file

OUT = pathlib.Path("audit/real_grid.json")
rows = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["id"] for r in rows}

g = Gateway(); g.login()
cams = json.loads(g.get("/cameras.json")[0].decode())
s = HLSSampler(g)
print(f"{len(cams)} cameras; {len(done)} already measured", flush=True)

for c in cams:
    cid = c["id"]
    if cid in done:
        continue
    row = {"id": cid, "name": c["name"]}
    try:
        pl = s.playlist(cid)
        pos = int(time.time()) % int(pl.total_s)
        m = s.sample(cid, pos, seconds=30)

        cap = cv2.VideoCapture(str(m))
        row["declared_fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
        cap.release()

        rep = measure_file(m, cid, max_frames=900)
        row.update({
            "width": rep.width, "height": rep.height,
            "frames": rep.frames_seen, "motion_events": rep.motion_events,
            "plate_px": rep.plate_px(), "capability": rep.capability_class(),
            "basis": rep.measurement_basis(), "hits": len(rep.hits),
            "brightness": rep.brightness, "night": rep.night,
            "segments_total": len(pl.segments),
            "hours": round(pl.total_s / 3600, 2),
        })
        print(f"  {cid:<7} {row['width']}x{row['height']} "
              f"plate={row['plate_px']} {row['capability']}", flush=True)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"  {cid:<7} FAILED {row['error'][:70]}", flush=True)
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=2))
    time.sleep(1.0)          # pace the load

s.cleanup()
print("done", flush=True)
