#!/usr/bin/env python3
"""
Survey the government grid: what can each camera actually deliver?

Their catalogue gives an id and a name. Everything here is measured from
the video: how big vehicles actually appear, and therefore what fraction
of passing vehicles could have their plate read at all.

The headline column is ANPR yield -- the share of vehicle sightings whose
plate clears the legibility threshold. Capability is not a yes/no on these
cameras: a vehicle directly beneath the mast is readable while the same
vehicle two lanes over is not, so the honest answer is a percentage.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, time, cv2, statistics
from sentinel.gateway import Gateway
from sentinel.hls import HLSSampler
from sentinel.detect import Detector
from sentinel.capability import CameraMotion, PLATE_TO_VEHICLE

OUT = pathlib.Path("audit/grid_survey.json")
PLATE_READ_PX, PLATE_MARGINAL_PX = 120, 80
SAMPLE_AT = 9 * 3600           # 09:00 into the recording: daylight traffic
FRAMES = 40

rows = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["id"] for r in rows}

g = Gateway(); g.login()
cams = json.loads(g.get("/cameras.json")[0].decode())
s = HLSSampler(g)
Detector._load()
print(f"{len(cams)} cameras, {len(done)} done, device={Detector.device()}", flush=True)

for c in cams:
    cid = c["id"]
    if cid in done:
        continue
    row = {"id": cid, "name": c["name"]}
    try:
        m = s.sample(cid, SAMPLE_AT, seconds=20)
        cap = cv2.VideoCapture(str(m))
        row["declared_fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)

        widths, people, cammot, n, used, bright = [], 0, CameraMotion(), 0, 0, []
        while used < FRAMES:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            if row.get("width") is None:
                row["height"], row["width"] = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            panning = cammot.update(gray)
            if n % 6:
                continue
            used += 1
            bright.append(float(gray.mean()))
            if panning:
                continue
            for d in Detector.detect(img, ("car","truck","bus","motorcycle","person"), 0.5):
                if d.label == "person":
                    people += 1
                else:
                    widths.append(d.width)
        cap.release()

        row["frames_analysed"] = used
        row["vehicles"] = len(widths)
        row["people"] = people
        row["pan_share"] = round(cammot.pan_share(), 3)
        row["is_ptz"] = cammot.is_ptz()
        row["brightness"] = round(statistics.fmean(bright), 1) if bright else None

        if widths:
            ws = sorted(widths)
            plate = [w * PLATE_TO_VEHICLE for w in ws]
            row["veh_w_p50"] = ws[len(ws)//2]
            row["veh_w_p90"] = ws[int(len(ws)*0.9)]
            row["plate_p50"] = round(plate[len(plate)//2], 1)
            row["plate_p90"] = round(plate[int(len(plate)*0.9)], 1)
            row["plate_max"] = round(plate[-1], 1)
            row["anpr_yield"] = round(sum(1 for p in plate if p >= PLATE_READ_PX)/len(plate), 4)
            row["marginal_yield"] = round(sum(1 for p in plate if p >= PLATE_MARGINAL_PX)/len(plate), 4)
            y = row["anpr_yield"]
            row["capability"] = ("plate-capable" if y >= 0.15 else
                                 "plate-occasional" if y >= 0.02 else
                                 "presence-only")
        else:
            row["capability"] = "no-vehicles-observed"

        print(f"  {cid:<7} {row.get('width')}x{row.get('height')} "
              f"veh={row['vehicles']:<5} plate_p90={row.get('plate_p90')} "
              f"yield={row.get('anpr_yield')} {row['capability']}"
              f"{' PTZ' if row['is_ptz'] else ''}", flush=True)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:90]}"
        print(f"  {cid:<7} FAILED {row['error'][:70]}", flush=True)
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=2))
    time.sleep(0.8)

s.cleanup()
print("done", flush=True)
