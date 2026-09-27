"""Measure burned-in clock drift for every camera; write results as we go."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, sys, urllib.request
from pathlib import Path
from sentinel.timesync import measure

out = Path("audit/clock_drift.json")
rows = json.loads(out.read_text()) if out.exists() else []
done = {r["id"] for r in rows}

cat = json.loads(urllib.request.urlopen("http://127.0.0.1:8080/api/ingest", timeout=10).read())
for c in cat["cameras"]:
    if c["id"] in done:
        continue
    tc = measure(c["rtsp_url"], c["id"], samples=6, seconds=60)
    rows.append({
        "id": c["id"],
        "drift_s": None if tc.drift_ms is None else round(tc.drift_ms / 1000.0, 1),
        "true_s": c["_synthetic_truth"]["clock_offset_s"],
        "confidence": tc.classify(), "reads": tc.reads, "samples": tc.samples,
        "note": tc.note,
    })
    out.write_text(json.dumps(rows, indent=2))
    print(f'  {c["id"]}: {rows[-1]["drift_s"]}s (true {rows[-1]["true_s"]}s) '
          f'{rows[-1]["confidence"]} reads={tc.reads}/{tc.samples}', flush=True)
