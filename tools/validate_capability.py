"""Measure every camera and score the measurement against known truth."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, urllib.request
from sentinel.capability import measure

cat = json.loads(urllib.request.urlopen("http://127.0.0.1:8080/api/ingest", timeout=10).read())
rows = []
for c in cat["cameras"]:
    truth = c["_synthetic_truth"]
    r = measure(c["rtsp_url"], c["id"], seconds=50)
    px, true_px = r.plate_px(), truth["plate_px"]
    err = None if px is None else round(100.0 * (px - true_px) / true_px, 1)
    rows.append({
        "id": c["id"], "res": f"{c['width']}x{c['height']}",
        "measured": px, "true": true_px, "err_pct": err,
        "cls": r.capability_class(), "basis": r.measurement_basis(),
        "hits": len(r.hits), "expect": truth["expect"],
    })
    print(f"  {c['id']} done: {px}px vs {true_px}px", flush=True)

print("\n" + "="*104)
print(f"{'camera':<8} {'resolution':<11} {'measured':>8} {'true':>6} {'err':>7}  "
      f"{'capability (measured)':<18} basis")
print("-"*104)
for r in rows:
    e = "n/a" if r["err_pct"] is None else f"{r['err_pct']:+.1f}%"
    print(f"{r['id']:<8} {r['res']:<11} {str(r['measured']):>8} {r['true']:>6} {e:>7}  "
          f"{r['cls']:<18} {r['basis']}")
print("="*104)
json.dump(rows, open("audit/capability_validation.json","w"), indent=2)
