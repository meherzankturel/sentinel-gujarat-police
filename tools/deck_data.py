#!/usr/bin/env python3
"""
Every number the deck shows, computed from the measurement files.

The deck makes claims about a police grid. Typing those numbers into
slides by hand is how a deck ends up disagreeing with the system it
describes -- and on 12 October the system is what gets run, not the deck.
So nothing in the presentation is hand-entered: this reads the same JSON
the registry was built from and emits what the slides render.

Sources, and they are printed on the slides that use them:

    audit/grid_survey.json          per-camera capability on the government grid
    audit/capability_validation.json  capability measurement vs known truth
    audit/clock_drift.json          measured clock drift
    audit/face_survey.json          inter-ocular pixels, per camera
    audit/upgrade_plan.json         what each shortfall would cost to close
    sentinel.db                     the registry itself, and the look-ahead
                                    nomination computed from it
"""
import json
import pathlib
import sqlite3
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"

# A plate needs roughly this many pixels across to be read at all. It is
# the single number the whole assignment argument turns on.
PLATE_READ_PX = 120
PLATE_MARGINAL_PX = 80

# Pixels between the eye centres before a face match is worth attempting.
# ISO/IEC 19794-5 puts 60px as the enrolment floor; this is the generous
# reading of it, and it is the figure sentinel/facecap.py classifies on.
FACE_FLOOR_PX = 40.0

# The look-ahead scenario the deck shows, matching tools/lookahead_demo.py.
LOOKAHEAD_BUDGET = 4
LOOKAHEAD_HORIZON_S = 600

# Planning assumptions carried from docs/scalability-plan.md section 8, so
# the deck and the cost model cannot disagree. One installed GPU node --
# server, accelerator, networking and commissioning -- against the ~Rs
# 1,600 cr / ~Rs 160 cr capital figures that document publishes. These are
# order-of-magnitude figures for comparing two designs, not a tender price,
# and the slide says so. What is robust is the ratio, which is set by the
# measured capability share and not by the per-node price at all.
# One installed node = one datacentre-class GPU plus its host server,
# amortised over five years. An A100/L40S-class card lands around Rs 8-13
# lakh and the server around Rs 8-12 lakh, so Rs 20 lakh is the midpoint of
# a defensible range rather than a figure chosen to make the gap look good.
#
# This was Rs 1 crore, which was not a costing -- it was reverse-engineered
# to match a total in the scalability plan that turned out to be arithmetic
# error. Every line of that table was out by 10x, and every error inflated
# the saving we were claiming. Corrected downward here and in the plan.
GPU_NODE_RUPEES = 20_00_000          # Rs 20 lakh per installed GPU node

# Continuous draw per node: GPU ~400 W plus host ~300 W.
NODE_WATTS = 700
ELECTRICITY_PER_KWH = 8              # Rs

# Installed cost per TB of hot storage, including redundancy, chassis and
# controllers -- not the price of a bare drive.
STORAGE_PER_TB_RUPEES = 12_000
HOT_RETENTION_H = 72                 # raw video expires; metadata does not


def _load(name):
    p = AUDIT / name
    if not p.exists():
        sys.exit(f"missing measurement file: {p}")
    return json.loads(p.read_text())


def grid():
    """The government grid, as surveyed."""
    rows = _load("grid_survey.json")
    ok = [r for r in rows if "error" not in r]
    measured = [r for r in ok if r.get("plate_p90") is not None]

    by_cap = {}
    for r in ok:
        by_cap[r["capability"]] = by_cap.get(r["capability"], 0) + 1

    plate_capable = [r for r in ok if r["capability"] == "plate-capable"]
    yields = [r["anpr_yield"] for r in measured]

    return {
        "surveyed": len(ok),
        "total": len(rows),
        "failed": len(rows) - len(ok),
        "by_capability": by_cap,
        "plate_capable": len(plate_capable),
        "plate_capable_share": round(len(plate_capable) / len(ok), 4) if ok else 0,
        # "1 camera in N" -- the headline
        "one_in": round(len(ok) / len(plate_capable), 1) if plate_capable else None,
        "median_plate_px": round(statistics.median(
            [r["plate_p50"] for r in measured]), 1) if measured else None,
        "best_plate_px": round(max(r["plate_max"] for r in measured), 1) if measured else None,
        "median_yield": round(statistics.median(yields), 4) if yields else None,
        "read_threshold_px": PLATE_READ_PX,
        "cameras": [
            {
                "id": r["id"],
                "name": r.get("name", ""),
                "res": f'{r.get("width")}x{r.get("height")}',
                "plate_p50": r.get("plate_p50"),
                "plate_p90": r.get("plate_p90"),
                "yield": r.get("anpr_yield"),
                "capability": r["capability"],
                "vehicles": r.get("vehicles"),
                "ptz": r.get("is_ptz", False),
            }
            for r in sorted(ok, key=lambda x: -(x.get("plate_p90") or 0))
        ],
    }


def validation():
    """Capability measurement against known ground truth."""
    rows = _load("capability_validation.json")
    errs = [abs(r["err_pct"]) for r in rows if r.get("err_pct") is not None]
    return {
        "n": len(rows),
        "worst_err_pct": round(max(errs), 1) if errs else None,
        "mean_err_pct": round(statistics.fmean(errs), 2) if errs else None,
        "rows": rows,
    }


def clocks():
    rows = _load("clock_drift.json")
    trusted = [r for r in rows if r.get("confidence") == "trusted"]
    drifts = [abs(r["drift_s"]) for r in rows if r.get("drift_s") is not None]
    return {
        "n": len(rows),
        "trusted": len(trusted),
        "max_abs_drift_s": round(max(drifts), 1) if drifts else None,
    }


def _import_root():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _cameras_by_grid(con):
    """
    Split the registry into the grids it actually contains.

    There are three, not two: the government grid, a local mock grid, and
    the entrant's own feed. A camera is placed in one only if it presents
    evidence of belonging there -- host in its URL, or the owning
    department -- and anything presenting none is counted as unclassified
    rather than given a home.

    This is not fussiness. An earlier version partitioned on
    `hls_url LIKE '%127.0.0.1%'` versus NOT LIKE, and the own-feed cameras
    carry no URL at all, so SQL's three-valued logic dropped them from both
    buckets while a sibling query defaulted them into "government". A deck
    that argues for measuring rather than assuming cannot be the thing that
    counts the entrant's own clips as the state's.

    The classifier itself is imported from sentinel/api.py so there is one
    definition of what a grid is, and the deck cannot drift from the API.
    """
    _import_root()
    from sentinel.api import grid_of

    by_grid = {}
    for r in con.execute("SELECT * FROM camera"):
        by_grid.setdefault(grid_of(dict(r)), []).append(dict(r))
    return by_grid


def registry():
    db = ROOT / "sentinel.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    n = lambda q: con.execute(q).fetchone()[0]
    by_grid = _cameras_by_grid(con)
    gov = by_grid.get("government", [])
    out = {
        "cameras": n("SELECT COUNT(*) FROM camera"),
        "government": len(gov),
        "own_feed": len(by_grid.get("own-feed", [])),
        "local_mock": len(by_grid.get("local-mock", [])),
        "unclassified": len(by_grid.get("unclassified", [])),
        # Departments are counted on the government grid, because that is
        # what the slide claims is federated. The entrant is not a
        # department of the state.
        "departments": len({c.get("department") for c in gov
                            if c.get("department")}),
        "departments_all": n("SELECT COUNT(DISTINCT department) FROM camera"),
        "sightings": n("SELECT COUNT(*) FROM sighting"),
        "alerts": n("SELECT COUNT(*) FROM alert"),
        "audit_entries": n("SELECT COUNT(*) FROM audit_log"),
    }
    con.close()
    return out


def scaling(g):
    """
    Statewide arithmetic, driven by the measured share rather than an
    assumed one. 80,000 cameras is the figure in the problem statement.
    """
    STATE = 80_000
    share = g["plate_capable_share"]
    return {
        "state_cameras": STATE,
        "heavy_share": round(share, 4),
        "heavy_cameras": int(round(STATE * share)),
        "saved_cameras": STATE - int(round(STATE * share)),
        # 40-60 concurrent 1080p30 streams per A100 in published reference
        # architectures; the midpoint is used and the range is stated.
        "gpus_all": int(round(STATE / 50)),
        "gpus_assigned": int(round(STATE * share / 50)),
        "mbps_per_camera": "2-5",
        "gbps_all": f"{STATE * 2 // 1000}-{STATE * 5 // 1000}",
        "gb_per_camera_day": 43,
        "pb_per_day": round(STATE * 43 / 1_000_000, 1),
        "pb_30_day": round(STATE * 43 * 30 / 1_000_000, 0),
    }


def faces():
    """
    Whether facial recognition is assignable anywhere on this grid.

    The survey measures inter-ocular pixels -- the distance between the eye
    centres -- derived from person detections through published
    anthropometry rather than from a face detector, because a face detector
    run on cameras that cannot resolve faces measures its own false
    positives. 40px is the floor for attempting a match at all.
    """
    rows = _load("face_survey.json")
    iods = [r["iod_p90"] for r in rows if r.get("iod_p90") is not None]
    if not iods:
        return {"n": len(rows)}
    best, worst = max(iods), min(iods)
    return {
        "n": len(rows),
        "floor_px": FACE_FLOOR_PX,
        "best_iod": round(best, 1),
        "worst_iod": round(worst, 1),
        # How far short the *best* camera on the grid is. This is the
        # number the argument rests on: if the best one is short, none
        # of them qualify.
        "best_short_by": round(FACE_FLOOR_PX / best, 1),
        "worst_short_by": round(FACE_FLOOR_PX / worst, 1),
        "people_measured": sum(r.get("people_seen", 0) for r in rows),
        "capable": sum(1 for r in rows if (r.get("iod_p90") or 0) >= FACE_FLOOR_PX),
        "rows": [
            {"id": r["camera_id"], "people": r.get("people_seen"),
             "body_p90": r.get("body_px_p90"), "iod": r.get("iod_p90"),
             "short_by": (round(FACE_FLOOR_PX / r["iod_p90"], 1)
                          if r.get("iod_p90") else None)}
            for r in sorted(rows, key=lambda x: -(x.get("iod_p90") or 0))
        ],
    }


def upgrades(g):
    """
    What it would take to make the grid capable, per camera.

    A department cannot act on "no". It can act on "this one is 1.4x short
    and a longer lens closes it". The counts below are the ones that would
    go into a procurement line.
    """
    plan = _load("upgrade_plan.json")
    # The plan covers every camera in the registry, own-feed included. The
    # deck's claim is about the government grid, so it is restricted to the
    # cameras the survey actually measured.
    gov = {c["id"] for c in g["cameras"]}
    plate = [s for s in plan["shortfalls"]
             if s["capability"] == "plate" and s["camera_id"] in gov]
    by = {}
    for s in plate:
        by[s["verdict"]] = by.get(s["verdict"], 0) + 1

    optics = sorted((s for s in plate if s["verdict"] == "optics"),
                    key=lambda s: s["factor"])
    met = by.get("met", 0)
    after = met + len(optics)
    total = len(plate)
    return {
        "total": total,
        "met": met,
        "optics": len(optics),
        "resite": by.get("resite", 0),
        "replace": by.get("replace", 0),
        "not_applicable": by.get("not-applicable", 0),
        # The headline: fit lenses on the optics band and the grid goes
        # from `met` plate-capable cameras to `after`.
        "after_capable": after,
        "gain": after - met,
        "best_factor": optics[0]["factor"] if optics else None,
        "worst_factor": optics[-1]["factor"] if optics else None,
        "share_before": round(met / total, 4) if total else 0,
        "share_after": round(after / total, 4) if total else 0,
        "rows": [
            {"id": s["camera_id"], "where": s["location"],
             "dept": s["department"], "have": s["have_px"],
             "factor": s["factor"],
             "lens": next((o["detail"] for o in s["options"]
                           if o["lever"] == "lens"), "")}
            for s in optics
        ],
    }


def lookahead():
    """
    The registry nominating which cameras to open next.

    Recomputed here rather than transcribed, so the slide cannot drift from
    what `tools/lookahead_demo.py` prints. The scenario is the one the
    cross-camera demonstration actually ran: a vehicle seen on cam01 and
    then cam05, travelling up the same corridor.
    """
    db = ROOT / "sentinel.db"
    if not db.exists():
        return {}
    _import_root()
    from sentinel.lookahead import Sighting, nominate

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    # Positively the government grid -- the look-ahead claim is about that
    # grid, so the mock and own-feed cameras must not be able to fill it.
    cams = _cameras_by_grid(con).get("government", [])
    con.close()
    by_id = {c["id"]: c for c in cams}
    if not {"cam01", "cam05"} <= by_id.keys():
        return {}

    a, b = by_id["cam05"], by_id["cam01"]
    go = lambda ss: nominate(ss, cams, budget=LOOKAHEAD_BUDGET,
                             horizon_s=LOOKAHEAD_HORIZON_S)

    # One sighting: no heading yet, so it is a ring and the budget bites.
    ring = go([Sighting("cam05", a["lat"], a["lon"], 0.0)])
    # Two: a heading exists, and the ring collapses to a cone.
    cone = go([Sighting("cam01", b["lat"], b["lon"], 0.0),
               Sighting("cam05", a["lat"], a["lon"], 420.0)])

    picked = [n.camera_id for n in cone.nominations]
    out = cone.as_dict()
    out["ring"] = ring.as_dict()
    out["cameras_in_registry"] = len(cams)
    out["hand_picked"] = "cam16"
    out["hand_picked_rank"] = (picked.index("cam16") + 1
                               if "cam16" in picked else None)
    return out


def costs(s):
    """
    The two designs priced, from the GPU counts already derived.

    Order-of-magnitude planning figures for comparing designs, not a tender
    response -- and the slide says so. Only the GPU line differs between
    the designs, so the difference between the designs *is* the GPU line.
    """
    CR = 1e7
    cr = lambda gpus: round(gpus * GPU_NODE_RUPEES / CR)
    power_cr = lambda gpus: round(
        gpus * (NODE_WATTS / 1000) * 24 * 365 * ELECTRICITY_PER_KWH / CR, 1)

    all_cr, assigned_cr = cr(s["gpus_all"]), cr(s["gpus_assigned"])
    hot_pb = round(s["state_cameras"] * s["gb_per_camera_day"]
                   * (HOT_RETENTION_H / 24) / 1_000_000, 1)
    return {
        "gpu_node_lakh": int(GPU_NODE_RUPEES / 1e5),
        "capital_all_cr": all_cr,
        "capital_assigned_cr": assigned_cr,
        "difference_cr": all_cr - assigned_cr,
        "ratio": round(all_cr / assigned_cr, 1) if assigned_cr else None,
        "power_all_cr_year": power_cr(s["gpus_all"]),
        "power_assigned_cr_year": power_cr(s["gpus_assigned"]),
        # Storage under the inverted retention policy. Identical in both
        # designs -- hot video is a function of camera count, not of which
        # analytic runs on it -- so it is not part of the difference.
        "hot_hours": HOT_RETENTION_H,
        "hot_pb": hot_pb,
        "hot_storage_cr": round(hot_pb * 1000 * STORAGE_PER_TB_RUPEES / CR, 1),
        "metadata_tb_year": 1,
    }


def build():
    g = grid()
    s = scaling(g)
    return {
        "grid": g,
        "validation": validation(),
        "clocks": clocks(),
        "registry": registry(),
        "scaling": s,
        "faces": faces(),
        "upgrades": upgrades(g),
        "lookahead": lookahead(),
        "costs": costs(s),
    }


if __name__ == "__main__":
    data = build()
    out = ROOT / "docs" / "deck" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))

    g, s = data["grid"], data["scaling"]
    print(f"surveyed          {g['surveyed']}/{g['total']}  ({g['failed']} failed)")
    print(f"capability        {g['by_capability']}")
    print(f"plate-capable     {g['plate_capable']}  = 1 in {g['one_in']}"
          f"  ({g['plate_capable_share']:.1%})")
    print(f"median plate px   {g['median_plate_px']}  (need {PLATE_READ_PX})")
    print(f"best plate px     {g['best_plate_px']}")
    print(f"validation        n={data['validation']['n']}"
          f" worst {data['validation']['worst_err_pct']}%")
    print(f"statewide heavy   {s['heavy_cameras']:,} of {s['state_cameras']:,}"
          f"  -> {s['gpus_assigned']:,} GPUs vs {s['gpus_all']:,}")
    u, f, c = data["upgrades"], data["faces"], data["costs"]
    print(f"upgrade plan      {u['met']} capable now -> {u['after_capable']}"
          f" if {u['optics']} lenses are fitted"
          f"  ({u['best_factor']}-{u['worst_factor']}x short)")
    print(f"faces             best {f.get('best_iod')}px IOD vs"
          f" {f.get('floor_px')}px floor -- short by"
          f" {f.get('best_short_by')}x; {f.get('capable')} cameras qualify")
    print(f"cost              Rs {c['capital_all_cr']:,} cr vs"
          f" Rs {c['capital_assigned_cr']:,} cr"
          f"  (difference Rs {c['difference_cr']:,} cr)")
    la = data["lookahead"]
    if la:
        print(f"look-ahead        {la['from_camera']} @ {la['heading']}deg,"
              f" {la['speed_kmh']} km/h -> {len(la['watch'])} nominated;"
              f" cam16 at rank {la['hand_picked_rank']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
