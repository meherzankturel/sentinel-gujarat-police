#!/usr/bin/env python3
"""
The camera net moves ahead of the vehicle.

The cross-camera demonstration designates a bus on cam05 and finds it again
on cam16. In that demonstration a human chose cam16. This is the same
question asked of the registry instead: given where the vehicle has been
seen, which cameras are worth opening next, and why?

That is the whole "open only the cameras you are actively processing"
constraint turned into a decision the system makes rather than one an
officer makes on the telephone.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sentinel.lookahead import Sighting, nominate
from sentinel.registry import connect

BUDGET = 4
HORIZON_S = 600


def registry_cameras(government_only=True):
    with connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT id, location_name, department, lat, lon, capability_class,"
            " hls_url FROM camera")]
    if government_only:
        rows = [r for r in rows if "127.0.0.1" not in (r.get("hls_url") or "")]
    return rows


def show(title, look):
    print("=" * 74)
    print(title)
    print("=" * 74)
    h = "unknown" if look.heading is None else f"{look.heading:.0f}deg"
    print(f"  last seen on      : {look.from_camera}")
    print(f"  direction         : {h}")
    print(f"  ground speed      : {look.speed_kmh:.0f} km/h  ({look.speed_basis})")
    print(f"  horizon / budget  : {look.horizon_s:.0f}s / {look.budget} open streams")
    print(f"  cameras considered: {look.considered}")
    print()
    if not look.nominations:
        print("  nothing nominated")
    for i, n in enumerate(look.nominations, 1):
        print(f"  {i}. {n.camera_id:<7} {n.capability:<20} "
              f"{n.km:>5.2f} km  ETA {n.eta_s/60:>4.1f} min")
        print(f"     {n.name} ({n.department})")
        for w in n.why:
            print(f"       - {w}")
    print(f"\n  {look.note}\n")


def main():
    cams = registry_cameras()
    by_id = {c["id"]: c for c in cams}
    print(f"{len(cams)} government cameras in the registry\n")

    # The cross-camera run: the bus is designated on cam05 at 08:00:03 and
    # reaches cam16 two seconds later. Those two overlap, so this asks the
    # question from cam05 alone -- one sighting, no heading yet.
    a = by_id["cam05"]
    one = [Sighting("cam05", a["lat"], a["lon"], 0.0)]
    show("ONE SIGHTING -- no direction known yet, so it is a ring",
         nominate(one, cams, budget=BUDGET, horizon_s=HORIZON_S))

    # Now with a direction: seen at cam01 then cam05, travelling north up
    # the same corridor. This is the case that matters -- a real second
    # sighting far enough away to establish a heading.
    b = by_id["cam01"]
    two = [Sighting("cam01", b["lat"], b["lon"], 0.0),
           Sighting("cam05", a["lat"], a["lon"], 420.0)]
    look = nominate(two, cams, budget=BUDGET, horizon_s=HORIZON_S)
    show("TWO SIGHTINGS -- cam01 then cam05, so the net points north", look)

    picked = [n.camera_id for n in look.nominations]
    if "cam16" in picked:
        print(f"  cam16 was nominated at position {picked.index('cam16') + 1} "
              f"of {len(picked)} -- the camera the cross-camera demonstration "
              f"searched by hand.")
    else:
        print("  cam16 was NOT nominated. Reported rather than hidden: the "
              "registry's own coordinates are recovered from camera names, "
              "and cam05/cam16 sit 151m apart with overlapping views, which "
              "is a co-observation rather than a journey.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
