#!/usr/bin/env python3
"""
sentinel.lookahead -- decide which cameras to open next.

The organisers' resources page states the constraint plainly: "Open only
the cameras you are actively processing." At 80,000 cameras that is not a
tuning note, it is the shape of the problem. Something has to choose, and
in a control room today that something is an officer with a telephone.

This module makes the registry do it. Given where a vehicle has been seen,
it nominates the cameras it is about to reach -- ranked, with an arrival
window, and never more than the number of streams actually affordable.

Three things decide a nomination:

  * Is the camera ahead?   Bearing from the last sighting, against the
                           direction of travel. A camera behind the vehicle
                           is not worth a stream.
  * Can it be reached?     Distance over measured ground speed, inside the
                           horizon the officer asked for.
  * What can it do there?  A plate-capable camera can confirm identity; a
                           presence-only camera can only corroborate. Both
                           are useful and they are not worth the same.

What this is not
----------------
It is not a prediction of where a vehicle will go. It is a ranked list of
where it *could* be seen next, which is a smaller and more defensible
claim. Roads bend, vehicles turn off, and drivers stop for tea.

The distances here are straight-line. A road route is always longer than
the great-circle line, so every arrival estimate is optimistic and every
reachable set is generous -- the error is deliberately in the direction of
watching a camera slightly too early rather than missing the vehicle. With
a road graph this becomes a routing problem and the same ranking applies;
that is a deployment upgrade, not a redesign, and it is stated in the
design document rather than implied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

# What each capability class is worth as a place to look next. A camera
# that can read a plate can end the question; one that can only see a shape
# can strengthen a route but never close it.
CONFIRM_VALUE = {
    "plate-capable": 1.00,
    "plate-occasional": 0.72,
    "plate-marginal": 0.72,
    "presence-only": 0.45,
    "no-vehicles-observed": 0.12,
    "unusable": 0.0,
}
DEFAULT_VALUE = 0.30              # unmeasured: worth something, not much

# A vehicle that has not been seen for longer than this is no longer
# usefully predictable from its last heading.
STALE_AFTER_S = 900.0

# Ground speed is measured between two sightings and then trusted. These
# bound it: a stationary reading makes every camera reachable, and an
# impossible one makes the whole grid reachable.
MIN_SPEED_KMH = 8.0
MAX_SPEED_KMH = 120.0
ASSUMED_SPEED_KMH = 35.0          # urban default when speed is unknowable


def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a_lat, a_lon, b_lat, b_lon) -> float:
    """Compass bearing from a to b, degrees clockwise from north."""
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dl = math.radians(b_lon - a_lon)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_between(a: float, b: float) -> float:
    """Smallest angle between two bearings, 0-180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class Sighting:
    """Where the vehicle was seen. Only what the ranking needs."""
    camera_id: str
    lat: float
    lon: float
    at_s: float                       # seconds on a single corrected clock


@dataclass
class Nomination:
    camera_id: str
    name: str = ""
    department: str = ""
    capability: str = "unknown"
    km: float = 0.0
    bearing: float = 0.0
    off_course: float = 0.0           # degrees away from direction of travel
    eta_s: float = 0.0
    score: float = 0.0
    why: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id, "name": self.name,
            "department": self.department, "capability": self.capability,
            "km": round(self.km, 2), "bearing": round(self.bearing),
            "off_course": round(self.off_course), "eta_s": round(self.eta_s),
            "score": round(self.score, 3), "why": list(self.why),
        }


@dataclass
class LookAhead:
    """The watch list, and the reasoning that produced it."""
    nominations: List[Nomination]
    heading: Optional[float]
    speed_kmh: float
    speed_basis: str
    from_camera: Optional[str]
    horizon_s: float
    budget: int
    considered: int
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "watch": [n.as_dict() for n in self.nominations],
            "heading": None if self.heading is None else round(self.heading),
            "speed_kmh": round(self.speed_kmh, 1),
            "speed_basis": self.speed_basis,
            "from_camera": self.from_camera,
            "horizon_s": self.horizon_s,
            "budget": self.budget,
            "considered": self.considered,
            "note": self.note,
        }


def _speed_from(sightings: Sequence[Sighting]) -> tuple[float, str]:
    """
    Ground speed between the last two sightings, bounded.

    Two cameras metres apart with overlapping views produce a nonsense
    speed -- that is the same co-observation the impossible-transit flag
    exists to catch -- so anything outside the bounds falls back to an
    urban default and says so rather than quietly using it.
    """
    if len(sightings) < 2:
        return ASSUMED_SPEED_KMH, "no second sighting; urban default"
    a, b = sightings[-2], sightings[-1]
    dt = b.at_s - a.at_s
    if dt <= 0:
        return ASSUMED_SPEED_KMH, "sightings not ordered in time; urban default"
    kmh = haversine_km(a.lat, a.lon, b.lat, b.lon) / (dt / 3600.0)
    if kmh < MIN_SPEED_KMH:
        return ASSUMED_SPEED_KMH, f"measured {kmh:.0f} km/h is too slow to project; urban default"
    if kmh > MAX_SPEED_KMH:
        return ASSUMED_SPEED_KMH, f"measured {kmh:.0f} km/h is not a road speed; urban default"
    return kmh, f"measured between {a.camera_id} and {b.camera_id}"


def _heading_from(sightings: Sequence[Sighting]) -> Optional[float]:
    """Direction of travel, or None when only one sighting exists."""
    for i in range(len(sightings) - 1, 0, -1):
        a, b = sightings[i - 1], sightings[i]
        if haversine_km(a.lat, a.lon, b.lat, b.lon) > 0.02:   # 20m of travel
            return bearing_deg(a.lat, a.lon, b.lat, b.lon)
    return None


def nominate(sightings: Sequence[Sighting],
             cameras: Iterable[dict],
             *,
             budget: int = 6,
             horizon_s: float = 600.0,
             now_s: Optional[float] = None,
             cone_deg: float = 75.0) -> LookAhead:
    """
    Rank the cameras worth opening next.

    `cameras` are registry rows: id, lat, lon, capability_class, and
    optionally location_name and department. Cameras the vehicle has
    already been seen on are excluded -- the question is where it goes, not
    where it has been.

    `budget` is the number of streams that can actually be opened at once.
    It is a hard limit, not a preference: the list is truncated to it, and
    what was dropped is reported rather than silently discarded.
    """
    sightings = [s for s in sightings if s.lat is not None and s.lon is not None]
    if not sightings:
        return LookAhead([], None, 0.0, "no sightings", None,
                         horizon_s, budget, 0,
                         "Nothing has been seen yet, so there is nowhere to look next.")

    sightings = sorted(sightings, key=lambda s: s.at_s)
    last = sightings[-1]
    now_s = last.at_s if now_s is None else now_s
    age = max(0.0, now_s - last.at_s)

    speed_kmh, basis = _speed_from(sightings)
    heading = _heading_from(sightings)
    seen = {s.camera_id for s in sightings}

    note = ""
    if age > STALE_AFTER_S:
        note = (f"Last seen {age/60:.0f} minutes ago. A heading that old does "
                f"not constrain much; treat this as a ring, not a direction.")
        heading = None

    # How far the vehicle could have gone by the end of the horizon. The
    # lower bound accounts for the time already elapsed since it was seen.
    reach_km = speed_kmh * ((age + horizon_s) / 3600.0)

    out: List[Nomination] = []
    considered = 0
    for cam in cameras:
        cid = cam.get("id")
        lat, lon = cam.get("lat"), cam.get("lon")
        if cid is None or lat is None or lon is None or cid in seen:
            continue
        considered += 1

        km = haversine_km(last.lat, last.lon, lat, lon)
        if km > reach_km:
            continue

        eta_s = max(0.0, (km / max(speed_kmh, 1e-6)) * 3600.0 - age)
        if eta_s > horizon_s:
            continue

        brg = bearing_deg(last.lat, last.lon, lat, lon)
        off = 0.0 if heading is None else angle_between(heading, brg)
        if heading is not None and off > cone_deg:
            continue

        cap = cam.get("capability_class") or "unknown"
        value = CONFIRM_VALUE.get(cap, DEFAULT_VALUE)

        # Ahead beats beside; soon beats later; a camera that can confirm
        # beats one that can only corroborate.
        course_fit = 1.0 if heading is None else max(0.0, 1.0 - (off / cone_deg) ** 1.6)
        time_fit = 1.0 - min(1.0, eta_s / horizon_s) * 0.7
        score = (0.35 + 0.65 * course_fit) * time_fit * (0.25 + 0.75 * value)

        why = []
        if heading is None:
            why.append("no heading yet, so every direction is in play")
        elif off < 20:
            why.append(f"directly ahead, {off:.0f}° off course")
        else:
            why.append(f"{off:.0f}° off the direction of travel")
        why.append(f"{km:.2f} km away, about {eta_s/60:.1f} min at {speed_kmh:.0f} km/h")
        if cap == "plate-capable":
            why.append("plate-capable: can confirm identity outright")
        elif cap in ("plate-occasional", "plate-marginal"):
            why.append("reads a plate sometimes: may confirm, may only corroborate")
        elif cap == "presence-only":
            why.append("presence-only: can corroborate the route, never confirm it")
        elif cap == "no-vehicles-observed":
            why.append("no vehicles seen here when surveyed; watched only if nothing better")

        out.append(Nomination(
            camera_id=cid, name=cam.get("location_name") or "",
            department=cam.get("department") or "", capability=cap,
            km=km, bearing=brg, off_course=off, eta_s=eta_s,
            score=score, why=why))

    out.sort(key=lambda n: -n.score)
    dropped = max(0, len(out) - budget)
    kept = out[:budget]

    if not note:
        if not out:
            note = ("Nothing on this grid is reachable inside the horizon. "
                    "Widen the horizon, or the vehicle has left the covered area.")
        elif dropped:
            note = (f"{dropped} further camera(s) are reachable but outside the "
                    f"budget of {budget} open streams.")
        else:
            note = f"All {len(out)} reachable camera(s) fit inside the budget."

    return LookAhead(nominations=kept, heading=heading, speed_kmh=speed_kmh,
                     speed_basis=basis, from_camera=last.camera_id,
                     horizon_s=horizon_s, budget=budget,
                     considered=considered, note=note)
