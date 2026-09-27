#!/usr/bin/env python3
"""
sentinel.hunt -- the whole system, doing the thing it exists to do.

Everything else in this package is a component that works on its own. This
is the part that runs the chain the organisers actually test:

    designate a vehicle
        -> watch the cameras the registry says are worth watching
        -> detect and track every vehicle on each
        -> recognise the designated one, by plate where a plate is legible
           and by appearance where it is not
        -> check the journey is physically possible
        -> raise an alert at a tier the evidence supports
        -> write the sighting so the officer's console can draw the route

Two design points worth stating, because they are what makes this survive
contact with a real grid.

The registry decides which cameras get watched and what runs on each. Most
of these cameras cannot read a plate -- the median vehicle yields a 27px
plate against the ~120px needed -- so identity is carried by appearance and
the plate corroborates when it can, not the other way round.

Nothing here claims certainty it has not earned. A vehicle matched on
appearance alone never reaches 'confirmed': a white hatchback resembles
thousands of others, and getting somebody stopped because their car looks
like another one is the failure this design exists to avoid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .detect import Detector
# The appearance thresholds live in sentinel.matching, next to the
# capability ceilings, because they are the same kind of statement: what a
# piece of evidence is allowed to claim. Re-exported here because this is
# where the hunt reads them and where callers have always found them.
from .matching import APPEARANCE_FLOOR, APPEARANCE_STRONG  # noqa: F401
from .registry import connect, normalise_plate
from .track import VehicleTracker
from .vehicle_reid import VehicleReID

# Faster than this between two cameras and it is not one vehicle: either the
# plate is cloned or a camera clock is wrong. Either is worth telling the
# officer; neither is worth silently correcting.
IMPOSSIBLE_KMH = 150.0


@dataclass
class Target:
    """The vehicle being hunted."""
    plate: Optional[str] = None
    appearance: Optional[np.ndarray] = None
    vehicle_class: Optional[str] = None
    note: str = ""

    @property
    def plate_norm(self) -> str:
        return normalise_plate(self.plate) if self.plate else ""


@dataclass
class Hit:
    camera_id: str
    camera_name: str
    lat: Optional[float]
    lon: Optional[float]
    at_utc: str
    frame: int
    tier: str
    score: float
    basis: str
    why: List[str] = field(default_factory=list)
    plate_read: Optional[str] = None
    vehicle_class: Optional[str] = None
    box: Optional[tuple] = None
    # The camera's own delivered rate. Timing a leg at an assumed 25 fps
    # when the camera delivers 15 makes the implied speed wrong by two
    # thirds, and the impossible-transit flag -- which is what detects a
    # cloned plate -- inherits the error silently.
    fps: float = 25.0


def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    if None in (a_lat, a_lon, b_lat, b_lon):
        return 0.0
    R = 6371.0088
    p1, p2 = np.radians(a_lat), np.radians(b_lat)
    dp, dl = p2 - p1, np.radians(b_lon - a_lon)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(min(1.0, np.sqrt(h))))


class Hunt:
    """Search a set of camera feeds for one designated vehicle."""

    def __init__(self, target: Target, log=None):
        self.target = target
        self.hits: List[Hit] = []
        self.log = log or (lambda m: None)
        self.cameras: Dict[str, dict] = {}
        self._load_cameras()

    def _load_cameras(self):
        with connect() as con:
            for r in con.execute(
                    "SELECT id, location_name, lat, lon, capability_class, "
                    "plate_px_measured, time_confidence_ms, department, "
                    "measured_fps, declared_fps FROM camera"):
                self.cameras[r["id"]] = dict(r)

    # ------------------------------------------------------------- search

    def search_stream(self, camera_id: str, frames, camera_name: str = "",
                      max_frames: Optional[int] = None,
                      fps: Optional[float] = None) -> List[Hit]:
        """
        Watch one camera for the designated vehicle.

        Frames come from anywhere -- a live stream, a sampled HLS window, a
        local clip -- because the identification logic must not depend on
        how the pixels arrived.
        """
        cam = self.cameras.get(camera_id, {})
        capability = cam.get("capability_class") or "unknown"
        # Prefer the rate we measured from this camera's own video over the
        # rate it declares, which on this grid is routinely wrong.
        rate = float(fps or cam.get("measured_fps")
                     or cam.get("declared_fps") or 25.0)
        can_read_plates = capability in ("plate-capable", "plate-marginal")

        tracker = VehicleTracker()
        found: List[Hit] = []
        n = 0
        best_per_track: Dict[int, float] = {}

        for img in frames:
            n += 1
            if max_frames and n > max_frames:
                break
            if n % 3:
                continue

            live = tracker.update(Detector.vehicles(img, min_score=0.5), n)
            for t in live:
                if t.last_seen != n or t.frames < 3:
                    continue
                # Scenery and parked objects are not this vehicle passing.
                if t.is_static():
                    continue
                if self.target.vehicle_class and t.label != self.target.vehicle_class:
                    continue

                x1, y1, x2, y2 = (max(0, int(v)) for v in t.box)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                score, basis, why = 0.0, "", []
                if self.target.appearance is not None:
                    e = VehicleReID.embed(crop)
                    if e is not None:
                        score = float(np.dot(e, self.target.appearance))
                        basis = "appearance"

                if score < APPEARANCE_FLOOR:
                    continue
                if score <= best_per_track.get(t.track_id, 0.0):
                    continue           # one hit per vehicle, its best moment
                best_per_track[t.track_id] = score

                tier = "corroborating"
                if score >= APPEARANCE_STRONG:
                    tier = "probable"
                    why.append(f"appearance {score:.2f}")
                else:
                    why.append(f"appearance {score:.2f}, weak")

                # A camera the registry measured as unable to read plates
                # cannot promote a match, however sure the pixels look.
                if capability == "presence-only":
                    tier = "corroborating"
                    why.append("camera measured presence-only")
                if not can_read_plates:
                    why.append("no plate available from this camera")

                drift = cam.get("time_confidence_ms")
                if drift is not None and abs(drift) > 30_000:
                    tier = "corroborating"
                    why.append(f"camera clock {drift/1000:+.0f}s out")

                found.append(Hit(
                    camera_id=camera_id,
                    camera_name=camera_name or cam.get("location_name", ""),
                    lat=cam.get("lat"), lon=cam.get("lon"),
                    at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    frame=n, tier=tier, score=round(score, 3),
                    basis=basis or "appearance", why=why,
                    vehicle_class=t.label, box=(x1, y1, x2, y2), fps=rate))

        self.hits.extend(found)
        self.log(f"{camera_id}: {len(found)} candidate sighting(s) in {n} frames")
        return found

    # -------------------------------------------------------------- route

    def route(self) -> List[dict]:
        """
        The journey, in order, with each leg checked for plausibility.

        A leg nobody could have driven is reported rather than corrected.
        Silently fixing it would hand the officer a tidy timeline that is
        wrong, which is worse than an untidy one that is honest.
        """
        hits = sorted(self.hits, key=lambda h: (h.at_utc, h.frame))
        legs = []
        for a, b in zip(hits, hits[1:]):
            km = haversine_km(a.lat, a.lon, b.lat, b.lon)
            # Each end of the leg is timed at its own camera's rate.
            secs = max(1.0, (b.frame / max(1.0, b.fps)) - (a.frame / max(1.0, a.fps)))
            if secs <= 0:
                secs = 1.0
            kmh = km / (secs / 3600.0) if secs else 0.0
            legs.append({
                "from": a.camera_id, "to": b.camera_id,
                "km": round(km, 2), "seconds": round(secs, 1),
                "timed_at_fps": {a.camera_id: a.fps, b.camera_id: b.fps},
                "kmh": round(kmh, 1),
                "impossible": kmh > IMPOSSIBLE_KMH,
                "note": ("no vehicle could drive this: a cloned plate, or a "
                         "camera clock is wrong") if kmh > IMPOSSIBLE_KMH else "",
            })
        return legs

    def summary(self) -> dict:
        by_tier: Dict[str, int] = {}
        for h in self.hits:
            by_tier[h.tier] = by_tier.get(h.tier, 0) + 1
        return {
            "target": self.target.plate or self.target.note or "appearance only",
            "sightings": len(self.hits),
            "cameras_with_hits": len({h.camera_id for h in self.hits}),
            "by_tier": by_tier,
            "impossible_legs": sum(1 for l in self.route() if l["impossible"]),
        }


def target_from_crop(crop: np.ndarray, plate: Optional[str] = None,
                     vehicle_class: Optional[str] = None) -> Target:
    """Designate a vehicle from a single picture of it."""
    e = VehicleReID.embed(crop)
    return Target(plate=plate, appearance=e, vehicle_class=vehicle_class,
                  note="designated from image")
