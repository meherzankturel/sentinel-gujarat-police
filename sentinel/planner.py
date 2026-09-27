#!/usr/bin/env python3
"""
sentinel.planner -- what would have to change for this camera to do the job.

Measuring a grid and reporting that most of it cannot read a plate is only
half an answer. A department cannot act on "no". It can act on "this camera
is three times short; a longer lens fixes it, and here is the one next to
it that needs re-siting instead."

So the registry's measurements are turned back into engineering changes.
Every analytic in this system has a pixels-on-target threshold -- 120 px
across a number plate, 40 px between the eyes -- and every camera has a
measured delivery. The ratio between them is the whole plan.

The physics, and why the output is a ratio and not a part number
----------------------------------------------------------------
Pixels on a target scale as

    px  ~  (object size x focal length x sensor pixels) / (sensor width x distance)

so pixels are linear in focal length, linear in horizontal resolution, and
inverse in distance. A camera three times short therefore needs three times
the focal length, or a third of the distance, or three times the horizontal
pixel count -- or any product of the three that reaches three.

What this system does NOT know is the current focal length, the mounting
height or the distance to the carriageway. The catalogue supplies none of
it, and on 16 September it stopped supplying even the resolution. Inventing
"fit a 12 mm lens" from a measurement that contains no millimetres would be
the same class of error as believing a specification sheet.

So the output is stated as a multiplier against what the camera does now,
which is exactly what the measurement supports. A surveyor standing under
the pole converts "3x the focal length" into a part number in about a
minute, and can check our arithmetic while they do it.

One consequence worth stating: a multiplier below about 2.5 is usually a
lens or an aim. Above 5 it is a new camera in a new place, because you
cannot buy your way out of a camera pointed at the wrong thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .facecap import FACE_CAPABLE_IOD, FACE_MARGINAL_IOD, iod_px

# Pixels across a number plate for a reliable read. The same figure the
# capability survey classifies against.
PLATE_READ_PX = 120.0

# Where a multiplier stops being a lens and starts being a building site.
OPTICS_ONLY = 2.5        # a longer lens, or aiming at a nearer lane
RESITE = 5.0             # move the camera, or put a second one closer


@dataclass
class Change:
    """One way to close the gap. Not exclusive -- these combine."""
    lever: str
    detail: str


@dataclass
class Shortfall:
    camera_id: str
    capability: str                 # "plate" | "face"
    location: str = ""
    department: str = ""
    have_px: Optional[float] = None
    need_px: float = 0.0
    factor: Optional[float] = None  # how many times short
    verdict: str = "unknown"        # met | optics | resite | replace | not-applicable | unknown
    options: List[Change] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id, "capability": self.capability,
            "location": self.location, "department": self.department,
            "have_px": None if self.have_px is None else round(self.have_px, 1),
            "need_px": round(self.need_px, 1),
            "factor": None if self.factor is None else round(self.factor, 2),
            "verdict": self.verdict,
            "options": [{"lever": c.lever, "detail": c.detail} for c in self.options],
            "note": self.note,
        }


def _verdict_for(factor: float) -> str:
    if factor <= 1.0:
        return "met"
    if factor <= OPTICS_ONLY:
        return "optics"
    if factor <= RESITE:
        return "resite"
    return "replace"


def _options_for(factor: float, width_px: Optional[int]) -> List[Change]:
    """
    The three levers, each expressed against what the camera does today.

    Resolution is listed last and deliberately unflattering: it is the lever
    procurement reaches for first, and it is the worst value of the three,
    because the pixel count goes up as the square of the gain while the
    thing you want goes up linearly.
    """
    n = math.ceil(factor * 10) / 10.0
    out = [
        Change("lens", f"{n:g}x the focal length, same mounting point"),
        Change("distance", f"aim at a lane {n:g}x closer, or move the camera "
                           f"to {1/n:.0%} of its present distance"),
    ]
    if width_px:
        need_w = int(math.ceil(width_px * n / 160) * 160)
        out.append(Change(
            "sensor",
            f"{n:g}x the horizontal resolution: {width_px}px -> ~{need_w}px "
            f"({n*n:.0f}x the pixel count for {n:g}x the detail -- the "
            f"weakest of the three levers)"))
    return out


def plate_shortfall(cam: dict) -> Shortfall:
    """What would have to change for this camera to read a number plate."""
    s = Shortfall(camera_id=cam.get("id", ""), capability="plate",
                  location=cam.get("location_name") or "",
                  department=cam.get("department") or "",
                  need_px=PLATE_READ_PX)
    have = cam.get("plate_px_measured")
    if cam.get("capability_class") == "no-vehicles-observed":
        s.verdict = "not-applicable"
        s.note = ("No vehicles were observed here when surveyed. Plate "
                  "reading is not the right ask of this camera; it is not "
                  "failing at it.")
        return s
    if not have:
        s.note = "Not measured yet. Survey it before planning a change."
        return s

    s.have_px = float(have)
    s.factor = PLATE_READ_PX / s.have_px
    s.verdict = _verdict_for(s.factor)
    if s.verdict == "met":
        s.note = (f"Already reads plates at {s.have_px:.0f}px. No change "
                  f"needed.")
        return s
    s.options = _options_for(s.factor, cam.get("width"))
    s.note = (f"{s.have_px:.0f}px of plate against the {PLATE_READ_PX:.0f}px "
              f"a read needs -- {s.factor:.1f}x short.")
    if s.verdict == "replace":
        s.note += (" At this margin a lens will not reach it: this is a new "
                   "camera, closer to the traffic.")
    return s


def face_shortfall(cam: dict, *, target: float = FACE_MARGINAL_IOD) -> Shortfall:
    """
    What would have to change for this camera to support face matching.

    `target` defaults to the floor for attempting a match at all, not to the
    threshold for doing it well. Planning against the higher figure is the
    honest default for a new installation, and is offered by the tool.
    """
    s = Shortfall(camera_id=cam.get("id", ""), capability="face",
                  location=cam.get("location_name") or "",
                  department=cam.get("department") or "",
                  need_px=target)
    have = cam.get("face_iod_px")
    if have is None and cam.get("body_px_p90"):
        have = iod_px(cam["body_px_p90"])
    if cam.get("face_class") == "no-people-observed":
        s.verdict = "not-applicable"
        s.note = ("No people of measurable size were seen here. This camera "
                  "watches a carriageway; face matching is not the right ask "
                  "of it.")
        return s
    if not have:
        s.note = "Not measured yet. Survey it before planning a change."
        return s

    s.have_px = float(have)
    s.factor = target / s.have_px
    s.verdict = _verdict_for(s.factor)
    if s.verdict == "met":
        s.note = f"Already resolves {s.have_px:.1f}px between the eyes."
        return s
    s.options = _options_for(s.factor, cam.get("width"))
    s.note = (f"{s.have_px:.1f}px between the eyes against the "
              f"{target:.0f}px floor -- {s.factor:.1f}x short.")
    if s.verdict in ("resite", "replace"):
        # For faces the answer above the optics band is almost always about
        # WHERE the camera is, not what is bolted to it. A traffic pole sees
        # the tops of heads at an angle no lens corrects, so "resite" and
        # "replace" get the same advice: put it where people walk past at
        # head height.
        s.note += (" Face matching is not reachable from this mounting point. "
                   "It needs a camera at head height on a walking route -- an "
                   "entrance, a ticket hall, a footbridge -- not a traffic "
                   "pole.")
    return s


def plan(cameras: Iterable[dict], *,
         face_target: float = FACE_MARGINAL_IOD) -> List[Shortfall]:
    """Every camera, both capabilities, cheapest fix first."""
    out: List[Shortfall] = []
    for cam in cameras:
        out.append(plate_shortfall(cam))
        out.append(face_shortfall(cam, target=face_target))
    order = {"optics": 0, "resite": 1, "replace": 2,
             "met": 3, "not-applicable": 4, "unknown": 5}
    out.sort(key=lambda s: (order.get(s.verdict, 9), s.factor or 1e9))
    return out


def summarise(shortfalls: List[Shortfall]) -> dict:
    """The counts a department would put in a budget line."""
    by = {}
    for s in shortfalls:
        by.setdefault(s.capability, {}).setdefault(s.verdict, 0)
        by[s.capability][s.verdict] += 1
    return by
