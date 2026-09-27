#!/usr/bin/env python3
"""
sentinel.reid -- follow a vehicle across cameras that cannot read its plate.

The organisers' live test case is "track a designated vehicle" across ~50
heterogeneous cameras. We measured their grid (audit/grid_survey.json) and
on most of it a number plate occupies 12-27 pixels of width. Reliable OCR
needs about 120. So on the majority of their cameras the plate is not
merely hard to read, it is not present in the signal, and any submission
whose tracking story is "we run ANPR everywhere" will stop dead the moment
the vehicle leaves the two or three cameras where a plate is legible.

The vehicle therefore has to be followed by what those cameras CAN see:
colour, class, proportion, and the plausibility of the journey between one
camera and the next. That is what this module does.

Two design positions worth defending to the panel:

1. The descriptor is explainable, not learned. A deep re-identification
   embedding would probably score a little better on a benchmark, and it
   would be useless in an evidence bundle: an officer cannot testify that
   "cosine distance 0.31" means two sightings are the same van. Every
   component here can be rendered in the UI -- the colour histogram is a
   bar chart, the spatial layout is a 2x3 swatch grid, the transit check
   is a speed in km/h. A defence lawyer can attack each one on its merits,
   which is the correct state of affairs.

2. Appearance alone is never allowed to reach 'confirmed'. A white
   hatchback looks like ten thousand other white hatchbacks in Ahmedabad.
   The tiers follow sentinel.matching: appearance on its own caps at
   'probable', and the top tier requires corroboration -- a partial plate
   fragment that agrees, or a tight transit fit on a corridor with no exit
   between the two cameras. Underclaiming here is the point; the next
   round is live in front of the State Crime Records Bureau.

Physics is used as a filter, not a hint. If matching a sighting at camera
A to one at camera B would require 300 km/h, that is not a weak match to
be ranked lower. It is a different vehicle, or a cloned plate, and the
system says so.

Licensing: OpenCV (Apache 2.0) and NumPy (BSD) only. Nothing AGPL, which
in a state procurement is a blocker rather than a footnote.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .matching import TIER_ORDER, _demote

# ---------------------------------------------------------------- constants

# Hue is quantised into 12 bins of 15 degrees (OpenCV packs 0-360 into
# 0-179). Finer bins do not survive a change of camera: white balance
# alone moves a measured hue by several degrees between two vendors'
# sensors pointed at the same car.
HUE_BINS = 12

# Below this saturation the hue value is numerical noise -- a grey pixel
# has no meaningful colour, and letting it vote puts a random spike in the
# histogram. Achromatic pixels are counted separately as black/grey/white,
# which matters because silver, white and black are the three commonest
# vehicle colours in India and they are precisely the ones a hue histogram
# cannot represent.
ACHROMATIC_SAT = 60
BLACK_V, GREY_V = 70, 175

# Every vehicle has glass, tyres and wheel-arch shadow, and all of it lands
# in the darkest bin. Left at full weight that bin gave a white van and a
# green truck a third of their colour mass in common -- a similarity floor
# under every pair on the road, which is measurable: it put the mean
# different-vehicle score at 0.34 with nothing but glass responsible. The
# bin is kept, because a genuinely black car is mostly this bin and we must
# still be able to say so, but it is discounted to reflect how little
# identity it carries.
DARK_BIN_WEIGHT = 0.45

# Hue bins are centred on the named colours rather than starting at zero, so
# that pure blue and pure green fall in the middle of "blue" and "green"
# instead of straddling a boundary. This is purely for the officer-facing
# label; matching is unaffected because neighbouring bins share mass anyway.
HUE_BIN_OFFSET = 7.5

# Shadow and specular highlight carry the illuminant's colour, not the
# vehicle's. A windscreen reflection is the sky; a wheel arch in shadow is
# almost black whatever the car is painted. Both are excluded outright.
SHADOW_V = 30
SPECULAR_V = 245

# Below this the descriptor is not trustworthy and is capped at
# 'corroborating' regardless of how well it scores. On a camera where the
# vehicle is 30 px wide, the "colour" being measured is mostly the codec's
# opinion of the colour.
MIN_BODY_WIDTH_PX = 40
MIN_BODY_PIXELS = 200

# The body region of a vehicle box. The lower third is wheels, shadow and
# road surface -- on every camera that is dark grey, so including it makes
# every vehicle in Gujarat look 30% more similar than it is. The side and
# top insets drop the box edges, where the detector's rectangle always
# includes some background.
BODY_TOP, BODY_BOTTOM = 0.06, 0.67
BODY_INSET_X = 0.08

# The spatial layout grid. 2x3 is chosen so a cell is still large enough to
# average meaningfully on a 60 px-wide vehicle; a 4x4 grid on that box is
# measuring individual macroblocks. It exists to separate a white van with
# a red stripe from a plain white van, and that is a coarse difference.
LAYOUT_ROWS, LAYOUT_COLS = 2, 3

# Detectors confuse these classes on CCTV geometry, so class disagreement is
# scored rather than treated as a veto. An SUV read as 'truck' on one camera
# and 'car' on the next is routine; a motorcycle read as a bus is not, and
# that pair is what the low numbers are for.
_CLASS_PAIRS = {
    ("car", "truck"): 0.45,
    ("car", "bus"): 0.15,
    ("truck", "bus"): 0.55,
    ("motorcycle", "bicycle"): 0.65,
    ("car", "motorcycle"): 0.05,
    ("car", "bicycle"): 0.05,
    ("truck", "motorcycle"): 0.0,
    ("truck", "bicycle"): 0.0,
    ("bus", "motorcycle"): 0.0,
    ("bus", "bicycle"): 0.0,
}

# Component weights. Colour dominates because it is the only component that
# survives a large change of viewing angle; aspect ratio is nearly worthless
# across cameras (a car seen head-on and in profile has different
# proportions) and is kept at a low weight only as a tie-breaker.
W_COLOUR, W_LAYOUT, W_ASPECT = 0.58, 0.30, 0.12

# A class mismatch scales the whole score rather than zeroing it, so that
# 'car' vs 'truck' on the same red vehicle still ranks above a white car.
CLASS_GATE_FLOOR = 0.25

# Transit plausibility. 140 km/h is above Gujarat's 120 km/h expressway
# limit with headroom for GPS-grade slop in the camera coordinates; a
# candidate requiring more than this is rejected outright rather than
# ranked low, because it is physically a different vehicle.
MAX_PLAUSIBLE_KMH = 140.0
FREE_FLOW_KMH = 80.0
CONGESTED_KMH = 15.0

# A vehicle that stopped for an hour is still the same vehicle, so a long
# gap must never reject a candidate -- it only stops the timing from being
# evidence for the match. Hence a floor rather than a decay to zero.
SLOW_FLOOR = 0.30
SLOW_DECAY_S = 1800.0

# Clock error is a real property of this grid: cam-08 runs +428 s. Slack of
# that size is added to the transit window so a bad clock cannot manufacture
# an "impossible" transit and throw away the true match -- the worst error
# this module could make.
BASE_TIME_SLACK_S = 3.0

# Above this much combined clock uncertainty, a transit time is too vague to
# corroborate anything, even on a single-route corridor.
TIGHT_FIT_MAX_SLACK_S = 30.0

COLOUR_NAMES = [
    "red", "orange", "yellow", "lime", "green", "teal",
    "cyan", "azure", "blue", "violet", "purple", "pink",
]
ACHROMATIC_NAMES = ["black", "silver/grey", "white"]
BIN_NAMES = COLOUR_NAMES + ACHROMATIC_NAMES


# ---------------------------------------------------------------- signature


@dataclass
class VehicleSignature:
    """
    A vehicle described in terms that survive a change of camera.

    Everything in here is meant to be shown to an officer. `colour_bins` is
    a bar chart, `layout` is a 2x3 grid of swatches, `dominant_colour` is a
    word. The one field deliberately excluded from cross-camera matching is
    `rel_area`: how large a vehicle appears depends on how far back the
    camera is mounted, which is exactly the variable the registry exists to
    record, so using it to compare two cameras would be measuring the
    mounting and calling it the vehicle.
    """

    vehicle_class: str
    colour_bins: List[float]          # 12 hue bins + black/grey/white
    layout: List[List[float]]         # per cell: hue_cos, hue_sin, sat, val
    aspect: float                     # box width / height
    rel_area: float                   # box area / frame area (NOT matched on)
    box_width_px: int
    pixels_used: int
    body_pixels: int
    camera_id: str = ""

    @property
    def dominant_colour(self) -> str:
        return BIN_NAMES[int(np.argmax(self.colour_bins))]

    @property
    def dominant_share(self) -> float:
        return round(float(max(self.colour_bins)), 3)

    @property
    def reliable(self) -> bool:
        """
        Whether this descriptor is worth believing at all.

        Two ways to fail: too few pixels on the vehicle to have a colour, or
        too few surviving the shadow/specular cut, which happens on a
        night camera where the whole vehicle is below SHADOW_V.
        """
        return (self.box_width_px >= MIN_BODY_WIDTH_PX
                and self.pixels_used >= MIN_BODY_PIXELS)

    def why_unreliable(self) -> Optional[str]:
        if self.box_width_px < MIN_BODY_WIDTH_PX:
            return (f"vehicle only {self.box_width_px}px wide "
                    f"(need {MIN_BODY_WIDTH_PX}px for a colour estimate)")
        if self.pixels_used < MIN_BODY_PIXELS:
            return (f"only {self.pixels_used} usable body pixels after "
                    "removing shadow and highlight")
        return None

    def describe(self) -> str:
        """One line an officer can read in a sighting list."""
        bits = [f"{self.dominant_colour} {self.vehicle_class}",
                f"{int(self.dominant_share * 100)}% of body",
                f"aspect {self.aspect:.2f}"]
        if not self.reliable:
            bits.append(f"LOW CONFIDENCE: {self.why_unreliable()}")
        return ", ".join(bits)

    def to_dict(self) -> dict:
        return {
            "vehicle_class": self.vehicle_class,
            "colour_bins": [round(v, 4) for v in self.colour_bins],
            "bin_names": BIN_NAMES,
            "layout": [[round(v, 4) for v in cell] for cell in self.layout],
            "aspect": round(self.aspect, 3),
            "rel_area": round(self.rel_area, 5),
            "box_width_px": self.box_width_px,
            "pixels_used": self.pixels_used,
            "body_pixels": self.body_pixels,
            "camera_id": self.camera_id,
            "dominant_colour": self.dominant_colour,
            "reliable": self.reliable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleSignature":
        return cls(
            vehicle_class=d["vehicle_class"],
            colour_bins=list(d["colour_bins"]),
            layout=[list(c) for c in d["layout"]],
            aspect=float(d["aspect"]),
            rel_area=float(d.get("rel_area", 0.0)),
            box_width_px=int(d.get("box_width_px", 0)),
            pixels_used=int(d.get("pixels_used", 0)),
            body_pixels=int(d.get("body_pixels", 0)),
            camera_id=d.get("camera_id", ""),
        )


def _body_region(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    y0, y1 = int(h * BODY_TOP), max(int(h * BODY_TOP) + 1, int(h * BODY_BOTTOM))
    x0, x1 = int(w * BODY_INSET_X), max(int(w * BODY_INSET_X) + 1,
                                        int(w * (1.0 - BODY_INSET_X)))
    return crop[y0:y1, x0:x1]


def _usable_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Pixels whose colour describes the paint rather than the weather.

    An absolute cut first, because "too dark to have a colour" is a property
    of the sensor and not of this particular vehicle. Then a percentile trim
    on what is left, which removes the residual shadow gradient down one
    flank without also removing a genuinely black car -- a fixed threshold
    alone would delete a black car entirely and then report it as white.
    """
    v = hsv[:, :, 2]
    m = (v >= SHADOW_V) & (v <= SPECULAR_V)
    kept = v[m]
    if kept.size >= MIN_BODY_PIXELS * 2:
        lo, hi = np.percentile(kept, [8, 92])
        if hi > lo:
            m &= (v >= lo) & (v <= hi)
    return m


def _smooth_colour_bins(bins: np.ndarray) -> np.ndarray:
    """
    Let neighbouring bins share mass.

    Without this, a red car whose measured hue sits on the boundary between
    bin 11 and bin 0 scores near zero against the same car photographed by a
    camera with slightly warmer white balance. The hue bins wrap because hue
    is circular; the achromatic bins do not, because white is not adjacent
    to black.
    """
    hue = bins[:HUE_BINS]
    smoothed_hue = (0.25 * np.roll(hue, 1) + 0.5 * hue + 0.25 * np.roll(hue, -1))
    ach = bins[HUE_BINS:]
    smoothed_ach = ach.copy()
    smoothed_ach[0] = 0.75 * ach[0] + 0.25 * ach[1]
    smoothed_ach[1] = 0.2 * ach[0] + 0.6 * ach[1] + 0.2 * ach[2]
    smoothed_ach[2] = 0.25 * ach[1] + 0.75 * ach[2]
    out = np.concatenate([smoothed_hue, smoothed_ach])
    out[HUE_BINS] *= DARK_BIN_WEIGHT
    s = out.sum()
    return out / s if s > 0 else out


def signature_from_crop(crop: np.ndarray, vehicle_class: str, *,
                        camera_id: str = "", box_width_px: Optional[int] = None,
                        frame_area: Optional[int] = None,
                        box_area: Optional[int] = None) -> VehicleSignature:
    """
    Build the descriptor from a cropped vehicle box.

    Kept separate from the frame-level entry point so the same code runs on
    a live detection, on a stored evidence crop, and in the tests -- a
    descriptor that behaves differently depending on how the pixels arrived
    is not evidence.
    """
    if crop is None or crop.size == 0:
        raise ValueError("empty crop")
    h, w = crop.shape[:2]
    body = _body_region(crop)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    mask = _usable_mask(hsv)

    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bins = np.zeros(HUE_BINS + 3, np.float64)

    sel = mask
    if not sel.any():
        # Everything was shadow or highlight. Fall back to the raw region so
        # a signature still exists; `reliable` will be False and the tiering
        # refuses to let it carry an alert.
        sel = np.ones_like(mask)

    hs, sats, vals = hh[sel], ss[sel], vv[sel]
    chroma = sats >= ACHROMATIC_SAT
    if chroma.any():
        centred = (hs[chroma].astype(np.float64) + HUE_BIN_OFFSET) % 180.0
        idx = np.clip((centred * HUE_BINS / 180.0).astype(np.int32),
                      0, HUE_BINS - 1)
        np.add.at(bins, idx, 1.0)
    ach_v = vals[~chroma]
    if ach_v.size:
        bins[HUE_BINS + 0] += float((ach_v < BLACK_V).sum())
        bins[HUE_BINS + 1] += float(((ach_v >= BLACK_V) & (ach_v < GREY_V)).sum())
        bins[HUE_BINS + 2] += float((ach_v >= GREY_V).sum())

    total = bins.sum()
    if total > 0:
        bins /= total
    bins = _smooth_colour_bins(bins)

    layout = _layout_cells(hsv, mask)

    return VehicleSignature(
        vehicle_class=vehicle_class,
        colour_bins=[float(x) for x in bins],
        layout=layout,
        aspect=float(w) / float(h) if h else 0.0,
        rel_area=(float(box_area if box_area is not None else w * h)
                  / float(frame_area)) if frame_area else 0.0,
        box_width_px=int(box_width_px if box_width_px is not None else w),
        pixels_used=int(mask.sum()),
        body_pixels=int(mask.size),
        camera_id=camera_id,
    )


def _layout_cells(hsv: np.ndarray, mask: np.ndarray) -> List[List[float]]:
    """
    Coarse spatial colour layout.

    Hue is stored as a unit vector rather than an angle so that averaging
    across the red wrap-around does not produce cyan, and it is weighted by
    saturation so a cell of grey bodywork does not assert a colour. Value is
    normalised against the region's own mean, because absolute brightness is
    the first thing to change between two cameras and the thing we least
    want the layout to encode -- what we want it to encode is "this cell is
    darker than the rest of the vehicle", which is a stripe or a window.
    """
    h, w = hsv.shape[:2]
    ref_v = float(hsv[:, :, 2][mask].mean()) if mask.any() else 1.0
    ref_v = max(ref_v, 1.0)
    cells: List[List[float]] = []
    for r in range(LAYOUT_ROWS):
        for c in range(LAYOUT_COLS):
            y0, y1 = h * r // LAYOUT_ROWS, max(h * (r + 1) // LAYOUT_ROWS, 1)
            x0, x1 = w * c // LAYOUT_COLS, max(w * (c + 1) // LAYOUT_COLS, 1)
            cell = hsv[y0:y1, x0:x1]
            cm = mask[y0:y1, x0:x1]
            if cell.size == 0 or not cm.any():
                cells.append([0.0, 0.0, 0.0, 0.0])
                continue
            ch = cell[:, :, 0][cm].astype(np.float64) * (2 * math.pi / 180.0)
            cs = cell[:, :, 1][cm].astype(np.float64) / 255.0
            cv_ = cell[:, :, 2][cm].astype(np.float64)
            wsum = cs.sum()
            if wsum > 1e-6:
                hue_cos = float((np.cos(ch) * cs).sum() / wsum)
                hue_sin = float((np.sin(ch) * cs).sum() / wsum)
            else:
                hue_cos = hue_sin = 0.0
            sat = float(cs.mean())
            val = float(np.clip(cv_.mean() / ref_v, 0.0, 2.0) / 2.0)
            # Chroma is scaled by mean saturation so a desaturated cell
            # contributes a short vector rather than a confident wrong hue.
            cells.append([hue_cos * sat, hue_sin * sat, sat, val])
    return cells


def signature(frame: np.ndarray, box: Sequence[float], vehicle_class: str,
              camera_id: str = "") -> VehicleSignature:
    """Build a descriptor for one sentinel.detect.Detection box in a frame."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("degenerate box")
    return signature_from_crop(
        frame[y1:y2, x1:x2], vehicle_class, camera_id=camera_id,
        box_width_px=x2 - x1, frame_area=w * h, box_area=(x2 - x1) * (y2 - y1))


# --------------------------------------------------------------- similarity


@dataclass
class SimilarityBreakdown:
    """
    Why the system thinks two sightings are the same vehicle, itemised.

    The breakdown is the deliverable, not a debugging aid. An evidence
    bundle that says "0.71" is not evidence; one that says "colour histogram
    overlaps 84%, both read as car, layout agrees 0.66, proportions differ
    by 12%" can be examined and disputed.
    """

    colour: float
    layout: float
    aspect: float
    vehicle_class: float
    overall: float
    reliable: bool
    notes: List[str] = field(default_factory=list)

    def why(self) -> str:
        bits = [f"colour {self.colour:.2f}", f"layout {self.layout:.2f}",
                f"class {self.vehicle_class:.2f}", f"shape {self.aspect:.2f}"]
        return "; ".join(bits + self.notes)

    def to_dict(self) -> dict:
        return {"colour": round(self.colour, 3), "layout": round(self.layout, 3),
                "aspect": round(self.aspect, 3),
                "vehicle_class": round(self.vehicle_class, 3),
                "overall": round(self.overall, 3), "reliable": self.reliable,
                "notes": list(self.notes)}


def class_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return _CLASS_PAIRS.get((a, b), _CLASS_PAIRS.get((b, a), 0.0))


def colour_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Histogram intersection: the share of colour mass the two vehicles have
    in common. Chosen over chi-square or Bhattacharyya because it is the one
    a human can be told in words -- "84% of the body colour matches" is a
    sentence an officer can repeat in court.
    """
    return float(np.minimum(np.asarray(a), np.asarray(b)).sum())


def layout_similarity(a: Sequence[Sequence[float]],
                      b: Sequence[Sequence[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    tot = 0.0
    for ca, cb in zip(a, b):
        # Chroma carries most of the weight; brightness the least, because
        # brightness is what differs most between two cameras looking at the
        # same car and least between two different cars of the same colour.
        d_chroma = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
        d_sat = abs(ca[2] - cb[2])
        d_val = abs(ca[3] - cb[3])
        d = 0.60 * min(d_chroma, 1.0) + 0.25 * d_sat + 0.15 * d_val
        tot += max(0.0, 1.0 - d)
    return tot / len(a)


def similarity(a: VehicleSignature, b: VehicleSignature) -> SimilarityBreakdown:
    """
    How alike two vehicle sightings look, 0..1, with the reasoning attached.

    Relative size is not a term. Two cameras at different mounting distances
    render the same vehicle at very different sizes, so including it would
    penalise exactly the cross-camera case this module exists to handle.
    """
    col = colour_similarity(a.colour_bins, b.colour_bins)
    lay = layout_similarity(a.layout, b.layout)
    cls = class_affinity(a.vehicle_class, b.vehicle_class)

    ratio = (min(a.aspect, b.aspect) / max(a.aspect, b.aspect)
             if a.aspect > 0 and b.aspect > 0 else 0.0)
    asp = max(0.0, (ratio - 0.5) / 0.5)      # 2:1 disagreement scores zero

    appearance = W_COLOUR * col + W_LAYOUT * lay + W_ASPECT * asp
    gate = CLASS_GATE_FLOOR + (1.0 - CLASS_GATE_FLOOR) * cls
    overall = appearance * gate

    notes: List[str] = []
    reliable = a.reliable and b.reliable
    for s in (a, b):
        if not s.reliable:
            notes.append(f"{s.camera_id or 'sighting'}: {s.why_unreliable()}")
    if a.vehicle_class != b.vehicle_class:
        notes.append(f"detected as {a.vehicle_class} and {b.vehicle_class}")

    return SimilarityBreakdown(colour=col, layout=lay, aspect=asp,
                               vehicle_class=cls, overall=float(overall),
                               reliable=reliable, notes=notes)


# ------------------------------------------------------------- plausibility


@dataclass
class CameraSite:
    """The registry facts this module needs about a camera."""
    id: str
    lat: Optional[float]
    lon: Optional[float]
    location_name: str = ""
    capability_class: str = "unknown"
    time_confidence_ms: float = 0.0
    time_confidence_note: str = ""

    def clock_slack_s(self) -> float:
        """
        How wrong this camera's clock might be, in seconds.

        Taken from the measured drift rather than assumed to be zero: cam-08
        on our own grid runs 428 s fast. Treating its timestamps as exact
        would make a perfectly ordinary journey look like teleportation.
        """
        return abs(float(self.time_confidence_ms or 0.0)) / 1000.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from the first camera to the second, degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def direction_consistency(a: CameraSite, b: CameraSite,
                          heading_deg: Optional[float]) -> Tuple[float, str]:
    """
    Was the vehicle even pointed at the next camera when it was last seen?

    A weak signal used weakly. A vehicle driving away from camera B can still
    reach camera B by turning round at the next junction, so a contrary
    heading discounts a candidate rather than refusing it -- the mistake to
    avoid is being clever about direction and losing a vehicle that simply
    took a different road.

    heading_deg is a true compass heading, which means it needs the camera's
    installed orientation to convert an on-screen motion vector into one.
    Until the registry records that per camera this stays None on real
    sightings and the term contributes nothing, by design rather than by
    accident.
    """
    if heading_deg is None or a.lat is None or a.lon is None \
            or b.lat is None or b.lon is None:
        return 1.0, ""
    if haversine_km(a.lat, a.lon, b.lat, b.lon) < 0.2:
        return 1.0, ""
    br = bearing_deg(a.lat, a.lon, b.lat, b.lon)
    off = abs((heading_deg - br + 180.0) % 360.0 - 180.0)
    if off <= 60.0:
        return 1.0, f"was heading toward that camera ({off:.0f}deg off bearing)"
    if off <= 120.0:
        return 0.82, f"was heading across, not toward, that camera ({off:.0f}deg off)"
    return 0.55, (f"was heading away from that camera ({off:.0f}deg off the "
                  "bearing); possible, but it would have had to turn round")


@dataclass
class TransitCheck:
    """The physics half of the decision, in units an officer recognises."""
    distance_km: float
    elapsed_s: float
    effective_s: float             # elapsed, widened by clock uncertainty
    implied_kmh: Optional[float]
    plausible: bool
    score: float                   # 0..1, how much the timing SUPPORTS the match
    reason: str

    def to_dict(self) -> dict:
        return {"distance_km": round(self.distance_km, 3),
                "elapsed_s": round(self.elapsed_s, 1),
                "implied_kmh": (round(self.implied_kmh, 1)
                                if self.implied_kmh is not None else None),
                "plausible": self.plausible, "score": round(self.score, 3),
                "reason": self.reason}


def transit_check(a: CameraSite, b: CameraSite, elapsed_s: float,
                  extra_slack_s: float = 0.0) -> TransitCheck:
    """
    Could a vehicle have got from camera A to camera B in this time?

    Asymmetric on purpose. Too fast is a hard rejection -- no road vehicle
    covers 40 km in four minutes, so the two sightings are different
    vehicles or the plate is cloned, and either way pretending otherwise
    puts a false route in front of an officer. Too slow is not a rejection
    at all, because a vehicle is allowed to park; it merely means the timing
    contributes nothing and the match has to stand on appearance alone.
    """
    elapsed_s = abs(float(elapsed_s))
    if a.lat is None or a.lon is None or b.lat is None or b.lon is None:
        return TransitCheck(0.0, elapsed_s, elapsed_s, None, True, 0.5,
                            "camera coordinates unknown; transit not checked")

    dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
    slack = a.clock_slack_s() + b.clock_slack_s() + BASE_TIME_SLACK_S + extra_slack_s
    # The generous end of the interval is used for the impossibility test so
    # that an untrusted clock can never be the thing that discards a true
    # match. Clock error should widen doubt, not create certainty.
    effective = elapsed_s + slack

    if dist < 0.02:
        return TransitCheck(dist, elapsed_s, effective, None, True, 1.0,
                            "cameras are effectively co-located")

    if effective <= 0:
        return TransitCheck(dist, elapsed_s, effective, None, False, 0.0,
                            f"{dist:.2f} km apart but sighted at the same instant")

    implied = dist / (effective / 3600.0)
    if implied > MAX_PLAUSIBLE_KMH:
        return TransitCheck(
            dist, elapsed_s, effective, implied, False, 0.0,
            f"would require {implied:.0f} km/h over {dist:.2f} km "
            f"(ceiling {MAX_PLAUSIBLE_KMH:.0f}) -- different vehicle or cloned plate")

    t_fast = dist / FREE_FLOW_KMH * 3600.0
    t_slow = dist / CONGESTED_KMH * 3600.0
    if effective < t_fast:
        # Legal but brisk: still supportive, just not the typical corridor
        # transit, so it is not given full weight.
        frac = (effective - dist / MAX_PLAUSIBLE_KMH * 3600.0) / max(
            1e-6, t_fast - dist / MAX_PLAUSIBLE_KMH * 3600.0)
        score = 0.5 + 0.5 * max(0.0, min(1.0, frac))
        reason = f"{implied:.0f} km/h over {dist:.2f} km -- fast but possible"
    elif effective <= t_slow:
        score = 1.0
        reason = (f"{implied:.0f} km/h over {dist:.2f} km -- "
                  "consistent with normal traffic")
    else:
        over = effective - t_slow
        score = SLOW_FLOOR + (1.0 - SLOW_FLOOR) * math.exp(-over / SLOW_DECAY_S)
        reason = (f"{elapsed_s / 60.0:.0f} min for {dist:.2f} km -- possible, "
                  "but the vehicle could have stopped anywhere; timing is "
                  "not evidence here")
    return TransitCheck(dist, elapsed_s, effective, implied, True, score, reason)


# ------------------------------------------------------------- corroboration


@dataclass
class Sighting:
    """One vehicle seen once, with everything needed to reason about it."""
    sighting_id: str
    camera_id: str
    at_epoch_s: float                    # clock-corrected where possible
    signature: VehicleSignature
    plate_fragment: Optional[str] = None     # may be partial, e.g. "GJ01AB"
    plate_confidence: float = 0.0
    heading_deg: Optional[float] = None      # true compass heading, if known
    frame_ref: Optional[str] = None


def plate_agreement(a: Optional[str], b: Optional[str]) -> Tuple[int, int]:
    """
    Characters two plate fragments agree and disagree on.

    Positional, because a partial read from a marginal camera is usually a
    contiguous run of the plate with the rest illegible rather than a
    scramble. '?' and '_' mark an unread position. Disagreements matter more
    than agreements: one solid contradiction is enough to say "not this
    vehicle" even when the appearance is a perfect match, and that is the
    cheapest true negative the system can buy.
    """
    if not a or not b:
        return 0, 0
    a, b = a.upper().strip(), b.upper().strip()
    same = diff = 0
    for ca, cb in zip(a, b):
        if ca in "?_ " or cb in "?_ ":
            continue
        if ca == cb:
            same += 1
        else:
            diff += 1
    return same, diff


# A pair of cameras between which there is no exit from the road. On such a
# corridor a tight transit time is genuine evidence rather than a
# coincidence, because the vehicle had nowhere else to be. This is a
# surveyed fact about the road network, not something the software can
# infer, so it is supplied rather than computed.
Corridor = Dict[str, object]


def single_route(corridor: Optional[Corridor], a: str, b: str) -> bool:
    if not corridor:
        return False
    pairs = corridor.get("single_route_pairs") or []
    return any({a, b} == set(p) for p in pairs)


@dataclass
class Corroboration:
    """Evidence beyond appearance, which is the only route to 'confirmed'."""
    plate_chars_agreeing: int = 0
    plate_chars_conflicting: int = 0
    tight_fit_on_single_route: bool = False

    def contradicts(self) -> bool:
        return self.plate_chars_conflicting > 0

    def supports(self) -> bool:
        # Four characters is roughly one field of an Indian plate (the state
        # and district code, or the four-digit serial). Fewer than that
        # matches too many vehicles in a district to be worth the top tier.
        return (self.plate_chars_agreeing >= 4
                or self.tight_fit_on_single_route)

    def why(self) -> str:
        bits = []
        if self.plate_chars_agreeing:
            bits.append(f"{self.plate_chars_agreeing} plate characters agree")
        if self.plate_chars_conflicting:
            bits.append(f"{self.plate_chars_conflicting} plate characters conflict")
        if self.tight_fit_on_single_route:
            bits.append("transit time fits a corridor with no exit between "
                        "the two cameras")
        return "; ".join(bits) or "no corroborating evidence"


# --------------------------------------------------------------------- tiers

# These are operating points read off a measured curve, not guesses. On
# 6,369 vehicle detections from three government clips (media/real), scoring
# 199k same-vehicle pairs against 120k pairs that are certainly different:
#
#     threshold   same-vehicle recall   false-match rate
#       0.62            0.70                 0.173
#       0.70            0.61                 0.104
#       0.80            0.55                 0.039
#       0.90            0.36                 0.005
#
# The first row is where this module started, and 17% is far too loose for a
# tier that is allowed to notify a control room -- that is alert fatigue with
# extra steps. 0.80 buys a 4% false-match rate for a third of the recall,
# which is the right trade when the cost of a false alert is an officer who
# stops reading them. The floor is where a match is worth SHOWING next to an
# existing track; it was never a threshold for believing one.
APPEARANCE_FLOOR = 0.55
APPEARANCE_PROBABLE = 0.80
APPEARANCE_CONFIRMED = 0.85


@dataclass
class ReidMatch:
    """A ranked candidate, with the whole argument for and against it."""
    candidate: Sighting
    appearance: SimilarityBreakdown
    transit: TransitCheck
    corroboration: Corroboration
    tier: str
    score: float
    why: str

    def notifies(self) -> bool:
        """Same contract as sentinel.matching.Match."""
        return self.tier in ("confirmed", "probable")

    def to_dict(self) -> dict:
        return {"sighting_id": self.candidate.sighting_id,
                "camera_id": self.candidate.camera_id,
                "at_epoch_s": self.candidate.at_epoch_s,
                "appearance": self.appearance.to_dict(),
                "transit": self.transit.to_dict(),
                "corroboration": self.corroboration.why(),
                "tier": self.tier, "score": round(self.score, 3),
                "why": self.why}


def assess(query: Sighting, cand: Sighting, sites: Mapping[str, CameraSite],
           corridor: Optional[Corridor] = None) -> ReidMatch:
    """Score and tier one candidate against the query."""
    app = similarity(query.signature, cand.signature)

    qs = sites.get(query.camera_id) or CameraSite(query.camera_id, None, None)
    cs = sites.get(cand.camera_id) or CameraSite(cand.camera_id, None, None)
    tr = transit_check(qs, cs, cand.at_epoch_s - query.at_epoch_s)

    same, diff = plate_agreement(query.plate_fragment, cand.plate_fragment)
    # A corridor time-fit is only evidence if the two clocks can be trusted
    # to the precision the argument rests on. cam-08 runs 428 s fast, which
    # is longer than the transit it would be corroborating -- so on that
    # camera the tight fit is an artefact of the clock, not a fact about the
    # vehicle, and it must not be allowed to reach 'confirmed'.
    clocks_tight = (qs.clock_slack_s() + cs.clock_slack_s()) <= TIGHT_FIT_MAX_SLACK_S
    corr = Corroboration(
        plate_chars_agreeing=same, plate_chars_conflicting=diff,
        tight_fit_on_single_route=(tr.score >= 0.85 and tr.plausible
                                   and clocks_tight
                                   and single_route(corridor, query.camera_id,
                                                    cand.camera_id)))

    dir_factor, dir_note = direction_consistency(qs, cs, query.heading_deg)

    # Timing modulates rather than dominates: a match that looks right and
    # is merely uninformative on timing should still outrank one that looks
    # wrong, so the transit term is folded in as a multiplier with a floor.
    score = app.overall * (0.55 + 0.45 * tr.score) * dir_factor

    why: List[str] = [app.why(), tr.reason]
    if dir_note:
        why.append(dir_note)
    tier = "corroborating"

    if not tr.plausible:
        why.append("rejected on transit time")
    elif corr.contradicts():
        # A plate that disagrees beats any amount of appearance similarity.
        # This is the one place the system is allowed to be certain, and it
        # is certain in the negative direction.
        why.append("plate fragments conflict -- not the same vehicle")
        score = 0.0
    elif not app.reliable:
        # Cameras too weak to describe a vehicle may corroborate a track
        # that already exists. They may never start one.
        why.append("descriptor below the reliability floor; corroboration only")
    elif app.overall >= APPEARANCE_CONFIRMED and corr.supports() and tr.score >= 0.6:
        tier = "confirmed"
        why.append(f"appearance plus corroboration ({corr.why()})")
    elif app.overall >= APPEARANCE_PROBABLE:
        tier = "probable"
        why.append("appearance alone -- a colour and class match is not an "
                   "identification, and cannot be confirmed without a plate")
    elif app.overall >= APPEARANCE_FLOOR:
        why.append("weak appearance match; corroboration only")
    else:
        why.append("appearance does not match")

    return ReidMatch(candidate=cand, appearance=app, transit=tr,
                     corroboration=corr, tier=tier, score=float(score),
                     why="; ".join(why))


@dataclass
class ReidResult:
    query: Sighting
    ranked: List[ReidMatch]
    implausible: List[ReidMatch]

    def best(self) -> Optional[ReidMatch]:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict:
        return {"query": {"sighting_id": self.query.sighting_id,
                          "camera_id": self.query.camera_id,
                          "signature": self.query.signature.to_dict()},
                "ranked": [m.to_dict() for m in self.ranked],
                "implausible": [m.to_dict() for m in self.implausible]}


def match_across_cameras(query: Sighting, candidates: Iterable[Sighting],
                         sites: Mapping[str, CameraSite], *,
                         corridor: Optional[Corridor] = None,
                         min_score: float = 0.0,
                         top_k: int = 25,
                         include_same_camera: bool = False) -> ReidResult:
    """
    Rank sightings from other cameras as possible re-appearances of the query.

    Candidates failing the transit check are not merely ranked last, they are
    moved out of the ranking entirely -- but they are still returned, because
    "this vehicle appears identical and could not possibly have made the
    journey" is an investigative lead in its own right and the officer should
    see it rather than have it silently dropped.
    """
    ranked: List[ReidMatch] = []
    impossible: List[ReidMatch] = []

    for c in candidates:
        if c.sighting_id == query.sighting_id:
            continue
        if not include_same_camera and c.camera_id == query.camera_id:
            continue
        m = assess(query, c, sites, corridor)
        if not m.transit.plausible:
            impossible.append(m)
        elif m.score >= min_score:
            ranked.append(m)

    ranked.sort(key=lambda m: (-TIER_ORDER.index(m.tier), -m.score))
    impossible.sort(key=lambda m: -m.appearance.overall)
    return ReidResult(query=query, ranked=ranked[:top_k],
                      implausible=impossible[:top_k])


def demote_for_clock(m: ReidMatch, note: str) -> ReidMatch:
    """
    Drop a match a tier because a camera's clock cannot be trusted.

    Kept as an explicit step rather than folded into assess() so that the
    officer's screen can show the demotion as its own line: the match was
    good, the clock was not, and here is which camera.
    """
    m.tier = _demote(m.tier)
    m.why += f"; demoted: {note}"
    return m


# ------------------------------------------------------------------ registry


def load_sites(con: sqlite3.Connection) -> Dict[str, CameraSite]:
    """Camera geometry and clock quality, straight from the registry."""
    con.row_factory = sqlite3.Row
    out: Dict[str, CameraSite] = {}
    for r in con.execute(
            "SELECT id, lat, lon, location_name, capability_class, "
            "time_confidence_ms, time_confidence_note FROM camera"):
        out[r["id"]] = CameraSite(
            id=r["id"], lat=r["lat"], lon=r["lon"],
            location_name=r["location_name"] or "",
            capability_class=r["capability_class"] or "unknown",
            time_confidence_ms=r["time_confidence_ms"] or 0.0,
            time_confidence_note=r["time_confidence_note"] or "")
    return out


# ------------------------------------------------------------------ measuring
#
# Everything below this line exists to put a number on the claims above.
# It is run, not asserted:  python -m sentinel.reid --help


def _bench_report(name: str, pairs_same: List[float], pairs_diff: List[float],
                  rank1: Optional[Tuple[int, int]] = None,
                  false_at: Optional[Tuple[int, int, float]] = None) -> dict:
    def stat(xs):
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 3),
                "p10": round(float(np.percentile(a, 10)), 3),
                "p50": round(float(np.percentile(a, 50)), 3),
                "p90": round(float(np.percentile(a, 90)), 3)}

    rep = {"bench": name, "same_vehicle": stat(pairs_same),
           "different_vehicle": stat(pairs_diff)}
    if rank1:
        hit, tot = rank1
        rep["rank1"] = {"correct": hit, "queries": tot,
                        "accuracy": round(hit / tot, 3) if tot else None}
    if false_at:
        fp, tot, thr = false_at
        rep["false_match_rate"] = {"threshold": thr, "false_matches": fp,
                                   "known_different_pairs": tot,
                                   "rate": round(fp / tot, 3) if tot else None}
    return rep


def _photometric(img: np.ndarray, gain: float, gamma: float,
                 wb: Tuple[float, float, float], blur: float,
                 noise: float, jpeg: int) -> np.ndarray:
    """
    Put one image through what a different camera would do to it.

    Gain, gamma, white balance, optics and codec: the four things that
    actually differ between two vendors' cameras pointed at the same road.
    Applied as a controlled transform so the descriptor's invariance can be
    measured rather than hoped for.
    """
    f = img.astype(np.float32) / 255.0
    f = np.clip(f * gain, 0, 1) ** gamma
    f[:, :, 0] *= wb[0]
    f[:, :, 1] *= wb[1]
    f[:, :, 2] *= wb[2]
    out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    if blur > 0:
        k = int(blur) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), blur)
    if noise > 0:
        out = np.clip(out.astype(np.int16)
                      + np.random.randn(*out.shape).astype(np.int16)
                      * int(noise), 0, 255).astype(np.uint8)
    if jpeg:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


# Eight cameras' worth of photometric character, spanning what the real grid
# survey showed: a bright well-exposed bridge camera through to a dim, soft,
# heavily compressed legacy unit.
BENCH_CAMERAS = [
    ("cam-A", 1.00, 1.00, (1.00, 1.00, 1.00), 0.0, 0, 0),
    ("cam-B", 0.82, 1.15, (1.10, 1.00, 0.92), 1.0, 3, 80),
    ("cam-C", 1.25, 0.88, (0.90, 1.00, 1.12), 0.0, 5, 70),
    ("cam-D", 0.65, 1.30, (1.05, 0.98, 0.95), 2.0, 8, 55),
    ("cam-E", 1.10, 0.95, (0.95, 1.05, 1.00), 1.0, 4, 65),
    ("cam-F", 0.75, 1.20, (1.15, 0.95, 0.88), 3.0, 10, 45),
    ("cam-G", 0.95, 1.05, (1.00, 1.02, 1.02), 0.0, 2, 85),
    ("cam-H", 0.55, 1.35, (1.08, 0.96, 0.90), 2.0, 12, 50),
]


def _draw_bench_vehicle(w: int, h: int, body_bgr, roof_bgr, stripe_bgr,
                        cls: str) -> np.ndarray:
    """
    A vehicle with an actual colour, which the local grid does not provide.

    grid/publish.py renders every vehicle on every camera with one hardcoded
    body colour, so it cannot exercise an appearance descriptor at all. This
    stands in only for the controlled-lighting benchmark; the numbers that
    matter come from real footage.
    """
    img = np.full((h, w, 3), 60, np.uint8)
    img[:, :] = (58, 58, 62)
    cv2.rectangle(img, (0, int(h * 0.30)), (w, int(h * 0.88)), body_bgr, -1)
    roof_h = int(h * 0.30) if cls != "truck" else int(h * 0.46)
    cv2.rectangle(img, (int(w * 0.22), int(h * 0.04)),
                  (int(w * 0.78), roof_h), roof_bgr, -1)
    cv2.rectangle(img, (int(w * 0.26), int(h * 0.10)),
                  (int(w * 0.74), int(roof_h * 0.92)), (52, 58, 64), -1)
    if stripe_bgr is not None:
        cv2.rectangle(img, (0, int(h * 0.56)), (w, int(h * 0.66)), stripe_bgr, -1)
    cv2.circle(img, (int(w * 0.24), int(h * 0.90)), max(2, int(h * 0.10)),
               (28, 28, 30), -1)
    cv2.circle(img, (int(w * 0.76), int(h * 0.90)), max(2, int(h * 0.10)),
               (28, 28, 30), -1)
    return img


# Colours weighted towards what is actually on an Indian road: white and
# silver dominate, which is also the hardest case for appearance re-ID and
# therefore the one the benchmark must not be allowed to dodge.
BENCH_FLEET = [
    ("white hatchback",  "car",        (236, 238, 238), (228, 230, 230), None),
    ("white sedan",      "car",        (240, 242, 240), (232, 234, 232), None),
    ("silver hatchback", "car",        (176, 178, 180), (168, 170, 172), None),
    ("silver sedan",     "car",        (168, 172, 176), (160, 164, 168), None),
    ("black sedan",      "car",        (38, 38, 40),    (30, 30, 32),    None),
    ("red hatchback",    "car",        (36, 34, 196),   (30, 28, 172),   None),
    ("blue sedan",       "car",        (168, 84, 34),   (150, 74, 30),   None),
    ("white van+stripe", "truck",      (238, 240, 238), (230, 232, 230), (40, 40, 200)),
    ("plain white van",  "truck",      (238, 240, 238), (230, 232, 230), None),
    ("yellow taxi",      "car",        (40, 196, 232),  (36, 176, 210),  None),
    ("green truck",      "truck",      (72, 132, 66),   (64, 118, 60),   None),
    ("red bus",          "bus",        (40, 42, 190),   (36, 38, 168),   None),
]


def bench_controlled(trials: int = 3, seed: int = 7) -> dict:
    """
    Discrimination and invariance under a controlled change of camera.

    This is the bench that answers "does the descriptor work at all". It is
    still synthetic and its numbers are an upper bound, not a forecast: real
    vehicles have specular paint, real cameras see them from different
    angles, and neither is modelled here.
    """
    rng = np.random.default_rng(seed)
    sigs: Dict[str, List[VehicleSignature]] = {}
    for name, cls, body, roof, stripe in BENCH_FLEET:
        sigs[name] = []
        for cam, gain, gamma, wb, blur, noise, jpeg in BENCH_CAMERAS:
            for _ in range(trials):
                # Size varies per camera because mounting distance does; the
                # descriptor must not be secretly keying on scale.
                w = int(rng.integers(70, 200))
                h = int(w * float(rng.uniform(0.40, 0.52)))
                img = _draw_bench_vehicle(w, h, body, roof, stripe, cls)
                img = _photometric(img, gain, gamma, wb, blur, noise, jpeg)
                sigs[name].append(signature_from_crop(
                    img, cls, camera_id=cam, box_width_px=w,
                    frame_area=1920 * 1080, box_area=w * h))

    same, diff = [], []
    names = list(sigs)
    for n in names:
        pool = sigs[n]
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if pool[i].camera_id != pool[j].camera_id:
                    same.append(similarity(pool[i], pool[j]).overall)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            for a in sigs[n1]:
                for b in sigs[n2]:
                    if a.camera_id != b.camera_id:
                        diff.append(similarity(a, b).overall)

    # Rank-1: each sighting queries the gallery of every other sighting from
    # a different camera. Correct if the top hit is the same vehicle.
    flat = [(n, s) for n in names for s in sigs[n]]
    hit = tot = 0
    for qn, qs in flat:
        gallery = [(n, s) for n, s in flat if s.camera_id != qs.camera_id]
        if not gallery:
            continue
        best = max(gallery, key=lambda t: similarity(qs, t[1]).overall)
        tot += 1
        hit += int(best[0] == qn)

    thr = APPEARANCE_PROBABLE
    fp = sum(1 for d in diff if d >= thr)
    rep = _bench_report("controlled lighting change (synthetic)", same, diff,
                        rank1=(hit, tot), false_at=(fp, len(diff), thr))
    rep["note"] = ("synthetic vehicles under measured photometric transforms; "
                   "no viewpoint change, no specular paint -- treat as a "
                   "ceiling, not a prediction")
    return rep


def _grid_boxes(img: np.ndarray, bgs) -> List[Tuple[int, int, int, int]]:
    """
    Vehicle boxes on the local grid, by background subtraction.

    sentinel.detect returns nothing at all on this grid -- its vehicles are
    drawn rectangles and a COCO-trained detector correctly declines to call
    them cars, at any threshold down to 0.05. Motion is the only way to get
    a box here, which is itself a finding: the local grid cannot stand in
    for real footage in any measurement that depends on a real detector.
    """
    mask = bgs.apply(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = img.shape[0] * img.shape[1]
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < area * 0.004 or w * h > area * 0.6 or w < h * 0.9:
            continue
        out.append((x, y, w, h))
    return out


def bench_grid(seconds: float = 46.0, cams: Optional[Sequence[str]] = None,
               host: str = "127.0.0.1") -> dict:
    """
    The local synthetic grid, where the ground truth is known.

    cam-01, cam-02, cam-06, cam-04 and cam-08 show plate GJ01AB1234;
    cam-03, cam-05 and cam-07 show different vehicles. Whether that ground
    truth is *reachable* by an appearance descriptor is a separate question,
    and the answer this bench returns.
    """
    from .stream import LiveStream

    cams = list(cams or [f"cam-0{i}" for i in range(1, 9)])
    truth = {"cam-01": "GJ01AB1234", "cam-02": "GJ01AB1234",
             "cam-06": "GJ01AB1234", "cam-04": "GJ01AB1234",
             "cam-08": "GJ01AB1234", "cam-03": "GJ18XY4477",
             "cam-05": "GJ05MN8899", "cam-07": "GJ18XY4477"}

    per_cam: Dict[str, List[VehicleSignature]] = {}
    # Frames delivered is recorded alongside observations because "no
    # vehicles seen" and "no frames arrived" look identical in a count of
    # zero, and they mean completely different things -- one is a statement
    # about the descriptor, the other about whether the camera was up.
    delivered: Dict[str, int] = {}
    for cam in cams:
        bgs = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32,
                                                 detectShadows=False)
        got: List[VehicleSignature] = []
        n = 0
        with LiveStream(f"rtsp://{host}:8554/stream/{cam}", cam) as s:
            for f in s.frames(max_seconds=seconds):
                n += 1
                boxes = _grid_boxes(f.image, bgs)
                if n < 25:
                    continue
                for (x, y, w, h) in boxes:
                    crop = f.image[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue
                    got.append(signature_from_crop(
                        crop, "car", camera_id=cam, box_width_px=w,
                        frame_area=f.image.shape[0] * f.image.shape[1],
                        box_area=w * h))
        per_cam[cam] = got
        delivered[cam] = n
        print(f"  {cam}: {n} frames delivered, {len(got)} vehicle observations",
              file=sys.stderr, flush=True)

    flat = [(truth.get(c, "?"), s) for c, ss in per_cam.items() for s in ss]
    same = [similarity(a, b).overall
            for i, (pa, a) in enumerate(flat) for pb, b in flat[i + 1:]
            if pa == pb and a.camera_id != b.camera_id]
    diff = [similarity(a, b).overall
            for i, (pa, a) in enumerate(flat) for pb, b in flat[i + 1:]
            if pa != pb and a.camera_id != b.camera_id]

    hit = tot = 0
    unanswerable = 0
    for qp, qs in flat:
        gal = [(p, s) for p, s in flat if s.camera_id != qs.camera_id]
        if not gal:
            continue
        # A vehicle that appears on only one camera has no right answer in a
        # cross-camera gallery. Counting those as failures would be scoring
        # the descriptor for the grid's decoys being single-sighting, so they
        # are excluded and reported separately.
        if not any(p == qp for p, _ in gal):
            unanswerable += 1
            continue
        best = max(gal, key=lambda t: similarity(qs, t[1]).overall)
        tot += 1
        hit += int(best[0] == qp)

    thr = APPEARANCE_PROBABLE
    fp = sum(1 for d in diff if d >= thr)
    rep = _bench_report("local synthetic grid", same, diff,
                        rank1=(hit, tot), false_at=(fp, len(diff), thr))
    rep["note"] = ("grid/publish.py draws every vehicle on every camera with "
                   "one hardcoded body colour (128,96,72) and one set of "
                   "proportions, so there is no appearance difference between "
                   "the target vehicle and the decoys. These numbers measure "
                   "the grid, not the descriptor.")
    rep["queries_with_no_possible_answer"] = unanswerable
    rep["per_camera"] = {k: {"frames_delivered": delivered.get(k, 0),
                             "observations": len(v)}
                         for k, v in per_cam.items()}
    dead = [k for k, v in delivered.items() if v == 0]
    if dead:
        rep["cameras_delivering_no_frames"] = dead
    return rep


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Obs:
    frame: int
    clip: str
    track: int
    sig: VehicleSignature
    box: Tuple[float, float, float, float]


def _track_clip(cap, clip: str, max_frames: int, stride: int,
                iou_thr: float, max_gap_frames: int,
                detect_scale: float = 0.5) -> List[_Obs]:
    """
    Identity ground truth that real footage can actually supply.

    There are no labels on government CCTV, so identity is recovered by
    linking detections through the scene on overlap alone. It is a weak
    tracker on purpose -- linking on appearance would make the benchmark
    circular, grading the descriptor with the descriptor. Overlap is
    independent of everything reid.py computes, so a track is evidence
    about identity that the thing under test did not get a vote in.
    """
    from .detect import Detection, Detector

    obs: List[_Obs] = []
    active: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
    next_id, idx = 0, 0
    while idx < max_frames:
        ok, img = cap.read()
        if not ok or img is None:
            break
        idx += 1
        if idx % stride:
            continue
        # Detection runs on a downscaled frame and the boxes are scaled back
        # up, but the descriptor is always cut from the full-resolution
        # pixels. Detection only needs to find the vehicle; the colour
        # measurement needs every pixel it can get.
        if detect_scale != 1.0:
            small = cv2.resize(img, None, fx=detect_scale, fy=detect_scale)
            dets = [Detection(d.label, d.score,
                              tuple(v / detect_scale for v in d.box))
                    for d in Detector.vehicles(small, min_score=0.55)]
        else:
            dets = Detector.vehicles(img, min_score=0.55)
        active = [t for t in active if idx - t[1] <= max_gap_frames]
        used = set()
        for d in dets:
            best, best_iou = None, iou_thr
            for k, (tid, last_f, box) in enumerate(active):
                if k in used:
                    continue
                v = _iou(d.box, box)
                if v > best_iou:
                    best, best_iou = k, v
            if best is None:
                tid = next_id
                next_id += 1
                active.append((tid, idx, tuple(d.box)))
            else:
                tid = active[best][0]
                active[best] = (tid, idx, tuple(d.box))
                used.add(best)
            if d.width < MIN_BODY_WIDTH_PX:
                continue
            try:
                obs.append(_Obs(idx, clip, tid,
                                signature(img, d.box, d.label, clip),
                                tuple(d.box)))
            except ValueError:
                continue
    return obs


def _rank1(queries, gallery) -> Tuple[int, int]:
    hit = tot = 0
    for k, q in queries:
        pool = [(gk, go) for gk, go in gallery if gk != k or go.frame != q.frame]
        if len(pool) < 2:
            continue
        best = max(pool, key=lambda t: similarity(q.sig, t[1].sig).overall)
        tot += 1
        hit += int(best[0] == k)
    return hit, tot


def _obs_to_json(o: "_Obs") -> dict:
    return {"frame": o.frame, "clip": o.clip, "track": o.track,
            "box": list(o.box), "sig": o.sig.to_dict()}


def _obs_from_json(d: dict) -> "_Obs":
    return _Obs(d["frame"], d["clip"], d["track"],
                VehicleSignature.from_dict(d["sig"]), tuple(d["box"]))


def _stitch_tracks(all_obs: List["_Obs"], stride: int,
                   max_gap_frames: int) -> int:
    """
    Rejoin one vehicle that the IoU tracker cut into several tracks.

    Overlap-only linking breaks every time a vehicle moves further than its
    own length between two sampled frames, which on a 30 fps clip sampled
    every third frame is most of them. The damage is not cosmetic: each
    fragment of a vehicle becomes a separate identity in the gallery, so the
    correct answer to a query is competing against four near-identical views
    of the very same car, and retrieving one of them is scored as an error.
    That deflates rank-1 for a reason that has nothing to do with the
    descriptor.

    Stitching is done on predicted motion alone -- where the box was going,
    and whether the next fragment starts there. Appearance is deliberately
    not consulted, because the ground truth must stay independent of the
    thing being measured.
    """
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(k):
        while parent.get(k, k) != k:
            parent[k] = parent.get(parent[k], parent[k])
            k = parent[k]
        return k

    by_track: Dict[Tuple[str, int], List[_Obs]] = {}
    for o in all_obs:
        by_track.setdefault((o.clip, o.track), []).append(o)
    for v in by_track.values():
        v.sort(key=lambda o: o.frame)

    def centre(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    merged = 0
    for clip in {o.clip for o in all_obs}:
        keys = sorted((k for k in by_track if k[0] == clip),
                      key=lambda k: by_track[k][0].frame)
        for i, ka in enumerate(keys):
            a = by_track[ka]
            ax, ay = centre(a[-1].box)
            aw = max(1.0, a[-1].box[2] - a[-1].box[0])
            if len(a) >= 2:
                pf = a[-2].frame
                px, py = centre(a[-2].box)
                dt0 = max(1, a[-1].frame - pf)
                vx, vy = (ax - px) / dt0, (ay - py) / dt0
            else:
                vx = vy = 0.0
            best, best_d = None, None
            for kb in keys[i + 1:]:
                b = by_track[kb]
                gap = b[0].frame - a[-1].frame
                if gap <= 0:
                    continue
                if gap > max_gap_frames:
                    break
                if find(ka) == find(kb):
                    continue
                bx, by_ = centre(b[0].box)
                d = math.hypot(ax + vx * gap - bx, ay + vy * gap - by_)
                if d < aw and (best_d is None or d < best_d):
                    best, best_d = kb, d
            if best is not None:
                parent[find(best)] = find(ka)
                merged += 1

    for o in all_obs:
        root = find((o.clip, o.track))
        o.track = root[1]
    return merged


def bench_real(paths: Sequence[str], max_frames: int = 2000, stride: int = 3,
               iou_thr: float = 0.2, min_separation: int = 12,
               max_pairs: int = 400_000, detect_scale: float = 0.5,
               cache: Optional[str] = None) -> dict:
    """
    Real traffic footage, with identity recovered by tracking.

    Two facts real footage gives for free, and they are the whole benchmark:
    a vehicle followed through a scene is certainly the same vehicle, and two
    vehicles visible in the same frame are certainly not. That is enough for
    a genuine rank-1 and a genuine false-match rate on real vehicles, without
    anyone hand-labelling a dataset.

    What it does NOT measure is a change of camera, because we have no
    vehicle known to appear in two of these clips. It measures the change of
    scale, viewing angle and illumination a vehicle undergoes while driving
    through one scene, which is the same kind of variation and a fair proxy
    -- but it is a proxy, and the number should be read as an optimistic one.
    """
    all_obs: List[_Obs] = []
    per_clip = {}
    # Detection is the expensive step by two orders of magnitude, so the
    # signatures are cached. Re-tuning a weight should not cost another pass
    # over the footage, or it will not get done.
    if cache and os.path.exists(cache):
        blob = json.loads(open(cache).read())
        all_obs = [_obs_from_json(d) for d in blob["observations"]]
        per_clip = blob["per_clip"]
        print(f"  loaded {len(all_obs)} cached observations from {cache}",
              file=sys.stderr, flush=True)
        paths = []
    for p in paths:
        clip = str(p).rsplit("/", 1)[-1]
        cap = cv2.VideoCapture(str(p))
        got = _track_clip(cap, clip, max_frames, stride, iou_thr, stride * 4,
                          detect_scale)
        cap.release()
        all_obs.extend(got)
        per_clip[clip] = {"observations": len(got),
                          "tracks": len({o.track for o in got})}
        print(f"  {clip}: {len(got)} usable detections, "
              f"{per_clip[clip]['tracks']} tracks", file=sys.stderr, flush=True)

    if cache and paths:
        with open(cache, "w") as fh:
            json.dump({"per_clip": per_clip,
                       "observations": [_obs_to_json(o) for o in all_obs]}, fh)

    tracks_before = len({(o.clip, o.track) for o in all_obs})
    merged = _stitch_tracks(all_obs, stride, stride * 8)
    tracks_after = len({(o.clip, o.track) for o in all_obs})

    def key(o):
        return (o.clip, o.track)

    # Same vehicle: two views of one track far enough apart in time that the
    # vehicle has actually moved and changed scale. Adjacent frames would
    # measure JPEG noise and report it as invariance.
    same = []
    by_track: Dict[Tuple[str, int], List[_Obs]] = {}
    for o in all_obs:
        by_track.setdefault(key(o), []).append(o)
    for obs in by_track.values():
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                if obs[j].frame - obs[i].frame >= min_separation:
                    same.append(similarity(obs[i].sig, obs[j].sig).overall)

    # Different vehicles, with certainty: simultaneous in one frame, or seen
    # on two different cameras.
    pairs = []
    for i, a in enumerate(all_obs):
        for b in all_obs[i + 1:]:
            if a.clip != b.clip:
                pairs.append((a, b))
            elif a.frame == b.frame and a.track != b.track:
                # Two boxes in one frame are two vehicles UNLESS they overlap,
                # in which case the detector has fired twice on one vehicle --
                # often as 'car' and 'truck' at once. Counting that as a pair
                # of different vehicles would score a vehicle against itself
                # and report the result as a false match.
                if _iou(a.box, b.box) < 0.3:
                    pairs.append((a, b))
    # Deterministic thinning rather than a random sample, so the reported
    # false-match rate is reproducible by anyone re-running the benchmark.
    step = max(1, len(pairs) // max_pairs)
    diff = [similarity(a.sig, b.sig).overall for a, b in pairs[::step]]

    # Rank-1 over a closed gallery of one view per vehicle. The query is the
    # last view of a track, the gallery holds the first view of every track
    # including distractors from the other camera.
    gallery = [(k, sorted(v, key=lambda o: o.frame)[0])
               for k, v in by_track.items() if len(v) >= 2]
    queries = []
    for k, views in by_track.items():
        views = sorted(views, key=lambda o: o.frame)
        if len(views) >= 2 and views[-1].frame - views[0].frame >= min_separation:
            queries.append((k, views[-1]))
    hit, tot = _rank1(queries, gallery)

    # Rank-1 split by how large the vehicle is in frame. This is the number
    # that decides where in the state grid appearance tracking can be
    # deployed at all: the survey found most government cameras render a
    # vehicle small, and if accuracy collapses below a width the registry can
    # measure, then the registry can also decide which cameras get this
    # analytic -- which is the whole control-plane argument, in a number.
    bands = {"40-79px": (40, 80), "80-159px": (80, 160),
             "160-319px": (160, 320), "320px+": (320, 10_000)}
    by_size = {}
    for label, (lo, hi) in bands.items():
        q = [(k, o) for k, o in queries if lo <= o.sig.box_width_px < hi]
        g = [(k, o) for k, o in gallery if lo <= o.sig.box_width_px < hi]
        h2, t2 = _rank1(q, g)
        by_size[label] = {"correct": h2, "queries": t2,
                          "accuracy": round(h2 / t2, 3) if t2 else None}

    thr = APPEARANCE_PROBABLE
    fp = sum(1 for d in diff if d >= thr)
    rep = _bench_report("real government footage", same, diff,
                        rank1=(hit, tot) if tot else None,
                        false_at=(fp, len(diff), thr) if diff else None)
    lens = sorted(len(v) for v in by_track.values())
    rep["clips"] = per_clip
    rep["gallery_size"] = len(gallery)
    # Track length is reported because it bounds how much this benchmark can
    # claim: short fragmented tracks make rank-1 pessimistic (one vehicle
    # split in two becomes its own distractor) and shrink the same-vehicle
    # pool. A reader should be able to see that rather than take it on trust.
    rep["track_length_median"] = lens[len(lens) // 2] if lens else 0
    rep["track_length_max"] = lens[-1] if lens else 0
    rep["rank1_by_vehicle_size"] = by_size
    rep["tracks_before_stitching"] = tracks_before
    rep["tracks_after_stitching"] = tracks_after
    rep["fragments_merged"] = merged
    rep["note"] = ("identity from IoU tracking, which is independent of the "
                   "descriptor under test. 'same vehicle' = two views of one "
                   "track at least "
                   f"{min_separation} frames apart; 'different vehicle' = "
                   "simultaneous in one frame, or on two different cameras. "
                   "Single-camera identity only: this measures scale, angle "
                   "and illumination change within a scene, not a change of "
                   "camera, so read rank-1 as an optimistic figure.")
    return rep


def main(argv=None):
    import argparse
    import glob

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--controlled", action="store_true",
                    help="synthetic controlled-lighting discrimination bench")
    ap.add_argument("--grid", action="store_true",
                    help="the local synthetic camera grid (needs ./grid/run.sh start)")
    ap.add_argument("--real", default="", help="glob of real clips, e.g. media/real/*.mp4")
    ap.add_argument("--seconds", type=float, default=46.0)
    ap.add_argument("--cache", default="",
                    help="reuse/store real-footage signatures (skips detection)")
    a = ap.parse_args(argv)

    out = []
    if a.controlled:
        out.append(bench_controlled())
    if a.grid:
        out.append(bench_grid(seconds=a.seconds))
    if a.real:
        files = sorted(glob.glob(a.real))
        if not files:
            out.append({"bench": "real footage", "error": f"no files at {a.real}"})
        else:
            out.append(bench_real(files, cache=a.cache or None))
    if not out:
        ap.error("choose at least one of --controlled / --grid / --real")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
