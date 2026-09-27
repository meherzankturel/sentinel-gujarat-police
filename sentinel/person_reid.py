#!/usr/bin/env python3
"""
sentinel.person_reid -- is that the same person I saw two cameras ago?

The problem statement names four watchlist categories: stolen vehicles,
blacklisted vehicles, wanted persons and missing persons. Everything built
so far serves the two vehicle categories, because a vehicle carries a
number plate and a plate is an identifier. A person carries no plate, and
on this grid a person carries no usable face either -- sentinel.facecap
measured 2.7-9.8 px between the eyes on the closest people, against a 40 px
floor for even attempting recognition. Short by a factor of four to fifteen.

So the two person categories cannot be served by identification. They can
be served by *association*, which is a different and much narrower claim:

    face recognition answers   "who is this person?"
    this module answers        "is this the same person I saw two cameras
                                ago, in the last few hours?"

The second question is the one an officer actually asks while a missing
child is still missing, and it is the one the pixels can support. The
module is built so that it cannot be mistaken for the first question, and
the limits below are enforced in code rather than written in a footnote.

What is measured, and why vertical bands
----------------------------------------
sentinel.reid describes a vehicle with a 2x3 grid of cells. That is right
for a vehicle and wrong for a person. A car is a rigid box whose left-right
structure holds still -- a stripe along the flank stays along the flank. A
walking person is an articulated object photographed from an arbitrary
angle: they turn, and left becomes right. Any horizontal structure in the
descriptor is destroyed by the person simply walking around a corner.

What survives a turn is the vertical order, because gravity fixes it. A
person wearing a red shirt above dark trousers presents red-above-dark from
the front, the back, and both sides. So the descriptor is three horizontal
bands stacked vertically:

    head    0.00-0.18 of box height   head and shoulders
    torso   0.18-0.55                 the upper garment
    legs    0.55-0.95                 the lower garment

and each band carries its own colour histogram, saturation and relative
brightness. The bottom 5% is dropped: it is feet, ground and contact
shadow, which is the road's colour on every camera.

The head band is measured and NOT matched on
--------------------------------------------
BAND_WEIGHTS gives the head band weight 0.0. This is deliberate and it is
the ethical boundary of the module, so it is a constant a reviewer can see
rather than an omission they have to notice.

The dominant colours of a head band are hair and skin. Matching people on
skin tone is both discriminatory and, at 13-49 px of head on this grid,
useless -- the measured value is mostly the illuminant and the codec. The
band is still computed, because it is a useful quality check (is the person
upright, is the head occluded), and it is still shown to an officer. It
contributes nothing to the score.

Nothing in this module touches a face. There is no facial landmark, no
embedding, no gallery of identified people. `PersonSignature.iod_px`
reports, from sentinel.facecap's published anthropometry, how many pixels
lie between the eyes in this very crop -- so an evidence bundle states, as
a number, that the face in it could not have been recognised.

The honest limits, all three enforced in code
---------------------------------------------
1. Clothing. This is a descriptor of what someone is wearing, so it dies
   when they change. FULL_STRENGTH_S (30 min) is where the timing costs
   nothing; PROBABLE_HORIZON_S (8 h, one shift or one day out) is the hard
   edge, and beyond it the match is capped at 'corroborating' whatever it
   scores. Across two days it is worthless and the module says so instead
   of quietly returning a number.

2. Crowds. Ten men in white shirts and dark trousers at a bus stand all
   match each other. So the candidate sighting carries the other people
   visible in the same frame, and if any of them matches the query nearly
   as well (CROWD_RIVAL_MARGIN) the tier is capped at 'corroborating'. A
   description that fits three people standing together is not a match, and
   this is the only way the software can know that -- the rival was in the
   frame, so it was measurable.

3. Never 'confirmed' on appearance. sentinel.reid allows a vehicle to reach
   'confirmed' when appearance is corroborated by agreeing plate characters
   or a no-exit corridor. For a person there is no equivalent: there is no
   plate, and a corridor time-fit narrows nothing when the walkway carries
   a thousand people an hour. So assess() has no path to 'confirmed' at
   all -- MAX_AUTOMATIC_TIER caps it at 'probable'. Confirmation is an act
   by a named human, officer_confirm(), which records who decided and on
   what basis. Misidentifying a person costs more than misidentifying a
   van, so the software declines to be the one that decides.

Physics is reused, not reinvented. The hard transit gate is
sentinel.reid.transit_check -- same 140 km/h ceiling, same clock-slack
widening, because a wanted person who steps into an auto-rickshaw travels
at vehicle speed. What is re-scored on top of it is the *supportive* band:
4 km/h over 300 m is the expected case for a pedestrian and scores full
marks, where reid would have called it suspiciously slow.

Licensing: OpenCV (Apache 2.0) and NumPy (BSD). No learned re-id model,
here by choice -- an officer can testify that the upper body was red and
the lower body dark, and cannot testify to a cosine distance.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .facecap import FACE_MARGINAL_IOD, iod_px
from .matching import TIER_ORDER, _cap_to, _demote
from .reid import (ACHROMATIC_SAT, BIN_NAMES, BLACK_V, GREY_V, HUE_BINS,
                   HUE_BIN_OFFSET, CameraSite, TransitCheck,
                   _smooth_colour_bins, _usable_mask, colour_similarity,
                   haversine_km, transit_check)

# ---------------------------------------------------------------- constants

# The refusal floor, in box height. The grid survey puts a person at
# 100-370 px tall on the cameras that see people at all. Below 64 px the
# torso band is about 23 px deep and the shoulders are 20 px across, which
# is one JPEG macroblock's opinion of a shirt.
#
# reid.py handles a too-small VEHICLE by returning a signature flagged
# unreliable, because an unreliable vehicle colour can still corroborate an
# existing track. This module refuses outright instead, and the difference
# is deliberate: an unreliable person descriptor is indistinguishable from
# "somebody in dark clothing", which fits half of Ahmedabad. Producing one
# invites a false identification of a human being, so the failure mode is
# to produce nothing.
MIN_BODY_PX = 64
MIN_BODY_WIDTH_PX = 22

# Usable pixels needed after the shadow/specular cut, per band and for the
# torso specifically. The torso is the identity; a signature without one is
# not a signature.
MIN_BAND_PIXELS = 100
MIN_TORSO_PIXELS = 150

# Person boxes include a lot of background -- a person is narrow and the
# detector's rectangle is not. The inset drops the vertical edges.
BODY_INSET_X = 0.14

BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("head", 0.00, 0.18),
    ("torso", 0.18, 0.55),
    ("legs", 0.55, 0.95),
)

# Head weight 0.0: see the docstring. Torso above legs because the upper
# garment is the larger, more varied and more often unoccluded surface --
# legs are hidden behind parked vehicles, market stalls and other people.
BAND_WEIGHTS = {"head": 0.0, "torso": 0.58, "legs": 0.42}

# Build is a tie-breaker only. A person box's aspect ratio changes with
# stride, with a raised arm and with camera elevation, so it cannot carry
# weight -- but it does separate an adult from a child, which matters for
# the missing-persons case.
W_BANDS, W_BUILD = 0.90, 0.10

# Within-band component weights. Colour dominates; relative brightness is
# lowest because it is what differs most between two cameras looking at the
# same person, and least between two different people in the same clothes.
BAND_HIST, BAND_SAT, BAND_VAL = 0.72, 0.16, 0.12

# The short horizon, stated in seconds so it is arguable.
FULL_STRENGTH_S = 1800.0            # 30 min: clothing certainly unchanged
PROBABLE_HORIZON_S = 8 * 3600.0     # one shift / one day out
HORIZON_FLOOR = 0.55                # multiplier at the edge of the horizon

# A rival this close to the best score means the description is not
# distinctive in that frame.
CROWD_RIVAL_MARGIN = 0.05

# Pedestrian speed bands for the supportive half of the transit check.
STROLL_KMH = 1.5
WALK_KMH = 7.0
RUN_KMH = 16.0
LOITER_FLOOR = 0.35
LOITER_DECAY_S = 1800.0

# Appearance operating points. Set from the measurement in bench_real()
# below, not chosen: on 300 frames of cam09_day.mp4 (Ahmedabad street,
# people on foot), same-person pairs at least 15 frames apart against
# pairs of people simultaneously visible in one frame --
#
#     threshold   same-person recall   false-match rate
#       0.55            0.94                0.365
#       0.70            0.76                0.130
#       0.78            0.58                0.055
#       0.85            0.29                0.011
#
# 0.78 is the operating point for 'probable', bought at 5.5% false matches
# for 58% recall. The floor at 0.55 is where a candidate is worth SHOWING
# beside an existing track; it was never a threshold for believing one.
APPEARANCE_FLOOR = 0.55
APPEARANCE_PROBABLE = 0.78

# No automatic path to 'confirmed'. Enforced, not intended.
MAX_AUTOMATIC_TIER = "probable"


class TooSmallToDescribe(ValueError):
    """
    Raised instead of returning a descriptor nobody should use.

    Carries the measurement so the officer's screen can say why a person
    on a particular camera produced no signature -- which is a fact about
    the camera, and belongs in the registry.
    """

    def __init__(self, message: str, *, body_height_px: int = 0,
                 body_width_px: int = 0, torso_pixels: int = 0):
        super().__init__(message)
        self.body_height_px = body_height_px
        self.body_width_px = body_width_px
        self.torso_pixels = torso_pixels


# --------------------------------------------------------------- signature


@dataclass
class PersonBand:
    """One horizontal slice of a person, in terms an officer can read."""

    name: str
    bins: List[float]          # 12 hue bins + black / grey / white
    sat: float                 # mean saturation, 0..1
    rel_val: float             # band brightness / whole-body brightness, 0..1
    pixels: int
    usable: bool

    @property
    def colour(self) -> str:
        return BIN_NAMES[int(np.argmax(self.bins))] if self.bins else "unknown"

    @property
    def share(self) -> float:
        return round(float(max(self.bins)), 3) if self.bins else 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "bins": [round(v, 4) for v in self.bins],
                "bin_names": BIN_NAMES, "sat": round(self.sat, 3),
                "rel_val": round(self.rel_val, 3), "pixels": self.pixels,
                "usable": self.usable, "colour": self.colour}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonBand":
        return cls(name=d["name"], bins=list(d["bins"]), sat=float(d["sat"]),
                   rel_val=float(d["rel_val"]), pixels=int(d["pixels"]),
                   usable=bool(d["usable"]))


@dataclass
class PersonSignature:
    """
    A person described by what they are wearing, and nothing else.

    Everything here renders: each band is a bar chart and a swatch, the
    build is a ratio, and `iod_px` is the number that says the face in this
    crop was never identifiable. There is no opaque field.
    """

    bands: List[PersonBand]
    aspect: float                # height / width
    body_height_px: int
    body_width_px: int
    pixels_used: int
    camera_id: str = ""

    def band(self, name: str) -> Optional[PersonBand]:
        for b in self.bands:
            if b.name == name:
                return b
        return None

    @property
    def matchable_bands(self) -> List[PersonBand]:
        """Bands that both carry weight and have enough pixels to trust."""
        return [b for b in self.bands
                if BAND_WEIGHTS.get(b.name, 0.0) > 0.0 and b.usable]

    @property
    def iod_px(self) -> float:
        """
        Pixels between the eyes in this crop, from facecap's anthropometry.

        Present so that any evidence bundle carrying this signature also
        carries the proof that a face was not used: on this grid the number
        comes out far below facecap's 40 px floor.
        """
        return iod_px(self.body_height_px)

    @property
    def face_identifiable(self) -> bool:
        return self.iod_px >= FACE_MARGINAL_IOD

    def describe(self) -> str:
        """One line for a sighting list."""
        bits = []
        for b in self.bands:
            if BAND_WEIGHTS.get(b.name, 0.0) <= 0.0:
                continue
            bits.append(f"{b.name} {b.colour}"
                        + ("" if b.usable else " (not enough pixels)"))
        bits.append(f"{self.body_height_px}px tall")
        if not self.face_identifiable:
            bits.append(f"face {self.iod_px:.1f}px between the eyes "
                        "-- not identifiable")
        return ", ".join(bits)

    def to_dict(self) -> dict:
        return {"bands": [b.to_dict() for b in self.bands],
                "aspect": round(self.aspect, 3),
                "body_height_px": self.body_height_px,
                "body_width_px": self.body_width_px,
                "pixels_used": self.pixels_used,
                "camera_id": self.camera_id,
                "iod_px": round(self.iod_px, 2),
                "face_identifiable": self.face_identifiable,
                "describe": self.describe()}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonSignature":
        return cls(bands=[PersonBand.from_dict(b) for b in d["bands"]],
                   aspect=float(d["aspect"]),
                   body_height_px=int(d["body_height_px"]),
                   body_width_px=int(d["body_width_px"]),
                   pixels_used=int(d.get("pixels_used", 0)),
                   camera_id=d.get("camera_id", ""))


def _colour_bins(hsv: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Colour histogram of the masked pixels, plus mean saturation and value.

    Identical quantisation to sentinel.reid -- 12 hue bins with the same
    offset, the same achromatic split, the same neighbour smoothing and the
    same discount on the darkest bin -- so that the two descriptors are
    read the same way in the UI and argued about on the same terms.
    """
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bins = np.zeros(HUE_BINS + 3, np.float64)
    if not mask.any():
        return bins, 0.0, 0.0

    hs, sats, vals = hh[mask], ss[mask], vv[mask]
    chroma = sats >= ACHROMATIC_SAT
    if chroma.any():
        centred = (hs[chroma].astype(np.float64) + HUE_BIN_OFFSET) % 180.0
        idx = np.clip((centred * HUE_BINS / 180.0).astype(np.int32),
                      0, HUE_BINS - 1)
        np.add.at(bins, idx, 1.0)
    ach = vals[~chroma]
    if ach.size:
        bins[HUE_BINS + 0] += float((ach < BLACK_V).sum())
        bins[HUE_BINS + 1] += float(((ach >= BLACK_V) & (ach < GREY_V)).sum())
        bins[HUE_BINS + 2] += float((ach >= GREY_V).sum())

    total = bins.sum()
    if total > 0:
        bins /= total
    return (_smooth_colour_bins(bins), float(sats.mean()) / 255.0,
            float(vals.mean()))


def signature_from_crop(crop: np.ndarray, *, camera_id: str = "",
                        body_height_px: Optional[int] = None,
                        body_width_px: Optional[int] = None) -> PersonSignature:
    """
    Build the descriptor from a cropped person box, or refuse.

    Refusal is a first-class outcome. TooSmallToDescribe is raised when the
    box is below MIN_BODY_PX, when it is too narrow, or when the torso band
    survives the shadow/specular cut with too few pixels -- which is what a
    night camera or a person standing in deep shade produces.
    """
    if crop is None or crop.size == 0:
        raise TooSmallToDescribe("empty crop")

    h, w = crop.shape[:2]
    height_px = int(body_height_px if body_height_px is not None else h)
    width_px = int(body_width_px if body_width_px is not None else w)

    if height_px < MIN_BODY_PX:
        raise TooSmallToDescribe(
            f"person only {height_px}px tall (need {MIN_BODY_PX}px before "
            "clothing colour means anything); no signature produced",
            body_height_px=height_px, body_width_px=width_px)
    if width_px < MIN_BODY_WIDTH_PX:
        raise TooSmallToDescribe(
            f"person only {width_px}px wide (need {MIN_BODY_WIDTH_PX}px); "
            "no signature produced",
            body_height_px=height_px, body_width_px=width_px)

    x0 = int(w * BODY_INSET_X)
    x1 = max(x0 + 1, int(w * (1.0 - BODY_INSET_X)))
    hsv_full = cv2.cvtColor(crop[:, x0:x1], cv2.COLOR_BGR2HSV)
    mask_full = _usable_mask(hsv_full)
    if not mask_full.any():
        mask_full = np.ones(hsv_full.shape[:2], bool)

    # Brightness reference for the whole body, so a band records "darker
    # than the rest of this person" rather than an absolute exposure -- the
    # first thing to change between two vendors' cameras.
    ref_v = max(1.0, float(hsv_full[:, :, 2][mask_full].mean()))

    bands: List[PersonBand] = []
    for name, top, bottom in BANDS:
        y0 = int(h * top)
        y1 = max(y0 + 1, int(h * bottom))
        hsv = hsv_full[y0:y1]
        m = mask_full[y0:y1]
        bins, sat, val = _colour_bins(hsv, m)
        px = int(m.sum())
        bands.append(PersonBand(
            name=name, bins=[float(x) for x in bins], sat=sat,
            rel_val=float(np.clip(val / ref_v, 0.0, 2.0) / 2.0),
            pixels=px, usable=px >= MIN_BAND_PIXELS))

    torso = next(b for b in bands if b.name == "torso")
    if torso.pixels < MIN_TORSO_PIXELS:
        raise TooSmallToDescribe(
            f"only {torso.pixels} usable torso pixels after removing shadow "
            f"and highlight (need {MIN_TORSO_PIXELS}); no signature produced",
            body_height_px=height_px, body_width_px=width_px,
            torso_pixels=torso.pixels)

    return PersonSignature(
        bands=bands,
        aspect=float(height_px) / float(width_px) if width_px else 0.0,
        body_height_px=height_px, body_width_px=width_px,
        pixels_used=int(mask_full.sum()), camera_id=camera_id)


def signature(frame: np.ndarray, box: Sequence[float],
              camera_id: str = "") -> PersonSignature:
    """Descriptor for one sentinel.detect person box in a frame."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise TooSmallToDescribe("degenerate box")
    return signature_from_crop(frame[y1:y2, x1:x2], camera_id=camera_id,
                               body_height_px=y2 - y1, body_width_px=x2 - x1)


# --------------------------------------------------------------- similarity


@dataclass
class PersonSimilarity:
    """Why the system thinks these are the same person, itemised by band."""

    bands: Dict[str, float]
    build: float
    overall: float
    bands_used: List[str]
    notes: List[str] = field(default_factory=list)

    def why(self) -> str:
        bits = [f"{k} {v:.2f}" for k, v in self.bands.items()
                if k in self.bands_used]
        bits.append(f"build {self.build:.2f}")
        return "; ".join(bits + self.notes)

    def to_dict(self) -> dict:
        return {"bands": {k: round(v, 3) for k, v in self.bands.items()},
                "build": round(self.build, 3),
                "overall": round(self.overall, 3),
                "bands_used": list(self.bands_used),
                "notes": list(self.notes)}


def band_similarity(a: PersonBand, b: PersonBand) -> float:
    """
    How alike one band of two people is.

    Histogram intersection carries most of it, for the reason reid gives:
    "78% of the upper-body colour matches" is a sentence a witness can
    repeat. Saturation separates a washed-out grey from a vivid colour of
    the same hue, and relative brightness separates a dark shirt from a
    pale one when neither has much hue at all.
    """
    hist = colour_similarity(a.bins, b.bins)
    sat = 1.0 - min(1.0, abs(a.sat - b.sat))
    val = 1.0 - min(1.0, abs(a.rel_val - b.rel_val))
    return BAND_HIST * hist + BAND_SAT * sat + BAND_VAL * val


def appearance(a: PersonSignature, b: PersonSignature) -> PersonSimilarity:
    """
    How alike two person sightings look, 0..1, with the reasoning attached.

    A band that either side could not measure is dropped and the remaining
    weights renormalised, rather than scored as a mismatch. Legs hidden
    behind a parked rickshaw are missing evidence, not contrary evidence,
    and the note records which band went missing so the officer sees the
    match was made on the torso alone.
    """
    scores: Dict[str, float] = {}
    notes: List[str] = []
    used: List[str] = []
    num = den = 0.0

    for name, _, _ in BANDS:
        ba, bb = a.band(name), b.band(name)
        if ba is None or bb is None:
            continue
        s = band_similarity(ba, bb)
        scores[name] = s
        wt = BAND_WEIGHTS.get(name, 0.0)
        if wt <= 0.0:
            continue
        if not (ba.usable and bb.usable):
            notes.append(f"{name} band not measurable on one of the two "
                         "sightings; excluded")
            continue
        used.append(name)
        num += wt * s
        den += wt

    band_score = num / den if den > 0 else 0.0
    if den <= 0.0:
        notes.append("no band could be compared; appearance says nothing here")

    ratio = (min(a.aspect, b.aspect) / max(a.aspect, b.aspect)
             if a.aspect > 0 and b.aspect > 0 else 0.0)
    build = max(0.0, (ratio - 0.5) / 0.5)

    overall = (W_BANDS * band_score + W_BUILD * build) if den > 0 else 0.0
    if "legs" not in used and den > 0:
        notes.append("matched on the upper body only")
    return PersonSimilarity(bands=scores, build=build, overall=float(overall),
                            bands_used=used, notes=notes)


# ------------------------------------------------------------- plausibility


def person_transit(a: CameraSite, b: CameraSite, elapsed_s: float,
                   extra_slack_s: float = 0.0) -> TransitCheck:
    """
    Could a PERSON have got from camera A to camera B in this time?

    The hard gate is sentinel.reid.transit_check, unchanged and imported,
    because a person who steps into an auto-rickshaw travels at vehicle
    speed and the physical ceiling is therefore the same 140 km/h -- and
    because the clock-slack widening in there (cam-08 runs 428 s fast) is
    the difference between doubting a match and discarding a true one.

    What is replaced is the supportive half. reid scores a transit against
    traffic flow, so 4 km/h across 300 m comes back as suspiciously slow.
    For a person on foot that is the expected case and it scores full
    marks. Above running pace the person must have used a vehicle: still
    possible, but the timing has stopped being evidence for anything.
    """
    base = transit_check(a, b, elapsed_s, extra_slack_s)
    if not base.plausible or base.implied_kmh is None:
        return base

    implied = base.implied_kmh
    dist = base.distance_km
    if implied <= STROLL_KMH:
        over = base.effective_s - dist / STROLL_KMH * 3600.0
        score = LOITER_FLOOR + (1.0 - LOITER_FLOOR) * math.exp(
            -max(0.0, over) / LOITER_DECAY_S)
        reason = (f"{base.elapsed_s / 60.0:.0f} min for {dist * 1000:.0f} m -- "
                  "possible, but the person could have waited anywhere; "
                  "timing is not evidence here")
    elif implied <= WALK_KMH:
        score, reason = 1.0, (f"{implied:.1f} km/h over {dist * 1000:.0f} m -- "
                              "consistent with walking")
    elif implied <= RUN_KMH:
        score, reason = 0.80, (f"{implied:.1f} km/h over {dist:.2f} km -- "
                               "running or cycling pace")
    else:
        score, reason = 0.45, (f"{implied:.0f} km/h over {dist:.2f} km -- "
                               "only possible in a vehicle; possible for a "
                               "person who took one, but the timing "
                               "corroborates nothing")
    return replace(base, score=float(score), reason=reason)


@dataclass
class Horizon:
    """How much the elapsed time has eaten into a clothing descriptor."""

    elapsed_s: float
    factor: float
    within_horizon: bool
    reason: str

    def to_dict(self) -> dict:
        return {"elapsed_s": round(self.elapsed_s, 1),
                "factor": round(self.factor, 3),
                "within_horizon": self.within_horizon, "reason": self.reason}


def horizon_check(elapsed_s: float) -> Horizon:
    """
    The explicit short-horizon rule.

    Appearance matching of a person is a statement about clothing. Inside
    half an hour that is as good as the descriptor gets. Out to eight hours
    -- a shift, a school day, an afternoon at a fair -- it decays but stays
    usable. Past that it is capped at 'corroborating' no matter what it
    scores, because the person has plausibly changed, and a system that
    offers a confident cross-day match on a shirt colour is lying.
    """
    e = abs(float(elapsed_s))
    if e <= FULL_STRENGTH_S:
        return Horizon(e, 1.0, True,
                       f"{e / 60.0:.0f} min apart -- clothing unchanged")
    if e <= PROBABLE_HORIZON_S:
        frac = (e - FULL_STRENGTH_S) / (PROBABLE_HORIZON_S - FULL_STRENGTH_S)
        return Horizon(e, 1.0 - (1.0 - HORIZON_FLOOR) * frac, True,
                       f"{e / 3600.0:.1f} h apart -- same day, clothing "
                       "probably unchanged")
    return Horizon(e, HORIZON_FLOOR, False,
                   f"{e / 3600.0:.1f} h apart -- beyond the "
                   f"{PROBABLE_HORIZON_S / 3600.0:.0f} h clothing horizon; "
                   "this describes an outfit, not a person")


# ------------------------------------------------------------------ crowding


def crowd_rivals(query: PersonSignature, co_visible: Iterable[PersonSignature],
                 target_score: float) -> List[float]:
    """
    Other people in the candidate's own frame who fit the query as well.

    This is the crowd limit made measurable rather than merely admitted.
    The rivals were standing in the same frame as the candidate, so they
    were detected, described and can be scored -- and if the description
    fits them too, then it is a description of a group and not of a person.
    """
    out = []
    for other in co_visible or ():
        s = appearance(query, other).overall
        if s >= APPEARANCE_FLOOR and s >= target_score - CROWD_RIVAL_MARGIN:
            out.append(s)
    return sorted(out, reverse=True)


# --------------------------------------------------------------- sightings


@dataclass
class PersonSighting:
    """
    One person seen once.

    `co_visible` holds the descriptors of the OTHER people detected in the
    same frame. It is not decoration: it is what lets the crowd limit be
    enforced instead of disclaimed.
    """

    sighting_id: str
    camera_id: str
    at_epoch_s: float
    signature: PersonSignature
    co_visible: List[PersonSignature] = field(default_factory=list)
    frame_ref: Optional[str] = None


@dataclass
class PersonMatch:
    """A ranked candidate, with the whole argument for and against it."""

    candidate: PersonSighting
    appearance: PersonSimilarity
    transit: TransitCheck
    horizon: Horizon
    rivals: List[float]
    tier: str
    score: float
    why: str
    confirmed_by: Optional[str] = None

    def notifies(self) -> bool:
        """Same contract as sentinel.matching.Match."""
        return self.tier in ("confirmed", "probable")

    def to_dict(self) -> dict:
        return {"sighting_id": self.candidate.sighting_id,
                "camera_id": self.candidate.camera_id,
                "at_epoch_s": self.candidate.at_epoch_s,
                "appearance": self.appearance.to_dict(),
                "transit": self.transit.to_dict(),
                "horizon": self.horizon.to_dict(),
                "rivals_in_frame": len(self.rivals),
                "tier": self.tier, "score": round(self.score, 3),
                "confirmed_by": self.confirmed_by, "why": self.why}


def assess(query: PersonSighting, cand: PersonSighting,
           sites: Mapping[str, CameraSite],
           extra_slack_s: float = 0.0) -> PersonMatch:
    """
    Score and tier one candidate against the query.

    There is no branch in here that produces 'confirmed', and the tier is
    passed through _cap_to(MAX_AUTOMATIC_TIER) on the way out so that no
    future edit can add one by accident.
    """
    app = appearance(query.signature, cand.signature)

    qs = sites.get(query.camera_id) or CameraSite(query.camera_id, None, None)
    cs = sites.get(cand.camera_id) or CameraSite(cand.camera_id, None, None)
    tr = person_transit(qs, cs, cand.at_epoch_s - query.at_epoch_s,
                        extra_slack_s)
    hz = horizon_check(cand.at_epoch_s - query.at_epoch_s)
    rivals = crowd_rivals(query.signature, cand.co_visible, app.overall)

    score = app.overall * (0.55 + 0.45 * tr.score) * hz.factor

    why: List[str] = [app.why(), tr.reason, hz.reason]
    tier = "corroborating"

    if not tr.plausible:
        why.append("rejected on transit time")
        score = 0.0
    elif not hz.within_horizon:
        why.append("beyond the clothing horizon; corroboration only")
    elif rivals:
        why.append(f"{len(rivals)} other people in that frame match this "
                   f"description as well (best {rivals[0]:.2f}) -- a "
                   "description that fits a group is not a match")
    elif app.overall >= APPEARANCE_PROBABLE:
        tier = "probable"
        why.append("appearance only -- clothing is not an identity, and this "
                   "system has no route to 'confirmed' for a person without "
                   "a human deciding")
    elif app.overall >= APPEARANCE_FLOOR:
        why.append("weak appearance match; corroboration only")
    else:
        why.append("appearance does not match")

    return PersonMatch(candidate=cand, appearance=app, transit=tr, horizon=hz,
                       rivals=rivals, tier=_cap_to(tier, MAX_AUTOMATIC_TIER),
                       score=float(score), why="; ".join(why))


def officer_confirm(match: PersonMatch, officer_id: str,
                    basis: str) -> PersonMatch:
    """
    The only route to 'confirmed' for a person: a named human decides.

    The basis is mandatory and free text, because what actually confirms a
    person is outside this module's reach -- an officer recognising a face
    on the full-resolution clip, a relative identifying a jacket, a vehicle
    at the scene whose plate was read. Recording who decided and on what
    grounds is what makes the bundle survive a courtroom; recording only
    'confirmed' is what makes it fail.
    """
    if not officer_id or not basis:
        raise ValueError("confirming a person requires an officer and a basis")
    match.tier = "confirmed"
    match.confirmed_by = officer_id
    match.why += (f"; confirmed by {officer_id} on {basis} -- the appearance "
                  "descriptor did not and cannot make this determination")
    return match


@dataclass
class PersonReidResult:
    query: PersonSighting
    ranked: List[PersonMatch]
    implausible: List[PersonMatch]

    def best(self) -> Optional[PersonMatch]:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict:
        return {"query": {"sighting_id": self.query.sighting_id,
                          "camera_id": self.query.camera_id,
                          "signature": self.query.signature.to_dict()},
                "ranked": [m.to_dict() for m in self.ranked],
                "implausible": [m.to_dict() for m in self.implausible]}


def match_across_cameras(query: PersonSighting,
                         candidates: Iterable[PersonSighting],
                         sites: Mapping[str, CameraSite], *,
                         min_score: float = 0.0, top_k: int = 25,
                         include_same_camera: bool = False) -> PersonReidResult:
    """
    Rank sightings from other cameras as possible re-appearances of the query.

    Candidates that fail the transit check are removed from the ranking
    rather than ranked last, and returned separately -- "this person looks
    identical and could not have made that journey" is a lead in its own
    right (two people dressed alike, or a clock that needs fixing) and the
    officer should see it rather than have it silently dropped.
    """
    ranked: List[PersonMatch] = []
    impossible: List[PersonMatch] = []

    for c in candidates:
        if c.sighting_id == query.sighting_id:
            continue
        if not include_same_camera and c.camera_id == query.camera_id:
            continue
        m = assess(query, c, sites)
        if not m.transit.plausible:
            impossible.append(m)
        elif m.score >= min_score:
            ranked.append(m)

    ranked.sort(key=lambda m: (-TIER_ORDER.index(m.tier), -m.score))
    impossible.sort(key=lambda m: -m.appearance.overall)
    return PersonReidResult(query=query, ranked=ranked[:top_k],
                            implausible=impossible[:top_k])


def demote_for_clock(m: PersonMatch, note: str) -> PersonMatch:
    """Drop a match a tier because a camera's clock cannot be trusted."""
    m.tier = _demote(m.tier)
    m.why += f"; demoted: {note}"
    return m


# ------------------------------------------------------------------ measuring
#
# Below this line is the measurement that the operating points above are read
# off. It is run, not asserted:  python -m sentinel.person_reid --help


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
    track: int
    sig: PersonSignature
    box: Tuple[float, float, float, float]


def _track_people(path: str, max_frames: int, stride: int, camera_id: str,
                  iou_thr: float = 0.25, detect_scale: float = 1.0,
                  min_score: float = 0.70) -> Tuple[List[_Obs], dict]:
    """
    Identity ground truth from overlap alone.

    Government CCTV comes with no labels, so identity is recovered by
    linking person detections through the scene on box overlap. Linking on
    appearance would make the measurement circular -- grading the
    descriptor with the descriptor. Overlap is independent of every number
    this module computes.

    Detection runs at full resolution here, unlike reid's vehicle bench: a
    pedestrian at 120 px tall halves to 60 px, which is below the detector's
    useful range and below this module's own refusal floor.
    """
    from .detect import Detection, Detector

    cap = cv2.VideoCapture(str(path))
    obs: List[_Obs] = []
    active: List[List] = []
    next_id = idx = 0
    refused = kept = frames_used = 0
    heights: List[int] = []

    while idx < max_frames:
        ok, img = cap.read()
        if not ok or img is None:
            break
        idx += 1
        if idx % stride:
            continue
        frames_used += 1
        if detect_scale != 1.0:
            small = cv2.resize(img, None, fx=detect_scale, fy=detect_scale)
            dets = [Detection(d.label, d.score,
                              tuple(v / detect_scale for v in d.box))
                    for d in Detector.people(small, min_score=min_score)]
        else:
            dets = Detector.people(img, min_score=min_score)

        active = [t for t in active if idx - t[1] <= stride * 4]
        used = set()
        for d in dets:
            heights.append(d.height)
            best, best_iou = None, iou_thr
            for k, t in enumerate(active):
                if k in used:
                    continue
                v = _iou(d.box, t[2])
                if v > best_iou:
                    best, best_iou = k, v
            if best is None:
                tid = next_id
                next_id += 1
                active.append([tid, idx, tuple(d.box)])
            else:
                tid = active[best][0]
                active[best] = [tid, idx, tuple(d.box)]
                used.add(best)
            try:
                obs.append(_Obs(idx, tid, signature(img, d.box, camera_id),
                                tuple(d.box)))
                kept += 1
            except TooSmallToDescribe:
                refused += 1
    cap.release()
    heights.sort()
    stats = {"frames_read": idx, "frames_analysed": frames_used,
             "detections": kept + refused, "signatures": kept,
             "refused_too_small": refused,
             "person_height_px_p10": heights[int(len(heights) * 0.1)] if heights else None,
             "person_height_px_p50": heights[len(heights) // 2] if heights else None,
             "person_height_px_p90": heights[int(len(heights) * 0.9)] if heights else None}
    return obs, stats


def bench_real(path: str, max_frames: int = 300, stride: int = 5,
               min_separation: int = 15, camera_id: str = "cam-09") -> dict:
    """
    Separation on real footage: same person over time vs two people at once.

    Two facts real footage gives for free. A person followed through a
    scene is certainly the same person; two people visible in one frame are
    certainly not. That is a genuine same/different split on real
    pedestrians without anyone hand-labelling anything.

    What it does NOT measure is a change of camera, because no person here
    is known to appear in two clips. It measures the change of scale, pose
    and illumination a person undergoes walking through one scene -- the
    same kind of variation, so a fair proxy, but read it as optimistic.
    """
    obs, stats = _track_people(path, max_frames, stride, camera_id)

    by_track: Dict[int, List[_Obs]] = {}
    for o in obs:
        by_track.setdefault(o.track, []).append(o)
    for v in by_track.values():
        v.sort(key=lambda o: o.frame)

    same: List[float] = []
    for v in by_track.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[j].frame - v[i].frame >= min_separation:
                    same.append(appearance(v[i].sig, v[j].sig).overall)

    diff: List[float] = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if a.frame != b.frame or a.track == b.track:
                continue
            if _iou(a.box, b.box) >= 0.3:
                continue          # detector fired twice on one person
            diff.append(appearance(a.sig, b.sig).overall)

    # Rank-1 over a closed gallery: query is the last view of a track, the
    # gallery holds the first view of every track that has one.
    gallery = [(k, v[0]) for k, v in by_track.items() if len(v) >= 2]
    hit = tot = 0
    for k, v in by_track.items():
        if len(v) < 2 or v[-1].frame - v[0].frame < min_separation:
            continue
        pool = [(gk, go) for gk, go in gallery
                if not (gk == k and go.frame == v[-1].frame)]
        if len(pool) < 2:
            continue
        best = max(pool, key=lambda t: appearance(v[-1].sig, t[1].sig).overall)
        tot += 1
        hit += int(best[0] == k)

    def stat(xs):
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": len(xs), "mean": round(float(a.mean()), 3),
                "p10": round(float(np.percentile(a, 10)), 3),
                "p50": round(float(np.percentile(a, 50)), 3),
                "p90": round(float(np.percentile(a, 90)), 3)}

    curve = []
    for thr in (0.55, 0.70, APPEARANCE_PROBABLE, 0.85):
        curve.append({
            "threshold": thr,
            "same_person_recall": (round(sum(1 for s in same if s >= thr)
                                         / len(same), 3) if same else None),
            "false_match_rate": (round(sum(1 for d in diff if d >= thr)
                                       / len(diff), 3) if diff else None)})

    return {"bench": "real government footage (people)", "clip": path,
            "capture": stats, "tracks": len(by_track),
            "same_person": stat(same), "different_person": stat(diff),
            "separation_of_means": (round(stat(same)["mean"]
                                         - stat(diff)["mean"], 3)
                                    if same and diff else None),
            "rank1": {"correct": hit, "queries": tot,
                      "accuracy": round(hit / tot, 3) if tot else None},
            "threshold_curve": curve,
            "note": ("identity from IoU tracking, independent of the "
                     "descriptor under test. 'same person' = two views of "
                     f"one track at least {min_separation} frames apart; "
                     "'different person' = simultaneously visible, "
                     "non-overlapping boxes. Single camera only, so this "
                     "measures pose/scale/illumination change within one "
                     "scene, not a change of camera.")}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--real", default="media/real/cam09_day.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args(argv)
    if not os.path.exists(a.real):
        ap.error(f"no clip at {a.real}")
    print(json.dumps(bench_real(a.real, max_frames=a.frames,
                                stride=a.stride), indent=2))


if __name__ == "__main__":
    main()
