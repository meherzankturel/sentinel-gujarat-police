#!/usr/bin/env python3
"""
sentinel.track -- follow each vehicle, and read its plate at the right moment.

Reading plates from arbitrary single frames is why our first measured yield
on cam01 was 1.7%. A vehicle crossing frame gives thirty to sixty separate
looks at the same plate, and they are not equally good: the plate grows as
the vehicle approaches, is sharpest at one particular moment, and is
motion-blurred at others. Picking a frame at random throws almost all of
that away.

So each vehicle is tracked, every look at its plate is kept, and the read
happens once per vehicle using the best evidence available:

  * best-frame selection -- read at closest approach, not at random
  * multi-frame fusion   -- stack aligned looks at the same plate, which
                            recovers detail no single frame holds

The second is ordinary super-resolution: sensor noise is independent
between frames while the plate itself is not, so averaging aligned crops
raises the signal and cancels the noise. It is what lets a plate that is
marginal in every individual frame become readable once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _character_flips(crop: np.ndarray) -> int:
    """
    How many times the crop alternates light/dark across its width.

    A row of characters does this many times -- measured at 22 to 37 on
    real plates. A bumper shadow or a blank panel does it once or twice.
    It is the cheapest available evidence that a crop contains writing,
    and it is what stops a wide smear being chosen over a sharp plate.
    """
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
        return 0
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = g < (float(g.mean()) - 20)
    col = dark.mean(axis=0)
    return int(np.sum(np.abs(np.diff((col > col.mean()).astype(np.int8)))))


@dataclass
class PlateLook:
    """One frame's view of a plate belonging to a tracked vehicle."""
    crop: np.ndarray
    width: int
    sharpness: float
    frame_index: int

    def quality(self) -> float:
        """
        How much this look is worth reading.

        Size still matters -- characters that were never sampled cannot be
        recovered -- but it is not the only thing, and the previous form
        made it the only thing by accident. It multiplied width by a
        sharpness term capped at 120, and on close footage every crop
        clears 120 comfortably (measured range here: 61 to 1776). The term
        saturated, quality collapsed to width alone, and the widest crop
        won every time.

        The widest crop is frequently the worst one. Where localisation is
        hard, the frames it gets wrong produce a smear across the bumper
        that is wider than the plate it was cut from. On one real track, 18
        of 29 looks read the registration correctly and the selected look
        was a 378px blur that read nothing.

        So: size, scaled by how sharp the crop actually is and by whether
        it carries characters at all. The sharpness term no longer
        saturates, and character evidence can veto a large soft smear.
        """
        sharp_fit = self.sharpness / (self.sharpness + 400.0)
        flip_fit = min(1.0, _character_flips(self.crop) / 20.0)
        return self.width * sharp_fit * flip_fit


# A detector will happily call a roadside hoarding a bus. An audit of real
# footage from these cameras found an "ADVERTISE HERE" billboard tracked as
# a bus under two separate ids, and parked auto-rickshaws tracked as
# vehicles. Every one of those becomes a false sighting on somebody's route,
# which is worse than missing a real one.
#
# The distinguishing property is not appearance -- a parked rickshaw looks
# exactly like a moving rickshaw -- it is displacement. A vehicle in traffic
# travels several times its own width across a track; scenery does not move
# at all, and a parked vehicle only jitters with the detector's box.
# Measured as SPEED, not total distance. Total distance punishes a track
# that was only seen briefly: a car glimpsed for six frames has not travelled
# far because it was barely watched, not because it is parked. Judging on
# total displacement threw away 39 of 65 real tracks on cam05.
#
# In widths-per-frame, the separation on real footage is stark: the
# billboard managed 0.0015, a vehicle in traffic manages 0.05 and upward.
STATIC_SPEED = 0.012           # own-widths per frame


@dataclass
class Track:
    track_id: int
    label: str
    box: Tuple[float, float, float, float]
    last_seen: int
    frames: int = 1
    looks: List[PlateLook] = field(default_factory=list)
    boxes: List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)
    appearance: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    split_at: Optional[int] = None      # frame where identity changed

    @property
    def width(self) -> int:
        return int(self.box[2] - self.box[0])

    def best_look(self) -> Optional[PlateLook]:
        return max(self.looks, key=lambda l: l.quality()) if self.looks else None

    # ------------------------------------------------------- plausibility

    def centres(self):
        return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for _, b in self.boxes]

    def displacement(self) -> float:
        """How far the object travelled, in multiples of its own width."""
        cs = self.centres()
        if len(cs) < 2:
            return 0.0
        widths = [b[2] - b[0] for _, b in self.boxes]
        w = float(np.median(widths)) if widths else 1.0
        if w <= 0:
            return 0.0
        span = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a in cs for b in (cs[0], cs[-1]))
        return span / w

    def speed(self) -> float:
        """Own-widths travelled per frame."""
        span = max(1, self.boxes[-1][0] - self.boxes[0][0]) if len(self.boxes) > 1 else 1
        return self.displacement() / span

    def is_static(self) -> bool:
        """
        Scenery, or something parked. Either way it is not traffic.

        Deliberately NOT used to delete the track. A parked vehicle carrying
        a watchlist plate is of great interest to an officer, and a system
        that silently drops it is worse than one that never looked. It is
        recorded and marked stationary, so it never enters a route as a
        passing sighting and never contributes movement it did not make.
        """
        return self.frames >= 8 and self.speed() < STATIC_SPEED

    def note_appearance(self, frame_index: int, crop: np.ndarray):
        """
        Keep a cheap colour signature per frame so an identity switch can be
        detected later. A full re-identification embedding per frame would be
        far too expensive to run on every camera; a colour histogram costs
        almost nothing and catches the failure we actually see -- a track
        sliding off a green rickshaw onto a dark car.
        """
        if crop is None or crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        self.appearance.append((frame_index, h))

    def identity_break(self, threshold: float = 0.45) -> Optional[int]:
        """
        The frame at which this track stopped being one object.

        Compared against a running average of what the track looked like
        rather than the immediately preceding frame: a vehicle changes
        gradually as it turns, and only a swap changes it abruptly.
        """
        if len(self.appearance) < 6:
            return None
        acc = self.appearance[0][1].copy()
        for i, (frame_index, h) in enumerate(self.appearance[1:], 1):
            sim = float(cv2.compareHist(acc.astype("float32"),
                                        h.astype("float32"), cv2.HISTCMP_CORREL))
            if i >= 3 and sim < threshold:
                return frame_index
            acc = 0.8 * acc + 0.2 * h
        return None


class VehicleTracker:
    """
    Deliberately simple association by overlap.

    A heavier tracker buys little here: cameras are fixed, vehicles move
    predictably across frame, and what matters downstream is only that the
    looks belonging to one vehicle stay grouped together. Complexity in the
    tracker would be complexity we then have to defend.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 12):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: Dict[int, Track] = {}
        self._retired: List[Track] = []
        self._next = 1

    def update(self, detections, frame_index: int) -> List[Track]:
        unmatched = list(detections)
        for t in list(self.tracks.values()):
            best, best_iou = None, self.iou_threshold
            for d in unmatched:
                s = iou(t.box, d.box)
                if s > best_iou:
                    best, best_iou = d, s
            if best is not None:
                t.box = best.box
                t.label = best.label
                t.last_seen = frame_index
                t.frames += 1
                t.boxes.append((frame_index, best.box))
                unmatched.remove(best)

        for d in unmatched:
            t = Track(self._next, d.label, d.box, frame_index)
            t.boxes.append((frame_index, d.box))
            self.tracks[self._next] = t
            self._next += 1

        # A track that leaves frame is retired, not discarded. Deleting it
        # here lost every vehicle that drove out of shot -- which is nearly
        # all of them -- and left only whatever happened to still be on
        # screen when the clip ended.
        for tid, t in list(self.tracks.items()):
            if frame_index - t.last_seen > self.max_missing:
                self._retired.append(t)
                del self.tracks[tid]
        return list(self.tracks.values())

    def finished(self, frame_index: int) -> List[Track]:
        """Drain the tracks that have left frame since the last call."""
        out, self._retired = self._retired, []
        return out


# ------------------------------------------------------------------ fusion


def fuse_looks(looks: List[PlateLook], target_h: int = 64,
               max_frames: int = 12) -> Optional[np.ndarray]:
    """
    Stack several looks at one plate into a single sharper image.

    The looks are upscaled to a common size, aligned to the best of them,
    and averaged. Alignment is essential -- averaging unaligned crops just
    blurs. Noise is independent frame to frame while the characters are
    not, so the characters survive the average and the noise does not.
    """
    if not looks:
        return None
    ordered = sorted(looks, key=lambda l: -l.quality())[:max_frames]
    ref = ordered[0]
    ar = max(1.5, ref.crop.shape[1] / max(1, ref.crop.shape[0]))
    size = (int(target_h * ar), target_h)

    def prep(c):
        g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, size, interpolation=cv2.INTER_CUBIC).astype(np.float32)

    base = prep(ref.crop)
    stack, weights = [base], [ref.quality()]

    for l in ordered[1:]:
        cand = prep(l.crop)
        try:
            # Translation only: within one vehicle pass the plate barely
            # rotates, and allowing more freedom lets a poor look drag the
            # alignment off rather than contribute to it.
            warp = np.eye(2, 3, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
            cv2.findTransformECC(base, cand, warp, cv2.MOTION_TRANSLATION, crit, None, 3)
            aligned = cv2.warpAffine(cand, warp, size,
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            continue                      # refused to align; drop it
        stack.append(aligned)
        weights.append(l.quality())

    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    fused = np.tensordot(w, np.stack(stack), axes=(0, 0))
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    # Fusion recovers detail but leaves it soft; a mild unsharp mask puts
    # the character edges back without amplifying what was never there.
    blur = cv2.GaussianBlur(fused, (0, 0), 1.2)
    return cv2.addWeighted(fused, 1.6, blur, -0.6, 0)
