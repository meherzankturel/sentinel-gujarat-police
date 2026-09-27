#!/usr/bin/env python3
"""
sentinel.capability -- what a camera can actually do, measured.

The registry every other team will build records what a camera IS:
resolution, codec, location. That is a specification sheet, and a
specification sheet cannot tell you whether a number plate is readable,
because readability depends on how far the camera sits from the road.

This module records what a camera DELIVERS. It watches the live feed,
finds moving vehicles, locates the plate on them, and measures how many
pixels wide that plate actually is. Capability is then derived from the
measurement, never from the resolution.

Our own grid proves why this matters: cam-01 and cam-04 are both
1920x1080. One reads plates. The other cannot, because it is mounted
further back. Any classifier working from resolution calls them
identical -- and the resolution-based one we started with did exactly
that, labelling cam-04 "plate-likely".

Thresholds, stated openly so a reviewer can challenge them:

    >= 120 px plate width   plate-capable      OCR is reliable
    80 - 120 px             plate-marginal     partial reads; corroboration
    < 80 px                 presence-only      count and track, do not read

Licensing: OpenCV (Apache 2.0) only. No AGPL component is used anywhere
in this pipeline, which matters because AGPL in a state deployment is a
procurement problem, not a footnote.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .stream import LiveStream

PLATE_CAPABLE_PX = 120
PLATE_MARGINAL_PX = 80

# An Indian single-row plate is 500 x 120 mm -> about 4.2 : 1.
PLATE_ASPECT_MIN, PLATE_ASPECT_MAX = 2.8, 5.8
# Plate width relative to vehicle width, used only as a fallback estimate
# (0.5 m plate on a ~1.8 m car).
PLATE_TO_VEHICLE = 0.28


@dataclass
class PlateHit:
    width_px: int
    height_px: int
    sharpness: float
    source: str            # "direct" (plate located) or "derived" (from vehicle)
    bbox: Tuple[int, int, int, int]


class CameraMotion:
    """
    Detects the camera moving, as opposed to things moving in front of it.

    Several of these cameras are PTZ units that pan. While one pans, every
    building and hoarding in frame registers as a moving object, motion
    analytics produce nonsense, and a plate detector goes hunting through
    signage. Knowing a camera pans is itself a registry fact: it tells you
    that background-subtraction analytics are only valid between moves.

    Global shift is measured by phase correlation, which is cheap and
    answers exactly the right question -- did the whole picture translate?
    """

    def __init__(self, shift_px: float = 2.0):
        self.prev = None
        self.shift_px = shift_px
        self.moving_frames = 0
        self.total_frames = 0
        self.max_shift = 0.0

    def update(self, gray: np.ndarray) -> bool:
        small = cv2.resize(gray, (320, 180)).astype(np.float32)
        moving = False
        if self.prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, small)
            mag = float(np.hypot(dx, dy))
            self.max_shift = max(self.max_shift, mag)
            moving = mag > self.shift_px
            self.total_frames += 1
            if moving:
                self.moving_frames += 1
        self.prev = small
        return moving

    def pan_share(self) -> float:
        return (self.moving_frames / self.total_frames) if self.total_frames else 0.0

    def is_ptz(self) -> bool:
        return self.pan_share() > 0.06


@dataclass
class CapabilityReport:
    camera_id: str
    frames_seen: int = 0
    motion_events: int = 0
    pan_share: float = 0.0
    is_ptz: bool = False
    hits: List[PlateHit] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    night: bool = False
    brightness: Optional[float] = None

    # ---- the registry columns ----
    def plate_px(self) -> Optional[int]:
        """
        Representative plate width. The 75th percentile, not the maximum:
        one lucky close pass should not promote a camera that usually
        sees plates too small to read.
        """
        direct = [h.width_px for h in self.hits if h.source == "direct"]
        pool = direct or [h.width_px for h in self.hits]
        if not pool:
            return None
        pool.sort()
        k = max(0, int(round(0.75 * (len(pool) - 1))))
        return int(pool[k])

    def measurement_basis(self) -> str:
        if any(h.source == "direct" for h in self.hits):
            return "direct plate localisation"
        if self.hits:
            return "derived from vehicle width"
        return "no vehicles observed"

    def capability_class(self) -> str:
        px = self.plate_px()
        if px is None:
            return "unknown" if self.frames_seen else "unusable"
        if px >= PLATE_CAPABLE_PX:
            cls = "plate-capable"
        elif px >= PLATE_MARGINAL_PX:
            cls = "plate-marginal"
        else:
            cls = "presence-only"
        if self.night and cls == "plate-capable":
            # Legible by day is not legible by night.
            cls = "plate-marginal"
        return cls

    def why(self) -> str:
        px = self.plate_px()
        if px is None:
            return f"no plate observed in {self.frames_seen} frames"
        bits = [f"plate {px}px ({self.measurement_basis()})",
                f"{len(self.hits)} observation(s)"]
        if self.night:
            bits.append("low light")
        return "; ".join(bits)


# --------------------------------------------------------------- detection


# A plate is a small panel low on a vehicle. Indian streets are full of
# bright rectangles carrying dark text -- shop boards, hoardings, road
# signs -- and on synthetic footage, where the only such rectangle was a
# plate, none of this mattered. On real footage the detector cheerfully
# measured a sign reading "...onal Bridg..." as a 304px number plate.
# These proportions are what separates a plate from a signboard.
PLATE_TO_VEHICLE_MIN = 0.10     # narrower than this and it is a detail
PLATE_TO_VEHICLE_MAX = 0.45     # wider than this and it is not on the car
PLATE_HEIGHT_MAX = 0.35         # of the vehicle box
PLATE_TOP_LIMIT = 0.35          # plates sit low; ignore the upper third


def _find_plate_in_roi(gray_roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the plate inside a vehicle box.

    Indian plates are a bright panel carrying dark characters, so we look
    for a bright quadrilateral of roughly 4:1 whose interior has strong
    horizontal edge energy (that is the lettering), and which sits low on
    the vehicle at a plausible fraction of its width. Deliberately a
    classical detector: nothing to train, nothing to license, and its
    failure mode is "found nothing" rather than a confident hallucination.
    """
    if gray_roi.size == 0 or min(gray_roi.shape[:2]) < 6:
        return None
    rh, rw = gray_roi.shape[:2]

    blur = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 3:
            continue
        ar = w / float(h)
        if not (PLATE_ASPECT_MIN <= ar <= PLATE_ASPECT_MAX):
            continue
        # Proportion and position on the vehicle, not just shape.
        if not (PLATE_TO_VEHICLE_MIN <= w / float(rw) <= PLATE_TO_VEHICLE_MAX):
            continue
        if h > rh * PLATE_HEIGHT_MAX:
            continue
        if (y + h / 2.0) < rh * PLATE_TOP_LIMIT:
            continue
        patch = gray_roi[y:y + h, x:x + w]
        if patch.size == 0:
            continue
        # bright panel, dark glyphs -> high mean, high internal contrast
        contrast = float(patch.std())
        brightness = float(patch.mean())
        if brightness < 90 or contrast < 18:
            continue
        score = (w * h) * (contrast / 128.0)
        if score > best_score:
            best, best_score = (x, y, w, h), score
    return best


def measure_frames(frames, camera_id: str) -> CapabilityReport:
    """
    Measure from any source of frames.

    Split out from measure() so the same logic runs against a live stream
    and against locally sampled HLS segments -- the measurement must not
    depend on how the pixels arrived.
    """
    rep = CapabilityReport(camera_id=camera_id)
    bgs = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32,
                                             detectShadows=False)
    cammot = CameraMotion()
    brightness = []
    for img in frames:
        rep.frames_seen += 1
        if rep.width is None:
            rep.height, rep.width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if rep.frames_seen % 5 == 0:
            brightness.append(float(gray.mean()))

        panning = cammot.update(gray)
        mask = bgs.apply(img)
        if rep.frames_seen < 25:
            continue
        if panning:
            # The picture itself moved. Nothing measured from this frame
            # describes traffic, so it is skipped rather than recorded.
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        frame_area = img.shape[0] * img.shape[1]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < frame_area * 0.004 or area > frame_area * 0.6:
                continue
            if w < h * 0.9:
                continue
            rep.motion_events += 1
            roi = gray[y:y + h, x:x + w]
            found = _find_plate_in_roi(roi)
            if found:
                px, py, pw, ph = found
                patch = gray[y + py:y + py + ph, x + px:x + px + pw]
                sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var()) \
                    if patch.size else 0.0
                rep.hits.append(PlateHit(pw, ph, sharp, "direct",
                                         (x + px, y + py, pw, ph)))
            else:
                est = int(w * PLATE_TO_VEHICLE)
                if est >= 6:
                    rep.hits.append(PlateHit(est, max(2, int(est * 0.24)),
                                             0.0, "derived", (x, y, w, h)))

    rep.pan_share = round(cammot.pan_share(), 3)
    rep.is_ptz = cammot.is_ptz()
    if brightness:
        rep.brightness = round(statistics.fmean(brightness), 1)
        rep.night = rep.brightness < 70
    return rep


def measure_file(path, camera_id: str, max_frames: int = 900) -> CapabilityReport:
    """Measure from a local video file or playlist."""
    def gen():
        cap = cv2.VideoCapture(str(path))
        try:
            n = 0
            while n < max_frames:
                ok, img = cap.read()
                if not ok or img is None:
                    return
                n += 1
                yield img
        finally:
            cap.release()
    return measure_frames(gen(), camera_id)


def measure(url: str, camera_id: str, seconds: float = 25.0,
            log=None) -> CapabilityReport:
    """Watch one camera and measure what it can deliver."""
    rep = CapabilityReport(camera_id=camera_id)
    log = log or (lambda m: None)

    # Static CCTV: background subtraction finds the vehicles without any
    # trained model, which is also the cheap "presence" analytic we assign
    # to weak cameras.
    bgs = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=32,
                                             detectShadows=False)
    brightness = []

    with LiveStream(url, camera_id, log=log) as s:
        for f in s.frames(max_seconds=seconds):
            img = f.image
            rep.frames_seen += 1
            if rep.width is None:
                rep.height, rep.width = img.shape[:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if rep.frames_seen % 5 == 0:
                brightness.append(float(gray.mean()))

            mask = bgs.apply(img)
            if rep.frames_seen < 25:          # let the model settle
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            frame_area = img.shape[0] * img.shape[1]
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                # plausible vehicle: big enough to matter, wider than tall
                if area < frame_area * 0.004 or area > frame_area * 0.6:
                    continue
                if w < h * 0.9:
                    continue
                rep.motion_events += 1

                roi = gray[y:y + h, x:x + w]
                found = _find_plate_in_roi(roi)
                if found:
                    px, py, pw, ph = found
                    patch = gray[y + py:y + py + ph, x + px:x + px + pw]
                    sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var()) \
                        if patch.size else 0.0
                    rep.hits.append(PlateHit(pw, ph, sharp, "direct",
                                             (x + px, y + py, pw, ph)))
                else:
                    est = int(w * PLATE_TO_VEHICLE)
                    if est >= 6:
                        rep.hits.append(PlateHit(est, max(2, int(est * 0.24)),
                                                 0.0, "derived", (x, y, w, h)))

    if brightness:
        rep.brightness = round(statistics.fmean(brightness), 1)
        rep.night = rep.brightness < 70
    return rep
