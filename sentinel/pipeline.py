#!/usr/bin/env python3
"""
sentinel.pipeline -- the service that actually watches a camera.

Everything else in this package is a component. This is the thing that
runs unattended: open a camera, detect vehicles, follow each one, read its
plate once using the best evidence that vehicle ever offered, match against
the watchlist, and write what it found.

Two decisions worth stating.

The registry decides whether plate reading is attempted at all. On a camera
measured at 12 pixels of plate width, an attempted read does not fail
politely -- it succeeds at producing a plausible wrong plate, and a wrong
plate on a police alert is worse than no plate.

A vehicle produces one sighting, not one per frame. Sixty frames of the
same car is one event to an officer, and sixty rows in an alert queue is
how a control room learns to stop reading the alert queue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .anpr import (PLATE_SHAPE, PlateRead, consensus,
                   locate_plate_in_vehicle, looks_like_plate,
                   score, _ocr_plate)
from .detect import Detector, osd_mask
from .track import PlateLook, Track, VehicleTracker, fuse_looks

READ_ATTEMPT_PX = 45          # below this there is nothing to attempt


@dataclass
class VehicleSighting:
    camera_id: str
    track_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    frames: int
    best_vehicle_px: int
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    plate_px: int = 0
    plate_tier: str = "none"
    read_method: str = "not attempted"
    looks: int = 0

    # Which frame the read actually came from, and where the vehicle sat in
    # it. The sighting's own timestamp is the track's *arrival*, which is
    # what a route wants; it is rarely the frame that read best. An officer
    # looking at a row asks a different question -- "show me the vehicle" --
    # and that question is answered by this frame, not by arrival. Carried
    # here so nothing has to guess at it afterwards.
    best_look_frame: Optional[int] = None
    best_look_box: Optional[tuple] = None
    best_look_crop: Optional["np.ndarray"] = None   # the pixels OCR read


class CameraWorker:
    """One camera, watched continuously."""

    def __init__(self, camera_id: str, capability: str = "unknown",
                 attempt_plates: bool = True, log: Optional[Callable] = None):
        self.camera_id = camera_id
        self.capability = capability
        self.attempt_plates = attempt_plates
        self.tracker = VehicleTracker()
        self.log = log or (lambda m: None)
        self.sightings: List[VehicleSighting] = []
        self._mask = None

    # ------------------------------------------------------------- frames

    def process(self, frames, max_frames: Optional[int] = None) -> List[VehicleSighting]:
        n = 0
        for img in frames:
            n += 1
            if max_frames and n > max_frames:
                break
            if self._mask is None:
                self._mask = osd_mask(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = [d for d in Detector.vehicles(img, min_score=0.5)
                    if self._outside_osd(d.box, img.shape)]
            live = self.tracker.update(dets, n)

            if self.attempt_plates:
                for t in live:
                    if t.last_seen != n:
                        continue
                    if t.width < READ_ATTEMPT_PX * 3:
                        continue
                    found = locate_plate_in_vehicle(gray, t.box)
                    if not found:
                        continue
                    px, py, pw, ph = found
                    if pw < READ_ATTEMPT_PX:
                        continue
                    crop = gray[py:py + ph, px:px + pw]
                    if crop.size == 0:
                        continue
                    # A look is picked for reading by its width, so one bad
                    # frame -- a bumper shadow, wider than the plate it sat
                    # under -- can outrank every good one and read as
                    # nothing. Keep only crops that carry characters.
                    if not looks_like_plate(crop):
                        continue
                    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
                    t.looks.append(PlateLook(crop.copy(), pw, sharp, n))

            for t in self.tracker.finished(n):
                self._close(t, n)

        for t in list(self.tracker.tracks.values()):
            self._close(t, n)
        return self.sightings

    def _outside_osd(self, box, shape) -> bool:
        """Ignore anything sitting inside the burned-in caption bands."""
        h = shape[0]
        _, y1, _, y2 = box
        return not (y2 < h * 0.09 or y1 > h * 0.88)

    # -------------------------------------------------------------- close

    def _close(self, t: Track, frame_index: int):
        if t.track_id in {s.track_id for s in self.sightings}:
            return
        if t.frames < 3:
            return                       # a flicker, not a vehicle

        s = VehicleSighting(
            camera_id=self.camera_id, track_id=t.track_id, vehicle_type=t.label,
            first_frame=t.boxes[0][0], last_frame=t.last_seen, frames=t.frames,
            best_vehicle_px=max(int(b[2] - b[0]) for _, b in t.boxes),
            looks=len(t.looks))

        best = t.best_look()
        if best is not None:
            # Read the best single look and the fusion of the best looks,
            # then keep whichever the OCR was more confident about. Fusion
            # usually wins on a distant plate and occasionally loses on a
            # very clean one, so we do not assume.
            candidates = []

            txt, conf = _ocr_plate(best.crop)
            if txt:
                candidates.append(("best frame", txt, conf, best.width, best.sharpness))

            fused = fuse_looks(t.looks)
            if fused is not None:
                ftxt, fconf = _ocr_plate(fused)
                if ftxt:
                    candidates.append(("fused %d looks" % min(12, len(t.looks)),
                                       ftxt, fconf, best.width, best.sharpness))

            if candidates:
                # Prefer the fused reading where it produced one. It draws on
                # evidence no single frame contains -- on cam09 it recovered
                # nine of ten characters from a plate that was 73px in every
                # individual frame -- and letting a lucky single frame
                # outrank it by confidence alone measured worse.
                # A string shaped like an Indian registration is stronger
                # evidence than one that is not, whatever produced it. Where
                # the plate's scale changes a lot across a track -- a vehicle
                # approaching the camera -- the fused image is built from
                # crops that do not align, and it degrades: on one real track
                # fusion returned SJISBEI659 while the best single frame read
                # GJ19BE3669. Preferring fusion blindly threw the good read
                # away. Well-formedness decides first; fusion still wins
                # among equals, which is where it was measured to help.
                well = [c for c in candidates if PLATE_SHAPE.match(c[1])]
                pool = well or candidates
                fused_first = [c for c in pool if c[0].startswith("fused")]
                method, txt, conf, pw, sharp = max(
                    fused_first or pool, key=lambda c: c[2])
                s.plate_text = txt
                s.plate_px = pw
                s.read_method = method
                s.plate_confidence = score(txt, conf, pw, sharp)
                r = PlateRead(txt, s.plate_confidence, pw, sharp, (0, 0, 0, 0),
                              conf, bool(__import__("re").match(
                                  r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", txt)))
                s.plate_tier = r.tier()
                s.best_look_frame = best.frame_index
                # The pixels the reader actually worked from. Where fusion
                # won, that is the fused image and not any single frame --
                # showing the best single look instead produced a crop of a
                # rear bumper captioned as the plate evidence, which is
                # exactly the sort of thing a defence solicitor is paid to
                # find. The fused image is what was read, so it is what is
                # kept.
                s.best_look_crop = (fused if fused is not None
                                    and method.startswith("fused")
                                    else best.crop)
                # The vehicle's own box in that frame, so the console can
                # show the car and not the whole road.
                at = min(t.boxes, key=lambda fb: abs(fb[0] - best.frame_index))
                s.best_look_box = tuple(int(v) for v in at[1])
        elif self.attempt_plates:
            s.read_method = "no plate large enough"

        self.sightings.append(s)
        self.log(f"{self.camera_id} track {t.track_id} {t.label} "
                 f"{s.frames}f plate={s.plate_text} ({s.read_method})")


def frames_from(path_or_url, max_frames: int = 2000):
    cap = cv2.VideoCapture(str(path_or_url))
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
