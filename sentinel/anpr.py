#!/usr/bin/env python3
"""
sentinel.anpr -- read number plates, and say honestly how sure we are.

A plate read is never a boolean here. Every read carries a confidence
derived from things we actually measured: how many pixels wide the plate
was, how sharp it was, and how confident the OCR was character by
character. That confidence is what lets the alerting layer distinguish a
sighting worth waking a control room for from one that should only
corroborate an existing track.

The registry decides whether this runs at all. On a camera measured at 44
pixels of plate width, reading is not attempted -- not because it would
fail politely, but because it would succeed at producing a plausible
wrong answer, and a wrong plate on a police alert is worse than no plate.

Detector and OCR are both permissively licensed: OpenCV (Apache 2.0) and
Tesseract (Apache 2.0). Nothing here is AGPL.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .capability import _find_plate_in_roi, PLATE_CAPABLE_PX, PLATE_MARGINAL_PX

# An Indian registration is SS DD LL NNNN: state, district, series, number.
# Matching is done on the normal form, so this is only used for scoring how
# plausible a read is.
PLATE_SHAPE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class PlateRead:
    text: str
    confidence: float           # 0-1, and deliberately conservative
    plate_px: int
    sharpness: float
    bbox: Tuple[int, int, int, int]
    ocr_conf: float
    well_formed: bool

    def tier(self) -> str:
        """
        What this read is allowed to do downstream.

        confirmed     -> may raise an alert on its own
        probable      -> may raise a lower-tier alert
        corroborating -> may only strengthen an existing track, never start one
        """
        if self.confidence >= 0.80 and self.well_formed:
            return "confirmed"
        if self.confidence >= 0.55:
            return "probable"
        return "corroborating"


def tidy_plate(text: str) -> str:
    """
    Trim a raw OCR string down to the registration inside it.

    A plate crop always carries a little of its surround -- a screw head,
    the border of the panel, a bumper edge -- and Tesseract renders those
    as extra characters. A real read of GJ03KH8510 came back as
    LGJ03KH0510: correct apart from one leading character that was never
    part of the plate. Discarding the whole read for that would be absurd.

    Shape is judged on the normalised form, because the raw string has a
    letter O where a digit belongs and so never matches the pattern.
    normalise_plate is a one-to-one character mapping, so indices still
    line up with the original text. Nothing is invented: only characters
    the OCR actually produced can survive.
    """
    from .registry import normalise_plate

    t = "".join(ch for ch in text.upper() if ch in PLATE_CHARS)
    norm = normalise_plate(t)
    if len(t) <= 10 and PLATE_SHAPE.match(norm):
        return t
    for length in (10, 9):
        for i in range(0, max(1, len(t) - length + 1)):
            if i + length > len(t):
                break
            if PLATE_SHAPE.match(norm[i:i + length]):
                return t[i:i + length]
    return t


def _ocr_variants(crop: np.ndarray) -> List[np.ndarray]:
    """
    Several preparations of the same crop.

    Measured on a real plate from cam09 (GJ03KH8510): no single preparation
    read it correctly, but different ones got different halves right --
    Otsu at a small scale returned GJ03KH0519, the unthresholded image at a
    large scale returned FG03KN8510. Between them every character was
    present. So we produce several and let them vote, rather than betting
    the read on one guess about what this particular plate needs.

    Scale matters more than expected: Tesseract wants a character band of
    roughly 48 pixels. Enlarging to 120px, which seemed the safe choice,
    was actively losing reads.
    """
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    out: List[np.ndarray] = []
    for target_h in (48, 96, 160):
        sc = target_h / max(1, g.shape[0])
        up = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(cv2.bilateralFilter(up, 5, 45, 45), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(th)
        out.append(cv2.bitwise_not(th))
        if target_h >= 96:
            out.append(up)                                   # unthresholded
            out.append(cv2.createCLAHE(3.0, (8, 8)).apply(up))
    return out


def _ocr_plate(crop: np.ndarray) -> Tuple[str, float]:
    """
    Read the plate crop, and say honestly how much to trust the reading.

    Confidence does NOT come from Tesseract here. This build reports a
    per-word confidence of exactly 0.000000 for every plate crop we give
    it, so a "keep the most confident reading" rule selected nothing at
    all. Relying on a number without checking it produces meant the reader
    silently returned empty on every vehicle.

    What we use instead is reproducibility: the crop is prepared several
    ways, each is read, and a string that several independent preparations
    agree on is worth more than one that appears once. That is a real
    signal about the image rather than a self-report from the recogniser.

    Scale is the setting that mattered most. Tesseract wants a character
    band of roughly 48 pixels; enlarging crops to 120px, which had seemed
    the safe choice, was quietly costing reads.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return "", 0.0

    seen: dict[str, int] = {}
    attempts = 0
    for prep in _ocr_variants(crop):
        padded = cv2.copyMakeBorder(
            prep, 14, 14, 14, 14, cv2.BORDER_CONSTANT,
            value=255 if prep.mean() > 127 else 0)
        ok, buf = cv2.imencode(".png", padded)
        if not ok:
            continue
        for psm in ("6", "7", "8"):
            attempts += 1
            try:
                r = subprocess.run(
                    ["tesseract", "stdin", "stdout", "--psm", psm, "-c",
                     f"tessedit_char_whitelist={PLATE_CHARS}"],
                    input=buf.tobytes(), capture_output=True, timeout=20)
            except Exception:
                continue
            raw = "".join(ch for ch in r.stdout.decode("utf-8", "replace").upper()
                          if ch in PLATE_CHARS)
            cand = tidy_plate(raw)
            # INATTTITITTT survived an earlier version of this filter purely
            # by being long enough. A string that cannot be a registration
            # is noise, and letting it through gives an officer something
            # to chase that was never on a vehicle.
            from .registry import normalise_plate as _n
            if 8 <= len(cand) <= 10 and PLATE_SHAPE.match(_n(cand)):
                seen[cand] = seen.get(cand, 0) + 1

    if not seen:
        return "", 0.0

    # Prefer a reading shaped like a registration, then one that recurred.
    def rank(item):
        text, count = item
        from .registry import normalise_plate
        shaped = 1 if PLATE_SHAPE.match(normalise_plate(text)) else 0
        return (shaped, count, -abs(len(text) - 10))

    text, count = max(seen.items(), key=rank)
    agreement = count / max(1, attempts)
    from .registry import normalise_plate
    shaped = bool(PLATE_SHAPE.match(normalise_plate(text)))
    conf = min(1.0, (0.35 + 0.65 * min(1.0, agreement * 4)) * (1.0 if shaped else 0.7))
    return text, round(conf, 3)


def score(text: str, ocr_conf: float, plate_px: int, sharpness: float) -> float:
    """
    Combine what we know into one conservative number.

    Pixels-on-target dominates, because it is the thing that physically
    limits whether the characters were ever recoverable. OCR confidence on
    a 40-pixel plate is confidence in a guess.
    """
    if not text:
        return 0.0

    px_factor = min(1.0, plate_px / float(PLATE_CAPABLE_PX))
    if plate_px < PLATE_MARGINAL_PX:
        px_factor *= 0.45                     # below this, treat as a hint

    len_factor = 1.0 if len(text) >= 9 else (0.75 if len(text) >= 7 else 0.45)
    shape_factor = 1.0 if PLATE_SHAPE.match(text) else 0.7
    sharp_factor = 1.0 if sharpness >= 60 else (0.85 if sharpness >= 25 else 0.6)

    return round(min(1.0, ocr_conf * px_factor * len_factor
                     * shape_factor * sharp_factor), 3)


def read_plates(frame: np.ndarray, fg_mask: np.ndarray,
                min_plate_px: int = PLATE_MARGINAL_PX) -> List[PlateRead]:
    """Find vehicles in the motion mask, locate their plates, read them."""
    out: List[PlateRead] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = frame.shape[:2]
    frame_area = fh * fw

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < frame_area * 0.004 or w * h > frame_area * 0.6:
            continue
        if w < h * 0.9:
            continue

        found = _find_plate_in_roi(gray[y:y + h, x:x + w])
        if not found:
            continue
        px, py, pw, ph = found
        if pw < min_plate_px:
            # Measured too small to read. Declining is the correct answer.
            continue

        crop = gray[y + py:y + py + ph, x + px:x + px + pw]
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        text, oconf = _ocr_plate(crop)
        if not text:
            continue

        out.append(PlateRead(
            text=text, ocr_conf=round(oconf, 3), plate_px=pw, sharpness=round(sharp, 1),
            bbox=(x + px, y + py, pw, ph),
            well_formed=bool(PLATE_SHAPE.match(text)),
            confidence=score(text, oconf, pw, sharp)))
    return out


@dataclass
class PlateConsensus:
    """One vehicle's plate, agreed across the frames it was visible in."""
    text: str
    confidence: float
    reads: int
    agreeing: int
    agreement: float
    best_plate_px: int
    best_sharpness: float
    variants: List[Tuple[str, int]]

    def tier(self) -> str:
        if self.confidence >= 0.80 and PLATE_SHAPE.match(self.text):
            return "confirmed"
        if self.confidence >= 0.55:
            return "probable"
        return "corroborating"


def consensus(reads: List[PlateRead]) -> Optional[PlateConsensus]:
    """
    Consolidate every read of one vehicle into a single verdict.

    A plate glimpsed once at 0.69 confidence is a guess. The same plate
    read thirty times, agreeing with itself while the vehicle crosses the
    frame, is evidence -- so repeated independent agreement raises
    confidence, and disagreement between frames holds it down. This is
    also what stops a single lucky frame on a weak camera from promoting
    itself: agreement there is low, so the ceiling stays low.
    """
    if not reads:
        return None

    from .matching import edit_distance

    counts: dict[str, int] = {}
    for r in reads:
        counts[r.text] = counts.get(r.text, 0) + 1
    variants = sorted(counts.items(), key=lambda kv: -kv[1])

    # A partial read is not evidence against a plate, it is less evidence
    # for it. "1AB1234" does not contradict "GJ01AB1234" -- the camera saw
    # the same characters and missed the leading two. Counting those as
    # disagreement made a genuinely capable camera look uncertain.
    full = [r for r in reads if len(r.text) >= 8]
    if not full:
        full = reads

    # Cluster reads that differ by a single character: one OCR slip in one
    # frame is the same reading, not a competing one.
    best_cluster, best_key = [], None
    for cand in {r.text for r in full}:
        cluster = [r for r in full if edit_distance(r.text, cand) <= 1]
        if len(cluster) > len(best_cluster):
            best_cluster, best_key = cluster, cand
    modal = best_cluster or full
    # Canonical spelling is the most confident member of the cluster.
    text = max(modal, key=lambda r: r.confidence).text
    agreeing = len(modal)

    base = sum(r.confidence for r in modal) / len(modal)
    agreement = agreeing / len(full)
    support = min(1.0, agreeing / 5.0)         # five agreeing reads is full support

    conf = base + (1.0 - base) * 0.6 * support * agreement

    return PlateConsensus(
        text=text, confidence=round(min(1.0, conf), 3),
        reads=len(reads), agreeing=agreeing, agreement=round(agreement, 3),
        best_plate_px=max(r.plate_px for r in modal),
        best_sharpness=max(r.sharpness for r in modal),
        variants=variants[:4])


# --------------------------------------------------- plate inside a vehicle


# The government grid delivers vehicles up to about 760px wide. Above that
# we are outside the regime every constant in locate_plate_in_vehicle was
# tuned against, and in practice well outside it: on a close, sunlit car the
# bonnet, bumper and plate all threshold to white and merge into a single
# blob, so the plate ends up with no contour of its own. Measured on a
# 1021px-wide vehicle, the shipped method does not place the true plate among
# its candidates at all.
CLOSE_VEHICLE_PX = 800


def _plate_ink(patch: np.ndarray) -> Tuple[float, int, float]:
    """
    What separates a plate from every other bright patch on a car.

    A bonnet highlight is bright and blank. A plate is bright and carries a
    row of dark characters, and those characters alternate along the row in
    a way a smear or a shadow does not. Both facts are measured here: the
    share of dark pixels, and how many times the dark/light run flips
    across the width.
    """
    if patch.size == 0 or min(patch.shape[:2]) < 6:
        return 0.0, 0, 0.0
    mean = float(patch.mean())
    dark = patch < (mean - 20)
    col = dark.mean(axis=0)
    flips = int(np.sum(np.abs(np.diff((col > col.mean()).astype(np.int8)))))
    return float(dark.mean()), flips, mean


def _locate_close_vehicle(roi_eq: np.ndarray, vw: int
                          ) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate a plate on a vehicle that fills the frame.

    Otsu keeps the light panels; opening drops the speckle. That reliably
    puts the plate among the candidates -- it was rank 20 of 100 by the old
    "widest bright thing" scoring, which is the bonnet. Ranking on ink
    rather than width puts it first, by a factor of four.
    """
    _, th = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    best, best_score = None, 0.0
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.06 * vw or h < 8:
            continue
        ar = w / float(h)
        if not (1.8 <= ar <= 7.0):
            continue
        rel = w / float(vw)
        if not (0.08 <= rel <= 0.55):
            continue

        patch = roi_eq[y:y + h, x:x + w]
        ink, flips, bright = _plate_ink(patch)
        fill = cv2.contourArea(c) / float(w * h)

        ar_fit = 1.0 / (1.0 + abs(ar - 4.3) * 0.8)
        rel_fit = 1.0 / (1.0 + abs(rel - 0.20) * 6.0)
        ink_fit = 1.0 if 0.10 <= ink <= 0.55 else 0.0
        flip_fit = min(1.0, flips / 14.0)          # ~10 characters ≈ 20 flips
        bright_fit = min(1.0, max(0.0, (bright - 70) / 90.0))
        rect_fit = min(1.0, fill / 0.75)

        score_ = (ar_fit * rel_fit * (0.15 + ink_fit) * (0.2 + flip_fit)
                  * (0.3 + bright_fit) * (0.4 + rect_fit))
        if score_ > best_score:
            best, best_score = (x, y, w, h), score_
    return best


def looks_like_plate(crop: np.ndarray) -> bool:
    """
    Is this crop a plate, or is it whatever else the localiser settled on?

    A located box is not evidence on its own. Where localisation is hard --
    a close, sunlit vehicle -- some frames yield the plate and some yield a
    bumper shadow or a windscreen reflection, and the shadow is often the
    wider of the two. Since a look is chosen for reading by its width, one
    bad frame can outrank a dozen good ones and the whole track reads as
    nothing.

    Measured across real crops: a true plate runs 22-37 light/dark flips
    across its width, because that is what a row of characters does. A
    shadow or a blank panel runs 1-3. The gate is set far below the
    observed floor so it rejects only the obviously characterless.
    """
    ink, flips, _ = _plate_ink(crop)
    return flips >= 8 and 0.08 <= ink <= 0.60


def locate_plate_in_vehicle(gray: np.ndarray,
                            box: Tuple[float, float, float, float]
                            ) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the plate inside a detected vehicle box, in full-frame coordinates.

    Searching the whole frame is what let a road sign and the camera's own
    caption be measured as number plates. Searching inside a vehicle box
    that a detector has already vouched for removes that entire class of
    mistake, and lets us apply proportions that only make sense relative to
    a vehicle: a plate is a small panel low on the back or front, never a
    third of the whole object.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    vw, vh = x2 - x1, y2 - y1
    if vw < 24 or vh < 18:
        return None

    # Plates sit low. Searching the upper half only finds windscreens and
    # roof boxes.
    top = y1 + int(vh * 0.35)
    roi = gray[top:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(roi)
    th = cv2.adaptiveThreshold(roi_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, -6)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

    # Above the scale the constants below were tuned for, use the method
    # built for that scale. Gated on the vehicle's size rather than on the
    # old method failing, because it fails by returning a confident wrong
    # box -- a windscreen, or the shadow under a bumper -- not by returning
    # nothing.
    if vw >= CLOSE_VEHICLE_PX:
        near = _locate_close_vehicle(roi_eq, roi.shape[1])
        if near:
            nx, ny, nw, nh = near
            return (x1 + nx, top + ny, nw, nh)

    best, best_score = None, 0.0
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 4:
            continue
        ar = w / float(h)
        if not (2.2 <= ar <= 6.5):
            continue
        rel = w / float(vw)
        if not (0.12 <= rel <= 0.60):
            continue

        patch = roi_eq[y:y + h, x:x + w]
        if patch.size == 0:
            continue
        # Characters produce strong vertical edges; a blank panel or a
        # bumper reflection does not.
        edges = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        edge_energy = float(np.mean(np.abs(edges)))
        contrast = float(patch.std())
        if edge_energy < 8 or contrast < 15:
            continue

        score = w * (edge_energy / 40.0) * (1.0 + contrast / 128.0)
        if score > best_score:
            best, best_score = (x1 + x, top + y, w, h), score
    return best
