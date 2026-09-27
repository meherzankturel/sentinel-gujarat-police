#!/usr/bin/env python3
"""
sentinel.timesync -- how far a camera's clock can be trusted.

A route reconstruction is a sequence of sightings ordered in time. If one
camera's burned-in clock is seven minutes fast, that camera's sighting
lands seven minutes out of place, and the reconstructed route is wrong in
a way nothing on screen reveals. The officer sees a clean timeline and has
no reason to doubt it.

So the registry measures drift instead of assuming it is zero: read the
clock the camera burns into its own picture, compare against true UTC at
the moment of capture, and store the difference. Sightings are then
corrected on the way in, and the correction is recorded rather than
applied silently.

OCR is Tesseract (Apache 2.0).
"""

from __future__ import annotations

import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .stream import LiveStream

# Most CCTV OSDs burn the stamp into a top or bottom strip.
STRIP_FRACTION = 0.10
# Separators are whatever the OCR made of them -- hyphen, em-dash, space,
# stray punctuation -- so match on the digits and tolerate the rest.
TS_RE = re.compile(
    r"(20\d{2})\D{1,4}(\d{1,2})\D{1,4}(\d{1,2})\D{1,5}"
    r"(\d{1,2})\D{1,3}(\d{2})\D{1,3}(\d{2})")

# Drift is computed from the time of day only, never the date.
#
# OCR misreads a date digit far more often than it misreads the clock, and
# a single wrong digit in the month throws the answer out by weeks -- which
# then gets rejected as implausible, losing a perfectly good reading of the
# time itself. Camera clock drift in the field is seconds to minutes, so
# comparing time-of-day and wrapping around midnight is both more robust
# and sufficient. A camera more than 12 hours out is separately unusable.
HALF_DAY_S = 12 * 3600

# Spread across samples beyond this means the reads disagree with each other,
# so the median is not trustworthy however tidy it looks.
AGREEMENT_MS = 4_000

# What this measures is the offset between the camera's burned-in clock and
# our clock at the moment we decode the frame, so it carries the encode and
# transport latency (about a second on this grid) as well as any real clock
# error. That combined figure is the operationally useful one: it is how far
# out a sighting would land on a timeline if we trusted the OSD blindly.


@dataclass
class TimeConfidence:
    camera_id: str
    samples: int = 0
    reads: int = 0
    drift_ms: Optional[float] = None
    spread_ms: Optional[float] = None
    note: str = ""

    def classify(self) -> str:
        """
        Confidence in the camera's own clock.

        Agreement between samples is checked as well as magnitude. A single
        OCR slip inside a small sample can survive the median and present a
        confident wrong drift -- cam-05 reported +19s that way, from one bad
        read out of three. A wide spread now means we say so rather than
        correct sightings by a number we do not actually believe.
        """
        if self.drift_ms is None:
            return "unknown"
        if self.spread_ms is not None and self.spread_ms > AGREEMENT_MS \
                and self.reads >= 3:
            return "low-agreement"
        d = abs(self.drift_ms)
        if d < 2_000:
            # Within the glass-to-reader latency floor: nothing to correct.
            return "trusted"
        if d < 30_000:
            return "correctable"
        return "unreliable"


def _ocr(img: np.ndarray, psm: str = "11") -> str:
    """
    Tesseract via the CLI, so there is no extra Python dependency to
    license. No character whitelist: constraining the alphabet made it
    coerce em-dashes into digits and invent plausible-looking wrong dates,
    which is the worst possible failure for a clock.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    try:
        r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", psm],
                           input=buf.tobytes(), capture_output=True, timeout=20)
        return r.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _read_stamp(strip: np.ndarray) -> Optional[tuple]:
    """Try both polarities and each page-segmentation mode; first parse wins."""
    for prep in _prep(strip):
        for psm in ("11", "7", "6"):
            text = (_ocr(prep, psm)
                    .replace("O", "0").replace("l", "1").replace("|", "1"))
            m = TS_RE.search(text)
            if m:
                return tuple(int(x) for x in m.groups())
    return None


def _prep(strip: np.ndarray) -> List[np.ndarray]:
    """
    Upscale and binarise the strip, and return BOTH polarities.

    Deciding polarity from the strip's average brightness does not work:
    a typical OSD is white text inside a small dark box sitting on an
    otherwise bright scene, so the average says "already dark on light"
    and leaves the clock inverted and unreadable. Rather than guess from
    a statistic that is wrong exactly where it matters, hand Tesseract
    both and keep whichever parses. OSDs in the field come both ways
    anyway.
    """
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 5, 40, 40)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [cv2.bitwise_not(th), th]


def measure(url: str, camera_id: str, samples: int = 6,
            seconds: float = 60.0, log=None) -> TimeConfidence:
    tc = TimeConfidence(camera_id=camera_id)
    drifts: List[float] = []

    with LiveStream(url, camera_id, log=log) as s:
        every = 0
        for f in s.frames(max_seconds=seconds):
            every += 1
            if every % 6:                       # OCR is slow, but a grainy
                continue                        # OSD needs several attempts
            true_utc = time.time()              # captured next to the frame
            img = f.image
            h = img.shape[0]
            band = max(18, int(h * STRIP_FRACTION))

            # Crop to the clock only. The camera label sits at the other end
            # of the same strip and OCR happily reads it as the timestamp.
            wid = img.shape[1]
            for strip in (img[:band, :int(wid * 0.55)],
                          img[-band:, :int(wid * 0.55)]):
                parsed = _read_stamp(strip)
                if not parsed:
                    continue
                y, mo, d, hh, mm, ss = parsed
                if not (1 <= mo <= 12 and 1 <= d <= 31 and hh <= 23 and mm <= 59
                        and ss <= 59):
                    continue

                shown_sod = hh * 3600 + mm * 60 + ss
                t = time.gmtime(true_utc)
                true_sod = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

                # Wrap to the nearest half day so a stamp either side of
                # midnight does not read as a 24-hour error.
                drift = (shown_sod - true_sod + HALF_DAY_S) % (2 * HALF_DAY_S) \
                    - HALF_DAY_S
                drifts.append(drift * 1000.0)
                tc.reads += 1
                break
            tc.samples += 1
            if tc.reads >= samples:
                break

    if drifts:
        # Reject reads far from the median before taking the final figure.
        if len(drifts) >= 4:
            med = statistics.median(drifts)
            kept = [d for d in drifts if abs(d - med) <= 5_000]
            if len(kept) >= 3:
                drifts = kept
        tc.drift_ms = round(statistics.median(drifts), 0)
        tc.spread_ms = round(max(drifts) - min(drifts), 0) if len(drifts) > 1 else 0.0
        tc.note = (f"OSD clock read {tc.reads}/{tc.samples} samples; "
                   f"median drift {tc.drift_ms/1000:+.1f}s")
    else:
        tc.note = f"no readable burned-in clock in {tc.samples} samples"
    return tc
