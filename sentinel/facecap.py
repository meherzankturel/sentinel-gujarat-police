#!/usr/bin/env python3
"""
sentinel.facecap -- can this camera support face recognition at all?

The problem statement asks for an analytics approach covering ANPR, facial
recognition and tracking. This module answers the facial recognition part
the same way the rest of the system answers every other question: by
measuring the camera rather than assuming it.

The number that decides it
--------------------------
Face recognition is limited by inter-ocular distance -- the pixels between
the centres of the eyes. It is the standard scale measure in the field
because it is the one dimension that survives pose: a head turns, a chin
tucks, but the eye line stays roughly rigid and roughly frontal for as long
as the face is recognisable at all.

    >= 60 px   recognition against a gallery is reasonable to attempt
    >= 40 px   an attempt is possible; expect degraded rank-1 accuracy
    <  40 px   there is no face here, only a smudge that resembles one

ISO/IEC 19794-5 puts 60 px between the eyes as the floor for an enrolment
image, and operational face-in-the-crowd deployments are generally
specified well above it. 40 px is where this system stops calling something
a face, and it is deliberately generous.

Why it is derived and not measured directly
-------------------------------------------
Measuring an eye line needs a face detector, and a face detector run on
cameras that cannot resolve faces returns exactly the confident nonsense
this module exists to prevent -- it would find "faces" in number plates,
wing mirrors and shrubs, and we would then be measuring our own false
positives. So the scale is derived from the person detections we already
trust, through stated anthropometry:

    head height   ~ standing height / 7.5      (adult, standing)
    head width    ~ 0.60 x head height
    inter-ocular  ~ 0.33 x head width

which composes to inter-ocular ~ standing height / 37.9. The ratio is
stated as a constant so a reviewer can disagree with it and recompute,
which is not true of a number that comes out of a detector.

The derivation is generous in two ways, and both favour the camera: it
assumes the person is standing upright and fully visible, and it assumes
they are facing the camera. A seated, walking-away or partially occluded
person yields less. So a camera this module calls face-unusable is not
marginal -- it is comfortably, unarguably short.

The separate axis
-----------------
Face capability is not plate capability. A camera on a gantry above a
carriageway can be excellent at plates and useless for faces; a camera at
door height in a ticket hall can be the reverse. They are measured and
stored separately, and the registry assigns each analytic against its own
axis. Collapsing them into one "quality" score is the same mistake as
believing a specification sheet.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

# Pixels between the eye centres.
FACE_CAPABLE_IOD = 60.0
FACE_MARGINAL_IOD = 40.0

# Composed from the three anthropometric ratios above. Stated as one
# constant so it can be argued with directly.
IOD_TO_BODY_HEIGHT = 1.0 / 37.9

# Below this a person box is too small for the ratio to mean anything; the
# detector's own box error dominates the measurement.
MIN_BODY_PX = 40

FACE_ANALYTIC_FOR = {
    "face-capable":  "face_match",      # gallery matching worth running
    "face-marginal": "face_match_low",  # attempt, corroborating only
    "face-unusable": "none",            # a face detector here invents faces
    "no-people-observed": "none",
    "unknown":       "probe",
}

# A face match from a marginal camera can never be more than corroborating,
# for the same reason a plate read from a presence-only camera cannot.
FACE_CEILING = {
    "face-capable":  "probable",
    "face-marginal": "corroborating",
    "face-unusable": "corroborating",
    "no-people-observed": "corroborating",
    "unknown":       "corroborating",
}


def iod_px(body_height_px: float) -> float:
    """Inter-ocular pixels implied by a person box of this height."""
    return max(0.0, float(body_height_px)) * IOD_TO_BODY_HEIGHT


def face_class(iod: Optional[float]) -> str:
    if iod is None:
        return "no-people-observed"
    if iod >= FACE_CAPABLE_IOD:
        return "face-capable"
    if iod >= FACE_MARGINAL_IOD:
        return "face-marginal"
    return "face-unusable"


@dataclass
class FaceReport:
    """What a camera can support for face recognition, and on what evidence."""
    camera_id: str
    people_seen: int = 0
    frames_analysed: int = 0
    body_px_p50: Optional[float] = None
    body_px_p90: Optional[float] = None
    iod_p50: Optional[float] = None
    iod_p90: Optional[float] = None
    face_class: str = "no-people-observed"
    basis: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "people_seen": self.people_seen,
            "frames_analysed": self.frames_analysed,
            "body_px_p50": None if self.body_px_p50 is None else round(self.body_px_p50, 1),
            "body_px_p90": None if self.body_px_p90 is None else round(self.body_px_p90, 1),
            "iod_p50": None if self.iod_p50 is None else round(self.iod_p50, 1),
            "iod_p90": None if self.iod_p90 is None else round(self.iod_p90, 1),
            "face_class": self.face_class,
            "basis": self.basis,
            "note": self.note,
        }


def report_from_heights(camera_id: str, heights: Iterable[float],
                        frames_analysed: int = 0) -> FaceReport:
    """
    Turn observed person-box heights into a face capability verdict.

    Judged on the 90th percentile rather than the median. A camera only has
    to resolve a face on its best pass, not its typical one -- an analytic
    that fires on the nearest person is still useful. Using the median here
    would fail cameras that are genuinely capable of the thing at the point
    where it matters.
    """
    hs = [float(h) for h in heights if h and h >= MIN_BODY_PX]
    r = FaceReport(camera_id=camera_id, frames_analysed=frames_analysed,
                   people_seen=len(hs))
    if not hs:
        r.basis = "no person large enough to measure"
        r.note = ("No people of measurable size were seen. This is not a "
                  "fault: many cameras on this grid watch carriageways.")
        return r

    hs.sort()
    r.body_px_p50 = hs[len(hs) // 2]
    r.body_px_p90 = hs[min(len(hs) - 1, int(len(hs) * 0.9))]
    r.iod_p50 = iod_px(r.body_px_p50)
    r.iod_p90 = iod_px(r.body_px_p90)
    r.face_class = face_class(r.iod_p90)
    r.basis = (f"inter-ocular derived from person height "
               f"(p90 {r.body_px_p90:.0f}px / {1/IOD_TO_BODY_HEIGHT:.1f})")

    if r.face_class == "face-unusable":
        short = FACE_MARGINAL_IOD / max(r.iod_p90, 1e-6)
        r.note = (f"{r.iod_p90:.1f}px between the eyes on the closest people; "
                  f"{FACE_MARGINAL_IOD:.0f}px is the floor for an attempt. "
                  f"Short by {short:.0f}x. A face detector pointed here does "
                  f"not recognise faces, it invents them.")
    elif r.face_class == "face-marginal":
        r.note = (f"{r.iod_p90:.1f}px between the eyes. An attempt is "
                  f"possible; a match from here can only corroborate.")
    else:
        r.note = (f"{r.iod_p90:.1f}px between the eyes on the closest people. "
                  f"Gallery matching is worth running here.")
    return r


def measure_frames(frames, camera_id: str, detector=None,
                   step: int = 5, min_score: float = 0.5) -> FaceReport:
    """
    Walk frames and measure the people in them.

    `detector` is injected so this can be tested without loading a model,
    and so the caller decides how expensive the detection is.
    """
    if detector is None:
        from .detect import Detector
        detector = lambda img: Detector.detect(img, ("person",), min_score)

    heights: List[float] = []
    n = used = 0
    for img in frames:
        n += 1
        if n % step:
            continue
        used += 1
        for d in detector(img):
            heights.append(getattr(d, "height", 0.0))
    return report_from_heights(camera_id, heights, frames_analysed=used)
