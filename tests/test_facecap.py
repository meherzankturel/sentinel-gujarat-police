"""
Face capability, and the vocabulary it has to fit into.

The interesting assertions here are the refusals. A face capability
measurement that is willing to call a 4-pixel eye line "marginal" is worse
than not measuring at all, because it licenses running a face matcher on a
camera that cannot support one -- and a face matcher with nothing to work
on does not fail quietly, it returns a name.
"""

import pytest

from sentinel import facecap
from sentinel.facecap import (FACE_CAPABLE_IOD, FACE_CEILING, FACE_MARGINAL_IOD,
                              FACE_ANALYTIC_FOR, face_class, iod_px,
                              report_from_heights)


class _Det:
    def __init__(self, height):
        self.height = height


def test_the_ratio_is_stated_and_composes_as_documented():
    """head/7.5, width 0.6 of height, eyes a third of width -> 1/37.9."""
    assert facecap.IOD_TO_BODY_HEIGHT == pytest.approx(1 / 37.9, rel=1e-6)
    # A 1.7m adult filling 379px of frame gives a 10px eye line.
    assert iod_px(379) == pytest.approx(10.0, abs=0.1)


@pytest.mark.parametrize("iod,expected", [
    (120, "face-capable"),
    (FACE_CAPABLE_IOD, "face-capable"),
    (FACE_CAPABLE_IOD - 0.1, "face-marginal"),
    (FACE_MARGINAL_IOD, "face-marginal"),
    (FACE_MARGINAL_IOD - 0.1, "face-unusable"),
    (4.3, "face-unusable"),
    (None, "no-people-observed"),
])
def test_the_thresholds_are_the_thresholds(iod, expected):
    assert face_class(iod) == expected


def test_a_government_grid_person_is_called_unusable_and_told_why():
    """
    cam05 measured 163px of body on its closest people, which is 4.3px
    between the eyes. Nothing about that is a face.
    """
    r = report_from_heights("cam05", [163] * 40)
    assert r.face_class == "face-unusable"
    assert r.iod_p90 == pytest.approx(4.3, abs=0.1)
    assert "invents them" in r.note
    assert "floor for an attempt" in r.note


def test_a_camera_that_could_do_it_is_not_refused():
    """The gate must be a measurement, not a policy of always saying no."""
    r = report_from_heights("doorway", [2400] * 20)     # ~63px eye line
    assert r.face_class == "face-capable"
    assert r.iod_p90 >= FACE_CAPABLE_IOD
    assert "worth running" in r.note


def test_judged_on_the_closest_people_not_the_typical_ones():
    """
    A camera only has to resolve a face on its best pass. Judging on the
    median would fail a capable camera because most people walk past it at
    a distance.
    """
    heights = [200] * 90 + [2400] * 10        # mostly far, occasionally close
    r = report_from_heights("mixed", heights)
    assert r.body_px_p90 > r.body_px_p50
    assert r.face_class == "face-capable"


def test_a_camera_watching_a_carriageway_is_not_called_faulty():
    r = report_from_heights("cam20", [])
    assert r.face_class == "no-people-observed"
    assert "not a fault" in r.note


def test_boxes_too_small_to_measure_are_discarded_not_counted():
    """Below 40px the detector's own box error dominates the ratio."""
    r = report_from_heights("noise", [5, 9, 12, 20])
    assert r.people_seen == 0
    assert r.face_class == "no-people-observed"


def test_measure_frames_uses_the_injected_detector():
    frames = [object()] * 20
    r = facecap.measure_frames(frames, "t", detector=lambda _: [_Det(150)], step=1)
    assert r.people_seen == 20
    assert r.face_class == "face-unusable"


# ---- the vocabulary has to be complete, or a camera falls through it ----

def test_every_face_class_has_an_analytic_and_a_ceiling():
    """
    The plate vocabulary drifted: survey_grid.py writes 'plate-occasional'
    and 'no-vehicles-observed', neither of which appears in ANALYTIC_FOR or
    CAPABILITY_CEILING, so nine real cameras fall through to defaults and
    two different parts of the system give them opposite ceilings. This
    asserts the face vocabulary cannot do the same.
    """
    classes = {"face-capable", "face-marginal", "face-unusable",
               "no-people-observed", "unknown"}
    assert classes <= set(FACE_ANALYTIC_FOR), "class with no assigned analytic"
    assert classes <= set(FACE_CEILING), "class with no tier ceiling"


def test_no_face_class_can_produce_a_confirmed_match():
    """
    Identity from a face is never as strong as identity from a plate on
    this system: a plate is a registered fact, a face match is a
    similarity. The best a face can do is corroborate a probable.
    """
    assert "confirmed" not in FACE_CEILING.values()
    assert FACE_CEILING["face-capable"] == "probable"


def test_an_unusable_camera_is_assigned_no_face_analytic():
    """The whole point: the registry must not schedule work that cannot
    produce evidence."""
    assert FACE_ANALYTIC_FOR["face-unusable"] == "none"
    assert FACE_ANALYTIC_FOR["no-people-observed"] == "none"
