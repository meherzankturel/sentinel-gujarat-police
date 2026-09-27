"""
The upgrade planner, tested where it would be tempting to overstate.

A planner that answers every question with "buy a better camera" is not a
plan, it is a shrug with a budget attached. A planner that promises a lens
will fix a camera pointed at the wrong road is worse: someone spends money
and the capability still is not there.

So the assertions below are mostly about the planner declining — refusing
to plan for a camera that was never asked the question, refusing to invent
a focal length it cannot know, and saying plainly when the gap is too wide
for optics.
"""

import pytest

from sentinel.facecap import FACE_CAPABLE_IOD, FACE_MARGINAL_IOD
from sentinel.planner import (OPTICS_ONLY, PLATE_READ_PX, RESITE,
                              face_shortfall, plan, plate_shortfall,
                              summarise)


def cam(**kw):
    base = dict(id="cam01", location_name="Test Road", department="Police",
                width=1920, capability_class="presence-only")
    base.update(kw)
    return base


# ---- plate ------------------------------------------------------------

def test_a_capable_camera_is_left_alone():
    s = plate_shortfall(cam(plate_px_measured=193, capability_class="plate-capable"))
    assert s.verdict == "met"
    assert s.options == []
    assert "No change needed" in s.note


def test_a_near_miss_is_a_lens_not_a_building_site():
    """cam05 at 89px is 1.35x short -- that is an aim or a longer lens."""
    s = plate_shortfall(cam(id="cam05", plate_px_measured=89))
    assert s.factor == pytest.approx(PLATE_READ_PX / 89, rel=1e-6)
    assert s.verdict == "optics"
    assert any(o.lever == "lens" for o in s.options)


def test_a_hopeless_camera_says_so_rather_than_selling_a_lens():
    """cam16 at 22px is 5.5x short. No lens reaches that from where it is."""
    s = plate_shortfall(cam(id="cam16", plate_px_measured=22))
    assert s.verdict == "replace"
    assert "a lens will not reach it" in s.note


def test_a_camera_that_watches_no_road_is_not_marked_as_failing():
    s = plate_shortfall(cam(id="cam20", capability_class="no-vehicles-observed"))
    assert s.verdict == "not-applicable"
    assert "not failing at it" in s.note
    assert s.options == []


def test_an_unmeasured_camera_is_not_planned_for():
    s = plate_shortfall(cam(plate_px_measured=None))
    assert s.verdict == "unknown"
    assert "Survey it" in s.note
    assert s.options == []


# ---- face -------------------------------------------------------------

def test_face_planning_uses_the_attempt_floor_by_default():
    s = face_shortfall(cam(face_iod_px=9.8))
    assert s.need_px == FACE_MARGINAL_IOD
    assert s.factor == pytest.approx(FACE_MARGINAL_IOD / 9.8, rel=1e-6)


def test_planning_to_the_higher_bar_is_available():
    s = face_shortfall(cam(face_iod_px=9.8), target=FACE_CAPABLE_IOD)
    assert s.need_px == FACE_CAPABLE_IOD
    assert s.factor > FACE_MARGINAL_IOD / 9.8


def test_a_traffic_pole_is_told_it_is_the_wrong_mounting_point():
    """
    The real answer for a 4x-short traffic camera is not a bigger lens, it
    is a camera somewhere people walk.
    """
    s = face_shortfall(cam(id="cam09", face_iod_px=9.8))
    # 9.8px against a 40px floor is 4.1x -- "resite", not "replace". For a
    # face that distinction barely matters: both mean the camera is in the
    # wrong place, and both get the same advice.
    assert s.verdict == "resite"
    assert "head height" in s.note
    assert "not a traffic pole" in s.note


def test_face_capability_can_be_derived_from_body_height():
    s = face_shortfall(cam(body_px_p90=370))
    assert s.have_px == pytest.approx(370 / 37.9, rel=1e-6)


def test_a_carriageway_camera_is_not_asked_about_faces():
    s = face_shortfall(cam(face_class="no-people-observed"))
    assert s.verdict == "not-applicable"
    assert "watches a carriageway" in s.note


# ---- the levers -------------------------------------------------------

def test_every_option_is_a_ratio_and_never_a_part_number():
    """
    The catalogue carries no focal length, no mounting height and no
    distance. Producing "fit a 12mm lens" from a measurement containing no
    millimetres is the same error as believing a spec sheet.
    """
    s = plate_shortfall(cam(plate_px_measured=60))
    joined = " ".join(o.detail for o in s.options)
    assert "mm" not in joined
    assert "x the focal length" in joined


def test_resolution_is_offered_but_named_as_the_worst_lever():
    """Pixel count rises as the square of the gain; detail rises linearly."""
    s = plate_shortfall(cam(plate_px_measured=60, width=1920))
    sensor = [o for o in s.options if o.lever == "sensor"]
    assert sensor, "resolution should still be offered"
    assert "weakest of the three" in sensor[0].detail


def test_no_sensor_option_when_the_resolution_is_unknown():
    s = plate_shortfall(cam(plate_px_measured=60, width=None))
    assert not [o for o in s.options if o.lever == "sensor"]


@pytest.mark.parametrize("have,expected", [
    (PLATE_READ_PX, "met"),
    (PLATE_READ_PX / OPTICS_ONLY, "optics"),
    (PLATE_READ_PX / (OPTICS_ONLY + 0.5), "resite"),
    (PLATE_READ_PX / (RESITE + 1), "replace"),
])
def test_the_bands_are_the_bands(have, expected):
    assert plate_shortfall(cam(plate_px_measured=have)).verdict == expected


# ---- the whole plan ---------------------------------------------------

def test_the_plan_puts_the_cheapest_fixes_first():
    cams = [
        cam(id="hopeless", plate_px_measured=10),
        cam(id="easy", plate_px_measured=100),
        cam(id="done", plate_px_measured=200, capability_class="plate-capable"),
    ]
    verdicts = [s.verdict for s in plan(cams) if s.capability == "plate"]
    assert verdicts.index("optics") < verdicts.index("replace")
    assert verdicts.index("replace") < verdicts.index("met")


def test_the_summary_counts_what_a_budget_line_needs():
    cams = [cam(id="a", plate_px_measured=100), cam(id="b", plate_px_measured=10)]
    got = summarise(plan(cams))
    assert got["plate"]["optics"] == 1
    assert got["plate"]["replace"] == 1


def test_every_shortfall_serialises():
    import json
    json.dumps([s.as_dict() for s in plan([cam(plate_px_measured=60,
                                               face_iod_px=9.8)])])
