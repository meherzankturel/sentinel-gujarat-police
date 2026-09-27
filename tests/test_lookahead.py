"""
Look-ahead, tested where it could get away with being wrong.

The feature nominates cameras to open next. Four of the assertions below
are about restraint rather than function: that a camera behind the vehicle
is never nominated, that the budget is a hard limit and not a preference,
that a nonsense ground speed is refused rather than projected, and that a
camera which cannot confirm identity never outranks one that can at the
same distance. All four would look fine in a live demonstration if they
were broken, because a plausible-looking watch list is indistinguishable
from a correct one until the vehicle fails to appear.
"""

import math

import pytest

from sentinel.lookahead import (ASSUMED_SPEED_KMH, LookAhead, Nomination,
                                Sighting, angle_between, bearing_deg,
                                haversine_km, nominate)

# A straight east-west road. One degree of longitude near this latitude is
# about 102 km, so these are ~1.02 km apart each.
LAT = 23.03


def cam(cid, dlon_km, capability="presence-only", dlat_km=0.0):
    return {
        "id": cid,
        "lat": LAT + dlat_km / 111.0,
        "lon": 72.50 + dlon_km / 102.0,
        "capability_class": capability,
        "location_name": cid.upper(),
        "department": "Police",
    }


def seen(cid, dlon_km, at_s, dlat_km=0.0):
    return Sighting(cid, LAT + dlat_km / 111.0, 72.50 + dlon_km / 102.0, at_s)


# Travelling east: seen at 0 km then at 1 km, 120 s apart -> 30 km/h.
EAST = [seen("c0", 0.0, 0.0), seen("c1", 1.0, 120.0)]


def test_a_camera_behind_the_vehicle_is_never_nominated():
    """A stream spent looking where the vehicle has been is a stream wasted."""
    ahead = cam("ahead", 2.0)
    behind = cam("behind", -2.0)
    r = nominate(EAST, [ahead, behind], horizon_s=1800)
    ids = [n.camera_id for n in r.nominations]
    assert "ahead" in ids
    assert "behind" not in ids


def test_heading_is_measured_not_assumed():
    r = nominate(EAST, [cam("x", 2.0)])
    assert r.heading is not None
    assert angle_between(r.heading, 90.0) < 5.0      # due east


def test_budget_is_a_hard_limit_and_the_remainder_is_reported():
    """You cannot open more streams than you have, and silence about that
    would read as 'these are all the cameras'."""
    # Deliberately not named c0/c1 -- those are the cameras the vehicle has
    # already been seen on, and excluding them is a different behaviour.
    cams = [cam(f"n{i}", 1.5 + i * 0.1) for i in range(12)]
    r = nominate(EAST, cams, budget=4, horizon_s=1800)
    assert len(r.nominations) == 4
    assert r.considered == 12
    assert "8 further camera(s)" in r.note


def test_a_confirming_camera_outranks_a_corroborating_one_at_equal_distance():
    """The whole registry argument is that cameras are not interchangeable."""
    weak = cam("weak", 2.0, "presence-only")
    strong = cam("strong", 2.0, "plate-capable")
    r = nominate(EAST, [weak, strong], horizon_s=1800)
    assert r.nominations[0].camera_id == "strong"


def test_capability_does_not_override_geography():
    """A plate-capable camera in the wrong direction is still the wrong camera."""
    near_ahead = cam("near", 1.4, "presence-only")
    far_behind = cam("far", -3.0, "plate-capable")
    r = nominate(EAST, [near_ahead, far_behind], horizon_s=1800)
    ids = [n.camera_id for n in r.nominations]
    assert ids == ["near"]


def test_an_impossible_speed_is_refused_rather_than_projected():
    """
    Two cameras with overlapping views seconds apart imply hundreds of
    km/h. Projecting that would make the whole state reachable.
    """
    co_observed = [seen("a", 0.0, 0.0), seen("b", 0.151, 2.4)]   # 226 km/h
    r = nominate(co_observed, [cam("x", 3.0)], horizon_s=600)
    assert r.speed_kmh == pytest.approx(ASSUMED_SPEED_KMH)
    assert "not a road speed" in r.speed_basis


def test_a_stationary_reading_is_refused_too():
    parked = [seen("a", 0.0, 0.0), seen("b", 0.001, 600.0)]
    r = nominate(parked, [cam("x", 1.0)], horizon_s=1800)
    assert r.speed_kmh == pytest.approx(ASSUMED_SPEED_KMH)
    assert "too slow" in r.speed_basis


def test_one_sighting_gives_a_ring_and_says_so():
    """With no second sighting there is no direction, and pretending
    otherwise would nominate a cone into empty road."""
    r = nominate([seen("only", 0.0, 0.0)],
                 [cam("e", 2.0), cam("w", -2.0)], horizon_s=1800)
    ids = {n.camera_id for n in r.nominations}
    assert ids == {"e", "w"}
    assert r.heading is None
    assert "no second sighting" in r.speed_basis


def test_a_stale_sighting_stops_pretending_to_know_the_direction():
    r = nominate(EAST, [cam("e", 2.0), cam("w", -2.0)],
                 now_s=120.0 + 3600.0, horizon_s=1800)
    assert r.heading is None
    assert "ring" in r.note


def test_cameras_already_seen_are_not_nominated_again():
    r = nominate(EAST, [cam("c1", 1.0), cam("c2", 2.0)], horizon_s=1800)
    assert [n.camera_id for n in r.nominations] == ["c2"]


def test_unreachable_cameras_are_excluded_and_explained():
    r = nominate(EAST, [cam("far", 400.0)], horizon_s=600)
    assert r.nominations == []
    assert "reachable" in r.note


def test_eta_accounts_for_time_already_elapsed():
    """A camera 1 km ahead is sooner if the vehicle has been driving since."""
    fresh = nominate(EAST, [cam("x", 2.0)], now_s=120.0, horizon_s=1800)
    later = nominate(EAST, [cam("x", 2.0)], now_s=180.0, horizon_s=1800)
    assert later.nominations[0].eta_s < fresh.nominations[0].eta_s


def test_every_nomination_states_why():
    r = nominate(EAST, [cam("x", 2.0, "plate-capable")], horizon_s=1800)
    n = r.nominations[0]
    assert n.why and any("ahead" in w or "off the direction" in w for w in n.why)
    assert any("confirm identity" in w for w in n.why)


def test_empty_input_is_handled_without_inventing_a_watch_list():
    r = nominate([], [cam("x", 1.0)])
    assert r.nominations == []
    assert "Nothing has been seen" in r.note


def test_as_dict_is_serialisable():
    import json
    r = nominate(EAST, [cam("x", 2.0, "plate-capable")], horizon_s=1800)
    json.dumps(r.as_dict())          # must not raise


def test_bearing_and_distance_agree_with_known_values():
    # Due north one degree of latitude is ~111 km at bearing 0.
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=0.5)
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=0.5)
    assert angle_between(350.0, 10.0) == pytest.approx(20.0)
