"""
Cross-camera vehicle re-identification, as executable claims.

The point of these tests is not coverage. It is that every claim the
submission makes about reid.py is checkable by someone who does not
believe us: that identical pixels score 1.0, that a red car and a white
truck do not, that a journey requiring 300 km/h is refused, and -- most
importantly -- that appearance alone can never produce a 'confirmed'
match. That last one is a promise to a police panel, so it is a test.
"""

import math

import cv2
import numpy as np
import pytest

from sentinel.reid import (
    APPEARANCE_PROBABLE, MAX_PLAUSIBLE_KMH, MIN_BODY_WIDTH_PX,
    CameraSite, Corroboration, Sighting, VehicleSignature,
    assess, class_affinity, colour_similarity, haversine_km,
    match_across_cameras, plate_agreement, signature, signature_from_crop,
    similarity, transit_check,
)


# --------------------------------------------------------------- fixtures


def vehicle(w=160, h=72, body=(40, 40, 200), roof=None, stripe=None):
    """
    A crude vehicle with a controllable colour.

    Deliberately not imported from grid/publish.py: that renders every
    vehicle in the grid with one hardcoded body colour, so it cannot
    produce two vehicles that differ in appearance at all.
    """
    roof = roof or tuple(int(c * 0.88) for c in body)
    img = np.full((h, w, 3), 58, np.uint8)
    cv2.rectangle(img, (0, int(h * 0.30)), (w, int(h * 0.88)), body, -1)
    cv2.rectangle(img, (int(w * 0.22), int(h * 0.04)),
                  (int(w * 0.78), int(h * 0.30)), roof, -1)
    cv2.rectangle(img, (int(w * 0.26), int(h * 0.09)),
                  (int(w * 0.74), int(h * 0.27)), (52, 58, 64), -1)
    if stripe:
        cv2.rectangle(img, (0, int(h * 0.56)), (w, int(h * 0.66)), stripe, -1)
    cv2.circle(img, (int(w * 0.24), int(h * 0.90)), int(h * 0.10), (28, 28, 30), -1)
    cv2.circle(img, (int(w * 0.76), int(h * 0.90)), int(h * 0.10), (28, 28, 30), -1)
    return img


RED_CAR = vehicle(body=(38, 36, 198))
WHITE_TRUCK = vehicle(w=180, h=76, body=(238, 240, 238))
BLUE_CAR = vehicle(body=(190, 90, 36))


def sig(img, cls="car", cam="cam-x"):
    return signature_from_crop(img, cls, camera_id=cam, box_width_px=img.shape[1],
                               frame_area=1920 * 1080,
                               box_area=img.shape[1] * img.shape[0])


# ----------------------------------------------------------- the descriptor


def test_identical_crops_score_one():
    a, b = sig(RED_CAR), sig(RED_CAR.copy(), cam="cam-y")
    s = similarity(a, b)
    assert s.overall == pytest.approx(1.0, abs=1e-6)
    assert s.colour == pytest.approx(1.0, abs=1e-6)
    assert s.layout == pytest.approx(1.0, abs=1e-6)


def test_red_car_and_white_truck_score_low():
    s = similarity(sig(RED_CAR, "car"), sig(WHITE_TRUCK, "truck", cam="cam-y"))
    assert s.overall < 0.30, s.why()
    # And the officer is told which component disagreed, not just a number.
    assert s.colour < 0.25
    assert "car" in s.why() and "truck" in s.why()


def test_colour_histogram_sums_to_one_and_names_the_colour():
    s = sig(RED_CAR)
    assert sum(s.colour_bins) == pytest.approx(1.0, abs=1e-6)
    assert s.dominant_colour == "red"
    assert sig(WHITE_TRUCK, "truck").dominant_colour == "white"


def test_descriptor_survives_a_change_of_exposure_and_white_balance():
    """
    The whole reason for using HSV and dropping shadow and highlight. A
    second camera with a stop less exposure and a warm white balance must
    still recognise the vehicle.
    """
    f = RED_CAR.astype(np.float32) / 255.0
    f = np.clip(f * 0.7, 0, 1) ** 1.2
    f[:, :, 0] *= 1.12
    f[:, :, 2] *= 0.92
    other = np.clip(f * 255, 0, 255).astype(np.uint8)
    s = similarity(sig(RED_CAR), sig(other, cam="cam-y"))
    assert s.overall > 0.60, s.why()


def test_spatial_layout_separates_a_striped_van_from_a_plain_one():
    """
    Two white vans, identical but for a red stripe. Colour histograms alone
    put them close together; the 2x3 layout is what pulls them apart, so
    the layout component specifically must drop.
    """
    striped = vehicle(w=180, h=76, body=(238, 240, 238), stripe=(40, 40, 200))
    plain = vehicle(w=180, h=76, body=(238, 240, 238))
    same = similarity(sig(striped, "truck"), sig(striped.copy(), "truck", "cam-y"))
    cross = similarity(sig(striped, "truck"), sig(plain, "truck", "cam-y"))
    assert cross.overall < same.overall
    assert cross.layout < same.layout


def test_relative_size_is_not_used_for_matching():
    """
    Mounting distance, not vehicle identity, decides how large a vehicle
    looks. A descriptor that matched on it would be comparing the cameras.
    """
    small = cv2.resize(RED_CAR, (60, 27))
    s = similarity(sig(RED_CAR), sig(small, cam="cam-y"))
    assert s.overall > 0.75, s.why()


def test_tiny_vehicle_is_marked_unreliable():
    """
    On most of the government grid a vehicle is a few dozen pixels wide.
    The descriptor must say so rather than assert a colour it cannot see.
    """
    tiny = cv2.resize(RED_CAR, (MIN_BODY_WIDTH_PX - 12, 16))
    s = sig(tiny)
    assert not s.reliable
    assert "px wide" in s.why_unreliable()
    assert "LOW CONFIDENCE" in s.describe()


def test_class_affinity_is_graded_not_binary():
    assert class_affinity("car", "car") == 1.0
    assert 0.2 < class_affinity("car", "truck") < 0.8   # SUV/pickup confusion
    assert class_affinity("motorcycle", "bus") == 0.0
    assert class_affinity("car", "truck") == class_affinity("truck", "car")


def test_colour_similarity_is_a_share_of_common_mass():
    a = [0.5, 0.5] + [0.0] * 13
    b = [0.5, 0.0, 0.5] + [0.0] * 12
    assert colour_similarity(a, b) == pytest.approx(0.5)


# ------------------------------------------------------- transit plausibility


GANDHINAGAR = CameraSite("cam-01", 23.1912, 72.6289, "Infocity Circle")
AHMEDABAD = CameraSite("cam-08", 23.0530, 72.6020, "Civil Hospital")


def test_haversine_matches_the_known_corridor_length():
    d = haversine_km(GANDHINAGAR.lat, GANDHINAGAR.lon, AHMEDABAD.lat, AHMEDABAD.lon)
    assert 14.0 < d < 17.0        # Infocity to Civil Hospital, ~15.5 km


def test_impossible_transit_is_rejected():
    """
    Fifteen kilometres in three minutes is 300 km/h. That is not a weak
    match to be ranked lower, it is a different vehicle or a cloned plate,
    and the system must refuse it outright.
    """
    t = transit_check(GANDHINAGAR, AHMEDABAD, elapsed_s=180)
    assert not t.plausible
    assert t.score == 0.0
    assert t.implied_kmh > MAX_PLAUSIBLE_KMH
    assert "km/h" in t.reason and "cloned plate" in t.reason


def test_normal_traffic_transit_is_fully_supported():
    t = transit_check(GANDHINAGAR, AHMEDABAD, elapsed_s=25 * 60)
    assert t.plausible
    assert t.score == pytest.approx(1.0)


def test_a_long_gap_is_possible_but_stops_being_evidence():
    """A vehicle is allowed to park. Timing then tells us nothing."""
    t = transit_check(GANDHINAGAR, AHMEDABAD, elapsed_s=6 * 3600)
    assert t.plausible
    assert 0.0 < t.score < 0.5
    assert "stopped" in t.reason


def test_an_untrusted_clock_cannot_manufacture_an_impossible_transit():
    """
    cam-08 on our own grid runs 428 s fast. Treating its timestamp as exact
    would make an ordinary 8-minute journey look like teleportation and
    discard the true match -- the worst error this module could make.
    """
    drifting = CameraSite("cam-08", AHMEDABAD.lat, AHMEDABAD.lon,
                          time_confidence_ms=428_500.0)
    strict = transit_check(GANDHINAGAR, AHMEDABAD, elapsed_s=300)
    forgiving = transit_check(GANDHINAGAR, drifting, elapsed_s=300)
    assert not strict.plausible
    assert forgiving.plausible
    assert forgiving.effective_s > forgiving.elapsed_s


def test_direction_discounts_a_vehicle_that_was_driving_away():
    """
    Heading is used weakly on purpose: a vehicle that drove away can still
    turn round, so a contrary heading must lower the score without removing
    the candidate. Losing a real vehicle costs more than ranking one badly.
    """
    from sentinel.reid import bearing_deg, direction_consistency
    br = bearing_deg(GANDHINAGAR.lat, GANDHINAGAR.lon, AHMEDABAD.lat, AHMEDABAD.lon)
    toward, _ = direction_consistency(GANDHINAGAR, AHMEDABAD, br)
    away, note = direction_consistency(GANDHINAGAR, AHMEDABAD, (br + 180) % 360)
    assert toward == 1.0
    assert 0.0 < away < 1.0
    assert "turn round" in note

    q = Sighting("q", "cam-01", 0.0, sig(RED_CAR, "car", "cam-01"),
                 heading_deg=(br + 180) % 360)
    c = sighting("c", "cam-08", 25 * 60, RED_CAR.copy())
    m = assess(q, c, SITES)
    assert m.tier == "probable"          # still a candidate, just discounted
    assert m.score < assess(sighting("q2", "cam-01", 0.0, RED_CAR),
                            c, SITES).score


def test_an_unknown_heading_changes_nothing():
    from sentinel.reid import direction_consistency
    assert direction_consistency(GANDHINAGAR, AHMEDABAD, None) == (1.0, "")


def test_co_located_cameras_skip_the_transit_check():
    a = CameraSite("a", 23.10, 72.60)
    b = CameraSite("b", 23.10, 72.60)
    t = transit_check(a, b, elapsed_s=1)
    assert t.plausible and t.score == 1.0


def test_unknown_coordinates_do_not_reject():
    """A camera missing a lat/lon is a registry gap, not a disproof."""
    t = transit_check(CameraSite("a", None, None), AHMEDABAD, elapsed_s=5)
    assert t.plausible
    assert "coordinates unknown" in t.reason


# --------------------------------------------------------------- plate hints


def test_plate_agreement_counts_agreements_and_conflicts():
    assert plate_agreement("GJ01AB1234", "GJ01AB1234") == (10, 0)
    assert plate_agreement("GJ01??????", "GJ01AB1234") == (4, 0)
    assert plate_agreement("GJ18AB1234", "GJ01AB1234") == (8, 2)
    assert plate_agreement(None, "GJ01AB1234") == (0, 0)


def test_four_agreeing_characters_corroborate_but_three_do_not():
    assert Corroboration(plate_chars_agreeing=4).supports()
    assert not Corroboration(plate_chars_agreeing=3).supports()


# ----------------------------------------------------------- honest tiering


SITES = {
    "cam-01": GANDHINAGAR,
    "cam-08": CameraSite("cam-08", AHMEDABAD.lat, AHMEDABAD.lon),
    "cam-06": CameraSite("cam-06", 23.108, 72.541, "SG Highway"),
}


def sighting(sid, cam, at, img, cls="car", plate=None):
    return Sighting(sighting_id=sid, camera_id=cam, at_epoch_s=at,
                    signature=sig(img, cls, cam), plate_fragment=plate)


def test_appearance_alone_never_reaches_confirmed():
    """
    The central promise of this module. A perfect appearance match, a
    perfectly plausible journey, and no plate: 'probable' at most, because
    a white hatchback looks like ten thousand other white hatchbacks.
    """
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    c = sighting("c", "cam-08", 25 * 60, RED_CAR.copy())
    m = assess(q, c, SITES)
    assert m.appearance.overall > 0.95
    assert m.transit.plausible
    assert m.tier == "probable"
    assert "cannot be confirmed without a plate" in m.why


def test_appearance_plus_an_agreeing_plate_fragment_reaches_confirmed():
    q = sighting("q", "cam-01", 0.0, RED_CAR, plate="GJ01AB1234")
    c = sighting("c", "cam-08", 25 * 60, RED_CAR.copy(), plate="GJ01AB????")
    m = assess(q, c, SITES)
    assert m.tier == "confirmed"
    assert m.corroboration.supports()


def test_a_conflicting_plate_beats_a_perfect_appearance_match():
    """The one place the system is allowed to be certain: in the negative."""
    q = sighting("q", "cam-01", 0.0, RED_CAR, plate="GJ01AB1234")
    c = sighting("c", "cam-08", 25 * 60, RED_CAR.copy(), plate="GJ18XY4477")
    m = assess(q, c, SITES)
    assert m.tier == "corroborating"
    assert m.score == 0.0
    assert "not the same vehicle" in m.why


def test_an_unreliable_descriptor_can_only_corroborate():
    tiny = cv2.resize(RED_CAR, (MIN_BODY_WIDTH_PX - 12, 16))
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    c = Sighting("c", "cam-08", 25 * 60, sig(tiny, "car", "cam-08"))
    m = assess(q, c, SITES)
    assert m.tier == "corroborating"
    assert "reliability floor" in m.why
    assert not m.notifies()


def test_corroborating_matches_never_notify():
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    c = sighting("c", "cam-08", 25 * 60, WHITE_TRUCK, "truck")
    m = assess(q, c, SITES)
    assert m.tier == "corroborating"
    assert not m.notifies()


# ----------------------------------------------------------------- ranking


def test_match_across_cameras_ranks_the_right_vehicle_first():
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    cands = [sighting("a", "cam-08", 25 * 60, WHITE_TRUCK, "truck"),
             sighting("b", "cam-08", 26 * 60, RED_CAR.copy()),
             sighting("c", "cam-06", 20 * 60, BLUE_CAR)]
    res = match_across_cameras(q, cands, SITES)
    assert res.best().candidate.sighting_id == "b"
    assert res.best().tier == "probable"


def test_the_impossible_candidate_is_moved_out_of_the_ranking_not_dropped():
    """
    A vehicle that looks identical but could not have made the journey is an
    investigative lead -- a clone, or a second identical vehicle. The
    officer must see it, so it is separated rather than silently discarded.
    """
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    teleport = sighting("t", "cam-08", 120.0, RED_CAR.copy())
    res = match_across_cameras(q, [teleport], SITES)
    assert res.ranked == []
    assert len(res.implausible) == 1
    assert res.implausible[0].appearance.overall > 0.95
    assert "cloned plate" in res.implausible[0].transit.reason


def test_same_camera_candidates_are_excluded_by_default():
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    same = sighting("s", "cam-01", 30.0, RED_CAR.copy())
    assert match_across_cameras(q, [same], SITES).ranked == []
    assert match_across_cameras(q, [same], SITES,
                                include_same_camera=True).ranked


def test_a_single_route_corridor_is_what_lets_timing_corroborate():
    """
    Without a plate, the only other route to 'confirmed' is a corridor with
    no exit between the two cameras plus a tight transit fit -- the vehicle
    had nowhere else to be. That is a surveyed fact about the road, so it
    must be supplied, never inferred.
    """
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    c = sighting("c", "cam-06", 12 * 60, RED_CAR.copy())
    corridor = {"single_route_pairs": [("cam-01", "cam-06")]}
    assert assess(q, c, SITES).tier == "probable"
    assert assess(q, c, SITES, corridor).tier == "confirmed"


def test_a_bad_clock_cannot_corroborate_a_corridor_transit():
    """
    cam-08's clock is 428 s out -- longer than the transit it would be
    corroborating. A "tight fit" measured with that clock is an artefact of
    the clock, so it must not buy the top tier.
    """
    q = sighting("q", "cam-01", 0.0, RED_CAR)
    c = sighting("c", "cam-06", 12 * 60, RED_CAR.copy())
    corridor = {"single_route_pairs": [("cam-01", "cam-06")]}
    drifting = dict(SITES)
    drifting["cam-06"] = CameraSite("cam-06", 23.108, 72.541,
                                    time_confidence_ms=428_500.0)
    assert assess(q, c, SITES, corridor).tier == "confirmed"
    assert assess(q, c, drifting, corridor).tier == "probable"


# --------------------------------------------------------------- plumbing


def test_signature_round_trips_through_json():
    """
    Stored signatures round-trip. to_dict() rounds for legibility -- these
    values end up in an evidence bundle a human reads -- so the tolerance
    here is the rounding, not the descriptor.
    """
    s = sig(RED_CAR)
    back = VehicleSignature.from_dict(s.to_dict())
    assert similarity(s, back).overall == pytest.approx(1.0, abs=1e-3)


def test_signature_from_a_frame_and_box_matches_the_crop():
    frame = np.full((1080, 1920, 3), 60, np.uint8)
    frame[400:472, 300:460] = RED_CAR
    a = signature(frame, (300, 400, 460, 472), "car", "cam-01")
    assert similarity(a, sig(RED_CAR)).overall == pytest.approx(1.0, abs=1e-6)
    assert a.rel_area == pytest.approx(160 * 72 / (1920 * 1080), rel=1e-3)


def test_registry_sites_load_from_the_database():
    """The camera coordinates driving the transit gate come from the registry."""
    import sqlite3
    from pathlib import Path
    from sentinel.reid import load_sites
    db = Path(__file__).resolve().parents[1] / "sentinel.db"
    if not db.exists():
        pytest.skip("no sentinel.db")
    sites = load_sites(sqlite3.connect(str(db)))
    # Existence is not enough: test_route_gates.py initialises an empty
    # registry at this same path, so in a full run the guard above sees a
    # file and lets an empty database through. What this test needs is a
    # populated one.
    if not sites:
        pytest.skip("sentinel.db has no cameras -- run tools/load_real_cameras.py")
    for s in sites.values():
        assert s.lat is None or -90 <= s.lat <= 90
    if "cam-08" in sites:
        # The grid's deliberately wrong clock must arrive as real slack.
        assert sites["cam-08"].clock_slack_s() > 60
