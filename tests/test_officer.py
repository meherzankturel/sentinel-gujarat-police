"""
The officer's console, tested where it makes claims.

Four of the assertions here are about honesty rather than function: that a
search without an authorising officer is refused rather than logged, that
the audit entry exists before the results are handed over, that a distance
we compute agrees with a distance computed a different way, and that the
integrity hash on an evidence bundle actually covers the bundle it is
printed on. A console that got any of those wrong would still look correct
in a demonstration.
"""

import functools
import math

import pytest
from fastapi import HTTPException

from sentinel import officer, registry
from sentinel.officer import bundle_hash, haversine_km

PLATE = "TS01AB1234"
CASE = "FIR-9001/2026"
OFFICER = "PSI Test"
AUTH = "DySP Test"

# Three cameras on one line of latitude, so the geometry in the test is
# independently checkable by hand and does not depend on the demo data.
CAMS = [("t-01", 23.0, 72.0, "plate-capable", 0.0),
        ("t-02", 23.0, 73.0, "plate-capable", 0.0),
        ("t-03", 23.0, 74.0, "presence-only", 0.0)]

#            camera, seconds from t0, plate as read, tier
SIGHTINGS = [("t-01",    0, "TS01AB1234", "confirmed"),
             ("t-02", 5400, "TS0IAB1Z34", "probable"),
             ("t-03", 6600, None,         "corroborating")]


def _iso(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"2026-09-09T{h:02d}:{m:02d}:{s:02d}Z"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """
    A database of its own. These tests write audit entries, and an audit
    log is append-only by design -- running them against the demo database
    would leave test searches permanently in the chain that the evidence
    bundles cite.
    """
    path = tmp_path / "test.db"
    registry.init(path)
    bound = functools.partial(registry.connect, path)
    monkeypatch.setattr(officer, "connect", bound)

    with bound() as con:
        for cid, lat, lon, cap, drift in CAMS:
            con.execute("INSERT INTO camera (id, department, location_name, "
                        "lat, lon, capability_class, plate_px_measured, "
                        "time_confidence_ms, time_confidence_note, "
                        "governance_class) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (cid, "Police", f"Test site {cid}", lat, lon, cap,
                         150 if cap == "plate-capable" else 40, drift,
                         "trusted: synthetic", "open"))
        con.execute("INSERT INTO watchlist_entry (entity_type, plate_number, "
                    "plate_normalised, reason, priority, case_ref) "
                    "VALUES ('vehicle',?,?,'stolen',1,?)",
                    (PLATE, registry.normalise_plate(PLATE), CASE))
        wid = con.execute("SELECT id FROM watchlist_entry").fetchone()["id"]

        for cid, offset, read, tier in SIGHTINGS:
            con.execute(
                "INSERT INTO sighting (camera_id, wallclock_utc, corrected_utc, "
                "plate_read, plate_normalised, plate_confidence, match_id, "
                "match_confidence, match_tier) VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, _iso(offset), _iso(offset), read,
                 registry.normalise_plate(read or PLATE),
                 0.9 if read else None, wid, 0.9 if read else 0.35, tier))

        # Another vehicle entirely, so a passing search is passing because
        # it selected, not because the table holds only one car.
        con.execute("INSERT INTO sighting (camera_id, wallclock_utc, "
                    "corrected_utc, plate_read, plate_normalised, match_tier) "
                    "VALUES ('t-01',?,?,'MH01ZZ9999','MH01ZZ9999','confirmed')",
                    (_iso(60), _iso(60)))
    return bound


def search(**kw):
    return officer.officer_search(plate=kw.pop("plate", PLATE),
                                  case_ref=kw.pop("case_ref", CASE),
                                  officer=kw.pop("officer", OFFICER),
                                  authorised_by=kw.pop("authorised_by", AUTH))


# 1 -------------------------------------------------- the gate is a gate


@pytest.mark.parametrize("missing", ["case_ref", "officer", "authorised_by"])
def test_search_is_refused_without_full_authorisation(db, missing):
    with pytest.raises(HTTPException) as e:
        search(**{missing: None})
    assert e.value.status_code == 400
    assert missing in e.value.detail["missing"]


def test_whitespace_is_not_a_case_reference(db):
    """A form that accepts a space has no gate, only a text box."""
    with pytest.raises(HTTPException):
        search(case_ref="   ")


def test_a_refused_search_reads_nothing_and_logs_nothing(db):
    """
    The refusal must happen before the query. Recording the attempt and
    returning the movements anyway would produce an immaculate audit trail
    of exactly the thing the trail exists to prevent.
    """
    with pytest.raises(HTTPException):
        search(case_ref=None)
    with db() as con:
        assert con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


# 2 ------------------------------------------------- the search is logged


def test_every_search_writes_an_audit_entry(db):
    out = search()
    with db() as con:
        row = con.execute("SELECT * FROM audit_log ORDER BY id DESC "
                          "LIMIT 1").fetchone()
        intact, broken = registry.verify_audit_chain(con)
    assert row["action"] == "officer_search_vehicle"
    assert row["case_ref"] == CASE
    assert row["actor"] == OFFICER
    assert row["authorised_by"] == AUTH
    # The count in the log has to be the count that was handed over, or the
    # log records a different search from the one that happened.
    assert row["result_count"] == out["count"] == 3
    assert intact and broken is None
    assert out["audit"]["entry_hash"] == row["hash"]


def test_the_audit_chain_detects_an_altered_entry(db):
    search()
    with db() as con:
        con.execute("UPDATE audit_log SET result_count = 99 WHERE id = 1")
    with db() as con:
        intact, broken = registry.verify_audit_chain(con)
    assert not intact and broken == 1


# 3 ------------------------------------------ the search finds the vehicle


def test_a_smudged_read_still_returns_the_vehicle(db):
    """
    t-02 stored TS0IAB1Z34. An equality search loses the vehicle on exactly
    the cameras where losing it matters, so this is the behaviour that
    justifies confusion-aware matching being in the search path and not
    only in the alerting path.
    """
    out = search()
    reads = [s["plate_read"] for s in out["sightings"]]
    assert "TS0IAB1Z34" in reads
    smudged = next(s for s in out["sightings"] if s["plate_read"] == "TS0IAB1Z34")
    assert not smudged["exact_read"]
    assert smudged["read_distance"] == 0.5


def test_other_vehicles_are_not_returned(db):
    out = search()
    assert all(s["plate_read"] != "MH01ZZ9999" for s in out["sightings"])


def test_a_camera_that_read_no_plate_never_reads_as_an_exact_match(db):
    """
    t-03 is presence-only and claimed nothing. Its attributed plate string
    equals the query, and it must still not present as a clean read.
    """
    out = search()
    attributed = next(s for s in out["sightings"] if s["camera_id"] == "t-03")
    assert attributed["plate_claimed"] is False
    assert attributed["exact_read"] is False
    assert attributed["read_distance"] is None
    assert attributed["tier"] == "corroborating"


def test_sightings_come_back_in_corrected_time_order(db):
    out = search()
    ats = [s["at"] for s in out["sightings"]]
    assert ats == sorted(ats)
    assert [s["seq"] for s in out["sightings"]] == [1, 2, 3]


# 4 ---------------------------------------------------- the route arithmetic


def test_haversine_agrees_with_a_known_distance():
    """
    One degree of longitude at the equator is a quarter-degree of the
    meridian quadrant: 2*pi*R/360. Checked to the metre.
    """
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.1949, abs=0.001)
    # Zero distance must be exactly zero, not a rounding artefact of asin.
    assert haversine_km(23.0225, 72.5714, 23.0225, 72.5714) == 0.0


def test_haversine_agrees_with_an_independent_formula():
    """
    Cross-checked against the spherical law of cosines rather than against
    a number copied from a map site. Two derivations agreeing is evidence;
    one derivation matching a constant we chose is not.
    """
    pairs = [(23.0225, 72.5714, 23.2156, 72.6369),   # Ahmedabad-Gandhinagar
             (23.0, 72.0, 23.0, 73.0),               # the test corridor
             (-33.8688, 151.2093, 51.5074, -0.1278)] # antipodal-ish, Sydney-London
    for la1, lo1, la2, lo2 in pairs:
        p1, p2 = math.radians(la1), math.radians(la2)
        cos_d = (math.sin(p1) * math.sin(p2) +
                 math.cos(p1) * math.cos(p2) * math.cos(math.radians(lo2 - lo1)))
        expected = 6371.0088 * math.acos(min(1.0, max(-1.0, cos_d)))
        assert haversine_km(la1, lo1, la2, lo2) == pytest.approx(expected, rel=1e-6)


def test_route_reports_distance_elapsed_and_implied_speed(db):
    r = officer.officer_route(plate=PLATE, case_ref=CASE, officer=OFFICER,
                              authorised_by=AUTH)
    legs = r["legs"]
    assert len(legs) == 2

    first = legs[0]
    # One degree of longitude along the 23rd parallel. On a sphere the
    # general haversine collapses to this closed form when both latitudes
    # are equal, so it is an exact expected value rather than an
    # approximation borrowed from a map site.
    expected_km = 2 * 6371.0088 * math.asin(
        math.cos(math.radians(23.0)) * math.sin(math.radians(1.0) / 2))
    assert expected_km == pytest.approx(102.35, abs=0.01)
    assert first["km"] == pytest.approx(expected_km, abs=0.001)
    assert first["seconds"] == 5400
    assert first["minutes"] == pytest.approx(90.0)
    assert first["implied_kmh"] == pytest.approx(first["km"] / 1.5, rel=1e-3)
    assert r["summary"]["distance_km"] == pytest.approx(
        legs[0]["km"] + legs[1]["km"], abs=0.01)
    assert r["summary"]["span_minutes"] == pytest.approx(110.0)


# 5 ------------------------------------------------- impossible transits


def test_a_plausible_leg_is_not_flagged(db):
    r = officer.officer_route(plate=PLATE, officer=OFFICER,
                                 case_ref=CASE, authorised_by=AUTH)
    assert r["legs"][0]["implied_kmh"] < officer.IMPOSSIBLE_KMH
    assert r["legs"][0]["impossible"] is False
    assert r["legs"][0]["explanations"] == []


def test_an_impossible_leg_is_flagged_and_explained(db):
    r = officer.officer_route(plate=PLATE, officer=OFFICER,
                                 case_ref=CASE, authorised_by=AUTH)
    bad = r["legs"][1]
    assert bad["seconds"] == 1200
    assert bad["implied_kmh"] > officer.IMPOSSIBLE_KMH
    assert bad["impossible"] is True
    assert r["summary"]["impossible_legs"] == 1
    # A flag with no candidate explanation tells an officer nothing about
    # what to do next.
    assert any("clone" in e.lower() for e in bad["explanations"])


def test_two_cameras_cannot_see_the_same_vehicle_at_the_same_second(db):
    with db() as con:
        con.execute("UPDATE sighting SET corrected_utc=?, wallclock_utc=? "
                    "WHERE camera_id='t-02'", (_iso(0), _iso(0)))
    r = officer.officer_route(plate=PLATE, officer=OFFICER,
                                 case_ref=CASE, authorised_by=AUTH)
    assert r["legs"][0]["impossible"] is True
    assert r["legs"][0]["implied_kmh"] is None


def test_slowing_the_journey_down_clears_the_flag(db):
    """The threshold is a threshold, not a property of this fixture."""
    with db() as con:
        con.execute("UPDATE sighting SET corrected_utc=?, wallclock_utc=? "
                    "WHERE camera_id='t-03'", (_iso(12000), _iso(12000)))
    r = officer.officer_route(plate=PLATE, officer=OFFICER,
                                 case_ref=CASE, authorised_by=AUTH)
    assert r["summary"]["impossible_legs"] == 0


# 6 ------------------------------------------------------------- evidence


def test_evidence_is_refused_without_full_authorisation(db):
    with pytest.raises(HTTPException) as e:
        officer.officer_evidence(plate=PLATE, case_ref=CASE, officer=OFFICER,
                                 authorised_by=None)
    assert e.value.status_code == 400


def test_evidence_bundle_carries_what_it_has_to_defend(db):
    search()
    out = officer.officer_evidence(plate=PLATE, case_ref=CASE, officer=OFFICER,
                                   authorised_by=AUTH)
    b = out["bundle"]
    assert len(b["sightings"]) == 3
    assert {c["id"] for c in b["cameras"]} == {"t-01", "t-02", "t-03"}
    # The measured capability is the part a defence solicitor attacks.
    assert all("plate_px_measured" in c and "capability_class" in c
               for c in b["cameras"])
    assert {s["tier"] for s in b["sightings"]} == {
        "confirmed", "probable", "corroborating"}
    assert b["route"]["legs"][1]["impossible"] is True
    assert b["audit_chain"]["case_entries_intact"] is True

    # The trail holds the search that preceded the export. It cannot hold
    # the export, whose log entry commits the hash of this very object, and
    # the bundle says so rather than leaving the gap unexplained.
    assert [e["action"] for e in b["audit_trail"]] == ["officer_search_vehicle"]
    assert all(e["case_ref"] == CASE for e in b["audit_trail"])
    assert "necessarily absent" in b["audit_trail_note"]


def test_the_printed_hash_actually_covers_the_bundle(db):
    """
    An integrity hash that does not cover the document it is printed on is
    worse than none, so this recomputes it from the returned object rather
    than trusting the field beside it.
    """
    out = officer.officer_evidence(plate=PLATE, case_ref=CASE, officer=OFFICER,
                                   authorised_by=AUTH)
    assert bundle_hash(out["bundle"]) == out["integrity"]["sha256"]


def test_hash_ignores_key_order_but_not_content():
    a = {"case": "X", "sightings": [{"tier": "confirmed", "plate": "GJ01AB1234"}]}
    b = {"sightings": [{"plate": "GJ01AB1234", "tier": "confirmed"}], "case": "X"}
    assert bundle_hash(a) == bundle_hash(b)

    demoted = {"case": "X",
               "sightings": [{"tier": "probable", "plate": "GJ01AB1234"}]}
    assert bundle_hash(demoted) != bundle_hash(a)


def test_evidence_hash_changes_when_the_evidence_changes(db):
    """
    Compared with the generation timestamp held fixed, so this is testing
    that the hash tracks the content and not merely that it tracks the
    clock.
    """
    def frozen():
        out = officer.officer_evidence(plate=PLATE, case_ref=CASE,
                                       officer=OFFICER, authorised_by=AUTH)
        body = dict(out["bundle"])
        body["generated_at"] = "FIXED"
        body["audit_trail"] = []          # grows with every export
        return bundle_hash(body)

    before = frozen()
    assert frozen() == before             # nothing changed, hash is stable

    with db() as con:
        con.execute("UPDATE sighting SET match_tier='confirmed' "
                    "WHERE camera_id='t-03'")
    assert frozen() != before


def test_producing_a_bundle_is_itself_recorded(db):
    out = officer.officer_evidence(plate=PLATE, case_ref=CASE, officer=OFFICER,
                                   authorised_by=AUTH)
    with db() as con:
        row = con.execute("SELECT * FROM audit_log WHERE action="
                          "'officer_export_evidence' ORDER BY id DESC "
                          "LIMIT 1").fetchone()
        intact, _ = registry.verify_audit_chain(con)
    assert row is not None
    assert out["integrity"]["sha256"] in row["query_params"]
    assert row["hash"] == out["integrity"]["audit_entry_hash"]
    assert intact


def test_no_sightings_is_a_404_not_an_empty_bundle(db):
    """An empty bundle with a valid hash is a document asserting nothing."""
    with pytest.raises(HTTPException) as e:
        officer.officer_evidence(plate="GJ99ZZ0000", case_ref=CASE,
                                 officer=OFFICER, authorised_by=AUTH)
    assert e.value.status_code == 404
