"""
The hunt, tested where it makes a claim it could get away with faking.

A hunt is the part of this system that reaches furthest: it opens cameras
nobody has searched, on the strength of one picture, and produces sightings
an officer may act on. Four of the assertions below are about honesty rather
than function -- that the query is refused before a frame is opened, that
the audit entry exists before the work is queued, that a camera the registry
measured as presence-only cannot produce a promoted match however convincing
the pixels are, and that the reason a tier was given is carried with it. All
four would still look correct in a live demonstration if they were wrong.
"""

import base64
import functools

import pytest
from fastapi import HTTPException

from sentinel import officer, registry
from sentinel.hunt import Hit

CASE = "FIR-9002/2026"
OFFICER = "PSI Test"
AUTH = "DySP Test"

# The registry rows the tier ceiling is computed from. Deliberately spread
# across the three cases that change the answer.
CAMS = [
    ("h-plate",    "plate-capable",  190,  -1000.0),
    ("h-presence", "presence-only",   40,  -1000.0),
    ("h-drifted",  "plate-capable",  190, 428500.0),
]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A database of its own; these tests write to an append-only chain."""
    path = tmp_path / "hunt.db"
    registry.init(path)
    bound = functools.partial(registry.connect, path)
    monkeypatch.setattr(officer, "connect", bound)
    with bound() as con:
        for cid, cap, px, drift in CAMS:
            con.execute("INSERT INTO camera (id, department, location_name, "
                        "lat, lon, capability_class, plate_px_measured, "
                        "time_confidence_ms, governance_class) "
                        "VALUES (?,?,?,?,?,?,?,?,'open')",
                        (cid, "Police", f"Site {cid}", 23.0, 72.0, cap, px, drift))
    return bound


@pytest.fixture()
def queued(monkeypatch):
    """
    Start hunts without running them. The worker decodes video; a unit test
    that waits on a decoder is a unit test nobody runs.
    """
    seen = []
    monkeypatch.setattr(officer, "_ensure_worker", lambda: None)
    monkeypatch.setattr(officer._JOB_QUEUE, "put", seen.append)
    return seen


def _camera():
    """
    A camera with footage on this machine, whichever they happen to be.

    Skips rather than fails when there is none. The footage is government
    CCTV of a public road and is deliberately not published with this
    repository, so a fresh clone has no media/ -- and ten red tests would
    say "this project is broken" when what they mean is "the video is not
    in the box". The reason is printed so nobody has to guess which.
    """
    cat = officer.footage_catalogue()
    if not cat:
        pytest.skip("no recorded windows in media/real -- this test needs "
                    "government footage, which is not redistributed")
    return sorted(cat)[0]


def _request(**kw):
    body = {"case_ref": CASE, "officer": OFFICER, "authorised_by": AUTH,
            "cameras": [_camera()], "preset": True}
    body.update(kw)
    return officer.HuntRequest(**body)


# 1 -------------------------------------------------- the gate is a gate


@pytest.mark.parametrize("missing", ["case_ref", "officer", "authorised_by"])
def test_a_hunt_is_refused_without_full_authorisation(db, queued, missing):
    with pytest.raises(HTTPException) as e:
        officer.officer_hunt(_request(**{missing: None}))
    assert e.value.status_code == 400
    assert missing in e.value.detail["missing"]
    assert not queued


def test_a_refused_hunt_opens_nothing_and_logs_nothing(db, queued):
    with pytest.raises(HTTPException):
        officer.officer_hunt(_request(officer=None))
    with db() as con:
        assert con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


def test_designation_is_reported_as_a_refusal_not_an_empty_result(db, queued):
    """
    A registration alone cannot drive a hunt on this grid, and the honest
    answer says why. Returning nothing found would tell the officer the
    vehicle was not on those cameras, which is a different claim entirely.
    """
    with pytest.raises(HTTPException) as e:
        officer.officer_hunt(_request(preset=False, plate="GJ01AB1234"))
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "designation_required"
    assert "remedy" in e.value.detail
    assert not queued


def test_a_camera_with_no_footage_is_refused_by_name(db, queued):
    with pytest.raises(HTTPException) as e:
        officer.officer_hunt(_request(cameras=["cam-99"]))
    assert e.value.detail["error"] == "no_footage"
    assert "cam-99" in e.value.detail["unavailable"]
    assert e.value.detail["available"]


# 2 ------------------------------------ the query is logged before the work


def test_the_query_is_on_the_chain_before_the_job_is_queued(db, queued):
    out = officer.officer_hunt(_request())
    with db() as con:
        row = con.execute("SELECT * FROM audit_log ORDER BY id DESC "
                          "LIMIT 1").fetchone()
        intact, broken = registry.verify_audit_chain(con)
    assert row["action"] == "officer_hunt_vehicle"
    assert row["case_ref"] == CASE and row["authorised_by"] == AUTH
    assert out["authorisation"]["entry_hash"] == row["hash"]
    assert intact and broken is None
    # One job queued, and it was queued after the entry existed.
    assert len(queued) == 1 and queued[0].id == out["hunt_id"]


def test_a_designation_read_is_case_linked_too(db, queued):
    """
    Reading a camera to offer vehicles is a read of the grid. A control that
    the hunt enforces and the picker bypasses is not a control.
    """
    req = officer.DesignateRequest(camera=_camera())
    with pytest.raises(HTTPException) as e:
        officer.officer_designate(req)
    assert e.value.detail["error"] == "authorisation_required"


# 3 -------------------------------- capability caps the tier, before the run


def test_the_ceiling_is_stated_before_any_footage_is_opened(db):
    with db() as con:
        rows = {r["id"]: dict(r) for r in con.execute("SELECT * FROM camera")}
    assert officer.tier_ceiling(rows["h-plate"])["tier"] == "probable"
    # Measured incapable of a plate: nothing it sees can be promoted.
    assert officer.tier_ceiling(rows["h-presence"])["tier"] == "corroborating"
    assert "presence-only" in officer.tier_ceiling(rows["h-presence"])["reason"]
    # A clock seven minutes out cannot be trusted to order a journey.
    assert officer.tier_ceiling(rows["h-drifted"])["tier"] == "corroborating"
    assert "clock" in officer.tier_ceiling(rows["h-drifted"])["reason"]


def test_the_reason_carries_the_measurement_that_capped_the_tier(db):
    """
    "Corroborating" on its own is an opinion. The officer is owed the
    measured number the decision was made from.
    """
    hit = Hit(camera_id="h-presence", camera_name="Site", lat=23.0, lon=72.0,
              at_utc="2026-09-10T00:00:00Z", frame=9, tier="corroborating",
              score=0.91, basis="appearance",
              why=["appearance 0.91", "camera measured presence-only"])
    with db() as con:
        cam = dict(con.execute("SELECT * FROM camera WHERE id = 'h-presence'"
                               ).fetchone())
    head, reasons = officer._hunt_reasons(hit, cam)
    assert "0.91" in head and "capped at corroborating" in head
    text = " ".join(r["text"] for r in reasons)
    assert "40px" in text and "120px" in text
    # An appearance match is never allowed to read as certainty.
    assert any(r["code"] == "ceiling" for r in reasons)
    assert "never reaches confirmed" in text


def test_a_strong_match_on_a_capable_camera_is_not_capped(db):
    hit = Hit(camera_id="h-plate", camera_name="Site", lat=23.0, lon=72.0,
              at_utc="2026-09-10T00:00:00Z", frame=9, tier="probable",
              score=0.88, basis="appearance", why=["appearance 0.88"])
    with db() as con:
        cam = dict(con.execute("SELECT * FROM camera WHERE id = 'h-plate'"
                               ).fetchone())
    head, _ = officer._hunt_reasons(hit, cam)
    assert "capped" not in head and "probable" in head


# 4 ------------------------------------------------------ the demo is local


def test_the_hunt_runs_from_footage_held_on_this_machine(db):
    """
    The government gateway is not reliably reachable from a demonstration
    room. Every camera offered must be answerable from a local file.

    Skips when there is no footage. This one reaches the catalogue directly
    rather than through the _camera() helper, so the earlier skip did not
    cover it and a clean clone still showed one red test -- which is the
    same misleading signal, just quieter for being alone.
    """
    catalogue = officer.footage_catalogue()
    if not catalogue:
        pytest.skip("no recorded windows in media/real -- this test needs "
                    "government footage, which is not redistributed")
    for cid, clip in catalogue.items():
        assert clip["frames"] > 0 and clip["fps"] > 0
        assert clip["path"].endswith(".mp4")
    assert "path" not in officer.hunt_sources()["cameras"][0]


def test_an_unreadable_designation_is_rejected_rather_than_hunted(db):
    with pytest.raises(HTTPException) as e:
        officer._decode_image(base64.b64encode(b"not a picture").decode())
    assert e.value.detail["error"] == "unreadable_image"


def test_a_frame_name_cannot_leave_the_hunt_directory(db):
    for bad in ("../../sentinel.db", "..%2Fsecret.jpg", "a/b.jpg", "x.png"):
        assert not officer._SAFE_IMAGE.match(bad)
    assert officer._SAFE_IMAGE.match("crop_cam09_36.jpg")
