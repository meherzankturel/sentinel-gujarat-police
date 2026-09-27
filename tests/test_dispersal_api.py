"""
The dispersal surface, tested where it makes claims.

Most of these assertions are about honesty rather than function. The
detector itself is tested in test_threat.py against flow fields with a
known shape; what can go wrong *here* is subtler and worse:

  * a rendered crowd served to an operator as a real detection,
  * an epicentre published as a point when it was measured as a region,
  * layers 2 and 3 served as if they had been validated, which they have
    not been and cannot be on this grid,
  * an alert that dispatches officers leaving no trace in the audit chain.

Each of those would look perfectly correct in a demonstration.
"""

import functools

import pytest
from fastapi.testclient import TestClient

from sentinel import api, registry

DETECTOR = "node-test-1"

#        id,      dept,        capability,             plate px, w,    h
CAMS = [("d-01", "Police",    "plate-capable",        192, 1920, 1080),
        ("d-02", "Municipal", "presence-only",         21, 1920, 1080),
        ("d-03", "Panchayat", "no-vehicles-observed", None,  854,  480)]


def _report(camera_id="d-02", **kw):
    body = {
        "camera_id": camera_id,
        "detector": DETECTOR,
        "source": "government-live",
        "pts_s": 31.4,
        "epicentre": [612, 331],
        "frame_w": 1280,
        "frame_h": 720,
        "surge_ratio": 2.4,
        "coherence": 0.52,
        "net_drift": 0.36,
        "movers": 11,
        "confidence": 0.71,
    }
    body.update(kw)
    return body


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    Its own database. These requests write to the audit chain, which is
    append-only by design, so running them against the demo database would
    leave test alerts permanently in the chain the evidence bundles cite.
    """
    path = tmp_path / "test.db"
    registry.init(path)
    monkeypatch.setattr(api, "connect", functools.partial(registry.connect, path))
    with registry.connect(path) as con:
        for cid, dept, cap, px, w, h in CAMS:
            con.execute("INSERT INTO camera (id, department, location_name, "
                        "lat, lon, capability_class, plate_px_measured, "
                        "width, height, governance_class) "
                        "VALUES (?,?,?,?,?,?,?,?,?,'open')",
                        (cid, dept, f"Site {cid}", 23.0, 72.0, cap, px, w, h))
    return TestClient(api.app)


# --------------------------------------------------------- the surface


def test_layer_one_is_offered_on_the_cameras_that_can_do_nothing_else(client):
    """
    The whole argument for layer 1 is that it runs where plate reading
    cannot. If the eligibility list ever collapses onto the plate-capable
    cameras, the analytic has stopped being interesting and the console
    should stop saying otherwise.
    """
    d = client.get("/api/dispersal").json()
    assert d["summary"]["cameras"] == len(CAMS)
    assert d["summary"]["eligible"] == len(CAMS)
    assert d["summary"]["eligible_without_plate_capability"] == 2
    by_id = {c["camera_id"]: c for c in d["cameras"]}
    assert by_id["d-03"]["eligible"] is True      # no plate ever measured
    assert by_id["d-03"]["plate_px"] is None


def test_eligibility_comes_from_layer_plan_not_from_a_local_rule(client):
    """
    Two writers with two rules is how the analytic vocabulary drifted once
    already, so the endpoint must agree with threat.layer_plan camera by
    camera rather than reimplement the gate.
    """
    from sentinel.threat import layer_plan
    for c in client.get("/api/dispersal").json()["cameras"]:
        plan = layer_plan(plate_px=c["plate_px"])
        assert c["eligible"] is bool(plan["dispersal"]["enabled"])
        assert c["why"] == plan["dispersal"]["why"]
        assert c["posture_gate"] == plan["posture"]
        assert c["weapon_gate"] == plan["weapon"]


def test_layers_two_and_three_are_never_served_as_validated(client):
    """
    Posture and weapon-in-hand have no footage to validate against on this
    grid. Their gates are published because a refusal is a useful output;
    an alert from either would be a claim nothing supports.
    """
    d = client.get("/api/dispersal").json()
    assert d["layer"] == 1
    text = d["validation"]["layers_2_and_3"].lower()
    assert "not validated" in text and "not deployed" in text
    # No layer-2/3 alert stream exists to be mistaken for one.
    assert set(a["source"] for a in d["alerts"]) <= set(api.DISPERSAL_SOURCES)


def test_the_true_positive_claim_is_labelled_as_simulation(client):
    v = client.get("/api/dispersal").json()["validation"]
    assert "rendered" in v["true_positives"].lower()
    assert "366" in v["false_positives"]          # real footage, real hours


# ----------------------------------------------------------- recording


def test_an_alert_is_recorded_and_written_to_the_audit_chain(client):
    r = client.post("/api/dispersal/alerts", json=_report())
    assert r.status_code == 201
    alert = r.json()["alert"]
    assert alert["camera_id"] == "d-02"
    assert alert["acknowledged_by"] is None

    a = client.get("/api/audit").json()
    assert a["chain_intact"] is True
    entry = a["entries"][0]
    assert entry["action"] == "dispersal_alert"
    assert entry["actor"] == DETECTOR
    # Not case-linked: a dispersal has no subject and no one to authorise.
    assert entry["case_ref"] is None
    assert entry["authorised_by"] is None


def test_recording_an_alert_needs_no_authorising_officer(client):
    """
    Deliberately different from /api/officer/search, which refuses without
    a case and an authorising officer. Nobody is being searched for here,
    and a control room cannot be made to wait for a case reference before
    it is told a crowd is running.
    """
    assert client.post("/api/dispersal/alerts", json=_report()).status_code == 201
    assert client.get("/api/officer/search",
                      params={"plate": "GJ01AB1234"}).status_code == 400


def test_an_alert_from_an_unknown_camera_is_refused(client):
    r = client.post("/api/dispersal/alerts", json=_report(camera_id="d-99"))
    assert r.status_code == 404


# ------------------------------------------------------------ honesty


def test_footage_provenance_is_mandatory_and_closed(client):
    r = client.post("/api/dispersal/alerts", json=_report(source="live"))
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unknown_source"


def test_a_simulated_alert_cannot_be_recorded_without_saying_so(client):
    """
    The one failure that would matter more than the feature: a rendered
    crowd shown to an officer as a real detection. Refused at the API
    boundary rather than left to the template.
    """
    r = client.post("/api/dispersal/alerts", json=_report(source="simulation"))
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "scene_note_required"

    ok = client.post("/api/dispersal/alerts", json=_report(
        source="simulation", scene_note="rendered crowd, grid/plaza.py seed 1"))
    assert ok.status_code == 201
    assert ok.json()["alert"]["simulated"] is True

    d = client.get("/api/dispersal").json()
    assert d["summary"]["simulated_alerts"] == 1
    assert d["summary"]["live_alerts"] == 0


def test_the_epicentre_is_published_as_a_region_not_a_point(client):
    """
    Measured at ~15% of the frame diagonal off the true origin. The API
    hands out the radius so no caller can draw a pinpoint by accident.
    """
    alert = client.post("/api/dispersal/alerts",
                        json=_report(frame_w=1280, frame_h=720)).json()["alert"]
    assert alert["epicentre_region_px"] == 220      # 0.15 * hypot(1280,720)
    assert "region" in alert["epicentre_basis"].lower()


# -------------------------------------------------------- acknowledging


def test_acknowledging_names_the_operator_and_is_audited(client):
    aid = client.post("/api/dispersal/alerts", json=_report()).json()["alert"]["id"]
    r = client.post(f"/api/dispersal/alerts/{aid}/acknowledge",
                    params={"officer": "PSI Test"})
    assert r.status_code == 200
    assert r.json()["alert"]["acknowledged_by"] == "PSI Test"

    assert client.get("/api/dispersal").json()["summary"]["unacknowledged"] == 0
    a = client.get("/api/audit").json()
    assert a["chain_intact"] is True
    assert a["entries"][0]["action"] == "acknowledge_dispersal"


def test_an_unnamed_acknowledgement_is_refused(client):
    aid = client.post("/api/dispersal/alerts", json=_report()).json()["alert"]["id"]
    assert client.post(f"/api/dispersal/alerts/{aid}/acknowledge").status_code == 422
    assert client.post("/api/dispersal/alerts/9999/acknowledge",
                       params={"officer": "PSI Test"}).status_code == 404
