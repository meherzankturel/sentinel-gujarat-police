"""
Every route that reads movement data, swept at the HTTP layer.

This file exists because of a hole it would have caught on day one.
/api/officer/route returned a citizen's entire movement history -- every
camera, location, department and corrected time -- to any caller with no
credentials at all, and the audit write was conditional on those same
missing credentials, so an unauthorised read left no trace in the chain.
It sat behind a one-word change to a URL next to a search that was
correctly gated.

The existing tests did not catch it because they call the route functions
directly, as Python. That tests the handler and never the routing: it
cannot tell you which paths FastAPI exposes, and it happily passes while
a handler skips the gate. Four of them called this one with no credentials
and asserted on the results, so the suite ratified the bypass rather than
finding it.

So this sweep is deliberately dumb. It enumerates the app's own routes
rather than a hand-written list -- a route added tomorrow is covered by
this file tomorrow, without anyone remembering to add it.
"""

import functools

import pytest
from fastapi.testclient import TestClient

from sentinel import api, officer, registry

CASE = "FIR-9100/2026"
OFFICER = "PSI Sweep"
AUTH = "DySP Sweep"
PLATE = "GJ01AB1234"

# Reading any of these reconstructs where a person has been. None may answer
# without a case, a searching officer and an authorising officer.
GATED = {
    "/api/officer/search": {"plate": PLATE},
    "/api/officer/route": {"plate": PLATE},
}

CREDENTIALS = {"officer": OFFICER, "case_ref": CASE, "authorised_by": AUTH}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "sweep.db"
    registry.init(path)
    bound = functools.partial(registry.connect, path)
    monkeypatch.setattr(officer, "connect", bound)
    monkeypatch.setattr(api, "connect", bound)
    return TestClient(api.app)


def test_the_gated_routes_refuse_without_credentials(client):
    for path, params in GATED.items():
        r = client.get(path, params=params)
        assert r.status_code == 400, f"{path} answered without credentials"
        assert r.json()["detail"]["error"] == "authorisation_required", path


@pytest.mark.parametrize("drop", ["officer", "case_ref", "authorised_by"])
def test_each_credential_is_required_on_its_own(client, drop):
    """All three, not any one of them. A case with no named officer is not
    an authorisation, it is a label."""
    for path, params in GATED.items():
        sent = dict(params, **{k: v for k, v in CREDENTIALS.items() if k != drop})
        r = client.get(path, params=sent)
        assert r.status_code == 400, f"{path} answered with {drop} missing"
        assert drop in r.json()["detail"]["missing"], path


def test_the_gated_routes_answer_with_full_credentials(client):
    """The gate must be a gate, not a wall."""
    for path, params in GATED.items():
        r = client.get(path, params=dict(params, **CREDENTIALS))
        assert r.status_code == 200, f"{path} refused a valid authorisation"


def test_an_authorised_read_is_always_written_to_the_chain(client):
    """
    The audit write used to be conditional on the same credentials the gate
    now requires, which meant the only reads that went unlogged were
    precisely the ones that should never have happened.
    """
    for path, params in GATED.items():
        before = _audit_count(client)
        assert before >= 0, "/api/audit did not answer; the check is vacuous"
        client.get(path, params=dict(params, **CREDENTIALS))
        assert _audit_count(client) > before, f"{path} left no audit entry"


def _audit_count(client) -> int:
    r = client.get("/api/audit")
    if r.status_code != 200:
        return -1
    body = r.json()
    entries = body.get("entries", body) if isinstance(body, dict) else body
    return len(entries) if isinstance(entries, list) else -1


def test_no_officer_route_is_left_ungated_by_accident(client):
    """
    Enumerated from the app rather than hand-listed, so a route added later
    is covered without anyone remembering to add it here. A new officer
    route that reads sightings must either appear in GATED or be added
    knowingly to the exemption below.
    """
    exempt = {
        "/api/officer/hunt/sources",   # lists local footage, no person data
        "/officer",                    # the page itself
    }
    found = set()
    for r in api.app.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/api/officer"):
            continue
        if "{" in path or path in exempt:
            continue
        if "GET" not in getattr(r, "methods", set()):
            continue
        found.add(path)

    unreviewed = found - set(GATED) - exempt
    assert not unreviewed, (
        f"ungated officer GET route(s): {sorted(unreviewed)}. Add the gate, "
        f"or list it in GATED/exempt with a reason.")


# ---- the registry must never inflate its own government count ----

def test_each_grid_is_classified_positively_and_never_defaults():
    """
    The own-feed cameras are local files and carry no stream URL. The old
    rule inferred the grid from that URL and defaulted to "government", so
    three of the entrant's own clips were counted as government cameras and
    the console reported 33 where the grid has 30.

    A camera must now present evidence of where it belongs. Anything that
    presents none is "unclassified" and says so, because inventing a home
    for it is how the count drifted.
    """
    from sentinel.api import grid_of
    assert grid_of({"hls_url": "https://cctv.corp8.cloud/cam01/index.m3u8"}) == "government"
    assert grid_of({"rtsp_url": "rtsp://127.0.0.1:8554/stream/cam-01"}) == "local-mock"
    assert grid_of({"hls_url": None, "department": "Entrant"}) == "own-feed"
    # The failure that mattered: no URL, no known department, no invention.
    assert grid_of({"hls_url": None, "department": "Police"}) == "unclassified"
    assert grid_of({}) == "unclassified"


def test_a_camera_with_no_url_is_never_called_government():
    """The padding direction, asserted directly."""
    from sentinel.api import grid_of
    for dept in ("Police", "Health", "GSRTC", "Municipal", "Panchayat", ""):
        assert grid_of({"department": dept}) != "government"
