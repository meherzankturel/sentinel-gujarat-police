"""
The hosted console, tested against the console it hosts.

This file exists because of a specific failure. The deployment used to
carry its own copy of the officer's endpoints -- a serverless function
caps at 250MB and the real module dragged in PyTorch, so a copy looked
like the only way -- and the copy drifted. It answered a search with
`summary.tiers` where the console reads `tiers`, the page threw on the
first result, and the officer's screen showed a header full of numbers
above an empty map. It was live on the public URL for a day.

Nothing here tests the search logic; `test_officer.py` does that. These
assert the things that drift silently:

    - the console's own fetch() calls all reach a route
    - the response carries the fields the console actually reads
    - an endpoint that cannot run here refuses with an explanation,
      rather than 404ing or raising
    - the hosted half still imports without the vision stack

The last one is the load-bearing assertion. Everything else follows from
the real endpoints being mounted, and they can only be mounted while
`sentinel.officer` imports in a function with no OpenCV and no torch.
"""

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
WEB = ROOT / "web"

CREDENTIALS = {"username": "reviewer", "password": "gujarat2026"}
QUERY = ("plate=GJ01AB1234&case_ref=FIR-0117/2026"
         "&officer=PSI+R.+Chauhan&authorised_by=DySP+M.+Parmar")

# The vision stack, and the bare fact that the hosted half must not touch
# it. Kept here as well as in tools/build_demo_site.py: the build refuses
# to publish a bundle that imports these, and this refuses to let the
# import creep back in between builds.
HEAVY = ("cv2", "torch", "torchvision", "numpy", "PIL")


@pytest.fixture(scope="module")
def client(monkeypatch_session=None):
    """The hosted app, loaded the way Vercel loads it."""
    if not (DEPLOY / "sentinel-demo.db").exists():
        pytest.skip("deploy/ not built — run tools/build_demo_site.py")

    import os
    os.environ.setdefault("SENTINEL_SECRET", "test-only-not-a-deployment-key")
    sys.path.insert(0, str(DEPLOY))
    spec = importlib.util.spec_from_file_location(
        "hosted_index", DEPLOY / "api" / "index.py")
    index = importlib.util.module_from_spec(spec)
    sys.modules["hosted_index"] = index
    spec.loader.exec_module(index)

    from fastapi.testclient import TestClient
    # https, because the session cookie is marked secure and an http test
    # client would silently drop it and make every assertion below a 401.
    c = TestClient(index.app, base_url="https://testserver")
    c.post("/login", data=CREDENTIALS, follow_redirects=False)
    return c


# ------------------------------------------------- the console's own calls


def console_calls() -> list[str]:
    """Every endpoint the two consoles fetch, taken from their source."""
    found = set()
    for page in ("index.html", "officer.html"):
        for path in re.findall(r'fetch\("(/[^"?]*)',
                               (WEB / page).read_text()):
            found.add(path.rstrip("/"))
    return sorted(found)


def test_every_endpoint_the_console_calls_exists(client):
    """
    A 404 here is the bug this file was written for. The evidence bundle
    -- the one action on the officer's screen that produces something a
    court sees -- 404'd on the deployment for a day, and the console
    swallowed it, because nothing checked that the page and the server
    agreed on what exists.
    """
    missing = []
    for path in console_calls():
        # Hunt results are fetched by id; any id exercises the same route.
        probe = path if not path.endswith("/hunt") else path
        for method in ("GET", "POST"):
            r = client.request(method, probe + "?" + QUERY)
            if r.status_code != 405:
                break
        if r.status_code == 404:
            missing.append(f"{path} -> 404")
    assert not missing, "the console calls endpoints the server does not have: " \
                        + ", ".join(missing)


# --------------------------------------------------------- the search shape


def test_search_carries_the_fields_the_console_reads(client):
    r = client.get(f"/api/officer/search?{QUERY}")
    assert r.status_code == 200
    d = r.json()

    # Read straight off officer.html: draw() uses search.tiers[t] and
    # search.departments.length, and every row in the timeline uses the
    # per-sighting tier. The drifted copy had none of the three.
    assert set(d["tiers"]) == {"confirmed", "probable", "corroborating"}
    assert isinstance(d["departments"], list) and d["departments"]
    assert d["sightings"], "the demo database should hold sightings"
    for s in d["sightings"]:
        assert s["tier"] in {"confirmed", "probable", "corroborating"}
        assert s["at"], "the timeline sorts and prints on `at`"
        assert s["seq"], "the map and the timeline select each other by seq"


def test_route_carries_the_summary_the_header_prints(client):
    summary = client.get(f"/api/officer/route?{QUERY}").json()["summary"]
    for field in ("sightings", "cameras", "distance_km", "span_minutes",
                  "impossible_legs"):
        assert field in summary, f"the header prints {field}"


# ------------------------------------------------------------ the governance


def test_a_search_without_an_authorising_officer_is_refused(client):
    r = client.get("/api/officer/search?plate=GJ01AB1234")
    assert r.status_code == 400
    assert set(r.json()["detail"]["missing"]) == {
        "officer", "case_ref", "authorised_by"}


def test_the_hosted_console_writes_the_audit_entry_it_promises(client):
    """
    The screen says "the query is written to the tamper-evident log before
    any result is returned". On the drifted copy that sentence was false:
    it answered searches and logged nothing. Of all the ways this
    deployment could be wrong, a governance claim it does not honour is
    the one that matters.
    """
    before = client.get("/api/audit").json()["count"]
    client.get(f"/api/officer/search?{QUERY}")
    after = client.get("/api/audit").json()
    assert after["count"] == before + 1
    assert after["chain_intact"], "the audit chain must verify after a write"


def test_evidence_bundle_is_produced_and_hashed(client):
    r = client.post(f"/api/officer/evidence?{QUERY}")
    assert r.status_code == 200
    d = r.json()
    assert d["integrity"]["algorithm"] == "sha256"
    assert len(d["integrity"]["sha256"]) == 64
    # The fields the bundle panel prints.
    b = d["bundle"]
    for field in ("case_ref", "prepared_by", "authorised_by", "subject",
                  "sightings", "route", "cameras", "audit_trail",
                  "audit_chain", "generated_at"):
        assert field in b


def test_session_is_required(client):
    from fastapi.testclient import TestClient
    anon = TestClient(client.app, base_url="https://testserver")
    for path in ("/api/cameras", "/api/officer/search?" + QUERY,
                 "/api/officer/hunt/sources"):
        assert anon.get(path).status_code == 401, path


# ------------------------------------------------------- what cannot run here


def test_the_hunt_refuses_with_an_explanation(client):
    """
    Not 404, not 500. The hunt needs a GPU this deployment does not have,
    and the console renders whatever it is told -- so it must be told
    something a reviewer can read.
    """
    for method, path in (("GET", "/api/officer/hunt/sources"),
                         ("POST", "/api/officer/hunt"),
                         ("POST", "/api/officer/hunt/designate"),
                         ("GET", "/api/officer/hunt/any-id")):
        r = client.request(method, path)
        assert r.status_code == 501, f"{method} {path} -> {r.status_code}"
        body = r.json()
        assert body["error"] == "not_available_on_hosted_demo"
        assert body["message"] and body["remedy"], \
            "a refusal with no explanation is just a failure"


# ------------------------------------------------- what a camera can do to people


def test_face_capability_reaches_the_console(client):
    """
    The brief names facial recognition, and the honest answer is a
    measurement rather than a feature. It is only an answer if a reviewer
    can see it, so it travels on the camera row like plate width does.
    """
    cams = client.get("/api/cameras").json()["cameras"]
    measured = [c for c in cams if c.get("face_iod_px") is not None]
    assert measured, "no camera carries a face measurement"

    for c in measured:
        assert c["face_class"] in {"face-capable", "face-marginal",
                                   "face-unusable"}
        assert c["person_body_px"] and c["face_people_seen"]
        # The derivation is inter-ocular ~ standing height / 37.9. If the
        # two ever disagree, one of them was typed rather than measured.
        assert abs(c["face_iod_px"] - c["person_body_px"] / 37.9) < 0.15

    # Every camera measured on this grid falls under the 40px floor. If a
    # future survey finds one that does not, this test should fail and be
    # read, because it changes what the platform is allowed to schedule.
    assert all(c["face_class"] == "face-unusable" for c in measured), (
        "a camera now clears the face floor — the deck, the HLD and the "
        "console all say none does")


# ------------------------------------------------ what must never be published


def test_no_still_is_published_for_a_withheld_sighting():
    """
    The stills are cut per read, and some reads belong to a member of the
    public whose parked car appears in the entrant's own street footage.
    Those sightings are withheld from the hosted database; publishing the
    matching image would hand over the same person by another route.

    Checked against the built deployment rather than the intention: the
    files on disk are what a reviewer can fetch.
    """
    import sqlite3
    published = DEPLOY / "site" / "assets" / "evidence"
    if not published.exists():
        pytest.skip("deploy/ not built — run tools/build_demo_site.py")

    con = sqlite3.connect(DEPLOY / "sentinel-demo.db")
    kept = {r[0] for r in con.execute(
        "SELECT evidence_ref FROM sighting WHERE evidence_ref IS NOT NULL")}
    con.close()

    orphans = [f.name for f in published.iterdir()
               if not any(f.name.startswith(stem + "-") for stem in kept)]
    assert not orphans, ("stills published that no surviving sighting "
                         "references: " + ", ".join(orphans))


def test_the_withheld_registration_appears_nowhere(client):
    """The plate itself, checked through the API a reviewer actually has."""
    for plate in ("GJ05JS8590", "GJO5JS8590", "CJ05JS8590"):
        r = client.get(f"/api/officer/search?plate={plate}&case_ref=FIR-0117/2026"
                       "&officer=PSI+R.+Chauhan&authorised_by=DySP+M.+Parmar")
        assert r.status_code == 200
        assert r.json()["sightings"] == [], f"{plate} is reachable"


# ------------------------------------------- the reason any of this is possible


def test_the_hosted_half_imports_without_the_vision_stack():
    """
    sentinel.officer defers cv2, torch, the detector, the tracker and the
    hunt so that the read-only half fits in a serverless function. If a
    module-level import of any of them comes back, the deployment cannot
    mount the real endpoints and the pressure to reimplement them returns
    with it. Fail here instead.
    """
    check = textwrap.dedent(f"""
        import sys, importlib.abc
        BLOCK = {HEAVY!r}
        class Deny(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError("heavy import: " + name)
                return None
        sys.meta_path.insert(0, Deny())
        from sentinel import officer
        assert officer.router.routes
    """)
    r = subprocess.run([sys.executable, "-c", check], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_deployment_runs_the_real_endpoints_not_a_copy(client):
    """
    The structural version of every assertion above: search, route and
    evidence are served by sentinel.officer's own functions, not by
    lookalikes defined in the deployment. Compared by module and name
    rather than by identity, because the deployment imports its own
    vendored copy of the package and the two objects are legitimately
    different -- what must never differ is which module wrote them.
    """
    wanted = {"/api/officer/search": "officer_search",
              "/api/officer/route": "officer_route",
              "/api/officer/evidence": "officer_evidence"}

    served = {}
    for route in client.app.routes:
        # FastAPI wraps an included router rather than flattening it.
        group = getattr(route, "original_router", None)
        for r in (group.routes if group else [route]):
            path = getattr(r, "path", None)
            if path in wanted:
                served[path] = r.endpoint

    for path, name in wanted.items():
        fn = served.get(path)
        assert fn is not None, f"{path} is not served at all"
        assert fn.__module__.endswith("sentinel.officer"), \
            f"{path} is served by {fn.__module__}, not sentinel.officer"
        assert fn.__name__ == name
