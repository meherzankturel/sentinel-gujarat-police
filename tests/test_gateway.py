"""
The gateway, tested against the two ways the grid actually broke.

Neither of these is hypothetical. On 16 September 2026 the documented
/api/ingest began returning 404 and the catalogue was found at
/cameras.json instead. And a 30-camera survey on 10 September completed
sixteen cameras, then took 403 on all fourteen that remained -- the
session had expired, but the survey recorded fourteen cameras as failed
and the registry carried that hole for six days.

Both failures produced results that looked like findings. A catalogue that
404s reads as "the grid is down"; a 403 mid-run reads as "these cameras
are unavailable". The tests below are for the recovery, because the
recovery is what stops a session problem from being written down as a
fact about a camera.
"""

import io
import json
import urllib.error

import pytest

from sentinel import gateway
from sentinel.gateway import Gateway


def _http_error(code):
    return urllib.error.HTTPError(
        "https://grid.test/x", code, "nope", {}, io.BytesIO(b""))


@pytest.fixture()
def gw(monkeypatch):
    monkeypatch.setattr(gateway, "load_env", lambda *a, **k: {
        "SENTINEL_HOST": "grid.test",
        "SENTINEL_EMAIL": "a@b.c",
        "SENTINEL_PASSWORD": "x",
    })
    g = Gateway()
    g._authed = True
    return g


def test_catalogue_falls_back_when_the_documented_path_is_gone(gw, monkeypatch):
    """404 on /api/ingest is not "no grid" -- it is "not there any more"."""
    seen = []

    def fake_get(path, **kw):
        seen.append(path)
        if path == "/cameras.json":
            raise _http_error(404)
        body = json.dumps([{"id": "cam01", "name": "01 Bridge"}]).encode()
        return body, f"https://grid.test{path}", 200

    monkeypatch.setattr(gw, "get", fake_get)
    cams = gw.catalogue()

    assert [c["id"] for c in cams] == ["cam01"]
    assert gw.catalogue_path == "/api/ingest"
    assert seen == ["/cameras.json", "/api/ingest"]


def test_catalogue_accepts_the_thinned_payload(gw, monkeypatch):
    """
    The grid stopped supplying location, codec and stream properties. Only
    the id is depended on; everything else is measured, not read.
    """
    body = json.dumps([{"id": "cam01", "name": "01 Bridge"},
                       {"id": "cam02", "name": "02 Circle"}]).encode()
    monkeypatch.setattr(gw, "get",
                        lambda p, **k: (body, f"https://grid.test{p}", 200))

    cams = gw.catalogue()
    assert len(cams) == 2
    assert all(set(c) == {"id", "name"} for c in cams)


def test_catalogue_raises_when_no_path_serves_it(gw, monkeypatch):
    monkeypatch.setattr(gw, "get",
                        lambda p, **k: (_ for _ in ()).throw(_http_error(404)))
    with pytest.raises(RuntimeError, match="no catalogue"):
        gw.catalogue()


def test_expired_session_logs_in_again_rather_than_failing_the_camera(
        gw, monkeypatch):
    """The 10 September failure: 403 must not be recorded as a camera fault."""
    calls = {"open": 0, "login": 0}

    class _Resp:
        status = 200

        def read(self):
            return b"#EXTM3U"

        def geturl(self):
            return "https://grid.test/cam17/index.m3u8"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(url, timeout=None):
        calls["open"] += 1
        if calls["open"] == 1:
            raise _http_error(403)
        return _Resp()

    def fake_login():
        calls["login"] += 1
        gw._authed = True
        return True

    monkeypatch.setattr(gw.opener, "open", fake_open)
    monkeypatch.setattr(gw, "login", fake_login)

    body, _, status = gw.get("/cam17/index.m3u8")

    assert status == 200 and body == b"#EXTM3U"
    assert calls["login"] == 1, "should have re-authenticated exactly once"
    assert calls["open"] == 2


def test_relogin_is_attempted_only_once(gw, monkeypatch):
    """A grid that 403s persistently must not become a login loop."""
    calls = {"open": 0, "login": 0}

    def fake_open(url, timeout=None):
        calls["open"] += 1
        raise _http_error(403)

    monkeypatch.setattr(gw.opener, "open", fake_open)
    monkeypatch.setattr(gw, "login", lambda: calls.__setitem__(
        "login", calls["login"] + 1) or True)

    with pytest.raises(urllib.error.HTTPError):
        gw.get("/cam17/index.m3u8", retries=3)

    assert calls["login"] == 1


def test_404_is_not_retried(gw, monkeypatch):
    """Backing off three times to re-learn that a path is wrong is waste."""
    calls = {"open": 0}

    def fake_open(url, timeout=None):
        calls["open"] += 1
        raise _http_error(404)

    monkeypatch.setattr(gw.opener, "open", fake_open)
    with pytest.raises(urllib.error.HTTPError):
        gw.get("/api/ingest", retries=3)

    assert calls["open"] == 1


def test_truncated_response_is_still_retried(gw, monkeypatch):
    """Their server truncates mid-download; that one must keep its retries."""
    import http.client
    calls = {"open": 0}

    class _Resp:
        status = 200

        def read(self):
            return b"ok"

        def geturl(self):
            return "https://grid.test/cam22/index.m3u8"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(url, timeout=None):
        calls["open"] += 1
        if calls["open"] < 3:
            raise http.client.IncompleteRead(b"partial", 500)
        return _Resp()

    monkeypatch.setattr(gw.opener, "open", fake_open)
    monkeypatch.setattr(gw, "login", lambda: True)
    monkeypatch.setattr(gateway, "CATALOGUE_PATHS", ("/cameras.json",))

    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)

    body, _, _ = gw.get("/cam22/index.m3u8", retries=3)
    assert body == b"ok" and calls["open"] == 3
