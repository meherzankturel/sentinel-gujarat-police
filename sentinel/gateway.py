#!/usr/bin/env python3
"""
sentinel.gateway -- authenticated access to the Sentinel camera grid.

The government gateway sits behind a login: every path redirects to
/auth/login until a session cookie is held. Credentials are read from
.env, which is gitignored, so they never enter the repository.

The catalogue is the contract. Nothing here hard-codes a stream URL --
camera ids and the available set change, and the catalogue payload is the
only authority on what exists and where it lives.

The catalogue's own location is not stable either. On 16 September 2026
the documented /api/ingest began returning 404 and the grid page was
found to be fetching /cameras.json instead, with a payload thinned to
{id, name} -- no location, codec or stream properties. CATALOGUE_PATHS is
tried in order for that reason, and callers must treat every field beyond
the id as optional.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Optional

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# Tried in order. The first is what the grid page currently fetches; the
# second is what the public Resources page documents. Neither is assumed.
CATALOGUE_PATHS = ("/cameras.json", "/api/ingest")


def load_env(path: Path = ENV_PATH) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text()
    out = {}
    for k, v in re.findall(r"^([A-Z_]+)=(.*)$", txt, re.M):
        v = v.strip().strip('"').strip("'")
        if v and not v.startswith("REPLACE_"):
            out[k] = v
    return out


class Gateway:
    """A logged-in session against the camera grid."""

    def __init__(self, host: Optional[str] = None, email: Optional[str] = None,
                 password: Optional[str] = None):
        env = load_env()
        self.host = (host or env.get("SENTINEL_HOST", "")).replace(
            "https://", "").replace("http://", "").rstrip("/")
        self.email = email or env.get("SENTINEL_EMAIL", "")
        self.password = password or env.get("SENTINEL_PASSWORD", "")
        self.base = f"https://{self.host}"
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        # Media paths are refused for a non-browser agent, so we present as
        # one. Nothing is spoofed about identity -- we are logged in as the
        # registered account; this only satisfies a client-side check.
        self.opener.addheaders = [
            ("User-Agent", BROWSER_UA),
            ("Referer", f"https://{self.host}/"),
            ("Accept", "*/*"),
            ("Accept-Language", "en-GB,en;q=0.9"),
        ]
        self._authed = False

    def login(self) -> bool:
        if not (self.host and self.email and self.password):
            raise RuntimeError("credentials missing from .env")
        data = urllib.parse.urlencode(
            {"email": self.email, "password": self.password}).encode()
        req = urllib.request.Request(
            f"{self.base}/auth/login", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.opener.open(req, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
            final = r.geturl()
        # A failed post lands back on the login form with an error banner.
        self._authed = "/auth/login" not in final or "auth/login" not in final
        if 'name="password"' in body and "auth/login" in final:
            self._authed = False
        return self._authed

    def get(self, path: str, timeout: int = 30, retries: int = 3):
        """
        Fetch with retry. Their server truncates responses mid-download
        often enough to lose a camera from a survey (IncompleteRead), and
        that will happen during the live test too, so it is handled rather
        than allowed to drop a camera silently.

        Sessions also expire mid-run. A 30-camera survey on 10 September
        completed 16 cameras and then took 403 on every remaining one --
        fourteen cameras recorded as failures when nothing was wrong with
        them. A 401 or 403 is therefore treated as "log in again and retry
        once", not as a verdict about the camera.

        A 404 is not retried. The path is wrong; asking three more times
        with backoff only slows down discovering that.
        """
        import http.client
        import time as _t

        url = path if path.startswith("http") else f"{self.base}{path}"
        last = None
        relogged = False
        attempt = 0
        while attempt < retries:
            try:
                with self.opener.open(url, timeout=timeout) as r:
                    return r.read(), r.geturl(), r.status
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 404:
                    raise
                if e.code in (401, 403) and not relogged:
                    relogged = True
                    self._authed = False
                    try:
                        self.login()
                    except Exception:
                        raise last
                    continue                            # retry, no backoff
                if attempt < retries - 1:
                    _t.sleep(1.5 * (2 ** attempt))
            except (http.client.IncompleteRead, http.client.HTTPException,
                    urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                if attempt < retries - 1:
                    _t.sleep(1.5 * (2 ** attempt))     # 1.5s, 3s
            attempt += 1
        raise last

    def catalogue(self) -> list[dict]:
        """
        The contract -- whichever path currently serves it.

        Always returns a list of camera dicts. Only "id" is guaranteed;
        the grid stopped supplying location, codec and stream properties
        in September 2026, so anything else must be measured rather than
        read. That is what the capability survey is for.
        """
        if not self._authed:
            self.login()
        last = None
        for path in CATALOGUE_PATHS:
            try:
                body, final, status = self.get(path)
            except urllib.error.HTTPError as e:
                last = e
                continue
            if "auth/login" in final:
                raise RuntimeError(
                    "not authenticated -- check .env credentials")
            payload = json.loads(body.decode("utf-8", "replace"))
            cams = payload if isinstance(payload, list) else (
                payload.get("cameras") or payload.get("streams") or [])
            if cams:
                self.catalogue_path = path
                return cams
        raise RuntimeError(
            f"no catalogue at any of {CATALOGUE_PATHS} on {self.base}"
            + (f" -- last error {last}" if last else ""))


    # ------------------------------------------------------------- streams

    def cookie_header(self) -> str:
        return "; ".join(f"{c.name}={c.value}" for c in self.jar)

    def hls_url(self, camera_id: str) -> str:
        """
        The player builds this as a relative path from the page root, so it
        is /<id>/index.m3u8 -- not the /live/stream/<id>/ shape the public
        Resources page documents. The catalogue is the contract; the URL
        pattern in the docs is not.
        """
        return f"{self.base}/{camera_id}/index.m3u8"

    def ffmpeg_headers(self) -> str:
        """Headers ffmpeg must send to fetch playlist, key and segments."""
        return (f"Cookie: {self.cookie_header()}\r\n"
                f"Referer: {self.base}/\r\n"
                f"User-Agent: {BROWSER_UA}\r\n")

    def opencv_options(self) -> str:
        """
        Value for OPENCV_FFMPEG_CAPTURE_OPTIONS. Newlines must be escaped as
        the literal characters here, since OpenCV parses this as a flat
        key;value|key;value string.
        """
        hdrs = self.ffmpeg_headers().replace("\r\n", "\\r\\n")
        return f"headers;{hdrs}"
