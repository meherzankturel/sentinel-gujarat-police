#!/usr/bin/env python3
"""
sentinel.hls -- sample a camera without dragging its whole day off the server.

The sandbox serves each camera as a 12-hour AES-128 encrypted HLS VOD.
Pointing ffmpeg at the playlist and asking it to seek makes it walk
segments from the beginning: slow for us and rude to a server we were
explicitly asked to pace our load against.

So we do the arithmetic ourselves -- work out which segments cover the
window we want, fetch only those, and hand ffmpeg a small local playlist.
A 30-second sample costs five segments instead of thousands.

The same code path works against a live rolling playlist: the only
difference is that the segment list is refetched rather than cached.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .gateway import Gateway


def _decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """
    HLS AES-128 is CBC with PKCS7 padding over the whole segment.

    We decrypt here rather than let ffmpeg do it, because ffmpeg refuses to
    open a key file whose extension is not a known media type -- and the
    alternative, allowed_extensions=ALL, loosens a security control on every
    stream we ever open just to read one key. Decrypting ourselves keeps
    that control intact and leaves plain segments on disk.
    """
    d = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    out = d.update(data) + d.finalize()
    if out:
        pad = out[-1]
        if 1 <= pad <= 16 and out[-pad:] == bytes([pad]) * pad:
            out = out[:-pad]
    return out


@dataclass
class Playlist:
    camera_id: str
    segments: List[Tuple[str, float]]      # (name, duration)
    total_s: float
    key_uri: Optional[str]
    key_iv: Optional[str]
    is_vod: bool

    def index_at(self, t: float) -> int:
        """Segment covering position t seconds into the recording."""
        t = t % self.total_s if self.total_s else 0
        acc = 0.0
        for i, (_, d) in enumerate(self.segments):
            if acc + d > t:
                return i
            acc += d
        return max(0, len(self.segments) - 1)


class HLSSampler:
    def __init__(self, gw: Gateway, workdir: Optional[Path] = None):
        self.gw = gw
        self.work = Path(workdir or tempfile.mkdtemp(prefix="sentinel-hls-"))
        self.work.mkdir(parents=True, exist_ok=True)
        self._playlists: dict[str, Playlist] = {}
        self._key: Optional[bytes] = None

    # ----------------------------------------------------------- playlist

    def playlist(self, camera_id: str, refresh: bool = False) -> Playlist:
        if not refresh and camera_id in self._playlists:
            return self._playlists[camera_id]
        body, _, _ = self.gw.get(f"/{camera_id}/index.m3u8", timeout=45)
        txt = body.decode("utf-8", "replace")

        segs: List[Tuple[str, float]] = []
        dur = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("#EXTINF:"):
                try:
                    dur = float(line[8:].split(",")[0])
                except ValueError:
                    dur = None
            elif line and not line.startswith("#") and dur is not None:
                segs.append((line, dur))
                dur = None

        k = re.search(r'#EXT-X-KEY:[^\n]*URI="([^"]+)"', txt)
        iv = re.search(r"#EXT-X-KEY:[^\n]*IV=(0x[0-9A-Fa-f]+)", txt)
        pl = Playlist(
            camera_id=camera_id, segments=segs,
            total_s=sum(d for _, d in segs),
            key_uri=k.group(1) if k else None,
            key_iv=iv.group(1) if iv else None,
            is_vod=("#EXT-X-ENDLIST" in txt))
        self._playlists[camera_id] = pl
        return pl

    def key(self, uri: str) -> bytes:
        if self._key is None:
            body, _, _ = self.gw.get(uri, timeout=20)
            self._key = body
        return self._key

    # ------------------------------------------------------------- sample

    def sample(self, camera_id: str, at_s: float, seconds: float = 30.0) -> Path:
        """
        Fetch just enough segments to cover `seconds` from position `at_s`
        and return a local playlist ffmpeg can open.
        """
        pl = self.playlist(camera_id)
        if not pl.segments:
            raise RuntimeError(f"{camera_id}: empty playlist")

        start = pl.index_at(at_s)
        want, got = seconds, []
        i = start
        while want > 0 and i < len(pl.segments):
            name, d = pl.segments[i]
            got.append((name, d))
            want -= d
            i += 1
        if not got:
            raise RuntimeError(f"{camera_id}: no segments at {at_s:.0f}s")

        outdir = self.work / camera_id
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)

        lines = ["#EXTM3U", "#EXT-X-VERSION:6", "#EXT-X-TARGETDURATION:10",
                 f"#EXT-X-MEDIA-SEQUENCE:{start}"]

        key = self.key(pl.key_uri) if pl.key_uri else None
        iv = bytes.fromhex((pl.key_iv or "0x" + "0" * 32)[2:])

        for name, d in got:
            body, _, _ = self.gw.get(f"/{camera_id}/{name}", timeout=60)
            if key:
                body = _decrypt(body, key, iv)
            (outdir / name).write_bytes(body)
            lines += [f"#EXTINF:{d:.6f},", name]
        lines.append("#EXT-X-ENDLIST")

        m3u8 = outdir / "sample.m3u8"
        m3u8.write_text("\n".join(lines) + "\n")
        return m3u8

    def cleanup(self):
        shutil.rmtree(self.work, ignore_errors=True)
