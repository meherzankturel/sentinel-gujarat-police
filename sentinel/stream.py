#!/usr/bin/env python3
"""
sentinel.stream -- the one way this system reads a camera.

Every checklist item the organisers published about stream handling is
enforced here, in one place, so no downstream code can quietly violate one:

  * RTSP is forced over TCP, and the transport is verifiable.
  * Timing comes from PTS, never from frame arrival time.
  * Inter-frame gaps are recorded, not treated as disconnects.
  * Reconnect uses exponential backoff, ~2s to ~30s, never a tight loop.
  * Decoder warnings at join are logged and survived, not fatal.
  * Captures are closed when done, because each client costs the gateway
    a full copy of the stream.

Why PTS matters, concretely: on connect the gateway replays its buffered
group-of-pictures, so the first couple of seconds arrive faster than real
time. Code that timestamps by arrival will compute impossible velocities
after every single reconnect, and a Kalman filter fed those numbers
produces confident nonsense. Feeding PTS deltas instead makes the join
burst harmless.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

# Must be set before cv2 is imported or it has no effect at all.
# 'timeout' is the current ffmpeg option; 'stimeout' was renamed years ago
# but older builds still expect it, so both are supplied.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;8000000|stimeout;8000000",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

JOIN_BURST_S = 2.5      # discard this much PTS before judging cadence
GAP_MS = 500            # a PTS jump beyond this is a coverage gap
BACKOFF_START_S = 2.0
BACKOFF_CAP_S = 30.0


@dataclass
class Frame:
    """A frame plus the only timestamp we trust."""
    image: np.ndarray
    pts_ms: float                  # stream presentation time
    index: int
    camera_id: str
    arrived_monotonic: float       # diagnostics ONLY -- never for motion


@dataclass
class StreamStats:
    frames: int = 0
    read_failures: int = 0
    reconnects: int = 0
    gap_count: int = 0
    gap_max_ms: float = 0.0
    pts_backwards: int = 0         # PTS that went down: reset or reorder
    first_frame_s: Optional[float] = None
    declared_fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    _pts: list = field(default_factory=list)

    def measured_fps(self) -> Optional[float]:
        """
        Real delivered rate, from PTS, with the join burst removed.

        Deliberately NOT computed from sorted PTS. Sorting would hide the
        reordering and resets we specifically want to know about; those are
        counted in pts_backwards instead.
        """
        if len(self._pts) < 12:
            return None
        origin = self._pts[0]
        usable = [p for p in self._pts if p - origin >= JOIN_BURST_S * 1000]
        if len(usable) < 12:
            return None
        span = usable[-1] - usable[0]
        return round((len(usable) - 1) / (span / 1000.0), 2) if span > 0 else None


class LiveStream:
    """
    A camera you can iterate. Reconnects on its own, and never lies to you
    about time.

        with LiveStream(url, "cam-01") as s:
            for frame in s.frames(max_seconds=20):
                ...
    """

    def __init__(self, url: str, camera_id: str = "", *,
                 open_timeout_s: float = 12.0,
                 stall_after_s: float = 8.0,
                 max_reconnects: int = 1_000_000,
                 log=None):
        self.url = url
        self.camera_id = camera_id or url.rsplit("/", 1)[-1]
        self.open_timeout_s = open_timeout_s
        self.stall_after_s = stall_after_s
        self.max_reconnects = max_reconnects
        self.stats = StreamStats()
        self.cap: Optional[cv2.VideoCapture] = None
        self._log = log or (lambda m: None)
        self._backoff = BACKOFF_START_S
        self._last_pts: Optional[float] = None

    # ---------------------------------------------------------------- open

    def _open(self) -> bool:
        self.close()
        t0 = time.monotonic()
        # Decoder complaints here are expected on a mid-stream join: the
        # first frames reference a keyframe we never received. They stop
        # once the first IDR arrives, so we log rather than abort.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            self._log(f"{self.camera_id}: open failed")
            return False
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Recorded for the registry's declared-vs-measured comparison.
        # Never used for timing: it is routinely wrong.
        self.stats.declared_fps = round(fps, 3) if fps and fps > 0 else None
        self._log(f"{self.camera_id}: opened in {time.monotonic()-t0:.2f}s")
        return True

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()      # each open capture costs the gateway
            except Exception:
                pass
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------- backoff

    def _sleep_backoff(self):
        wait = min(self._backoff, BACKOFF_CAP_S)
        self._log(f"{self.camera_id}: reconnecting in {wait:.1f}s")
        time.sleep(wait)
        self._backoff = min(self._backoff * 2, BACKOFF_CAP_S)

    def _reset_backoff(self):
        self._backoff = BACKOFF_START_S

    # -------------------------------------------------------------- frames

    def frames(self, max_seconds: Optional[float] = None,
               max_frames: Optional[int] = None) -> Iterator[Frame]:
        started = time.monotonic()
        idx = 0
        last_good = time.monotonic()

        while True:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                return
            if max_frames is not None and idx >= max_frames:
                return

            if self.cap is None:
                if self.stats.reconnects > self.max_reconnects:
                    return
                if not self._open():
                    self.stats.reconnects += 1
                    self._sleep_backoff()
                    continue
                last_good = time.monotonic()

            ok, img = self.cap.read()
            now = time.monotonic()

            if not ok or img is None:
                self.stats.read_failures += 1
                # A gap is not a disconnect. Only a sustained silence is.
                if now - last_good > self.stall_after_s:
                    self._log(f"{self.camera_id}: no frames for "
                              f"{self.stall_after_s}s, treating as dropped")
                    self.close()
                    self.stats.reconnects += 1
                    self._sleep_backoff()
                else:
                    time.sleep(0.02)
                continue

            last_good = now
            self._reset_backoff()      # a real frame means we are healthy

            if self.stats.first_frame_s is None:
                self.stats.first_frame_s = round(now - started, 2)
                self.stats.height, self.stats.width = img.shape[0], img.shape[1]

            pts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pts > 0:
                if self._last_pts is not None:
                    d = pts - self._last_pts
                    if d < 0:
                        # Reordering or a stream restart. Counted, not hidden.
                        self.stats.pts_backwards += 1
                    elif d > GAP_MS:
                        self.stats.gap_count += 1
                        self.stats.gap_max_ms = max(self.stats.gap_max_ms, d)
                self._last_pts = pts
                self.stats._pts.append(pts)

            self.stats.frames += 1
            idx += 1
            yield Frame(image=img, pts_ms=pts, index=idx,
                        camera_id=self.camera_id, arrived_monotonic=now)


def force_tcp_env() -> str:
    """The capture options actually in effect. Printed by the tests."""
    return os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
