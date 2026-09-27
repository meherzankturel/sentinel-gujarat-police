#!/usr/bin/env python3
"""
publish.py -- run the local synthetic camera grid.

Why this exists
---------------
The government gateway (cctv.corp8.cloud) is not always reachable, and even
when it is, we cannot restart one of their feeds. Their own pre-submission
checklist requires "reconnect with backoff implemented and tested by
restarting a feed" -- which is untestable against someone else's server.

So we run our own grid. It mirrors the documented Sentinel contract:

    rtsp://<host>:8554/stream/<id>
    http://<host>:8889/stream/<id>/whep
    http://<host>:8888/stream/<id>/index.m3u8

Each camera is rendered frame by frame in Python and piped to ffmpeg, which
encodes and pushes to MediaMTX. Rendering in Python rather than with ffmpeg
filters is deliberate: it lets us set the number-plate width in pixels
exactly, and plate pixels -- not resolution -- is what decides whether a
camera can actually read a plate. That distinction is the whole thesis.

Usage:
    python3 grid/publish.py                 # all cameras
    python3 grid/publish.py --only cam-01,cam-04
    python3 grid/publish.py --list
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFS = ROOT / "cameras.json"

RTSP_HOST = os.environ.get("GRID_HOST", "127.0.0.1")
RTSP_PORT = int(os.environ.get("GRID_RTSP_PORT", "8554"))

# One vehicle passes every CYCLE_S seconds, visible for PASS_S of them.
# Per-camera phase offsets make the same plate appear at staggered times,
# which is what turns eight independent cameras into a traversable route.
CYCLE_S = 40.0
PASS_S = 6.0

PHASE = {                 # seconds into the cycle at which the car appears
    "cam-01": 0.0,        # Infocity Circle      -- first sighting
    "cam-02": 7.0,        # Sector 21
    "cam-06": 14.0,       # SG Highway
    "cam-04": 21.0,       # Chiloda overpass
    "cam-08": 28.0,       # Civil Hospital       -- last sighting
    "cam-03": 3.0,        # different vehicle
    "cam-05": 11.0,       # different vehicle
    "cam-07": 19.0,       # different vehicle
}


# ----------------------------------------------------------------- rendering


def make_background(w, h):
    """Static road scene. Built once per camera, then copied per frame."""
    bg = np.zeros((h, w, 3), np.uint8)

    horizon = int(h * 0.34)
    # sky / far background, subtly graded
    for y in range(horizon):
        v = 92 + int(38 * (y / max(1, horizon)))
        bg[y, :] = (v + 12, v + 6, v - 6)
    # asphalt
    bg[horizon:, :] = (58, 58, 62)

    # kerb line
    cv2.line(bg, (0, horizon), (w, horizon), (74, 78, 82), max(1, h // 220))

    # lane markings: dashed centre line low in frame, where vehicles pass
    lane_y = int(h * 0.74)
    dash = max(12, w // 26)
    gap = dash
    x = 0
    while x < w:
        cv2.rectangle(bg, (x, lane_y), (x + dash, lane_y + max(2, h // 200)),
                      (168, 168, 172), -1)
        x += dash + gap

    # a couple of static roadside objects, so the scene is not featureless
    cv2.rectangle(bg, (int(w * 0.06), horizon - int(h * 0.16)),
                  (int(w * 0.09), horizon), (70, 72, 76), -1)
    cv2.rectangle(bg, (int(w * 0.88), horizon - int(h * 0.20)),
                  (int(w * 0.91), horizon), (70, 72, 76), -1)
    return bg


def fit_text_scale(text, target_w, font, thickness):
    """Find the cv2 font scale whose rendered width is closest to target_w."""
    lo, hi = 0.05, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        (tw, _), _ = cv2.getTextSize(text, font, mid, thickness)
        if tw > target_w:
            hi = mid
        else:
            lo = mid
    return lo


def draw_vehicle(frame, cx, cy, plate_px, plate_text):
    """
    Draw a vehicle whose number plate is exactly plate_px wide.
    Everything else is scaled off the plate, so plate_px is the single
    knob that decides legibility -- exactly as mounting distance is in
    the real world.
    """
    plate_w = int(plate_px)
    plate_h = max(4, int(round(plate_px * 0.24)))     # Indian plate ratio
    veh_w = int(plate_w * 4.3)
    veh_h = int(veh_w * 0.44)

    x0, y0 = int(cx - veh_w / 2), int(cy - veh_h / 2)
    x1, y1 = x0 + veh_w, y0 + veh_h

    # body
    cv2.rectangle(frame, (x0, y0 + int(veh_h * 0.28)), (x1, y1), (128, 96, 72), -1)
    # cabin
    cv2.rectangle(frame, (x0 + int(veh_w * 0.20), y0),
                  (x1 - int(veh_w * 0.18), y0 + int(veh_h * 0.34)),
                  (112, 84, 62), -1)
    # windscreen
    cv2.rectangle(frame, (x0 + int(veh_w * 0.24), y0 + int(veh_h * 0.05)),
                  (x1 - int(veh_w * 0.22), y0 + int(veh_h * 0.30)),
                  (52, 58, 64), -1)
    # wheels
    r = max(3, int(veh_h * 0.20))
    cv2.circle(frame, (x0 + int(veh_w * 0.24), y1), r, (28, 28, 30), -1)
    cv2.circle(frame, (x1 - int(veh_w * 0.24), y1), r, (28, 28, 30), -1)
    # headlights
    lw = max(2, int(veh_w * 0.07))
    cv2.rectangle(frame, (x1 - lw - 2, y0 + int(veh_h * 0.40)),
                  (x1 - 2, y0 + int(veh_h * 0.52)), (196, 214, 226), -1)

    # ---- number plate: the measured thing ----
    px = int(cx - plate_w / 2)
    py = int(y1 - plate_h - max(2, int(veh_h * 0.12)))
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (238, 240, 238), -1)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (40, 40, 40), 1)

    font = cv2.FONT_HERSHEY_DUPLEX
    thick = 1 if plate_h < 26 else 2
    scale = fit_text_scale(plate_text, plate_w * 0.88, font, thick)
    (tw, th), _ = cv2.getTextSize(plate_text, font, scale, thick)
    cv2.putText(frame, plate_text,
                (px + (plate_w - tw) // 2, py + (plate_h + th) // 2),
                font, scale, (18, 18, 18), thick, cv2.LINE_AA)
    return (px, py, plate_w, plate_h)


def draw_osd(frame, cam_id, clock_offset_s):
    """
    Burned-in clock, as almost every deployed CCTV camera has.
    clock_offset_s lets a camera's clock be deliberately wrong, which is
    what the registry's time-confidence field exists to detect.
    """
    h, w = frame.shape[:2]
    t = time.gmtime(time.time() + clock_offset_s)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", t)

    # Sized to be plainly legible, because that is what a burned-in
    # timestamp is for. Rendering it small enough that OCR guesses at the
    # digits would be testing our reader against a defect no real camera
    # has -- Tesseract was reading the hour "07" as "67" at the old size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, w / 1900.0 * 0.95)
    thick = 2 if w >= 900 else 1
    (tw, th), _ = cv2.getTextSize(stamp, font, scale, thick)
    pad = max(4, int(w * 0.006))

    cv2.rectangle(frame, (pad, pad), (pad * 2 + tw, pad * 2 + th + 6), (0, 0, 0), -1)
    cv2.putText(frame, stamp, (pad + pad // 2, pad + th + 3),
                font, scale, (235, 235, 235), thick, cv2.LINE_AA)

    (cw, ch), _ = cv2.getTextSize(cam_id, font, scale, thick)
    cv2.rectangle(frame, (w - cw - pad * 3, pad),
                  (w - pad, pad * 2 + ch + 6), (0, 0, 0), -1)
    cv2.putText(frame, cam_id, (w - cw - pad * 2, pad + ch + 3),
                font, scale, (235, 235, 235), thick, cv2.LINE_AA)


class NoisePool:
    """Pre-generated grain. Generating fresh noise per 1080p frame is slow."""

    def __init__(self, h, w, sigma, n=12):
        self.tiles = [
            (np.random.randn(h, w, 1) * sigma).astype(np.int16)
            for _ in range(n)
        ] if sigma > 0 else []
        self.i = 0

    def apply(self, frame):
        if not self.tiles:
            return frame
        t = self.tiles[self.i % len(self.tiles)]
        self.i += 1
        return np.clip(frame.astype(np.int16) + t, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ ffmpeg


def ffmpeg_cmd(cam):
    w, h, fps = cam["width"], cam["height"], cam["fps"]
    url = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/stream/{cam['id']}"
    gop = max(1, int(fps * 2))

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
    ]
    if cam["codec"] == "h265":
        cmd += ["-c:v", "libx265", "-preset", "ultrafast",
                "-x265-params", f"keyint={gop}:min-keyint={gop}:log-level=none",
                "-tag:v", "hvc1"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-g", str(gop), "-keyint_min", str(gop)]

    cmd += ["-pix_fmt", "yuv420p", "-r", str(fps),
            "-f", "rtsp", "-rtsp_transport", "tcp", url]
    return cmd, url


def run_camera(cam, stop):
    w, h, fps = cam["width"], cam["height"], cam["fps"]
    bg = make_background(w, h)
    noise = NoisePool(h, w, cam.get("grain", 0))
    blur = float(cam.get("blur", 0) or 0)
    bright = int(cam.get("brightness", 0) or 0)
    phase = PHASE.get(cam["id"], 0.0)
    lane_y = int(h * 0.66)

    cmd, url = ffmpeg_cmd(cam)
    print(f"  {cam['id']:<8} {w}x{h} @{fps} {cam['codec']:<5} "
          f"plate={cam['plate_px']}px -> {url}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    period = 1.0 / fps
    n = 0
    t0 = time.monotonic()
    try:
        while not stop["v"]:
            frame = bg.copy()

            # vehicle pass, phase-shifted per camera
            cyc = (time.time() + phase) % CYCLE_S
            if cyc < PASS_S:
                veh_w = int(cam["plate_px"] * 4.3)
                span = w + veh_w * 2
                cx = int(-veh_w + (cyc / PASS_S) * span)
                draw_vehicle(frame, cx, lane_y, cam["plate_px"], cam["plate"])

            if blur > 0:
                k = int(blur * 4) * 2 + 1
                frame = cv2.GaussianBlur(frame, (k, k), blur)
            if bright:
                frame = np.clip(frame.astype(np.int16) + bright, 0, 255).astype(np.uint8)
            frame = noise.apply(frame)

            # OSD drawn last so it stays crisp, as on a real encoder
            draw_osd(frame, cam["id"], cam.get("clock_offset_s", 0))

            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                break

            n += 1
            target = t0 + n * period
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -2.0:          # fell badly behind; resync
                t0 = time.monotonic()
                n = 0
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    defs = json.loads(DEFS.read_text())["cameras"]
    if args.list:
        for c in defs:
            print(f"{c['id']}  {c['department']:<10} {c['width']}x{c['height']} "
                  f"{c['fps']}fps {c['codec']}  plate={c['plate_px']}px  {c['expect']}")
        return
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        defs = [c for c in defs if c["id"] in want]
    if not defs:
        sys.exit("no cameras selected")

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))

    # One camera per call runs in this process. More than one is fanned out
    # to child processes rather than threads: frame generation is numpy work
    # holding the GIL, so threading starves the high-resolution cameras until
    # they under-feed their encoder and the stream visibly corrupts. A real
    # ingest node separates cameras across cores for the same reason.
    if len(defs) == 1:
        print(f"Publishing {defs[0]['id']} to "
              f"rtsp://{RTSP_HOST}:{RTSP_PORT}/stream/{defs[0]['id']}")
        run_camera(defs[0], stop)
        return

    print(f"Publishing {len(defs)} cameras to "
          f"rtsp://{RTSP_HOST}:{RTSP_PORT}/stream/<id>  (one process each)")
    kids = []
    for c in defs:
        kids.append(subprocess.Popen(
            [sys.executable, __file__, "--only", c["id"]]))
        time.sleep(0.35)

    try:
        while not stop["v"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop["v"] = True
    print("\nstopping...")
    for k in kids:
        k.terminate()
    for k in kids:
        try:
            k.wait(timeout=5)
        except Exception:
            k.kill()


if __name__ == "__main__":
    main()
