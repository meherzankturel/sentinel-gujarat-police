#!/usr/bin/env python3
"""
plaza.py -- the crowd cameras of the local grid (cam-09, cam-10).

Why a second publisher
----------------------
publish.py renders a road: one vehicle, one number plate, one measurable
quantity. The threat layer needs the opposite scene -- a plaza full of
people with no vehicle in it -- and it needs something publish.py cannot
give: a written-down ground truth. Panic dispersal is only worth claiming
if we can say how often it fired when nothing happened and how many
seconds after the event it fired when something did.

So this renders ~25 to 35 people milling at walking pace, and every
CYCLE_S seconds they all run away from one point for FLIGHT_S seconds.
The moment each dispersal starts and the point they run from are computed
from the clock by dispersal_events(), so the detector can be scored
against a schedule it was never shown.

What this is and is not
-----------------------
This is a development target and a scoring harness. It is NOT evidence
that dispersal detection works on real crowds: figures rendered with
OpenCV primitives move more consistently than people do, and the camera
never wobbles. What it does prove is that the detector separates the
three cases that matter -- milling, a surge without divergence, and a
surge with divergence -- through a real H.264 encode, an RTSP hop, and a
decode, with the grain and blur of the camera it is standing in for.

The negative control is more useful than the positive one: cam-01 to
cam-08 publish vehicles crossing the frame, which is a large surge with
no divergence, and the detector must stay silent on all of them.

Usage:
    ./.venv/bin/python grid/plaza.py                 # cam-09 and cam-10
    ./.venv/bin/python grid/plaza.py --only cam-09
    ./.venv/bin/python grid/plaza.py --schedule      # print ground truth
"""

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFS = ROOT / "cameras.json"
sys.path.insert(0, str(ROOT))

# Reused rather than copied: the crowd cameras must carry the same burned-in
# clock and the same grain pipeline as the road cameras, or the threat layer
# would be tested against a cleaner picture than the rest of the grid.
from publish import NoisePool, draw_osd, ffmpeg_cmd, RTSP_HOST, RTSP_PORT  # noqa: E402

CYCLE_S = 60.0        # one dispersal per minute
EVENT_AT = 30.0       # ... at this offset into the cycle, so a detector
                      # joining at t=0 has 30s of ordinary crowd first
FLIGHT_S = 8.0        # how long people keep running
REACTION_MAX_S = 1.1  # not everyone turns at the same instant


# --------------------------------------------------------------- ground truth


def _rand_for(cam_id: str, cycle: int) -> np.random.RandomState:
    """
    Per-event randomness that both the renderer and the scorer can derive
    from the clock alone. Seeding from a hash rather than a counter means
    a scorer that joins the stream late still knows where the last event
    was supposed to be.
    """
    h = hashlib.sha256(f"{cam_id}:{cycle}".encode()).digest()
    return np.random.RandomState(int.from_bytes(h[:4], "big"))


def epicentre_for(cam_id: str, cycle: int):
    """Normalised plaza coordinates of the cycle'th dispersal."""
    r = _rand_for(cam_id, cycle)
    return float(r.uniform(0.22, 0.78)), float(r.uniform(0.30, 0.85))


def dispersal_events(t0: float, t1: float, cam_id: str):
    """Every event that starts in [t0, t1). The ground truth, in seconds."""
    out = []
    c = int(math.floor((t0 - EVENT_AT) / CYCLE_S))
    while True:
        start = c * CYCLE_S + EVENT_AT
        if start >= t1:
            break
        if start >= t0:
            out.append({"cycle": c, "start": start,
                        "end": start + FLIGHT_S,
                        "epicentre_norm": epicentre_for(cam_id, c)})
        c += 1
    return out


def event_state(t: float, cam_id: str):
    """(active, seconds_since_start, epicentre_norm) at time t."""
    c = int(math.floor((t - EVENT_AT) / CYCLE_S))
    start = c * CYCLE_S + EVENT_AT
    el = t - start
    if 0.0 <= el < FLIGHT_S:
        return True, el, epicentre_for(cam_id, c)
    return False, el, None


# ------------------------------------------------------------------ scene


class Plaza:
    """
    A crowd, in normalised plaza coordinates: x across, y into the frame
    with 0 far and 1 near. Speeds are in plaza units per second, so a
    person near the camera covers more pixels than one at the back at the
    same walking pace -- which is exactly the perspective effect that
    makes a naive pixel-velocity threshold useless and a relative-to-its-
    own-baseline one necessary.
    """

    WALK = 0.055
    RUN = 0.235

    def __init__(self, cam, seed=0):
        self.cam = cam
        self.w = cam["width"]
        self.h = cam["height"]
        self.cam_id = cam["id"]
        self.n = int(cam.get("crowd", 26))
        self.rng = np.random.RandomState(seed if seed is not None
                                         else abs(hash(self.cam_id)) % 2**31)
        self.bg = self._background()

        r = self.rng
        self.px = r.uniform(0.05, 0.95, self.n)
        self.py = r.uniform(0.06, 0.98, self.n)
        self.heading = r.uniform(0, 2 * math.pi, self.n)
        self.pace = r.uniform(0.55, 1.35, self.n)      # personal walk speed
        self.retarget = r.uniform(0, 2.5, self.n)      # time until next turn
        self.phase = r.uniform(0, 2 * math.pi, self.n)  # gait phase
        self.reaction = r.uniform(0.0, REACTION_MAX_S, self.n)
        self.colour = [tuple(int(v) for v in c) for c in
                       r.randint(45, 215, size=(self.n, 3))]
        self.speed = np.zeros(self.n)

    # ------------------------------------------------------------ geometry

    def to_px(self, x, y):
        """Plaza coordinates to image coordinates, with perspective."""
        y_far, y_near = self.h * 0.44, self.h * 0.98
        py = y_far + (y_near - y_far) * y
        k = 0.40 + 0.60 * y                     # the plaza narrows with depth
        px = self.w * 0.5 + (x - 0.5) * self.w * k
        return px, py

    def person_height(self, y):
        return self.h * (0.085 + 0.155 * y)

    # ---------------------------------------------------------- simulation

    def step(self, t, dt):
        """Advance the crowd. t is absolute seconds; the event is on the clock."""
        active, elapsed, epi = event_state(t, self.cam_id)
        r = self.rng

        if active and epi is not None:
            ex, ey = epi
            dx = self.px - ex
            dy = self.py - ey
            d = np.hypot(dx, dy)
            d[d < 1e-3] = 1e-3
            # Away from the epicentre, with a personal reaction delay and a
            # short acceleration. A crowd that all turned on the same frame
            # would be an easier signal than any real one.
            gone = np.clip((elapsed - self.reaction) / 0.45, 0.0, 1.0)
            target = np.arctan2(dy, dx) + r.normal(0, 0.16, self.n)
            turn = gone > 0
            self.heading = np.where(turn, target, self.heading)
            self.speed = self.WALK * self.pace + \
                gone * (self.RUN * self.pace - self.WALK * self.pace)
        else:
            # Milling: wander, with pauses. Nobody in a plaza walks in a
            # straight line for ten seconds.
            self.retarget -= dt
            due = self.retarget <= 0
            if due.any():
                k = int(due.sum())
                self.heading[due] += r.uniform(-2.2, 2.2, k)
                self.retarget[due] = r.uniform(1.2, 3.4, k)
                self.pace[due] = r.uniform(0.0, 1.35, k)   # 0 = standing still
            self.speed = self.WALK * self.pace

        vx = np.cos(self.heading) * self.speed
        vy = np.sin(self.heading) * self.speed * 0.55   # depth foreshortening
        self.px += vx * dt
        self.py += vy * dt

        # The plaza has walls. People stopping at the edge is also what
        # ends a real dispersal: the flow dies as the exits fill.
        for arr, lo, hi in ((self.px, 0.02, 0.98), (self.py, 0.03, 0.99)):
            np.clip(arr, lo, hi, out=arr)
        self.heading[(self.px <= 0.021) | (self.px >= 0.979)] += math.pi
        self.phase += dt * (5.0 + 22.0 * self.speed)

    # ----------------------------------------------------------- rendering

    def _background(self):
        w, h = self.w, self.h
        bg = np.zeros((h, w, 3), np.uint8)
        horizon = int(h * 0.40)
        for y in range(horizon):
            v = 96 + int(30 * (y / max(1, horizon)))
            bg[y, :] = (v + 14, v + 8, v - 4)
        bg[horizon:, :] = (104, 106, 110)

        # Paving, drawn in perspective. Texture matters: a featureless floor
        # gives optical flow nothing to be wrong about, which would flatter
        # the detector.
        for i in range(1, 15):
            f = i / 15.0
            y = int(horizon + (h - horizon) * (f ** 1.7))
            cv2.line(bg, (0, y), (w, y), (94, 96, 100), 1)
        for i in range(-8, 9):
            x0 = int(w * 0.5 + i * w * 0.03)
            x1 = int(w * 0.5 + i * w * 0.115)
            cv2.line(bg, (x0, horizon), (x1, h), (94, 96, 100), 1)

        # Shelter roof and pillars: static structure, so a panning test has
        # something to correlate against.
        cv2.rectangle(bg, (0, int(h * 0.30)), (w, horizon), (78, 80, 86), -1)
        for fx in (0.12, 0.38, 0.62, 0.88):
            x = int(w * fx)
            cv2.rectangle(bg, (x - int(w * 0.006), int(h * 0.30)),
                          (x + int(w * 0.006), int(h * 0.52)), (66, 68, 74), -1)
        cv2.rectangle(bg, (int(w * 0.03), int(h * 0.33)),
                      (int(w * 0.22), int(h * 0.39)), (58, 92, 128), -1)
        return bg

    def render(self):
        frame = self.bg.copy()
        order = np.argsort(self.py)          # far first, so near ones occlude
        for i in order:
            self._draw_person(frame, float(self.px[i]), float(self.py[i]),
                              float(self.phase[i]), float(self.speed[i]),
                              self.colour[i])
        return frame

    def _draw_person(self, frame, x, y, phase, speed, colour):
        cx, cy = self.to_px(x, y)
        s = self.person_height(y)
        if s < 8:
            return
        cx, cy = int(cx), int(cy)
        th = max(1, int(s * 0.055))
        stride = math.sin(phase) * (0.10 + 1.4 * speed) * s
        swing = -stride * 0.7

        cv2.ellipse(frame, (cx, cy), (int(s * 0.17), int(s * 0.05)), 0, 0, 360,
                    (86, 88, 92), -1)
        hip = int(cy - s * 0.46)
        sh = int(cy - s * 0.80)
        # legs
        cv2.line(frame, (cx, hip), (int(cx + stride), cy), (54, 54, 62), th + 1)
        cv2.line(frame, (cx, hip), (int(cx - stride), cy), (44, 44, 52), th + 1)
        # torso
        cv2.line(frame, (cx, hip), (cx, sh), colour, max(2, int(s * 0.17)))
        # arms
        cv2.line(frame, (cx, sh), (int(cx + swing), int(sh + s * 0.30)),
                 colour, th)
        cv2.line(frame, (cx, sh), (int(cx - swing), int(sh + s * 0.30)),
                 colour, th)
        # head
        cv2.circle(frame, (cx, int(sh - s * 0.10)), max(2, int(s * 0.095)),
                   (98, 116, 146), -1)


# ------------------------------------------------------------ frame source


def frames(cam, seconds=180.0, fps=None, seed=0, t0=0.0, degrade=True):
    """
    Deterministic offline frames, with the same degradation the published
    stream carries. Scoring runs against this: it is the only way to know
    the event start to the frame rather than to the second.
    """
    fps = fps or cam["fps"]
    plaza = Plaza(cam, seed=seed)
    noise = NoisePool(cam["height"], cam["width"], cam.get("grain", 0)) \
        if degrade else None
    blur = float(cam.get("blur", 0) or 0) if degrade else 0
    bright = int(cam.get("brightness", 0) or 0) if degrade else 0

    dt = 1.0 / fps
    n = int(seconds * fps)
    t = t0
    for _ in range(n):
        plaza.step(t, dt)
        frame = plaza.render()
        if blur > 0:
            k = int(blur * 4) * 2 + 1
            frame = cv2.GaussianBlur(frame, (k, k), blur)
        if bright:
            frame = np.clip(frame.astype(np.int16) + bright, 0, 255).astype(np.uint8)
        if noise is not None:
            frame = noise.apply(frame)
        yield t, frame
        t += dt


# ---------------------------------------------------------------- publish


def run_camera(cam, stop):
    w, h, fps = cam["width"], cam["height"], cam["fps"]
    plaza = Plaza(cam, seed=abs(hash(cam["id"])) % 2**31)
    noise = NoisePool(h, w, cam.get("grain", 0))
    blur = float(cam.get("blur", 0) or 0)

    cmd, url = ffmpeg_cmd(cam)
    print(f"  {cam['id']:<8} {w}x{h} @{fps} {cam['codec']:<5} "
          f"crowd={plaza.n} -> {url}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    period = 1.0 / fps
    n = 0
    t0 = time.monotonic()
    try:
        while not stop["v"]:
            # Wall clock, not frame count: the scorer derives the event
            # schedule from the same clock, so it can join the stream at
            # any moment and still know what should be happening.
            now = time.time()
            plaza.step(now, period)
            frame = plaza.render()
            if blur > 0:
                k = int(blur * 4) * 2 + 1
                frame = cv2.GaussianBlur(frame, (k, k), blur)
            frame = noise.apply(frame)
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
            elif slack < -2.0:
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
    ap.add_argument("--schedule", action="store_true",
                    help="print the next few dispersal events and exit")
    args = ap.parse_args()

    defs = [c for c in json.loads(DEFS.read_text())["cameras"]
            if c.get("scene") == "plaza"]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        defs = [c for c in defs if c["id"] in want]
    if not defs:
        sys.exit("no plaza cameras selected")

    if args.schedule:
        now = time.time()
        for c in defs:
            print(c["id"])
            for e in dispersal_events(now, now + 240, c["id"]):
                print(f"  in {e['start'] - now:6.1f}s  epicentre "
                      f"{e['epicentre_norm'][0]:.2f},{e['epicentre_norm'][1]:.2f}")
        return

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))

    # One process per camera, for the reason publish.py documents: frame
    # generation is numpy work holding the GIL and threads starve the
    # high-resolution camera until its encoder under-feeds.
    if len(defs) == 1:
        run_camera(defs[0], stop)
        return

    kids = [subprocess.Popen([sys.executable, __file__, "--only", c["id"]])
            for c in defs]
    try:
        while not stop["v"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    for k in kids:
        k.terminate()


if __name__ == "__main__":
    main()
