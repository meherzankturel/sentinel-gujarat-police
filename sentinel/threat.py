#!/usr/bin/env python3
"""
sentinel.threat -- detecting an attack you cannot see.

A knife blade is about 20 cm. On this grid a number plate -- 50 cm of
retro-reflective panel, the easiest object in the scene to find -- was
measured at 12 to 27 px on the government cameras. Scale that down: the
hand holding a knife is roughly 5 px across, and the knife itself is a
smear a couple of pixels wide. It is not in the image. A weapon detector
pointed at that feed is not detecting weapons, it is hallucinating them,
and every alert it raises costs a control room a dispatch.

But you do not need to see the weapon to detect the attack.

When a knife comes out in a public place, twenty people run away from one
point at the same instant. That signal is enormous: it covers a third of
the frame, it lasts several seconds, and it survives compression, glare,
night and a 640x480 sensor. It needs no trained model at all -- only the
observation that the crowd's velocity field suddenly grew AND started
pointing away from a common origin.

So this module is three layers, each gated on what the camera can
physically support, and the registry decides which ones are allowed to
run:

    1  PANIC DISPERSAL   every camera        dense optical flow, no model
    2  THREAT POSTURE    body-sized pixels   COCO keypoints, geometry only
    3  WEAPON IN HAND    the sharpest only   COCO detector near a wrist

The gate is the point. Layer 3 running everywhere is what produces the
alert flood that kills these deployments; layer 3 running on the four
cameras that can actually resolve a hand is a usable analytic.

Two deliberate positions, both of which will be defended in the writeup:

  * Layer 2 reads body geometry -- limb angles and the height of a wrist
    relative to the shoulders. It never identifies a face and never infers
    emotion or intent from an expression. Affect recognition is contested
    science and, applied by a police system to the public, is a liability
    rather than a feature. The face keypoints the model returns are used
    only as a "top of the head" height reference and are never stored,
    compared or exported.
  * A dispersal alert is a reason to look at a camera, not a conclusion
    about a crime. It is raised as an operator prompt with the epicentre
    marked, so the seven seconds an officer would spend finding the
    disturbance in the frame are spent deciding about it instead.

Licensing: OpenCV (Apache 2.0) and torchvision (BSD-3-Clause). Nothing
here is AGPL, which in a state procurement is a blocker rather than a
footnote.

What has actually been measured, so that nothing here is read as more
than it is:

  * Layer 1's false positive rate is measured on real capture off the
    government grid -- 366 s across three cameras, day and night, with
    traffic and pedestrians and no dispersal anywhere in it. Zero alerts.
    Of 6995 scored frames, 33% cleared the surge gate on their own and
    48% cleared the crowd gate, but only 2.0% cleared the radial gate and
    only 9 frames cleared all three at once, never twice in a row. The
    conjunction is doing the work, not a high threshold.
  * Layer 1's true positive rate and latency are measured only against
    rendered crowds (grid/plaza.py). That is simulation, and the figure
    should be read as "the detector responds to the thing it is aimed at",
    not as a recall estimate for real crowds.
  * Layers 2 and 3 have no real footage to validate against: there is no
    government-grid capture of an attack, and there will not be one. What
    IS measured on real frames is the only thing that can be -- how many
    pixels these cameras put on a body and on a hand -- and that is what
    layer_plan() gates on.

The measured limit of layer 1: it needs the crowd to surround the point
they run from. Fed a whole frame moving outward from an off-centre origin
-- a zoom, a dolly, a camera being carried forward -- net drift reads 0.73
and it refuses, correctly. Fed a real dispersal, which is a local disc of
outward motion in a frame that is otherwise still, it fires at every one
of fifteen epicentre positions tested including hard against the frame
edge. Three attempts to widen that gate -- sector-balanced drift,
epicentre-local statistics, and an opposed-motion floor -- were each
measured against real footage and rendered crowds, and each separated the
two cases worse than the gate already in place. They are not in the code.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .capability import CameraMotion
from .detect import Detector, WEAPON_CLASSES

# ------------------------------------------------------------------ tuning
#
# Every threshold below is stated as a number a reviewer can argue with,
# and the ones that matter were set from measured distributions on the
# plaza scene rather than picked to make a demo pass.

FLOW_WIDTH = 320          # flow is computed on a 320px-wide copy: Farneback
                          # at 1080p costs ~40x more for no extra signal,
                          # because a running crowd is a low-frequency event
NOISE_FLOOR_PX = 0.35     # per-frame displacement below this is sensor grain
BASELINE_S = 8.0          # rolling window the surge is judged against
WARMUP_S = 4.0            # never fire before a baseline exists
SURGE_RATIO = 1.7         # crowd speed vs its own baseline: walk -> run
MIN_SPEED_FRAC = 0.02     # of flow width per second; stops a near-static
                          # scene turning grain into an infinite ratio
COHERENCE_FIRE = 0.35     # how radial the field must be (see below)
NET_DRIFT_MAX = 0.6       # above this the motion energy is going one way, and
                          # one-way is a vehicle or a procession, not panic
MIN_MOVERS = 3            # a dispersal is a crowd: several separate things
MIN_MOVING_FRAC = 0.12    # ... or one large moving mass, for a dense crowd
                          # where the individuals cannot be separated
CONFIRM_FRAMES = 3        # consecutive frames before an alert is raised
COOLDOWN_S = 12.0         # one event is one alert, not two hundred

# Layer gates. A person 80px tall gives each limb about 8px of leverage,
# which is the floor at which keypoint regression stops being noise.
MIN_BODY_PX = 80
MIN_HAND_PX = 8.0

# Physical scale bridges, so the registry's existing measured column can
# gate layers it was never measured for. A number plate is 0.5 m, an adult
# is ~1.7 m, a hand is ~0.10 m across.
BODY_PER_PLATE = 3.4
HAND_PER_PLATE = 0.2


# ===================================================================
# Layer 1 -- panic dispersal
# ===================================================================


@dataclass
class FlowStats:
    """One frame-pair of crowd motion, reduced to five numbers."""
    speed_px_frame: float      # mean displacement of the moving pixels
    speed_px_s: float          # the same, per second of PTS
    moving_frac: float
    divergence: float          # peak of the smoothed divergence field
    coherence: float           # -1..1, how radial the field is (see below)
    net_drift: float           # 0..1, how much of the motion energy is one-way
    movers: int                # separate moving regions: a crowd, or one bus
    epicentre: Optional[Tuple[float, float]]   # in flow-image coordinates


def analyse_flow(flow: np.ndarray, dt: float = 1.0,
                 noise_floor: float = NOISE_FLOOR_PX) -> FlowStats:
    """
    Reduce a dense flow field to the two things that separate panic from
    ordinary movement, plus where it came from.

    Surge alone is not panic: a bus pulling out of frame is a large, fast,
    entirely ordinary motion event. Divergence alone is not panic either:
    a crowd leaving a platform in both directions does it every four
    minutes, slowly, all day.

    Divergence is measured as div(v) = du/dx + dv/dy. For a crowd running
    outward from a point at roughly constant speed the field is v = k*r_hat
    and div = k/r, which has a sharp maximum exactly at the origin -- so
    the peak of the divergence field is the epicentre, for free.

    Divergence alone is a poor firing test, because its magnitude scales
    with how fast the crowd is moving and with the resolution it was
    measured at. So the decision is taken on 'coherence' instead: the
    magnitude-weighted mean cosine between each moving pixel's flow and
    the direction pointing away from the epicentre. That is scale-free.

    Coherence alone is not enough either, and the counter-example is the
    bus. A solid object translating across the frame produces a divergence
    ridge along its trailing edge, and every pixel of the bus is then on
    one side of that ridge moving away from it -- which scores as coherent
    outward flow. So 'net_drift' is measured alongside: the vector sum of
    the flow divided by the sum of its magnitudes, which is 1 when
    everything moves the same way and near 0 when motion is balanced
    across directions. A crowd scattering from a point has people going
    both ways at once and cannot have high net drift; a bus, a train or a
    procession has almost nothing else.

    Net drift is weighted by the square of the speed, not by the speed.
    That is not cosmetic: on a grainy camera the sensor noise clears the
    motion floor across the whole frame, and thousands of slow random
    vectors drag a linear average back towards "balanced" whatever the
    vehicle in the middle is doing. Measured on the grid's own road
    cameras, linear weighting put a vehicle pass at 0.62 while energy
    weighting put it at 0.83, against 0.36 for a real dispersal.
    """
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    # A panning camera adds a constant vector to every pixel. Divergence
    # is already blind to that; the radial test is not, so the global
    # median -- the background's own motion -- is removed first. This is
    # the cheap complement to the phase-correlation pan detector, which
    # rejects the frame entirely once the pan is large.
    u -= float(np.median(u))
    v -= float(np.median(v))

    mag = cv2.magnitude(u, v)
    moving = mag > noise_floor
    n_moving = int(np.count_nonzero(moving))
    frac = n_moving / float(mag.size)
    dt = max(dt, 1e-3)
    if n_moving < 12:
        return FlowStats(0.0, 0.0, frac, 0.0, 0.0, 1.0, 0, None)

    m = mag[moving]
    speed = float(m.mean())
    net = math.hypot(float((u[moving] * m).sum()), float((v[moving] * m).sum())) \
        / max(float((m * m).sum()), 1e-6)

    # How many separate things are moving. One vehicle is one blob; a
    # crowd scattering is a dozen. Counted on the fastest motion only, so
    # that grain -- which is everywhere -- is not counted as a mover.
    strong = (mag > max(noise_floor, 0.6 * float(np.percentile(mag, 99.0)))) \
        .astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    area_min = max(4, int(0.0004 * mag.size))
    movers = int(sum(1 for i in range(1, n_cc)
                     if cc_stats[i, cv2.CC_STAT_AREA] >= area_min))

    # scale=1/8 undoes the Sobel kernel's gain, so this is a real spatial
    # derivative rather than a number eight times too large.
    du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    h, w = u.shape
    # Smoothing at ~2% of frame width: an epicentre is a region the size of
    # the space around one person, not a pixel.
    div = cv2.GaussianBlur(du_dx + dv_dy, (0, 0), sigmaX=max(2.0, w * 0.02))

    peak = float(div.max())
    iy, ix = np.unravel_index(int(np.argmax(div)), div.shape)
    ex, ey = float(ix), float(iy)
    if peak > 0:
        # The argmax of a smoothed field jitters between neighbouring
        # pixels frame to frame; the centroid of the ridge does not, and a
        # stable epicentre is what an operator is being pointed at.
        strong = div > peak * 0.6
        wsum = float(div[strong].sum())
        if wsum > 0:
            sy, sx = np.nonzero(strong)
            ex = float((sx * div[strong]).sum() / wsum)
            ey = float((sy * div[strong]).sum() / wsum)

    ys, xs = np.nonzero(moving)
    rx = xs - ex
    ry = ys - ey
    r = np.sqrt(rx * rx + ry * ry)
    ok = r > 1e-3
    coherence = 0.0
    if np.count_nonzero(ok) >= 12:
        uu = u[moving][ok]
        vv = v[moving][ok]
        mm = m[ok]
        cos = (uu * rx[ok] + vv * ry[ok]) / (r[ok] * np.maximum(mm, 1e-6))
        # Weighted by speed: the people actually running decide this, not
        # the hundreds of near-still background pixels that scraped past
        # the noise floor.
        coherence = float((cos * mm).sum() / mm.sum())

    return FlowStats(speed, speed / dt, frac, peak, coherence,
                     min(1.0, net), movers, (ex, ey))


@dataclass
class DispersalEvent:
    camera_id: str
    t: float                          # PTS seconds, never arrival time
    epicentre: Tuple[int, int]        # full-resolution image coordinates
    speed_px_s: float
    baseline_px_s: float
    surge_ratio: float
    coherence: float
    net_drift: float
    movers: int
    divergence: float
    moving_frac: float
    confidence: float

    def explain(self) -> str:
        return (f"crowd speed {self.speed_px_s:.0f} px/s vs baseline "
                f"{self.baseline_px_s:.0f} px/s (x{self.surge_ratio:.1f}), "
                f"flow {self.coherence:.2f} radial from "
                f"({self.epicentre[0]}, {self.epicentre[1]})")


class DispersalDetector:
    """
    Watches one camera for a crowd suddenly running away from one point.

    No model, no weights, no training data, and therefore nothing to
    licence and nothing to explain to a court about how it was trained.
    It works on the worst camera on the grid, which is the entire reason
    it is layer 1.
    """

    def __init__(self, camera_id: str = "", *,
                 flow_width: int = FLOW_WIDTH,
                 surge_ratio: float = SURGE_RATIO,
                 coherence_fire: float = COHERENCE_FIRE,
                 net_drift_max: float = NET_DRIFT_MAX,
                 confirm_frames: int = CONFIRM_FRAMES,
                 cooldown_s: float = COOLDOWN_S,
                 noise_floor: float = NOISE_FLOOR_PX):
        self.camera_id = camera_id
        self.flow_width = flow_width
        self.surge_ratio = surge_ratio
        self.coherence_fire = coherence_fire
        self.net_drift_max = net_drift_max
        self.confirm_frames = confirm_frames
        self.cooldown_s = cooldown_s
        self.noise_floor = noise_floor

        self.cammot = CameraMotion()
        self.hist: Deque[Tuple[float, float]] = deque()
        self.prev_small: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None
        self.first_t: Optional[float] = None
        self.last_fire: float = -1e9
        self.pending = 0
        self.scale = 1.0
        self.panning = False
        self.frames_skipped_panning = 0
        self.last_stats: Optional[FlowStats] = None
        self.last_ratio: float = 0.0

    # ---------------------------------------------------------- pipeline

    def update(self, frame: np.ndarray, t_s: float) -> Optional[DispersalEvent]:
        """
        Feed one frame. t_s must be PTS seconds: a crowd's acceleration is
        the measurement, and arrival-time deltas after a reconnect would
        manufacture one out of the gateway's GOP replay.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        self.scale = self.flow_width / float(w)
        small = cv2.resize(gray, (self.flow_width,
                                  max(2, int(round(h * self.scale)))))

        self.panning = self.cammot.update(gray)
        if self.panning:
            # Half of these are PTZ units. While one pans, every building
            # in frame moves outward from the direction of travel, which
            # is a textbook diverging field. Nothing measured during a pan
            # describes the crowd, so the pair is dropped and flow starts
            # again from the new viewpoint.
            self.frames_skipped_panning += 1
            self.prev_small = small
            self.prev_t = t_s
            return None

        prev, prev_t = self.prev_small, self.prev_t
        self.prev_small = small
        self.prev_t = t_s
        if prev is None or prev_t is None or t_s <= prev_t:
            return None

        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return self.ingest_flow(flow, t_s, t_s - prev_t)

    # ------------------------------------------------------------- core

    def ingest_flow(self, flow: np.ndarray, t_s: float,
                    dt: float) -> Optional[DispersalEvent]:
        """
        The decision, separated from where the pixels came from so it can
        be tested against flow fields with known shape rather than against
        footage we rendered ourselves and then graded ourselves on.
        """
        st = analyse_flow(flow, dt, self.noise_floor)
        self.last_stats = st
        if self.first_t is None:
            self.first_t = t_s

        min_speed = MIN_SPEED_FRAC * self.flow_width
        base = self._baseline()
        self.hist.append((t_s, st.speed_px_s))
        while self.hist and t_s - self.hist[0][0] > BASELINE_S:
            self.hist.popleft()

        warm = (t_s - self.first_t) >= WARMUP_S and len(self.hist) >= 8
        if base is None or not warm:
            self.last_ratio = 0.0
            return None

        ratio = st.speed_px_s / max(base, min_speed * 0.5)
        self.last_ratio = ratio
        surge = ratio >= self.surge_ratio and st.speed_px_s >= min_speed
        radial = (st.coherence >= self.coherence_fire
                  and st.divergence > 0
                  and st.net_drift <= self.net_drift_max)
        # Crowd scale, stated as an either/or on purpose. Several separate
        # movers is the sparse case; one large moving mass is the dense
        # one, where a packed crowd cannot be segmented into people at all
        # and requiring separate blobs would make the detector fail
        # precisely where the crowd is biggest.
        crowd = (st.movers >= MIN_MOVERS or st.moving_frac >= MIN_MOVING_FRAC)

        if not (surge and radial and crowd):
            # One frame of disagreement clears the run. A genuine dispersal
            # holds both conditions for seconds; grain does not hold either
            # for three consecutive frames.
            self.pending = 0
            return None

        self.pending += 1
        if self.pending < self.confirm_frames:
            return None
        if t_s - self.last_fire < self.cooldown_s:
            return None

        self.last_fire = t_s
        self.pending = 0
        ex, ey = st.epicentre or (0.0, 0.0)
        conf = 0.5 * _clip01((ratio - 1.0) / 1.5) + \
               0.5 * _clip01((st.coherence - 0.2) / 0.5)
        return DispersalEvent(
            camera_id=self.camera_id,
            t=t_s,
            epicentre=(int(round(ex / self.scale)), int(round(ey / self.scale))),
            speed_px_s=round(st.speed_px_s / self.scale, 1),
            baseline_px_s=round(base / self.scale, 1),
            surge_ratio=round(ratio, 2),
            coherence=round(st.coherence, 3),
            net_drift=round(st.net_drift, 3),
            movers=st.movers,
            divergence=round(st.divergence, 4),
            moving_frac=round(st.moving_frac, 4),
            confidence=round(conf, 3),
        )

    def _baseline(self) -> Optional[float]:
        """
        Median, not mean. A crowd's normal speed is not normally
        distributed -- one bus crossing the frame drags a mean up far
        enough to mask the surge that follows it.
        """
        if len(self.hist) < 8:
            return None
        return float(np.median([s for _, s in self.hist]))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def annotate(frame: np.ndarray, ev: DispersalEvent) -> np.ndarray:
    """Mark the epicentre for an operator (and for our own eyeballs)."""
    out = frame.copy()
    h, w = out.shape[:2]
    x, y = ev.epicentre
    r = max(14, int(min(h, w) * 0.05))
    cv2.circle(out, (x, y), r, (40, 40, 235), 2)
    cv2.circle(out, (x, y), r * 2, (40, 40, 235), 1)
    cv2.drawMarker(out, (x, y), (40, 40, 235), cv2.MARKER_CROSS, r, 2)
    txt = f"DISPERSAL  x{ev.surge_ratio:.1f} speed  {ev.coherence:.2f} radial"
    cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (60, 220, 255), 2, cv2.LINE_AA)
    return out


# ===================================================================
# Layer 2 -- threat posture
# ===================================================================

# COCO keypoint order, as torchvision returns it.
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
# Used ONLY as a height reference for "above the head". Never encoded,
# never compared between people, never exported.
HEAD_REF = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


@dataclass
class Posture:
    box: Tuple[float, float, float, float]
    score: float
    body_px: int                      # box height: the layer's own gate
    flags: List[str] = field(default_factory=list)
    wrists: List[Tuple[float, float]] = field(default_factory=list)
    valid: bool = True                # false when the person is too small
    note: str = ""

    def explain(self) -> str:
        if not self.valid:
            return f"person {self.body_px}px tall -- below posture gate"
        return ", ".join(self.flags) if self.flags else "neutral"


class PostureAnalyser:
    """
    Body geometry from COCO keypoints. Two postures are reported:

      arm-raised-overhead -- a wrist above the top of the head with the
        elbow above the shoulder. That is an overhead strike or a thrown
        object; it is also, honestly, a person hailing a rickshaw, which
        is why it is corroboration for layer 1 and not an alert of its own.

      on-ground -- the torso axis lying within 35 degrees of horizontal,
        or a bounding box wider than it is tall. Someone has gone down.

    Nothing here touches identity or expression. The face keypoints are
    used as a height reference and discarded.
    """

    _model = None
    _device = None

    @classmethod
    def _load(cls):
        if cls._model is not None:
            return
        import torch
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn,
            KeypointRCNN_ResNet50_FPN_Weights as W)

        cls._model = keypointrcnn_resnet50_fpn(
            weights=W.COCO_V1, box_score_thresh=0.6)
        cls._model.eval()
        cls._device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        cls._model.to(cls._device)

    @classmethod
    def device(cls) -> str:
        cls._load()
        return cls._device

    @classmethod
    def analyse(cls, frame: np.ndarray, min_score: float = 0.75,
                min_body_px: int = MIN_BODY_PX) -> List[Posture]:
        import torch
        cls._load()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            out = cls._model([t.to(cls._device)])[0]

        res: List[Posture] = []
        for box, score, kps, kscores in zip(out["boxes"], out["scores"],
                                            out["keypoints"],
                                            out["keypoints_scores"]):
            s = float(score)
            if s < min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            body_px = int(round(y2 - y1))
            k = kps.detach().cpu().numpy()[:, :2]
            ks = kscores.detach().cpu().numpy()
            p = Posture(box=(x1, y1, x2, y2), score=round(s, 3),
                        body_px=body_px)
            if body_px < min_body_px:
                # Reported, not silently dropped: "this camera saw people
                # but cannot judge posture on them" is a registry fact.
                p.valid = False
                p.note = f"body {body_px}px < {min_body_px}px gate"
                res.append(p)
                continue
            p.flags = cls._flags(k, ks, (x1, y1, x2, y2))
            for name in ("left_wrist", "right_wrist"):
                if ks[KP[name]] > 2.0:
                    p.wrists.append((float(k[KP[name]][0]),
                                     float(k[KP[name]][1])))
            res.append(p)
        return res

    @staticmethod
    def _flags(k: np.ndarray, ks: np.ndarray, box) -> List[str]:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        flags: List[str] = []

        def vis(name, thr=2.0):
            return ks[KP[name]] > thr

        def pt(name):
            return k[KP[name]]

        shoulders = [pt(n) for n in ("left_shoulder", "right_shoulder")
                     if vis(n)]
        hips = [pt(n) for n in ("left_hip", "right_hip") if vis(n)]

        # --- overhead strike -------------------------------------------
        head_y = min([pt(n)[1] for n in HEAD_REF if vis(n)], default=None)
        if head_y is None and shoulders:
            # Head not visible: estimate its top from the shoulder line,
            # because a wrist above the shoulders by a third of the body
            # is overhead whether or not the face was resolved.
            head_y = float(np.mean([p[1] for p in shoulders])) - 0.12 * bh
        if head_y is not None and shoulders:
            sh_y = float(np.mean([p[1] for p in shoulders]))
            for w_name, e_name in (("left_wrist", "left_elbow"),
                                   ("right_wrist", "right_elbow")):
                if not vis(w_name):
                    continue
                wy = pt(w_name)[1]
                elbow_above = (not vis(e_name)) or pt(e_name)[1] < sh_y
                if wy < head_y - 0.05 * bh and elbow_above:
                    flags.append("arm-raised-overhead")
                    break

        # --- gone to ground --------------------------------------------
        horizontal = bw > bh * 1.15
        if shoulders and hips:
            sx = float(np.mean([p[0] for p in shoulders]))
            sy = float(np.mean([p[1] for p in shoulders]))
            hx = float(np.mean([p[0] for p in hips]))
            hy = float(np.mean([p[1] for p in hips]))
            axis = math.degrees(math.atan2(abs(hx - sx), abs(hy - sy) + 1e-6))
            if axis > 55.0:
                horizontal = True
        if horizontal:
            flags.append("on-ground")
        return flags


# ===================================================================
# Layer 3 -- weapon in hand
# ===================================================================


@dataclass
class WeaponSighting:
    label: str
    score: float
    box: Tuple[float, float, float, float]
    weapon_px: int                   # longest side of the detection
    near_hand: bool
    hand_px: float                   # pixels across the nearest hand
    valid: bool                      # is this camera entitled to an opinion
    note: str = ""

    def explain(self) -> str:
        if not self.valid:
            return (f"{self.label} @{self.score:.2f}, {self.weapon_px}px, "
                    f"hand ~{self.hand_px:.1f}px -- BELOW EVIDENCE FLOOR, "
                    f"reported as unreliable")
        return (f"{self.label} @{self.score:.2f}, {self.weapon_px}px, "
                f"{'in hand region' if self.near_hand else 'not near a hand'}")


def hand_px_from_body_px(body_px: float) -> float:
    """A hand is about 10 cm across on a 170 cm body."""
    return float(body_px) * 0.06


def hand_px_from_plate_px(plate_px: float) -> float:
    """
    Bridge to the column the registry already measures.

    We measure plate width per camera because a plate is the easiest
    object in a street scene to localise. A plate is 0.5 m and a hand is
    0.1 m, so a camera measured at 27px of plate has a 5px hand -- which
    is the whole argument for gating layer 3, expressed in a number the
    registry already holds for every camera on the grid.
    """
    return float(plate_px) * HAND_PER_PLATE


class WeaponInHand:
    """
    COCO 'knife', 'baseball bat' and 'scissors' near a detected hand.

    Included with its own honesty gate rather than left out: the useful
    output on most cameras is not a detection, it is the statement that
    this camera has five pixels on a hand and therefore cannot support
    the analytic at all. A sighting below the floor is still returned,
    marked invalid, so the system can show an operator why it refused.
    """

    @classmethod
    def scan(cls, frame: np.ndarray, postures: Optional[Sequence[Posture]] = None,
             min_score: float = 0.5) -> List[WeaponSighting]:
        people = Detector.people(frame, min_score=0.45)
        weapons = Detector.detect(frame, WEAPON_CLASSES, min_score=min_score)

        hands: List[Tuple[float, float, float]] = []   # x, y, hand_px
        if postures:
            for p in postures:
                for (wx, wy) in p.wrists:
                    hands.append((wx, wy, hand_px_from_body_px(p.body_px)))
        if not hands:
            # No keypoints available on this camera: fall back to the
            # person box, treating the whole upper body as the hand region.
            # Deliberately looser, and the looseness is reported.
            for d in people:
                x1, y1, x2, y2 = d.box
                hands.append(((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.55,
                              hand_px_from_body_px(y2 - y1)))

        out: List[WeaponSighting] = []
        for d in weapons:
            x1, y1, x2, y2 = d.box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            wpx = int(round(max(x2 - x1, y2 - y1)))
            near, hpx, best = False, 0.0, 1e18
            for (hx, hy, hp) in hands:
                dist = math.hypot(cx - hx, cy - hy)
                if dist < best:
                    best, hpx = dist, hp
                # Within four hand-widths of a wrist is "in the hand" at
                # any camera scale, which keeps the test resolution-free.
                if dist <= max(24.0, hp * 4.0):
                    near = True
            valid = hpx >= MIN_HAND_PX
            out.append(WeaponSighting(
                label=d.label, score=d.score, box=d.box, weapon_px=wpx,
                near_hand=near, hand_px=round(hpx, 1), valid=valid,
                note="" if valid else
                     f"hand ~{hpx:.1f}px < {MIN_HAND_PX:.0f}px floor"))
        return out


# ===================================================================
# The gate itself
# ===================================================================


def layer_plan(plate_px: Optional[float] = None,
               body_px: Optional[float] = None,
               is_ptz: bool = False,
               night: bool = False) -> Dict[str, Dict[str, object]]:
    """
    Which threat layers this camera is allowed to run, and why.

    This is the registry acting as a control plane rather than a map. The
    inputs are columns already measured per camera: plate pixels, observed
    body height, whether the unit pans, whether it is usable at night.
    """
    if body_px is None and plate_px:
        body_px = plate_px * BODY_PER_PLATE
    hand_px = (hand_px_from_body_px(body_px) if body_px
               else (hand_px_from_plate_px(plate_px) if plate_px else None))

    plan: Dict[str, Dict[str, object]] = {}

    plan["dispersal"] = {
        "enabled": True,
        "why": ("crowd-scale motion survives any resolution on this grid"
                + ("; PTZ -- suppressed while panning" if is_ptz else "")),
    }

    if body_px is None:
        plan["posture"] = {"enabled": False,
                           "why": "no person observed; body pixels unmeasured"}
    elif body_px >= MIN_BODY_PX:
        plan["posture"] = {"enabled": True,
                           "why": f"body ~{body_px:.0f}px >= {MIN_BODY_PX}px"}
    else:
        plan["posture"] = {"enabled": False,
                           "why": f"body ~{body_px:.0f}px < {MIN_BODY_PX}px; "
                                  "limb geometry would be noise"}

    if hand_px is None:
        plan["weapon"] = {"enabled": False,
                          "why": "scale unmeasured on this camera"}
    elif hand_px >= MIN_HAND_PX and not night:
        plan["weapon"] = {"enabled": True,
                          "why": f"hand ~{hand_px:.0f}px >= {MIN_HAND_PX:.0f}px"}
    else:
        plan["weapon"] = {
            "enabled": False,
            "why": (f"hand ~{hand_px:.1f}px < {MIN_HAND_PX:.0f}px -- a 20cm "
                    "blade is under 2px here and is not in the image"
                    if hand_px < MIN_HAND_PX else
                    "low light: weapon-class detections unreliable"),
        }
    return plan


# ===================================================================
# Running it
# ===================================================================


def watch(source: str, camera_id: str = "", *, max_seconds: Optional[float] = None,
          save_dir: Optional[str] = None,
          on_event=None) -> List[DispersalEvent]:
    """
    Run layer 1 over one source and return the dispersals it found.

    `source` is a path or an RTSP URL. Timing comes from the decoder's PTS
    rather than the wall clock, because on a live stream the gateway
    replays its buffered GOP on connect and a detector timing by arrival
    would read that burst as the whole crowd accelerating at once.
    """
    if source.startswith("rtsp://"):
        # The organisers' rule, not a preference: UDP loses packets across
        # NAT and the half-decoded frames that result look exactly like a
        # model defect.
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source)
    det = DispersalDetector(camera_id or source.rsplit("/", 1)[-1])
    events: List[DispersalEvent] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if max_seconds is not None and t > max_seconds:
                break
            ev = det.update(frame, t)
            if ev is None:
                continue
            events.append(ev)
            if save_dir:
                cv2.imwrite(f"{save_dir}/dispersal_{det.camera_id}_"
                            f"{len(events)}.png", annotate(frame, ev))
            if on_event:
                on_event(ev, frame)
    finally:
        cap.release()
    return events


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse, json as _json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("source", help="video file or rtsp:// URL")
    ap.add_argument("--camera-id", default="")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--save-dir", default=None,
                    help="write an annotated frame per alert")
    ap.add_argument("--plan", action="store_true",
                    help="also report which layers this camera may run, "
                         "measured from the people in its first frames")
    a = ap.parse_args(argv)

    if a.plan:
        cap = cv2.VideoCapture(a.source)
        ok, frame = cap.read()
        cap.release()
        if ok:
            heights = sorted(d.height for d in Detector.people(frame, min_score=0.5))
            body = heights[len(heights) // 2] if heights else None
            print(_json.dumps({"people_seen": len(heights),
                               "body_px_p50": body,
                               "plan": layer_plan(body_px=body)}, indent=2))

    evs = watch(a.source, a.camera_id, max_seconds=a.seconds, save_dir=a.save_dir)
    for e in evs:
        print(_json.dumps({"camera": e.camera_id, "t_s": round(e.t, 2),
                           "epicentre": list(e.epicentre),
                           "confidence": e.confidence,
                           "why": e.explain()}))
    print(f"{len(evs)} dispersal alert(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
