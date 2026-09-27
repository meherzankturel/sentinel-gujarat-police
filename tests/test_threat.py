"""
The dispersal detector, tested on flow fields with a known shape.

The point of testing at this level is that a flow field can be written
down. "Twenty people ran away from (140, 90)" is a formula; footage of it
is an approximation of the formula that we would also have rendered
ourselves, and grading our own rendering is how a measurement in this
project already went wrong once -- a plate detector validated only on
synthetic frames went on to measure a road sign as a number plate.

So the three cases that decide whether this analytic is worth anything
are asserted here as arithmetic:

    a surge with no divergence      -- a bus leaving  -- must NOT fire
    divergence with no surge        -- a normal crowd -- must NOT fire
    both at once                    -- panic          -- must fire

and the scoring against rendered plaza footage, with true/false positive
counts and detection latency, is a separate exercise run against
grid/plaza.py.
"""

import numpy as np
import pytest

from sentinel.threat import (
    DispersalDetector, KP, KEYPOINTS, MIN_HAND_PX, NET_DRIFT_MAX,
    PostureAnalyser, analyse_flow, hand_px_from_plate_px, layer_plan,
)

H, W = 180, 320
FPS = 15.0
DT = 1.0 / FPS


# ------------------------------------------------------------------ fields


def radial(cx, cy, speed, h=H, w=W, sign=1.0):
    """
    Every pixel in the frame moving directly away from (cx, cy) at a
    constant speed, to the edges, forever.

    Useful for checking the geometry -- coherence, and where the epicentre
    lands -- but note that no crowd produces this. It is a zoom, not a
    dispersal, and off-centre it is genuinely one-way motion. Use
    localised() for anything that asks whether the detector fires.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = xs - cx, ys - cy
    r = np.hypot(rx, ry)
    r[r < 1e-6] = 1e-6
    return np.dstack([sign * speed * rx / r, sign * speed * ry / r]).astype(np.float32)


def localised(cx, cy, speed, R=70.0, h=H, w=W):
    """
    What a crowd scattering actually puts in the flow field: a disc of
    outward motion around the point, fading out at the edge of it where
    people were too far away to have reacted yet.

    The difference from radial() is not cosmetic. A dispersal is a local
    event in a frame that is mostly still, and the stillness of the rest
    of the frame is part of the signature.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = xs - cx, ys - cy
    r = np.maximum(np.hypot(rx, ry), 1e-6)
    amp = speed * np.exp(-(r / R) ** 2)
    return np.dstack([amp * rx / r, amp * ry / r]).astype(np.float32)


def milling(rng, speed, h=H, w=W):
    """
    A crowd going nowhere in particular: a smooth low-frequency random
    field, which is what dense flow over a plaza of wandering people
    actually looks like once it is downsampled.
    """
    import cv2
    small = rng.normal(0, 1, (9, 16, 2)).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mag = np.maximum(np.hypot(f[..., 0], f[..., 1]), 1e-6)
    return (f / mag[..., None] * speed).astype(np.float32)


def bus(speed, h=H, w=W):
    """A single large object crossing the frame: fast, one-way, ordinary."""
    f = np.zeros((h, w, 2), np.float32)
    f[int(h * 0.30):int(h * 0.85), int(w * 0.10):int(w * 0.75), 0] = speed
    return f


def feed(det, field_fn, seconds, t0=0.0):
    """Run the detector over a stretch of time; return the fire times."""
    fires = []
    n = int(seconds * FPS)
    for i in range(n):
        t = t0 + i * DT
        ev = det.ingest_flow(field_fn(i), t, DT)
        if ev:
            fires.append((t, ev))
    return fires


# ----------------------------------------------------------- the measures


def test_radial_field_is_coherent_and_locates_its_origin():
    st = analyse_flow(radial(140.0, 90.0, 2.0), DT)
    assert st.coherence > 0.9
    assert st.divergence > 0
    ex, ey = st.epicentre
    assert abs(ex - 140) < 12 and abs(ey - 90) < 12


def test_a_whole_frame_zoom_is_read_as_one_way_and_that_is_correct():
    """
    The measured limit of the net-drift gate, pinned here so that nobody
    loosens it to make a test pass.

    Sweeping the origin of a whole-frame outward field across a 320x180
    image measures net drift 0.006 at the centre, 0.50 at 44% of width and
    0.73 at 66%. That is frame geometry: with the origin off centre, most
    of the frame lies on one side of it and genuinely does all move the
    same way. Above 0.6 the detector refuses, which is the right answer --
    a whole frame moving outward is a zoom or a dolly, not twenty people.

    A real dispersal does not do this, because a real dispersal is local;
    see the localised() sweep below, which fires at every position tested
    including hard against the frame edge.
    """
    assert analyse_flow(radial(160.0, 90.0, 2.0), DT).net_drift < 0.1
    off = analyse_flow(radial(210.0, 60.0, 2.6), DT)
    assert off.coherence > 0.8              # it does look radial
    assert off.net_drift > NET_DRIFT_MAX    # and it is still refused


def test_converging_field_reads_as_negative_coherence():
    """A crowd gathering to watch a fight is the opposite sign, not panic."""
    st = analyse_flow(radial(140.0, 90.0, 2.0, sign=-1.0), DT)
    assert st.coherence < 0.0


def test_one_object_crossing_the_frame_looks_radial_but_is_all_one_way():
    """
    The trap this detector has to survive. A solid object leaves a
    divergence ridge along its trailing edge and every pixel of it is then
    'moving away' from that ridge, so coherence alone says panic. Net
    drift is what says bus.
    """
    st = analyse_flow(bus(3.0), DT)
    assert st.coherence > 0.5           # it really does look radial
    assert st.net_drift > 0.9           # and it really is one-way


def test_a_pan_offset_does_not_destroy_the_radial_measure():
    """Camera translation is a constant added to every pixel; remove it."""
    f = radial(160.0, 90.0, 2.0)
    f[..., 0] += 6.0
    f[..., 1] += 2.0
    st = analyse_flow(f, DT)
    assert st.coherence > 0.85


# ------------------------------------------------------------- the gating


def test_pure_surge_does_not_fire():
    rng = np.random.RandomState(7)
    det = DispersalDetector("t")
    base = [milling(rng, 0.6) for _ in range(60)]
    assert not feed(det, lambda i: base[i % 60], 12.0)

    # Speed jumps six-fold, and stays up for longer than any real vehicle
    # pass. There is no epicentre, so there is no alert.
    surge = bus(3.5)
    fires = feed(det, lambda i: surge, 10.0, t0=12.0)
    assert fires == [], f"fired on a one-way surge: {fires}"


def test_a_crowd_all_moving_the_same_way_does_not_fire():
    """A procession, or the camera itself panning. Fast, but not panic."""
    rng = np.random.RandomState(11)
    det = DispersalDetector("t")
    base = [milling(rng, 0.6) for _ in range(60)]
    feed(det, lambda i: base[i % 60], 12.0)
    march = np.zeros((H, W, 2), np.float32)
    march[..., 0] = 4.0
    assert feed(det, lambda i: march, 10.0, t0=12.0) == []


def test_pure_divergence_does_not_fire():
    """
    A station concourse empties outward all day long. Divergence with no
    change of pace is the normal state of a public space, and the rolling
    baseline is what makes it uninteresting.
    """
    det = DispersalDetector("t")
    steady = localised(150.0, 95.0, 2.6)
    assert feed(det, lambda i: steady, 40.0) == []


def test_surge_and_divergence_together_fire():
    rng = np.random.RandomState(3)
    det = DispersalDetector("t")
    base = [milling(rng, 0.6) for _ in range(60)]
    feed(det, lambda i: base[i % 60], 12.0)

    panic = localised(210.0, 60.0, 3.2)
    fires = feed(det, lambda i: panic, 4.0, t0=12.0)
    assert fires, "did not fire on a surge diverging from a point"
    t, ev = fires[0]
    assert t - 12.0 < 0.6                       # within nine frames
    assert abs(ev.epicentre[0] - 210) < 20
    assert abs(ev.epicentre[1] - 60) < 20
    assert ev.surge_ratio > 1.7
    assert ev.coherence > 0.35
    assert ev.confidence > 0.5


@pytest.mark.parametrize("cx", [30, 80, 160, 240, 290])
@pytest.mark.parametrize("cy", [45, 90, 135])
def test_it_fires_wherever_in_frame_the_dispersal_happens(cx, cy):
    """
    An attack does not politely occur in the middle of the picture. It
    happens by a shopfront, at a gate, at the edge of what the camera can
    see, and a detector that only works in the centre third is a detector
    that misses most of the frame.

    All fifteen positions fire, and the epicentre comes back exact,
    because divergence peaks at the origin regardless of where in the
    image that origin sits.
    """
    rng = np.random.RandomState(3)
    det = DispersalDetector("t")
    base = [milling(rng, 0.6) for _ in range(60)]
    feed(det, lambda i: base[i % 60], 12.0)

    panic = localised(float(cx), float(cy), 3.2)
    fires = feed(det, lambda i: panic, 4.0, t0=12.0)
    assert fires, f"missed a dispersal at ({cx}, {cy})"
    _, ev = fires[0]
    assert abs(ev.epicentre[0] - cx) < 12
    assert abs(ev.epicentre[1] - cy) < 12


def test_one_event_raises_one_alert():
    """Alert fatigue is what kills these deployments; 8s of panic is 1 alert."""
    rng = np.random.RandomState(5)
    det = DispersalDetector("t")
    base = [milling(rng, 0.6) for _ in range(60)]
    feed(det, lambda i: base[i % 60], 12.0)
    fires = feed(det, lambda i: localised(160.0, 90.0, 3.2), 8.0, t0=12.0)
    assert len(fires) == 1


def test_nothing_fires_before_a_baseline_exists():
    """The first seconds after a reconnect are not evidence of anything."""
    det = DispersalDetector("t")
    assert feed(det, lambda i: localised(160.0, 90.0, 3.6), 2.0) == []


# --------------------------------------------------------- end to end (px)


def test_epicentre_is_reported_in_full_resolution_image_coordinates():
    """
    The same event, through real frames and real optical flow, on a
    640x360 camera: the alert must point at a pixel an operator can look
    at, not at a coordinate in the 320px working copy.
    """
    import cv2
    rng = np.random.RandomState(2)
    w, h = 640, 360
    n = 40
    px = rng.uniform(0.08, 0.92, n) * w
    py = rng.uniform(0.15, 0.92, n) * h
    head = rng.uniform(0, 2 * np.pi, n)
    ex, ey = w * 0.62, h * 0.40

    def draw():
        img = np.full((h, w, 3), 70, np.uint8)
        img[:int(h * 0.12)] = 40
        for i in range(n):
            cv2.circle(img, (int(px[i]), int(py[i])), 5, (200, 200, 205), -1)
        return img

    det = DispersalDetector("cam-x")
    t = 0.0
    for _ in range(int(10 * FPS)):                 # milling
        px[:] = np.clip(px + np.cos(head) * 0.8, 4, w - 5)
        py[:] = np.clip(py + np.sin(head) * 0.5, 4, h - 5)
        head += rng.normal(0, 0.25, n)
        det.update(draw(), t)
        t += DT

    fired = None
    for _ in range(int(3 * FPS)):                  # everyone runs from (ex, ey)
        d = np.hypot(px - ex, py - ey)
        px[:] = np.clip(px + (px - ex) / d * 4.0, 2, w - 3)
        py[:] = np.clip(py + (py - ey) / d * 4.0, 2, h - 3)
        ev = det.update(draw(), t)
        fired = fired or ev
        t += DT

    assert fired is not None, "no alert on a rendered dispersal"
    assert abs(fired.epicentre[0] - ex) < w * 0.15
    assert abs(fired.epicentre[1] - ey) < h * 0.20


# ------------------------------------------------- layers 2 and 3, gating


def _skeleton(x=100.0, y=100.0, bw=40.0, bh=180.0):
    """An upright person, in the keypoint order torchvision returns."""
    k = np.zeros((17, 2), np.float32)
    cx = x + bw / 2
    put = lambda n, px, py: k.__setitem__(KP[n], (px, py))
    put("nose", cx, y + 0.06 * bh)
    put("left_eye", cx - 4, y + 0.05 * bh)
    put("right_eye", cx + 4, y + 0.05 * bh)
    put("left_ear", cx - 7, y + 0.06 * bh)
    put("right_ear", cx + 7, y + 0.06 * bh)
    put("left_shoulder", cx - 12, y + 0.22 * bh)
    put("right_shoulder", cx + 12, y + 0.22 * bh)
    put("left_elbow", cx - 16, y + 0.38 * bh)
    put("right_elbow", cx + 16, y + 0.38 * bh)
    put("left_wrist", cx - 18, y + 0.54 * bh)
    put("right_wrist", cx + 18, y + 0.54 * bh)
    put("left_hip", cx - 9, y + 0.55 * bh)
    put("right_hip", cx + 9, y + 0.55 * bh)
    put("left_knee", cx - 9, y + 0.78 * bh)
    put("right_knee", cx + 9, y + 0.78 * bh)
    put("left_ankle", cx - 9, y + 0.98 * bh)
    put("right_ankle", cx + 9, y + 0.98 * bh)
    return k, np.full(17, 6.0, np.float32), (x, y, x + bw, y + bh)


def test_standing_person_has_no_threat_posture():
    k, ks, box = _skeleton()
    assert PostureAnalyser._flags(k, ks, box) == []


def test_wrist_above_the_head_reads_as_an_overhead_strike():
    k, ks, box = _skeleton()
    for w_, e_ in (("right_wrist", "right_elbow"),):
        k[KP[e_]] = (k[KP[e_]][0], 100 + 0.14 * 180)      # elbow above shoulder
        k[KP[w_]] = (k[KP[w_]][0], 100 - 0.06 * 180)      # wrist above head
    assert "arm-raised-overhead" in PostureAnalyser._flags(k, ks, box)


def test_a_person_lying_down_reads_as_on_ground():
    k, ks, _ = _skeleton()
    # Torso axis rotated to horizontal: shoulders and hips side by side.
    k[KP["left_shoulder"]] = (100, 200)
    k[KP["right_shoulder"]] = (100, 208)
    k[KP["left_hip"]] = (190, 200)
    k[KP["right_hip"]] = (190, 208)
    assert "on-ground" in PostureAnalyser._flags(k, ks, (100, 190, 260, 220))


def test_weapon_layer_is_refused_on_a_camera_that_cannot_resolve_a_hand():
    """
    The government cameras measured 12-27px of number plate. A plate is
    0.5m and a hand is 0.1m, so the hand is 2-5px and the knife inside it
    does not exist in the image. The registry must say so rather than
    running the detector and reporting whatever it hallucinates.
    """
    assert hand_px_from_plate_px(27) < MIN_HAND_PX
    plan = layer_plan(plate_px=27)
    assert plan["weapon"]["enabled"] is False
    assert plan["dispersal"]["enabled"] is True          # always available


def test_weapon_layer_is_allowed_where_the_hand_is_actually_resolved():
    plan = layer_plan(body_px=400)
    assert plan["weapon"]["enabled"] is True
    assert plan["posture"]["enabled"] is True


def test_posture_layer_is_refused_on_body_sized_pixels_it_does_not_have():
    plan = layer_plan(body_px=40)
    assert plan["posture"]["enabled"] is False
    assert plan["dispersal"]["enabled"] is True
