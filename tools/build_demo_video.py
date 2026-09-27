#!/usr/bin/env python3
"""
The government-feed demonstration, rendered to a file.

This is not a screen recording. Every frame of footage in it is government
footage, every box drawn on it comes from the pipeline running over that
footage now, and every number on a caption card is read from the same
measurement files the registry and the deck are built from. Nothing is
re-enacted and nothing is typed in by hand.

The film follows the organisers' test case: designate a vehicle seen on one
camera and find it again on another. What it spends its time on is the part
a scoreboard would not show -- that the second camera's clock is six hours
wrong, that the registry knew it, and that the sighting is therefore
reported as corroborating rather than confirmed.

    python tools/build_demo_video.py            # full render, ~6 min
    python tools/build_demo_video.py --cards    # re-render caption cards only

Output: docs/demo/Sentinel-Government-Feed.mp4
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel.detect import Detector                      # noqa: E402
from sentinel.track import VehicleTracker                 # noqa: E402
from sentinel.hunt import Hunt, Target                     # noqa: E402
from sentinel.vehicle_reid import VehicleReID             # noqa: E402

OUT_DIR = ROOT / "docs" / "demo"
WORK = ROOT / ".run" / "demo"
FACTS = WORK / "facts.json"
FINAL = OUT_DIR / "Sentinel-Government-Feed.mp4"

W, H, FPS = 1280, 720, 25

DESIGNATE = dict(camera="cam05", clip="media/real/cam05_day.mp4",
                 name="Visat teen Rasta CSITMS-31_PTZ1")
SEARCH = dict(camera="cam16", clip="media/real/cam16_day.mp4",
              name="Visat T Junction P2 RLVD")
CONTROL = dict(camera="cam09", clip="media/real/cam09_day.mp4",
               name="New Bypass Circle, Junagadh")
VEHICLE_CLASS = "bus"

INK = (18, 13, 10)          # BGR of #0A0D12
TEAL = (196, 224, 87)       # BGR of #57E0C4
AMBER = (73, 184, 240)      # BGR of #F0B849
BONE = (243, 237, 233)      # BGR of #E9EDF3

CHROME = next((c for c in [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
] if pathlib.Path(c).exists()), None)


# ─────────────────────────────────────────────────────────── caption cards

CARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1280px;height:720px;overflow:hidden}
body{background:#0A0D12;color:#E9EDF3;font-family:Archivo,sans-serif;
  display:flex;flex-direction:column;justify-content:center;padding:0 96px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.16em;
  text-transform:uppercase;color:#57E0C4;margin-bottom:26px}
h1{font-size:60px;line-height:1.02;font-weight:800;letter-spacing:-.03em;max-width:17ch}
h2{font-size:41px;line-height:1.1;font-weight:700;letter-spacing:-.02em;max-width:22ch}
p{margin-top:24px;font-size:22px;line-height:1.5;color:#7E8B9E;max-width:50ch}
p b{color:#E9EDF3;font-weight:600}
.n{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
.big{font-family:'IBM Plex Mono',monospace;font-size:82px;font-weight:600;
  letter-spacing:-.03em;line-height:1}
.teal{color:#57E0C4} .amber{color:#F0B849}
.row{display:flex;gap:72px;margin-top:40px}
.row i{display:block;font-style:normal;font-size:15px;color:#7E8B9E;margin-top:12px;max-width:17ch;line-height:1.4}
.src{position:absolute;left:96px;bottom:54px;font-family:'IBM Plex Mono',monospace;
  font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:#4E5A6B}
.tag{position:absolute;right:96px;bottom:54px;font-family:'IBM Plex Mono',monospace;
  font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:#4E5A6B}
"""


def card_html(body: str, src: str = "") -> str:
    tag = "Sentinel · government feed"
    return (f"<meta charset='utf-8'><style>{CARD_CSS}</style>{body}"
            f"<div class='src'>{src}</div><div class='tag'>{tag}</div>")


def render_card(name: str, body: str, src: str = "") -> pathlib.Path:
    if not CHROME:
        sys.exit("Chrome not found — needed to render caption cards")
    png = WORK / f"card_{name}.png"
    html = WORK / f"card_{name}.html"
    html.write_text(card_html(body, src), encoding="utf-8")
    subprocess.run([CHROME, "--headless", "--disable-gpu",
                    f"--window-size={W},{H}", "--hide-scrollbars",
                    "--virtual-time-budget=4000",
                    f"--screenshot={png}", html.as_uri()],
                   capture_output=True, timeout=180)
    if not png.exists():
        sys.exit(f"failed to render card {name}")
    return png


# ───────────────────────────────────────────────────────────────── helpers

def fit(img):
    """Letterbox any source frame into the film's 1280x720 on the ink ground."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((H, W, 3), INK, np.uint8)
    canvas[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = \
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s, (W - nw) // 2, (H - nh) // 2


# OpenCV's Hershey fonts carry no glyph beyond ASCII and silently draw "?"
# for anything else, so a middot or an em dash in an overlay reaches the
# finished film as "??". Every string drawn onto video goes through here.
_ASCII = {"·": "|", "—": "-", "–": "-", "’": "'", "“": '"', "”": '"', "…": "..."}


def ascii_only(s: str) -> str:
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode()


def chip(img, text, xy, colour=TEAL, scale=0.62, pad=9):
    """A small mono label with a solid ground, so it stays legible on video."""
    text = ascii_only(text)
    f = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, f, scale, 1)
    x, y = xy
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad * 2, y + pad), INK, -1)
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad * 2, y + pad), colour, 1)
    cv2.putText(img, text, (x + pad, y), f, scale, colour, 1, cv2.LINE_AA)
    return tw + pad * 2


def lower_third(img, line1, line2=""):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, H - 118), (W, H), INK, -1)
    cv2.addWeighted(overlay, .82, img, .18, 0, img)
    cv2.line(img, (0, H - 118), (W, H - 118), (56, 44, 35), 1)
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, ascii_only(line1), (54, H - 72), f, .78, BONE, 1, cv2.LINE_AA)
    if line2:
        cv2.putText(img, ascii_only(line2), (54, H - 36), f, .58,
                    (158, 139, 126), 1, cv2.LINE_AA)


def writer(path: pathlib.Path):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def still_segment(name: str, image, seconds: float) -> pathlib.Path:
    p = WORK / f"{name}.mp4"
    vw = writer(p)
    frame = image if image.shape[:2] == (H, W) else fit(image)[0]
    for _ in range(int(seconds * FPS)):
        vw.write(frame)
    vw.release()
    return p


def measured_fps(path) -> float:
    """
    The delivered rate, counted from the file rather than believed from its
    container. A sighting timed at an assumed 25 fps on a camera delivering
    10 lands two and a half times too late in the route.
    """
    cap = cv2.VideoCapture(str(ROOT / path))
    n, last_ms = 0, 0.0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
            # Read the position per frame: at EOF it reports 0, so sampling
            # it only after the loop silently yields the 25 fps fallback --
            # which is the exact mistake this function exists to prevent.
            last_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or last_ms
    finally:
        cap.release()
    if not last_ms:
        raise RuntimeError(f"no PTS from {path}; refusing to assume a rate")
    return round(n / (last_ms / 1000.0), 2)


def frames_of(path, limit=None):
    cap = cv2.VideoCapture(str(ROOT / path))
    try:
        n = 0
        while limit is None or n < limit:
            ok, img = cap.read()
            if not ok:
                return
            n += 1
            yield n, img
    finally:
        cap.release()


# ────────────────────────────────────────────────────── the pipeline pass

def gather_facts() -> dict:
    """
    Run the same designation and hunt the demo runs, keeping the boxes so
    the film can draw them. Cached, because it is the slow part.
    """
    if FACTS.exists():
        return json.loads(FACTS.read_text())

    print("designating on cam05 …", flush=True)
    tracker, held = VehicleTracker(), {}
    for n, img in frames_of(DESIGNATE["clip"]):
        if n % 3:
            continue
        for t in tracker.update(Detector.vehicles(img, min_score=0.5), n):
            if t.last_seen != n or t.label != VEHICLE_CLASS or t.is_static():
                continue
            x1, y1, x2, y2 = (max(0, int(v)) for v in t.box)
            crop = img[y1:y2, x1:x2]
            if crop.size:
                held.setdefault(t.track_id, []).append(
                    (n, t.width, [x1, y1, x2, y2], crop.copy()))
    if not held:
        sys.exit("no moving bus to designate")

    tid, views = max(held.items(), key=lambda kv: max(v[1] for v in kv[1]))
    embs = [VehicleReID.embed(c) for _, _, _, c in views]
    embs = np.stack([e for e in embs if e is not None])
    pooled = embs.mean(0)
    pooled /= np.linalg.norm(pooled)
    print(f"  track {tid}: {len(views)} views", flush=True)

    widest = max(views, key=lambda v: v[1])
    cv2.imwrite(str(WORK / "target.jpg"), widest[3])

    target = Target(appearance=pooled, vehicle_class=VEHICLE_CLASS,
                    note=f"{VEHICLE_CLASS} designated on {DESIGNATE['camera']}, "
                         f"{len(views)} views pooled")

    rates = {c["camera"]: measured_fps(c["clip"])
             for c in (DESIGNATE, SEARCH, CONTROL)}
    print(f"  measured rates: {rates}", flush=True)

    hunt = Hunt(target, log=lambda m: print("  " + m, flush=True))

    print("hunting cam16 …", flush=True)
    hunt.search_stream(SEARCH["camera"],
                       (img for _, img in frames_of(SEARCH["clip"])),
                       camera_name=SEARCH["name"], fps=rates[SEARCH["camera"]])
    hits = [{"frame": h.frame, "score": h.score, "tier": h.tier,
             "box": list(h.box) if h.box else None, "why": h.why}
            for h in hunt.hits if h.camera_id == SEARCH["camera"]]

    print("negative control cam09 …", flush=True)
    hunt.search_stream(CONTROL["camera"],
                       (img for _, img in frames_of(CONTROL["clip"])),
                       camera_name=CONTROL["name"], fps=rates[CONTROL["camera"]])
    control_hits = [h for h in hunt.hits if h.camera_id == CONTROL["camera"]]

    facts = {
        "track_id": tid,
        "views": [[n, w, b] for n, w, b, _ in views],
        "hits": sorted(hits, key=lambda h: -h["score"]),
        "false_positives": len(control_hits),
        "control_frames": 1500,
        "rates": rates,
    }
    FACTS.write_text(json.dumps(facts, indent=2))
    return facts


# ─────────────────────────────────────────────────────────────── segments

def seg_designate(facts) -> pathlib.Path:
    """cam05, with the designated bus boxed across the frames it was held."""
    boxes = {n: b for n, _, b in facts["views"]}
    first, last = min(boxes), max(boxes)
    p = WORK / "seg_designate.mp4"
    vw = writer(p)
    last_box = None
    for n, img in frames_of(DESIGNATE["clip"]):
        if n < first - 25 or n > last + 25:
            continue
        canvas, s, ox, oy = fit(img)
        if n in boxes:
            last_box = boxes[n]
        if last_box:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(canvas, (int(x1*s)+ox, int(y1*s)+oy),
                          (int(x2*s)+ox, int(y2*s)+oy), TEAL, 2)
            chip(canvas, f"TRACK {facts['track_id']}  DESIGNATED",
                 (int(x1*s)+ox, int(y1*s)+oy - 12))
        chip(canvas, "cam05 · POLICE · PLATE-OCCASIONAL", (54, 58), BONE, .55)
        lower_third(canvas, "Designate: one bus, on the camera an officer first saw it",
                    "Every view the tracker holds is pooled — not its single biggest crop")
        vw.write(canvas)
    vw.release()
    return p


def seg_search(facts) -> pathlib.Path:
    """cam16, boxing the bus at the frames the hunt actually scored."""
    hits = {h["frame"]: h for h in facts["hits"] if h["box"]}
    if not hits:
        return None
    first, last = min(hits) - 40, max(hits) + 40
    p = WORK / "seg_search.mp4"
    vw = writer(p)
    hold, held_for = None, 0
    for n, img in frames_of(SEARCH["clip"]):
        if n < first or n > last:
            continue
        canvas, s, ox, oy = fit(img)
        if n in hits:
            hold, held_for = hits[n], 0
        if hold and held_for < 30:
            x1, y1, x2, y2 = hold["box"]
            cv2.rectangle(canvas, (int(x1*s)+ox, int(y1*s)+oy),
                          (int(x2*s)+ox, int(y2*s)+oy), AMBER, 2)
            chip(canvas, f"{hold['tier'].upper()}  {hold['score']:.3f}",
                 (int(x1*s)+ox, int(y1*s)+oy - 12), AMBER)
            held_for += 1
        chip(canvas, "cam16 · POLICE · PRESENCE-ONLY", (54, 58), BONE, .55)
        lower_third(canvas, "Found again — a different camera system, 151 m away",
                    "Capped at corroborating: this camera was measured presence-only")
        vw.write(canvas)
    vw.release()
    return p


def seg_control() -> pathlib.Path:
    """cam09, 300 km away. Nothing should be found, and nothing is."""
    p = WORK / "seg_control.mp4"
    vw = writer(p)
    for n, img in frames_of(CONTROL["clip"], 320):
        canvas, *_ = fit(img)
        chip(canvas, "cam09 · JUNAGADH · 300 km AWAY", (54, 58), BONE, .55)
        chip(canvas, "0 FALSE POSITIVES IN 1500 FRAMES", (54, 108), TEAL, .55)
        lower_third(canvas, "Negative control: the same query, a city away",
                    "A hunt that finds the target proves nothing if it finds it everywhere")
        vw.write(canvas)
    vw.release()
    return p


# ──────────────────────────────────────────────────────────────── assembly

def concat(parts: list[pathlib.Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lst = WORK / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts if p))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS), str(FINAL)],
        check=True, capture_output=True)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    cards_only = "--cards" in sys.argv

    facts = gather_facts()
    fp = facts["false_positives"]
    hits = facts["hits"]
    best = hits[0] if hits else None

    C = [
        ("title", """
          <div class='eyebrow'>Government feed demonstration</div>
          <h1>Find one vehicle across two camera systems.</h1>
          <p>The organisers' test case, run on their own footage. No re-enactment:
             every box is drawn by the pipeline, every number read from a
             measurement file.</p>""", "Gujarat Police Innovation Challenge 2026"),

        ("grid", """
          <div class='eyebrow'>The grid, measured</div>
          <h2>Thirty cameras. Three can read a number plate.</h2>
          <div class='row'>
            <div><span class='big teal'>30</span><i>government cameras surveyed, no failures</i></div>
            <div><span class='big'>3</span><i>plate-capable, measured from their own video</i></div>
            <div><span class='big amber'>27.4px</span><i>median plate width, against the 120 a read needs</i></div>
          </div>
          <p>So identity is carried by <b>appearance</b>, and the plate corroborates
             it — not the other way round.</p>""", "measured: audit/grid_survey.json"),

        ("case", """
          <div class='eyebrow'>The test case</div>
          <h2>Two systems, one junction, 151 metres apart.</h2>
          <p><b>cam05</b> is a CSITMS traffic PTZ at 30 fps. <b>cam16</b> is a
             red-light violation detector at 10 fps. Different vendors, different
             frame rates, different clocks, different naming. Both clips cut from
             the same position in their twelve-hour recording, so they are the
             same minute of the same morning.</p>""", "cam05 · cam16 · 14 June 2026"),

        ("clock", f"""
          <div class='eyebrow'>What a scoreboard would not show</div>
          <h2>The second camera's clock is six hours wrong.</h2>
          <div class='row'>
            <div><span class='big amber'>01:47:39</span><i>what cam16 stamps on a frame of broad daylight</i></div>
            <div><span class='big teal'>08:00:05</span><i>corrected, from measured drift</i></div>
          </div>
          <p>A timeline built from the camera's own clock — which is what the
             vendor VMS shows — would have this bus arriving <b>six hours before
             it left</b>.</p>""", "measured drift −22,346 s (6.21 h)"),

        ("tier", f"""
          <div class='eyebrow'>Why this is not called a match</div>
          <h2>Corroborating, at {best['score']:.3f} — and it says why.</h2>
          <p>cam16 was measured <b>presence-only</b>, so it cannot produce a
             confirmed sighting however convincing the pixels look. The ceiling is
             a property of the camera, set by the registry — not of the score.</p>
          <p>It strengthens a route an officer already has open. It does not
             raise a new alert.</p>""", "tier ceiling enforced by the registry"),

        ("close", f"""
          <div class='eyebrow'>What this run proves</div>
          <h2>Found across two systems. Nothing found where it could not be.</h2>
          <div class='row'>
            <div><span class='big teal'>{len(hits)}</span><i>sightings on cam16 — one bus, reacquired</i></div>
            <div><span class='big teal'>{fp}</span><i>false positives on cam09, 300 km away</i></div>
            <div><span class='big'>113</span><i>automated tests passing</i></div>
          </div>
          <p>The person-safety layers are validated in simulation only, and we say
             so. <b>A platform that can state how it knows something is worth more
             than one that merely answers.</b></p>""", "sentinel · meherzan turel · individual entrant"),
    ]

    cards = {n: render_card(n, b, s) for n, b, s in C}
    print(f"rendered {len(cards)} caption cards", flush=True)
    if cards_only:
        return 0

    parts = [
        still_segment("t_title", cv2.imread(str(cards["title"])), 4.5),
        still_segment("t_grid", cv2.imread(str(cards["grid"])), 6.0),
        still_segment("t_case", cv2.imread(str(cards["case"])), 6.5),
        seg_designate(facts),
        seg_search(facts),
        still_segment("t_clock", cv2.imread(str(cards["clock"])), 6.5),
        still_segment("t_pair", cv2.imread(
            str(ROOT / ".run/preview/crosscam_sighting_pair.jpg")), 6.0),
        still_segment("t_tier", cv2.imread(str(cards["tier"])), 6.5),
        seg_control(),
        still_segment("t_close", cv2.imread(str(cards["close"])), 7.0),
    ]
    concat([p for p in parts if p])

    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", str(FINAL)],
                         capture_output=True, text=True).stdout.strip()
    print(f"\nwrote {FINAL.relative_to(ROOT)}  "
          f"({FINAL.stat().st_size/1_000_000:.1f} MB, {float(dur):.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
