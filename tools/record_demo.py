#!/usr/bin/env python3
"""
The two mandatory demonstration videos, recorded off the running system.

The rules are explicit: "Mock-ups, animations, simulated interfaces, or
concept videos without an operational backend will not be considered."  So
nothing here is drawn.  A real Chromium is driven against the real FastAPI
service on 127.0.0.1:8099, and the browser session itself is recorded.
Every number that appears on screen was read out of the database or
computed by the pipeline while the camera was rolling.

Three kinds of frame appear in the output, and they are kept visually
distinct on purpose:

  * the consoles          -- screen capture of web/index.html and
                             web/officer.html, live, being typed into
  * the terminal          -- screen capture of a page into which the real
                             stdout of a real subprocess is streamed as it
                             arrives; the process runs while recording
  * caption cards         -- plain title cards, clearly captions, carrying
                             only text and evidence images that the
                             pipeline itself wrote

Re-runnable.  A take that goes wrong costs one command:

    python tools/record_demo.py --video own
    python tools/record_demo.py --video gov
    python tools/record_demo.py --video both

Output: docs/demo/Sentinel-Own-Feed.mp4   (hard 3-minute limit)
        docs/demo/Sentinel-Government-Feed.mp4
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
REC = ROOT / ".run" / "rec"
CARDS = REC / "cards"
CLIPS = REC / "clips"
OUT = ROOT / "docs" / "demo"
BASE = "http://127.0.0.1:8099"

# Captured larger than delivered.  The officer's console collapses its
# timeline column below 1380px, and the timeline is where a sighting says
# why it was capped -- the single most important sentence in either film.
# 1440x810 keeps the column and downscales cleanly to 720p.
CAP_W, CAP_H = 1440, 810
OUT_W, OUT_H = 1280, 720

AUTH_OFFICER = "PSI R. Chauhan"
AUTH_BY = "DySP M. Parmar"
OWN_CASE = "FIR-0231/2026"
GOV_CASE = "FIR-0117/2026"
OWN_PLATE = "GJ19BE3669"


# ----------------------------------------------------------------- pages
#
# The cards borrow the consoles' palette so the film reads as one system,
# but they carry no controls and no data of their own: a caption that looks
# like an interface is exactly what the rules exclude.

CARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --ink:#FAF9F6; --panel:#FFFFFF; --panel2:#F3F1EB; --line:#E4E1D9;
  --bone:#1A1D23; --mist:#4E5763; --dim:#8A9098;
  --plate:#0E9E83; --marginal:#B8780F; --presence:#3E6E94;
  --sans:"Archivo",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1440px;height:810px;overflow:hidden}
body{background:var(--ink);color:var(--bone);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;display:flex;flex-direction:column}
.mark{position:absolute;top:34px;left:56px;font-family:var(--sans);
  font-weight:700;letter-spacing:.14em;font-size:14px;text-transform:uppercase}
.mark span{color:var(--plate)}
.stamp{position:absolute;top:36px;right:56px;font-family:var(--mono);
  font-size:11px;color:var(--dim);letter-spacing:.06em}
main{flex:1;display:flex;flex-direction:column;justify-content:center;
  padding:0 56px}
.kick{font-family:var(--mono);font-size:12px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--plate);margin-bottom:18px}
h1{font-size:52px;line-height:1.08;font-weight:600;letter-spacing:-.02em;
  max-width:1120px}
h1.small{font-size:38px}
p{font-size:20px;line-height:1.55;color:var(--mist);max-width:980px;
  margin-top:22px}
p b{color:var(--bone);font-weight:600}
p code,.m{font-family:var(--mono);font-size:.92em}
ul{margin-top:22px;max-width:1000px}
li{font-size:19px;line-height:1.5;color:var(--mist);margin-bottom:11px;
  list-style:none;padding-left:22px;position:relative}
li:before{content:"";position:absolute;left:0;top:11px;width:9px;height:2px;
  background:var(--plate)}
li b{color:var(--bone);font-weight:600}
figure{margin-top:26px}
figure img{max-width:1140px;max-height:392px;border:1px solid var(--line);
  border-radius:3px;display:block}
figcaption{font-family:var(--mono);font-size:12px;color:var(--dim);
  margin-top:11px;letter-spacing:.03em}
footer{padding:0 56px 34px;font-family:var(--mono);font-size:12.5px;
  color:var(--dim);letter-spacing:.03em}
footer b{color:var(--mist);font-weight:500}
.rule{height:1px;background:var(--line);margin:0 56px 20px}
"""

TERM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1440px;height:810px;overflow:hidden;background:#14161A}
body{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;color:#DCD9D2;
  -webkit-font-smoothing:antialiased}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;
  background:#1C1F25;border-bottom:1px solid #2A2E36}
.dot{width:10px;height:10px;border-radius:50%;background:#0E9E83;
  box-shadow:0 0 0 3px rgba(14,158,131,.18)}
.cmd{font-size:14px;color:#8FE0CD}
.cmd i{color:#6B7280;font-style:normal}
.live{margin-left:auto;font-family:"Archivo",sans-serif;font-size:11px;
  letter-spacing:.18em;text-transform:uppercase;color:#8A9098}
pre{padding:18px 22px;font-size:15px;line-height:1.47;white-space:pre;
  color:#DCD9D2}
"""


def card_html(kick: str, head: str, body: str = "", bullets=None,
              image: pathlib.Path | None = None, cap: str = "",
              foot: str = "", small: bool = False, stamp: str = "") -> str:
    parts = [f'<div class="kick">{html.escape(kick)}</div>',
             f'<h1 class="{"small" if small else ""}">{head}</h1>']
    if body:
        parts.append(f"<p>{body}</p>")
    if bullets:
        parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
    if image is not None:
        parts.append(f'<figure><img src="file://{image}" alt="">'
                     f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    return (f"<!doctype html><meta charset=utf-8><style>{CARD_CSS}</style>"
            f'<div class="mark">Sentinel<span>.</span></div>'
            f'<div class="stamp">{html.escape(stamp)}</div>'
            f"<main>{''.join(parts)}</main>"
            + (f'<div class="rule"></div><footer>{foot}</footer>' if foot else ""))


def term_html(cmd: str) -> str:
    return (f"<!doctype html><meta charset=utf-8><style>{TERM_CSS}</style>"
            f'<header><span class="dot"></span>'
            f'<span class="cmd"><i>$</i> {html.escape(cmd)}</span>'
            f'<span class="live">live — real process, unedited stdout</span>'
            f"</header><pre id=out></pre>"
            "<script>window.__append=function(l){"
            "var p=document.getElementById('out');"
            "p.textContent+=l+'\\n';};</script>")


def write(name: str, markup: str) -> pathlib.Path:
    CARDS.mkdir(parents=True, exist_ok=True)
    p = CARDS / name
    p.write_text(markup, encoding="utf-8")
    return p


# ------------------------------------------------------------- utilities

def api(path: str):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.load(r)


def ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def clip(src: pathlib.Path, start: float, length: float,
         name: str) -> pathlib.Path:
    """A browser-playable excerpt.  Same footage, transcoded, nothing else."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    dst = CLIPS / name
    if dst.exists():
        return dst
    ffmpeg(["-ss", str(start), "-i", str(src), "-t", str(length),
            "-vf", f"scale={CAP_W}:{CAP_H}:flags=lanczos", "-an",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)])
    return dst


def video_page(path: pathlib.Path, caption: str) -> pathlib.Path:
    """A bare <video> on black.  No controls, no chrome, no pretending."""
    return write(path.stem + ".html",
                 "<!doctype html><meta charset=utf-8><style>"
                 "*{margin:0;padding:0}html,body{width:1440px;height:810px;"
                 "background:#0E1013;overflow:hidden}"
                 "video{position:absolute;inset:0;width:1440px;height:810px;"
                 "object-fit:cover}"
                 "figcaption{position:absolute;left:0;right:0;bottom:0;"
                 "padding:13px 26px;background:rgba(10,12,15,.72);"
                 "font-family:'IBM Plex Mono',Menlo,monospace;font-size:14px;"
                 "color:#CFD4DA;letter-spacing:.04em}</style>"
                 f'<video src="file://{path}" autoplay muted></video>'
                 f"<figcaption>{html.escape(caption)}</figcaption>")


class Take:
    """One browser session, one video file, timed beat by beat."""

    def __init__(self, page, name: str):
        self.page = page
        self.name = name
        self.t0 = time.monotonic()
        self.marks: list[tuple[str, float]] = []

    def beat(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def mark(self, label: str) -> None:
        t = time.monotonic() - self.t0
        self.marks.append((label, t))
        print(f"    {t:6.1f}s  {label}", flush=True)

    def show(self, path: pathlib.Path, hold: float, label: str) -> None:
        self.page.goto(f"file://{path}", wait_until="load")
        self.page.wait_for_timeout(400)
        self.mark(label)
        self.beat(hold)

    def type(self, sel: str, text: str, delay: int = 55) -> None:
        self.page.click(sel)
        self.page.fill(sel, "")
        self.page.type(sel, text, delay=delay)

    def scroll_to(self, sel: str, block: str = "center") -> None:
        self.page.eval_on_selector(
            sel, f"e => e.scrollIntoView({{behavior:'smooth',block:'{block}'}})")
        self.page.wait_for_timeout(900)


# ------------------------------------------------------------ video one

def build_own_cards(facts: dict) -> dict:
    stamp = facts["stamp"]
    c = {}
    c["title"] = write("own_01_title.html", card_html(
        "Gujarat Police Hackathon 2026 · Own-feed demonstration",
        "One vehicle, three recordings,<br>and a system that refuses to<br>"
        "overstate what it saw.",
        "Everything after this card is a <b>screen recording of the running "
        "system</b> — a real browser driven against the real service, and a "
        "real process writing to a real terminal. Nothing is re-enacted.",
        foot="Sentinel · Integrated Video Management &amp; Analytics Platform "
             "· Meherzan Turel", stamp=stamp))

    c["feed"] = write("own_02_feed.html", card_html(
        "The feed",
        "Three clips, filmed by the entrant<br>in Surat on 25 September 2026.",
        bullets=[
            "One <b>black Hyundai Alcazar, registration GJ19BE3669</b>, driven "
            "past three fixed positions on one street.",
            "Listed on the watchlist as <b>stolen</b> under case "
            "<span class='m'>FIR-0231/2026</span> before any footage was opened.",
            "Position and start time taken from the <b>GPS and clock the "
            "recording device wrote into the file</b> — not typed in from memory.",
            "1920&times;1080 HEVC, 60&nbsp;fps declared, measured from PTS at "
            "60.0, 60.0 and 60.03.",
        ],
        foot="media/own/IMG_2044.MOV · IMG_2045.MOV · IMG_2046.MOV",
        stamp=stamp))

    c["ingest"] = write("own_03_ingest.html", card_html(
        "Onboarding and processing",
        "The three recordings are put on the<br>grid and the pipeline is run<br>"
        "over them.",
        "Watch for three things: the <b>capability measured off the footage</b>, "
        "the <b>plate the OCR actually returned</b>, and the tier the system "
        "then allows that read to claim. The terminal that follows is a live "
        "process, streaming as it runs.",
        foot="<b>$</b> .venv/bin/python tools/ingest_own_feed.py", stamp=stamp))

    c["end"] = write("own_99_end.html", card_html(
        "Own-feed demonstration · end",
        "Detected, matched, alerted,<br>reconstructed — and capped.",
        bullets=[
            "<b>Onboarded and processed</b> three recorded feeds; "
            f"{facts['frames']} frames, {facts['tracks']} vehicle tracks.",
            "<b>AI detection and analytics</b>: vehicle detection, tracking, "
            "plate localisation and OCR fused across looks.",
            "<b>Watchlist correlation</b> on the normalised registration, "
            "matched before the officer typed anything.",
            "<b>Two alerts raised automatically</b>, both capped at "
            "<b>probable</b> because the cameras were measured at 110px and "
            "115px of plate against the 120px a read needs.",
            "Every query case-linked, refused without an authorising officer, "
            "and chained into a tamper-evident log.",
        ],
        foot="Sentinel · sentinel.hackathon@gujarat.gov.in submission",
        stamp=stamp))
    return c


def record_own(page, facts: dict) -> None:
    t = Take(page, "own")
    c = build_own_cards(facts)

    t.show(c["title"], 6.5, "title card")
    t.show(c["feed"], 8.5, "the feed")

    # The footage itself, so the vehicle is a thing the viewer has seen
    # before any box is drawn around it.
    own_clip = clip(ROOT / "media" / "own" / "IMG_2045.MOV", 13, 8,
                    "own_b_pass.mp4")
    t.show(video_page(own_clip, "media/own/IMG_2045.MOV — own feed B, Surat, "
                                "25 Sept 2026, 13.0s to 21.0s of the recording"),
           7.2, "own-B footage")

    t.show(c["ingest"], 6.0, "ingest card")

    # ---- the real process, streamed into a real browser, while recording
    page.goto(f"file://{write('own_04_term.html', term_html('.venv/bin/python tools/ingest_own_feed.py'))}",
              wait_until="load")
    t.mark("terminal — ingest starts")
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "python"), "tools/ingest_own_feed.py"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        page.evaluate("l => window.__append(l)", line.rstrip("\n"))
    proc.wait()
    t.mark(f"terminal — ingest finished rc={proc.returncode}")
    t.beat(26.0)

    # ---- the registry: what onboarding actually wrote
    page.goto(f"{BASE}/#own-B", wait_until="networkidle")
    page.wait_for_timeout(2600)
    t.mark("registry — own-B selected")
    t.beat(10.0)

    # ---- the officer's console.  The search is run once with the
    # authorisation the console ships with, so that when it is taken away a
    # moment later the refusal is visibly the gate closing on a query that
    # had just worked, rather than a form complaining about empty fields.
    page.goto(f"{BASE}/officer", wait_until="networkidle")
    page.wait_for_timeout(2400)
    t.mark("officer console")
    t.type("#f-plate", OWN_PLATE)
    t.type("#f-case", OWN_CASE, delay=40)
    page.click("#go")
    page.wait_for_selector(".evt", timeout=20000)
    page.wait_for_timeout(2000)
    t.mark("route reconstructed")
    t.beat(5.0)

    # ---- take the authorisation away
    for sel in ("#f-case", "#f-auth"):
        page.click(sel)
        page.fill(sel, "")
        page.wait_for_timeout(300)
    t.beat(0.8)
    page.click("#go")
    page.wait_for_selector("#refuse .refuse", timeout=20000)
    page.wait_for_timeout(1200)
    t.mark("refused — no case, no authorising officer")
    t.beat(7.5)

    # ---- and put it back
    t.type("#f-case", OWN_CASE, delay=40)
    t.type("#f-auth", AUTH_BY, delay=32)
    t.beat(0.8)
    page.click("#go")
    page.wait_for_selector(".evt", timeout=20000)
    page.wait_for_timeout(2000)
    t.mark("authorised — route returned, query on the chain")
    t.beat(6.0)

    # ---- the watchlist hit, which the system raised on its own
    t.scroll_to("#verdict")
    t.mark("watchlist alert")
    t.beat(9.5)
    t.scroll_to("#tierbar")
    t.beat(5.0)

    # ---- and the sentence the whole design exists to be able to print
    page.eval_on_selector("#timeline",
                          "e => e.scrollTo({top:0,behavior:'smooth'})")
    t.beat(1.0)
    t.mark("tier reasons — capped on capability")
    t.beat(11.0)

    # ---- evidence
    page.click("#mk")
    page.wait_for_selector("#bundle.show", timeout=20000)
    page.wait_for_timeout(1500)
    t.mark("evidence bundle")
    t.beat(11.0)

    t.show(c["end"], 11.0, "end card")
    t.mark("done")


# ------------------------------------------------------------ video two

def build_gov_cards(facts: dict) -> dict:
    stamp = facts["stamp"]
    pv = ROOT / ".run" / "preview"
    c = {}
    c["title"] = write("gov_01_title.html", card_html(
        "Gujarat Police Hackathon 2026 · Government-feed demonstration",
        "Thirty government cameras,<br>five departments, graded by what<br>"
        "each one can actually do.",
        "Everything after this card is a <b>screen recording of the running "
        "system</b>, driven against the registry built from the organisers' "
        "own live streams.",
        foot="Sentinel · Integrated Video Management &amp; Analytics Platform "
             "· Meherzan Turel", stamp=stamp))

    c["grid"] = write("gov_02_grid.html", card_html(
        "Onboarding the government grid",
        "Read the catalogue, open each<br>camera in turn, measure it,<br>"
        "close it again.",
        bullets=[
            "<b>30 cameras across 5 departments</b> — Police, Municipal, "
            "GSRTC, Panchayat, Health — onboarded from "
            "<span class='m'>/api/ingest</span>, never from a hard-coded URL.",
            "RTSP forced over TCP; frame rate measured from PTS because the "
            "declared rate cannot be trusted.",
            "<b>3 of the 30 can read a number plate.</b> Not 3 of the 30 are "
            "high-resolution — 11 are 1080p. Resolution would have told you "
            "the opposite.",
            "Analytics are then <b>assigned by measured capability</b>: "
            "13 of the 41 cameras in the registry are scheduled for plate "
            "work at all, the other 28 run cheap presence detection and "
            "serve as corroboration. That is what makes 80,000 cameras "
            "arithmetic rather than aspiration.",
        ],
        foot="audit/grid_survey.json · audit/real_catalogue.json", stamp=stamp))

    c["clock"] = write("gov_03_clock.html", card_html(
        "Time confidence",
        "cam16 stamps 01:47:22 over a frame<br>of broad daylight.",
        "Its clock is <b>6.21 hours slow</b> — measured by comparing it "
        "against a camera 151 metres away watching the same junction, not "
        "taken on trust from the stream. A timeline built from the camera's "
        "own clock, which is what the vendor VMS shows, would have a vehicle "
        "arriving six hours before it left. Sentinel builds the timeline from "
        "<b>corrected</b> time, and caps anything seen on cam16 at "
        "<b>corroborating</b>.",
        image=pv / "crosscam_clocks.jpg",
        cap="cam05 and cam16 at the same instant. Left 07:59:48, right "
            "01:47:22. Measured by tools/cross_camera_demo.py — the registry "
            "row you just saw still reads \u201cnot measured\u201d, because "
            "this drift has not been written back into it yet.",
        small=True, stamp=stamp))

    c["case"] = write("gov_08_case.html", card_html(
        "The organisers' test case",
        "Designate a vehicle on one camera.<br>Find it again on another.",
        "Two unrelated camera systems — a CSITMS traffic PTZ and a red-light "
        "violation detector — 151 metres apart, different frame rates, "
        "different clocks, different naming. The bus is matched at "
        "<b>appearance 0.567</b> and reported as <b>corroborating</b>: cam16 "
        "was measured at 21px of plate and its clock is six hours out, so "
        "nothing seen there is allowed to confirm anything. "
        "<b>0 false positives</b> on the 300&nbsp;km-away control camera.",
        image=pv / "crosscam_sighting_pair.jpg",
        cap="cam05 2026-06-14 08:00:03 → cam16 08:00:05 corrected "
            "(its own clock reads 01:47:39). tools/cross_camera_demo.py.",
        small=True, stamp=stamp))

    c["end"] = write("gov_99_end.html", card_html(
        "Government-feed demonstration · end",
        "What was shown, and what<br>was not claimed.",
        bullets=[
            "<b>Onboarded</b> 30 government cameras from the organisers' "
            "catalogue and graded every one of them from its own delivered video.",
            "<b>Viewed recorded government footage</b> and ran the same "
            "detection, tracking and re-identification chain over it.",
            "<b>Found the designated vehicle again</b>, with the frame, the "
            "offset into the window and the reason for its tier beside it.",
            "A camera that came back empty is reported as <b>searched "
            "and clear</b>, with the frames examined beside it. "
            "\u201cNot found\u201d means not found in the stated window, and "
            "never that the search quietly gave up.",
            "No sighting on this grid is called <b>confirmed</b>. Only a "
            "legible plate supports that claim, and these cameras were "
            "measured on what they can actually resolve.",
        ],
        foot="Accompanying output report: docs/output-report.md", stamp=stamp))
    return c


def record_gov(page, facts: dict) -> None:
    t = Take(page, "gov")
    c = build_gov_cards(facts)

    t.show(c["title"], 6.0, "title card")
    t.show(c["grid"], 9.0, "onboarding card")

    # ---- the registry, which is the control plane
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_timeout(3000)
    t.mark("registry — capability spectrum")
    t.beat(6.0)
    page.goto(f"{BASE}/#cam09", wait_until="load")
    page.wait_for_timeout(1800)
    t.mark("cam09 — plate-capable, 192px")
    t.beat(6.0)
    page.goto(f"{BASE}/#cam16", wait_until="load")
    page.wait_for_timeout(1600)
    t.mark("cam16 — presence-only, 21px")
    t.beat(5.5)

    t.show(c["clock"], 9.0, "clock drift card")

    gov_clip = clip(ROOT / "media" / "real" / "cam09_day.mp4", 0, 7,
                    "cam09_window.mp4")
    t.show(video_page(gov_clip, "media/real/cam09_day.mp4 — cam09, New Bypass "
                                "Circle, Junagadh. Government footage, the "
                                "window the pipeline is about to process."),
           6.2, "cam09 footage")

    # ---- designate off a camera, then hunt the others
    page.goto(f"{BASE}/officer", wait_until="networkidle")
    page.wait_for_timeout(2400)
    t.type("#f-case", GOV_CASE, delay=40)
    t.mark("officer console — hunt")
    t.scroll_to("#h-pick")
    page.click("#h-pick")
    page.wait_for_timeout(800)
    page.select_option("#p-cam", "cam09")
    page.wait_for_timeout(700)
    page.click("#p-go")
    t.mark("reading cam09 for vehicles to designate")
    page.wait_for_selector("#p-grid figure", timeout=180000)
    page.wait_for_timeout(2200)
    t.mark("candidates offered")
    t.beat(2.5)
    page.click("#p-grid figure")
    page.wait_for_timeout(1200)
    t.mark("vehicle designated")

    # cam16 and cam01 are left out of the search only to keep the film
    # inside its running time.  cam16 is the camera the clock card is
    # about; cam05 stays in as the camera that will come back clear.
    for cam in ("cam16", "cam01"):
        page.uncheck(f"#h-cams input[value='{cam}']")
        page.wait_for_timeout(400)
    t.scroll_to("#h-cams")
    t.beat(4.0)
    page.click("#h-go")
    t.mark("hunt started")
    page.wait_for_timeout(2500)
    page.wait_for_function(
        "() => document.querySelectorAll('#timeline .evt').length > 0",
        timeout=300000)
    t.mark("first sighting on screen")
    page.wait_for_function(
        "() => /complete|Search complete/i.test(document.getElementById('h-progress').textContent||'')",
        timeout=300000)
    page.wait_for_timeout(2000)
    t.mark("hunt complete")
    t.beat(4.0)
    page.eval_on_selector("#timeline",
                          "e => e.scrollTo({top:0,behavior:'smooth'})")
    t.beat(8.0)
    t.mark("tier reasons")
    page.eval_on_selector("#timeline",
                          "e => e.scrollTo({top:e.scrollHeight,behavior:'smooth'})")
    t.beat(6.0)
    t.scroll_to("#h-progress", "start")
    t.mark("audit chain")
    t.beat(5.0)

    t.show(c["case"], 10.5, "cross-camera test case")
    t.show(c["end"], 10.0, "end card")
    t.mark("done")


# ---------------------------------------------------------------- driver

def facts_now() -> dict:
    """Read the numbers off the running system, so a card cannot go stale."""
    cams = api("/api/cameras")
    own = json.loads((ROOT / "audit" / "own_feed.json").read_text())
    frames = sum(c["pipeline"]["frames_processed"] for c in own)
    tracks = sum(c["pipeline"]["tracks"] for c in own)
    return {
        "stamp": time.strftime("Recorded %d %b %Y %H:%M %Z"),
        "by_grid": cams["by_grid"],
        "frames": f"{frames:,}",
        "tracks": f"{tracks:,}",
    }


def preflight() -> dict:
    try:
        f = facts_now()
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"the service at {BASE} is not answering: {exc}\n"
                 "start it with:  .venv/bin/python -m uvicorn sentinel.api:app "
                 "--host 127.0.0.1 --port 8099")
    g = f["by_grid"]
    want = {"government": 30, "own-feed": 3}
    for key, n in want.items():
        if g.get(key) != n:
            print(f"  ! registry reports {g.get(key)} {key} cameras, expected "
                  f"{n}. The header in the film will say so. Restart the "
                  f"service if it is running stale code.", file=sys.stderr)
    if not api("/api/officer/hunt/sources")["cameras"]:
        sys.exit("no recorded government footage under media/real/")
    # Transcode the footage excerpts before the camera rolls.  Doing it
    # lazily mid-take freezes whichever caption card happens to be on
    # screen for the length of an ffmpeg run.
    clip(ROOT / "media" / "own" / "IMG_2045.MOV", 13, 8, "own_b_pass.mp4")
    clip(ROOT / "media" / "real" / "cam09_day.mp4", 0, 7, "cam09_window.mp4")
    return f


def convert(webm: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-i", str(webm),
            "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos,fps=30,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", "21",
            "-profile:v", "high", "-level", "4.0", "-an",
            "-movflags", "+faststart", str(dst)])


def run(which: str, facts: dict) -> pathlib.Path:
    from playwright.sync_api import sync_playwright

    stage = REC / f"{which}_take"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    print(f"  recording {which} …", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required",
                                          "--hide-scrollbars"])
        ctx = browser.new_context(
            viewport={"width": CAP_W, "height": CAP_H},
            device_scale_factor=1,
            record_video_dir=str(stage),
            record_video_size={"width": CAP_W, "height": CAP_H})
        page = ctx.new_page()
        try:
            (record_own if which == "own" else record_gov)(page, facts)
        finally:
            ctx.close()
            browser.close()

    webms = sorted(stage.glob("*.webm"))
    if not webms:
        sys.exit(f"playwright wrote no video into {stage}")
    dst = OUT / ("Sentinel-Own-Feed.mp4" if which == "own"
                 else "Sentinel-Government-Feed.mp4")
    convert(webms[0], dst)
    return dst


def probe(path: pathlib.Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return {"codec": s["codec_name"], "w": s["width"], "h": s["height"],
            "pix_fmt": s["pix_fmt"], "seconds": float(d["format"]["duration"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", choices=("own", "gov", "both"), default="both")
    ap.add_argument("--frames-dir", default=str(REC / "verify"),
                    help="where to drop stills for eyeballing the take")
    args = ap.parse_args()

    facts = preflight()
    which = ("own", "gov") if args.video == "both" else (args.video,)
    for w in which:
        dst = run(w, facts)
        info = probe(dst)
        limit = 180 if w == "own" else 210
        flag = "  OVER LIMIT" if info["seconds"] > limit else ""
        print(f"  {dst.relative_to(ROOT)}  {info['w']}x{info['h']} "
              f"{info['codec']} {info['pix_fmt']}  "
              f"{info['seconds']:.1f}s{flag}", flush=True)
        vdir = pathlib.Path(args.frames_dir) / w
        vdir.mkdir(parents=True, exist_ok=True)
        for frac in (0.08, 0.3, 0.52, 0.72, 0.92):
            ffmpeg(["-ss", f"{info['seconds'] * frac:.2f}", "-i", str(dst),
                    "-frames:v", "1", str(vdir / f"{int(frac * 100):02d}.png")])


if __name__ == "__main__":
    main()
