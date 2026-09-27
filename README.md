# Sentinel

**A camera registry that records what each camera can actually do — measured from its video, not read from a spec sheet.**

Gujarat Police Hackathon Innovation Challenge 2026 · Integrated Video
Management & Analytics Platform · submitted by Meherzan Turel as an
individual entrant.

---

## The finding this is built on

Gujarat operates ~80,000 cameras across 26 departments. Everyone building
for this problem assumes the cameras can do what their specifications say.

I surveyed all 30 cameras in the government sandbox. A number plate needs
roughly **120 pixels** of width to be read. The median vehicle on this grid
presents **27.4 px**.

**Three cameras in thirty can read a number plate.**

Eleven of the thirty are 1920×1080, and only three of those eleven are
among them. The clearest pair in the data:

| Camera | Resolution | Plate width delivered |
|---|---|---:|
| `cam16` Visat P2 | 1920×1080 | **22 px** |
| `cam22` BK Mervada | 854×480 | **63 px** |

A fifth of the pixels, three times the plate. Resolution predicts
nothing; mounting distance and angle predict everything, and neither
appears on a specification sheet.

So this system **measures each camera, then assigns analytics
accordingly**. Plate reading runs only where a plate is legible. Weak
cameras run cheap presence detection and serve as corroboration. On the
measured distribution that is 10 % of cameras on heavy analytics rather
than 100 % — ~160 datacentre GPUs statewide instead of ~1,600.

---

## What runs

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m uvicorn sentinel.api:app --port 8099
```

**Python 3.12.** Not 3.13 or 3.14: numpy has no wheel for them yet and
pip falls back to compiling it from source, which fails without a
toolchain. Established by installing into an empty environment rather
than trusting the one this was developed in.

The suite is **209 tests and every one of them passes** where its
fixtures exist — including the organisers' own pre-submission checklist,
which is written here as executable acceptance tests rather than as a
list somebody ticked.

A fresh clone reports **189 passed, 21 skipped**, verified by cloning into
an empty directory and running it. Every skip names its own reason: ten
need the local mock grid started (`./grid/run.sh start`), nine need
government footage, which is CCTV of public roads and is not redistributed
here, one needs a populated registry, and one needs the evidence stills,
which are cut from footage that is not redistributed either. Start the
grid and the checklist tests run — that is the fastest way to see the
ingest rules enforced rather than asserted.

- `http://127.0.0.1:8099/` — camera registry on a map, coloured by
  **measured capability**, not by online status
- `http://127.0.0.1:8099/officer` — the officer's console: search a
  registration, get a reconstructed route with a confidence and a stated
  reason per sighting, and produce an evidence bundle

Government-grid access needs credentials in `.env` (gitignored, never
committed). Everything above runs without them against the bundled
footage.

---

## The test case, on their footage

Designate a vehicle seen on one government camera, find it again on
another:

| | |
|---|---|
| Found on a second camera system | **2 sightings**, best score 0.567 |
| Tier awarded | **corroborating** — not confirmed |
| False positives on a camera 300 km away | **0** in 1500 frames |

`cam16` is measured **presence-only**, so the registry caps it at
corroborating however convincing the pixels look. The ceiling is a
property of the camera, not of the score.

And `cam16` stamps **01:47:39** on a frame of broad daylight — its clock
is **6.21 hours** wrong. A timeline built from the camera's own clock,
which is what the vendor VMS shows, would have the bus arriving six hours
before it left. The registry measured the drift, so the route is corrected
and the sighting demoted rather than silently mis-ordered.

```bash
python tools/cross_camera_demo.py     # the above, with both controls
python tools/run_hunt_demo.py         # hunt from a single photograph
```

---

## Layout

| | |
|---|---|
| `sentinel/` | registry, capability, timesync, detect, track, re-id, ANPR, matching, hunt, threat, officer console, API |
| `tools/` | grid survey, registry loader, demos, deck and video builders |
| `tests/` | 173 tests. The eight pre-submission checklist items are covered too, but skip unless the local mock grid is running |
| `audit/` | the measurements every claim rests on |
| `docs/` | high-level design, output report, presentation, demonstration film |
| `live_audit.py` | the original grid probe: connects, samples, measures, disconnects |

## Documents

- [`docs/high-level-design.md`](docs/high-level-design.md) — architecture and the arithmetic behind it
- [`docs/scalability-plan.md`](docs/scalability-plan.md) — scaling to 80,000 cameras: sizing, retention, HA/DR, costs, rollout
- [`docs/output-report.md`](docs/output-report.md) — results, with the source file for each figure
- `docs/deck/Sentinel-Presentation.pdf` — 16 slides, generated from the measurement files
- The demonstration films are submitted to the organisers directly. They
  are government CCTV of public roads and are not republished here.

Nothing in the presentation or the report is typed by hand:
`tools/deck_data.py` computes every figure from `audit/`, so the documents
cannot drift away from the system they describe.

---

## Engineering notes

Both of these were discovered against the live grid, and both produced
results that looked like findings:

- **The catalogue moved.** The documented `/api/ingest` returns 404; the
  grid serves `/cameras.json`, thinned to `{id, name}` — no location,
  codec or stream URLs. A registry that *reads* capability from the
  catalogue cannot be built at all. This one measures instead, so the
  change cost nothing.
- **A session expiry was recorded as camera failure.** A survey completed
  16 cameras, then took 403 on the remaining 14 and wrote them down as
  failed cameras. They were fine. The client now re-authenticates once on
  401/403 and retries; a 404 is not retried.

Frame rates come from PTS, never `CAP_PROP_FPS`. `cam05`'s container
declares `r_frame_rate=250/1` while `CAP_PROP_FPS` on the same file
reports 29.96 — two fields, one file, eight times apart. Measured
delivery is 29.98. RTSP is forced over TCP. Decoder warnings at join
are logged, not fatal. Reconnect uses exponential backoff, tested by
restarting a feed.

**Licensing:** OpenCV (Apache-2.0) and torchvision (BSD-3-Clause). No
AGPL — Ultralytics YOLO was rejected on those grounds, because in a state
procurement that is a blocker rather than a footnote.

---

## What is proven, and what is designed

Proven with tests and on real footage: ingest and stream handling,
capability measurement (2.6 % worst error against known ground truth —
measured on the eight local mock streams, where the true plate width is
known because I rendered it), clock-drift detection, the tamper-evident audit chain, tiered
matching, cross-camera re-identification, the officer's console.

Designed but **not** proven on real footage: the person-safety layers for
threat posture and weapon-in-hand — there is no government-grid footage of
an attack and there will not be one; statewide scaling, which is
arithmetic and architecture rather than running software; and real VMS
vendor SDK integration.

That distinction is stated here, in the design document, in the report and
on the final slide, because the next stage is production feeds in front of
the State Crime Records Bureau, and anything overstated now becomes a
problem on 12 October.
