# Sentinel — Output Report

**Integrated Video Management & Analytics Platform**
Gujarat Police Hackathon Innovation Challenge 2026
Submitted by Meherzan Turel, individual entrant

Accompanies: `Sentinel-Own-Feed.mp4`, `Sentinel-Government-Feed.mp4`,
`Sentinel-Presentation.pdf`, `high-level-design.md`,
`scalability-plan.md`.

---

## 1. What this report is

The results of running the submitted system against the organisers' own
camera grid, with the figures each claim rests on and the file each figure
came from.

Every number below is reproducible from this repository. The presentation
is generated from the same files by `tools/build_deck.py`, so the deck and
this report cannot disagree with each other or with the system.

---

## 2. The grid, as measured

All 30 government cameras were surveyed — resolution, delivered frame
rate, vehicle scale, plate-pixel yield, brightness and PTZ behaviour —
sampled at 09:00 in each twelve-hour recording. Daylight; median frame
brightness 110 of 255.

**30 of 30 surveyed. No failures.**

| Capability | Cameras | What it means |
|---|---:|---|
| plate-capable | 3 | ANPR is worth running |
| plate-occasional | 2 | Plate corroborates, never leads |
| presence-only | 18 | Detect, track, re-identify |
| no vehicles observed | 7 | Working cameras, not watching roads |

Source: `audit/grid_survey.json`

### The finding

A plate needs roughly **120 px** of width to be read. Across this grid the
median vehicle presents **27.4 px**.

**Three cameras in thirty can read a number plate — 1 in 10.**

| Camera | Resolution | Plate p90 | ANPR yield |
|---|---|---:|---:|
| cam18 Rajkot City | 1920×1080 | 272 px | 21.7 % |
| cam17 Rajkot Bus Port | 1920×1080 | 201 px | 31.2 % |
| cam09 New Bypass, Junagadh | 1920×1080 | 193 px | 54.6 % |
| cam05 Visat Teen Rasta | 1920×1080 | 89 px | 2.8 % |
| cam12 Adalaj Tollnaka | 1280×720 | 68 px | 5.4 % |
| cam22 BK Mervada Tran Rasta | 854×480 | 63 px | 0 % |
| cam16 Visat P2 | 1920×1080 | 22 px | 0 % |

**Resolution predicts nothing.** Eleven cameras on this grid are
1920×1080; three of those eleven are plate-capable. The clearest pair in
the data: **cam16 delivers 22 px of plate at 1920×1080, while cam22
delivers 63 px at 854×480** — a fifth of the pixels, three times the
plate. What decides it is mounting distance, angle and lane offset, and
none of those appear on a specification sheet or in a procurement
document.

Seven cameras observed no vehicles at all. They are not faulty; they point
where vehicles do not go. A platform that assumes every camera is a
traffic camera wastes a quarter of this estate before it starts.

### Is the measurement itself trustworthy?

Capability measurement was checked against eight cameras of known
ground-truth plate width. These are the eight local mock streams, not
government cameras: the true plate width is known there because the system
rendered it. That makes this a check of the measurement arithmetic, not of
its behaviour on real optics.

| | |
|---|---|
| Cameras validated | 8 |
| Mean absolute error | **1.02 %** |
| Worst absolute error | **2.6 %** |

Source: `audit/capability_validation.json`

---

## 3. The own-feed demonstration

Three clips filmed by the entrant on a Surat street on **25 September
2026**, of one **black Hyundai Alcazar, registration GJ19BE3669**, listed
as stolen under **FIR-0231/2026**. The clips are entered into the registry
as three cameras — `own-A`, `own-B`, `own-C` — and the same pipeline that
runs on the government grid is run over them.

Source: `tools/ingest_own_feed.py`, `audit/own_feed.json`, and the live
registry. Rerunning the tool reproduces every figure below.

### The cameras, measured not declared

Position and start time are read from the **GPS and creation time the
recording device wrote into the QuickTime container**. They are not
surveyed and not typed in from memory; consumer GNSS carries an error of
tens of metres, and every screen that shows these cameras says
*"position from the recording's own GPS, not surveyed"* on the location
name itself.

Capability is measured by the **same method and the same thresholds**
`tools/survey_grid.py` applies to the 30 government cameras — vehicle
width from the detector, scaled to plate width, then the share of passing
vehicles clearing 120 px. The yardstick was deliberately not re-tuned for
my own footage.

| Camera | Plate p90 | ANPR yield | Vehicles sampled | Class |
|---|---:|---:|---:|---|
| own-A | 105.6 px | 6.9 % | 392 | plate-occasional |
| own-B | 110.6 px | 5.2 % | 459 | plate-occasional |
| own-C | 115.4 px | 7.3 % | 177 | plate-occasional |

All three clips are 1920×1080 HEVC at a declared 60 fps; measured delivery
from PTS was 60.0, 60.0 and 60.03. **All three came out plate-occasional**
— 105 to 115 px of plate against the 120 px threshold. My own footage,
filmed deliberately, is by this system's measure a *weak* plate camera.
That is the result, and it is the reason the next table reads the way it
does.

### What the pipeline found

| Camera | Frames | Vehicle tracks | Plate reads |
|---|---:|---:|---:|
| own-A | 2,247 | 270 | **0** |
| own-B | 2,086 | 368 | 5 |
| own-C | 1,069 | 114 | 1 |

**own-A produced 270 vehicle tracks and not one legible plate.** Nothing
was written to the database for it and it does not appear on the route. An
empty leg is the honest output of a camera that saw traffic it could not
read, and reporting it is the point: a demonstration in which all three
of the entrant's own cameras happen to work is a demonstration that has
been arranged.

### The two reads of the target, and why both are demoted

| | own-B | own-C |
|---|---|---|
| Time (UTC / IST) | 06:10:51Z / 11:40:51 | 06:21:07Z / 11:51:07 |
| Plate read | GJ19BE3669 | GJ19BE3669 |
| Plate width | **504 px** | **285 px** |
| Read confidence | **1.00** | 0.87 |
| Read method | fused 10 looks | fused 12 looks |
| Read tier from the pixels | confirmed | confirmed |
| **Tier the system awarded** | **probable** | **probable** |
| Camera's measured class | plate-occasional (110 px) | plate-occasional (115 px) |

Both reads are exact, both are on plate widths four times the read
threshold, and **one of them is a 504-pixel plate read at 100 %
confidence**. Both were nevertheless **capped to `probable`**, because the
ceiling is a property of the *camera*, not of the score: `own-B` and
`own-C` measured plate-occasional, and a plate-occasional camera cannot
issue a confirmed match however convincing the individual read looks.

This is the same rule that demoted the government-grid sighting in §4, and
it was applied here against my own interest, on my own footage, on the
one result the demonstration exists to show. A system whose tier ceiling
bends when the answer is the one you wanted is not measuring anything.

Two alerts were raised, both at `probable`, both requiring an officer to
confirm rather than dispatching on their own.

### The other traffic on the road

Four further plates were read on `own-B` from vehicles that are not on
any watchlist, and each was **stored exactly as the OCR returned it**:

| Time | Read as | Plate px | Confidence | Tier | Matched |
|---|---|---:|---:|---|---|
| 06:10:36Z | `C_05JS####` | 187 | 0.44 | corroborating | — |
| 06:10:37Z | `G_O5JS####` | 220 | 0.37 | corroborating | — |
| 06:10:48Z | `G_05JS####` | 208 | 0.52 | corroborating | — |
| 06:10:56Z | `G_O5JS####` | 133 | 0.31 | corroborating | — |

*Redacted here. All four belong to one uninvolved member of the public
whose car was parked in the entrant's own street footage; the character
positions that carry the finding are shown and the rest is masked. The
full reads exist in the working database and are withheld from the
published one — see `tools/build_demo_site.py`. A submission arguing that
privacy is a property of the architecture does not print a stranger's
registration in its own report.*

Three of the four normalise to the same registration — the O/0 confusion
is repaired by the normal form — and the fourth reads a **G as a C**,
which the normal form does not repair and which is therefore left
standing. None of them were rewritten to a clean string. Writing the
tidy plate would launder a degraded read into an exact match and show an
officer a certainty the camera never had.

They are kept rather than discarded because a search that returns
GJ19BE3669 and not these is only meaningful if these are in the table to
be excluded.

### The route

`officer_route` for GJ19BE3669, under case `FIR-0231/2026`, officer
*PSI R. Chauhan*, authorised by *DySP M. Parmar* — all three required, or
the search is refused:

| | |
|---|---|
| Sightings | 2, on 2 cameras |
| Leg | own-B 06:10:51Z → own-C 06:21:07Z |
| Distance | **0.29 km** in **616 s** (10.3 min) |
| Implied speed | 1.7 km/h |
| **Impossible legs** | **0** |
| Threshold | 150 km/h |

No clock correction is applied. The recorder is a phone kept by network
time and I have **not** measured it against an external reference the way
`tools/measure_clocks.py` measures a camera's burned-in clock. The
registry therefore records the time confidence as *unmeasured* rather than
as zero drift — which reads as "no grounds to demote", not as "good".

### What the own feed does and does not show

It shows the whole chain on footage whose ground truth I control: plate
read → watchlist match → tiered alert → route on the map → evidence
bundle. It does **not** show a confirmed-tier match, because the cameras
I filmed with did not earn one. That is the more useful result.

---

## 4. The test case

The organisers' Step 7 asks for a designated vehicle tracked across
heterogeneous cameras. This was run on government footage.

**Setup.** `cam05` is a CSITMS traffic PTZ delivering 29.98 fps. `cam16`
is a red-light violation detector delivering 10.01 fps. Different vendors,
different frame rates, different clocks, different naming conventions,
watching one junction 151 m apart. Both clips were cut from the same
position in their twelve-hour recordings, so they cover the same minute of
the same morning — 14 June 2026.

Frame rates were measured from PTS, never read from the container. `cam05`'s
container declares `r_frame_rate=250/1`; `CAP_PROP_FPS` on the same file
reports 29.96; measured delivery is 29.98. Two header fields eight times
apart is why neither is trusted.

**Result.**

| | |
|---|---|
| Vehicle designated | one BRTS bus, on cam05, 53 pooled views |
| Found again on cam16 | **2 sightings**, best score **0.567** |
| Tier awarded | **corroborating** |
| False positives on cam09 (300 km away, 1500 frames) | **0** |

The two sightings on cam16 are one bus, not two: the tracker lost and
reacquired it mid-crossing. The report says so rather than counting twice.

### Why the sighting is not called a match

`cam16` was measured **presence-only**. Its tier ceiling is set by the
registry, so it cannot produce a confirmed sighting however convincing the
pixels look. The ceiling is a property of the camera, not of the score.

The sighting strengthens a route an officer already has open. It does not
raise a new alert. That distinction is the alert-fatigue argument made
operational.

### The clock

`cam16` stamps **2026-06-14 01:47:39** on a frame of broad daylight. Its
measured drift is **−22,346 s (6.21 hours)**.

A timeline built from the camera's own burned-in clock — which is what the
vendor VMS displays — would place this bus at the second camera **six
hours before it passed the first**. The registry measured the drift, so
the corrected route reads 08:00:03 → 08:00:05, and the sighting is
demoted rather than silently mis-ordered.

### The impossible-transit flag

The corrected leg is 0.15 km in 2.4 s — 230 km/h — and is flagged
impossible. This is correct behaviour and the registry explains it: these
two cameras have overlapping views 151 m apart, so this is not a journey
between two points but one vehicle co-observed by two systems seconds
apart. A registry that knows only coordinates would report a bus at
230 km/h. One that knows the cameras overlap reads it properly.

The same flag is what detects a cloned plate when the cameras genuinely
are far apart.

Source: `tools/cross_camera_demo.py` — rerunning it reproduces these
figures. The run log and evidence frames are government footage and are
supplied to the organisers rather than published.

---

## 5. Second run: hunt from a still image

A separate run designates a vehicle from a single photograph rather than
from a track, then hunts two cameras.

| | |
|---|---|
| Positive control (camera it was seen on) | 2 sightings — 1 probable, 1 corroborating |
| Negative control (cam01, 300 km away) | **0 false positives** |
| Impossible legs | 0 |

A hunt that finds the target proves nothing on its own if it also finds
the target everywhere else. The negative control is the number that decides
whether an officer can act on the result.

Source: `tools/run_hunt_demo.py` — rerunning it reproduces these figures.

---

## 6. Governance, as exercised

| | |
|---|---|
| Audit entries chained in the working system | 124 |
| Searches possible without an authorising officer | **0** |
| Departments federated in one registry | 5 |

Every query and export is hash-chained to its predecessor. Altering an
entry breaks every hash after it and the console reports the chain as
broken — verified by test, not asserted.

The registry reports **41 cameras: 30 government, 8 local mock streams,
3 own-feed clips**. The split is shown rather than totalled, because a
registry that reports 41 without saying 11 of them are its own test
cameras is padding its own headline.

*Known inconsistency, stated rather than hidden:* `tools/deck_data.py`
still partitions the registry by stream host into "government" and
"own_feed", which predates the three own-feed clips being added. Its
`registry.government` (30) and `registry.own_feed` (8) sum to 38, not to
the 41 it also reports. The three clips are the difference. The figures
above are taken from the database directly.

---

## 7. Scaling arithmetic

Driven by the measured 10 % heavy share, not an assumed one.

| | All cameras | Capability-assigned |
|---|---:|---:|
| Datacentre GPUs | ~1,600 | **~160** |
| Cameras on heavy analytics | 80,000 | **8,000** |

Bandwidth if centralised: 160–400 Gbps sustained. Storage: 3.4 PB per day,
103 PB at 30-day retention — not viable, so metadata and sightings are
retained indefinitely, evidence clips per case, raw video on a short
rolling window per department policy.

The full treatment — compute tiering, low-bandwidth sites, hot/warm/cold
retention, load balancing and horizontal scaling, monitoring and health
checks, HA/backup/DR, cybersecurity, and indicative capital and operating
cost for both designs — is `docs/scalability-plan.md`.

*Correction, stated rather than quietly dropped:* an earlier draft put the
all-cameras figure at 264,000 GPUs. That was extrapolated from a laptop GPU
and overstated the requirement by roughly 150×. A number nobody can
reproduce is worth less than no number.

---

## 8. Engineering conformance

All eight items on the organisers' pre-submission checklist pass as
automated tests, including reconnect-on-feed-restart, which cannot be
tested against a government feed at all.

**210 passed, nothing skipped** with the local mock grid running
(`./grid/run.sh start`, then `python -m pytest`). Without the grid the ten
checklist tests skip and say why; a clean clone of the public repository,
which also lacks the government footage and the measurement files, reports
189 passed, 21 skipped, every skip naming its own reason.

Two live-grid failures were found and fixed during this work, both of
which had produced results that looked like findings:

- **The catalogue moved.** The documented `/api/ingest` began returning
  404; the grid serves `/cameras.json`, thinned to `{id, name}` — no
  location, codec or stream URLs. A registry that reads capability from the
  catalogue could not be built at all. Mine measures instead, so the change
  cost nothing.
- **A session expiry was recorded as camera failure.** A survey completed
  16 cameras then took 403 on the remaining 14, which were written down as
  failed cameras. They were fine. The client now re-authenticates once on
  401/403 and retries; a 404 is not retried.

Licensing: OpenCV (Apache-2.0) and torchvision (BSD-3-Clause). No AGPL
anywhere in the stack, which in a state procurement is a blocker rather
than a footnote.

---

## 9. What is proven, and what is designed

Stated explicitly, because the next stage is production feeds in front of
the State Crime Records Bureau and anything overstated now becomes a
problem on 12 October.

**Proven, with tests and on real footage**

- Ingest and stream handling; all eight checklist items as automated tests
- Capability measurement, 2.6 % worst error against known ground truth
  (on the eight local mock streams, where the true width is known)
- Clock-drift detection — caught a deliberate 7-minute error to within 1 second
- Tamper-evident audit chain
- Tiered matching and the capability tier ceiling — including capping a
  504 px, 100 %-confidence read to *probable* on my own footage (§3)
- Cross-camera vehicle re-identification on two government camera systems
- The officer's console, end to end
- The own-feed chain end to end on real filmed footage: measured
  capability, plate read, watchlist match, tiered alert, route, evidence

**Measured on the real grid**

- Capability and ANPR yield for all 30 cameras, no failures
- Transport and catalogue behaviour, including both failures above
- Per-frame inference cost
- **What it would take to fix each camera.** `sentinel/planner.py` turns
  the measured shortfall into an engineering change, stated as a
  multiplier against what the camera does now rather than as an invented
  part number — the catalogue supplies no focal length, mounting height or
  distance, so none is claimed. Across the 33 cameras the plan covers —
  the 30 government cameras plus the 3 own-feed clips — for plate reading:
  3 already plate-capable, 11 a lens or a different aim, 8
  move-it-or-add-one-closer, 4 a new camera in a new place, 7 the wrong
  question for that camera. `tools/plan_upgrades.py`,
  `audit/upgrade_plan.json`.
- **Which cameras to open next.** `sentinel/lookahead.py` answers the
  organisers' "open only the cameras you are actively processing" as a
  decision the registry makes: bearing from the last sightings, distance
  over measured ground speed, ranked by what the camera can actually do,
  capped at the stream budget. Given the cross-camera run's first two
  sightings it nominated **cam16 at rank 1 of 1 reachable** — the camera a
  human chose by hand in §4. `tools/lookahead_demo.py`.

**Measured and deliberately not built**

- Facial recognition. The brief asks for it; the cameras cannot support
  it. Inter-ocular distance on the four cameras measured runs 2.7 to
  9.8 px against a 40 px floor for even attempting a match — the best
  camera on the grid is short by a factor of four. The registry therefore
  assigns face matching to no camera, and would assign it automatically to
  any camera that cleared the threshold. `sentinel/facecap.py`,
  `audit/face_survey.json`.

**Designed and demonstrated in simulation, not proven on real footage**

- Person-safety layers 2 and 3 (threat posture, weapon-in-hand). There is
  no government-grid footage of an attack and there will not be one.
  Layer 1's false-positive rate *is* measured on real capture — 366 s
  across three cameras, day and night, zero alerts — but its true-positive
  rate is measured only against rendered crowds, and is reported as such.
- Statewide scaling — arithmetic and architecture, not running software
- Live clock-drift measurement. The sandbox serves twelve-hour recordings,
  not live streams, so a drift figure taken from them would not mean what
  it appears to mean.
- Real VMS vendor SDK integration

---

## 10. Reproducing this report

```bash
./grid/run.sh start                   # the ten checklist tests need this
python -m pytest                      # 210 passed, nothing skipped
python tools/survey_grid.py           # re-survey the 30 government cameras
python tools/load_real_cameras.py     # load the survey into the registry
python tools/ingest_own_feed.py       # section 3, from audit/own_feed.json
python tools/ingest_own_feed.py --fresh   # ...re-reading the three clips
python tools/cross_camera_demo.py     # the test case, with both controls
python tools/run_hunt_demo.py         # hunt from a still image
python tools/survey_faces.py          # inter-ocular measurement, section 9
python tools/survey_faces.py --load   # ...written onto the camera rows
python tools/plan_upgrades.py         # per-camera upgrade plan
python tools/lookahead_demo.py        # which cameras to open next
python tools/deck_data.py             # recompute every figure used above
python tools/build_deck.py --pdf      # rebuild the presentation
python tools/record_demo.py --video both  # re-record both demonstration films
python tools/build_docs_pdf.py        # this report, the design and the plan as PDFs
```

The own-feed section can be checked directly against the running system:

```python
from sentinel.officer import officer_search, officer_route
auth = dict(case_ref="FIR-0231/2026", officer="PSI R. Chauhan",
            authorised_by="DySP M. Parmar")
officer_search(plate="GJ19BE3669", **auth)   # 2 sightings, both probable
officer_route(plate="GJ19BE3669", **auth)    # 0.29 km, 0 impossible legs
```

Omitting any one of the three authorisation fields is refused rather than
answered.

Credentials for the government grid are read from `.env`, which is
gitignored and has never entered the repository.

---

*Five times during development a component produced a confident wrong
answer that only revealed itself when the output was inspected rather than
trusted: a clock read as two years out; a phantom 19-second drift from one
bad sample; a road sign measured as a 304-pixel number plate; one vehicle
counted where there were dozens; and an entire camera survey conducted,
unknowingly, on night footage. Every one was caught by looking at pixels
instead of believing a number. That is the argument this submission makes
about the product itself.*
