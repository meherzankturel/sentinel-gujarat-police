# Sentinel — High-Level Design

**Integrated Video Management & Analytics Platform**
Gujarat Police Hackathon Innovation Challenge 2026
Submitted by Meherzan Turel, individual entrant

---

## 1. The problem, restated

Gujarat operates roughly 80,000 cameras across 26 departments, bought at
different times from different vendors, running different VMS software.
They do not interoperate. The state wants live feeds matched against
watchlists — stolen vehicles, wanted persons, missing persons — with
real-time alerts, across the infrastructure it already owns.

The obvious design is to ingest everything, run analytics on everything,
and alert on every match. That design fails three times over, and the
failures are arithmetic rather than opinion. Section 7 does the sums.

My answer is a **capability registry that acts as a control plane**: the
system measures what each camera can actually deliver, and assigns
analytics accordingly. Cameras are not equal, and a platform that treats
them as equal wastes its capacity on cameras that cannot produce evidence
while missing the ones that can.

---

## 2. What I found in the sandbox

Every claim in this section was measured, not assumed. Several contradict
the published documentation, and each cost a day's worth of wrong
assumptions to discover.

**The catalogue is nearly empty.** The Resources page states that
`GET /api/ingest` returns "every camera with its id, location, codec, live
status, stream properties, and all three URLs" and that "the catalogue is
the contract". In practice `/api/ingest` returns 404. The real endpoint is
`/cameras.json`, and every record contains exactly two fields:

```json
{ "id": "cam01", "name": "01 Chiman bhai Bridge" }
```

No location. No codec. No resolution. No stream URLs. **A registry that
depends on the catalogue cannot be built.** Mine measures instead, which
is why the discovery cost me nothing.

**Only one transport is reachable.** RTSP (8554) and WebRTC (8889) do not
respond from the public internet; the origin sits behind a proxy that
forwards only 80/443. HLS over HTTPS is the sole working path — which is
also why the portal's own player shows a black frame.

**The feeds are recordings, not live.** Each camera serves a 12.00-hour
AES-128 encrypted VOD of 7,200 segments carrying `EXT-X-ENDLIST`. The
portal simulates liveness by seeking to `(now mod duration)`. Frames carry
a burned-in stamp of 14 June 2026.

I have nonetheless built the client to be correct against live streams:
the pre-submission checklist is met in full, and the 12–13 October test
may well be live. Where a capability cannot be demonstrated against
recordings — clock-drift measurement, for one — I say so rather than
present a number that does not mean what it appears to mean.

**Most of the grid cannot read a number plate.** See section 4.

---

## 3. Architecture

```
  Cameras (30 sandbox / ~80,000 statewide, 26 departments)
        │
        ▼
  Ingest — one hardened path
        forced TCP · PTS-only timing · gap tolerance
        reconnect with backoff 2s→30s · segment-level sampling
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │  CAPABILITY REGISTRY — the control plane      │
  │  measured per camera, from its own video      │
  └───────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
  Vehicle analytics             Person-safety analytics
   detect & track  (all)         crowd dispersal  (all)
   re-identify     (most)        threat posture   (most)
   read plate      (few)         weapon in hand   (few)
        │                              │
        └──────────────┬───────────────┘
                       ▼
        Matching — tiered confidence, alert budget
                       │
                       ▼
        Officer's console — route, prediction, evidence
                       │
        Governance: case-linked access, tamper-evident log,
                    masking on export, per-department retention
```

The registry is the only component that decides what runs where. Every
analytic is assigned; none is run blindly.

---

## 4. The capability registry

### What is measured

Per camera, from its own video rather than its specification:

| Field | Why it exists |
|---|---|
| Delivered resolution | Declared resolution is frequently wrong |
| Real frame rate, from PTS | `CAP_PROP_FPS` disagrees with delivery |
| **Vehicle scale → plate-pixel yield** | The column that decides ANPR |
| PTZ pan share | A moving camera invalidates motion analytics |
| Night usability | Legible by day is not legible by night |
| Clock confidence | A wrong clock silently corrupts a timeline |
| Governance class | Health footage carries extra duty of care |

### The finding that shapes everything

Reliable plate reading needs roughly 120 pixels of plate width. All 30
government cameras were surveyed, sampled at 09:00 in each recording —
daylight, median frame brightness 110 of 255. Nothing below was inferred
from a specification.

| Camera | Resolution | Plate p90 | ANPR yield | Verdict |
|---|---|---:|---:|---|
| cam18 Rajkot City | 1920×1080 | 272 px | **21.7 %** | plate-capable |
| cam17 Rajkot Bus Port | 1920×1080 | 201 px | **31.2 %** | plate-capable |
| cam09 New Bypass, Junagadh | 1920×1080 | 193 px | **54.6 %** | plate-capable |
| cam05 Visat Teen Rasta | 1920×1080 | 89 px | 2.8 % | occasional |
| cam12 Adalaj Tollnaka | 1280×720 | 68 px | 5.4 % | occasional |
| cam22 BK Mervada Tran Rasta | 854×480 | 63 px | 0 % | presence-only |
| cam02 Janpath | 1920×1080 | 63 px | 1.6 % | presence-only |
| cam14 Delight RLVD | 1920×1080 | 45 px | 0.6 % | presence-only |
| cam16 Visat P2 | 1920×1080 | 22 px | 0 % | presence-only |
| 14 further cameras | mixed | ≤ 48 px | 0 % | presence-only |
| 7 cameras | mixed | — | — | no vehicles observed |

**Three cameras in thirty can read a number plate. Eleven cameras are
1920×1080, and only three of those eleven are among them.**

The clearest single comparison in the table: **cam16 is 1920×1080 and
yields 22 px of plate. cam22 is 854×480 and yields 63 px** — a quarter of
the pixels, three times the plate. Resolution predicts nothing. Mounting
distance and angle predict everything, and neither appears on any
specification sheet or in any procurement document.

Seven cameras observed no vehicles at all in the sampled window. They are
not broken; they are pointed at places vehicles do not go. A watchlist
platform that assumes every camera is a traffic camera wastes a quarter of
the estate before it starts.

This is why capability on a real grid is a **yield**, not a yes/no — a
vehicle directly beneath the mast is readable while the same vehicle two
lanes over is not. The yield is the number that tells Gujarat Police where
ANPR is worth deploying.

*An earlier survey of this grid covered only 16 cameras and was sampled,
unknowingly, at a position that was night-time in most recordings. It is
superseded here rather than quietly amended: the ordering it reported —
that mounting distance dominates resolution — survived re-measurement in
daylight across the full thirty, but its figures did not, and the figures
are what a reviewer would have checked.*

### Consequence for the live test

The organisers' test case asks entrants to **track a designated vehicle**,
not to read its plate. On this grid those are different problems. An
entry built on ANPR alone will demonstrate beautifully on a close-up and
produce nothing on the government feed. Identity must therefore be carried
by appearance — type, colour, size, direction, timing — with the plate
serving as corroboration on the minority of cameras that can supply one.

---

## 5. Analytics, assigned by capability

Capability is not one number. A camera on a gantry over a carriageway can
be excellent at number plates and useless for faces; a camera at door
height in a ticket hall is the reverse. So the registry measures and
stores the axes separately, and each analytic is assigned against its own.

| Camera capability | Vehicle analytic | Person-safety analytic |
|---|---|---|
| plate-capable | full ANPR + tracking | weapon detection |
| plate-occasional | ANPR as corroboration | threat posture |
| presence-only | detect, track, re-identify | crowd dispersal |
| no vehicles observed | presence detection | crowd dispersal |
| unusable | none | none |

The person-safety lane deserves a note. A knife is roughly 20 cm; on a
camera whose plates measure 27 px, a knife in a hand is about 8 px. It is
not in the image, and weapon detection there is fiction. But **you do not
need to see the weapon to detect the attack** — when a weapon appears in a
public place, a crowd disperses from a point, and that signal is large,
model-free, and visible on the worst camera on the network. It turns a
camera that can do nothing else into a violence sensor.

### 5.1 Facial recognition

The brief asks for an analytics approach covering ANPR, facial recognition
and tracking. Facial recognition is designed into this architecture and
assigned by the registry exactly as ANPR is — against a measured axis of
its own.

**The measurement.** Face recognition is limited by inter-ocular distance,
the pixels between the eye centres. ISO/IEC 19794-5 puts 60 px as the floor
for an enrolment image; this system treats 40 px as the floor for even
attempting a match, which is generous. Because running a face detector on
cameras that cannot resolve faces would measure my own false positives, the
scale is derived from person detections I already trust, through stated
anthropometry — head height ≈ standing height / 7.5, head width ≈ 0.6 ×
head height, inter-ocular ≈ 0.33 × head width, composing to standing
height / 37.9. The constant is published in `sentinel/facecap.py` so a
reviewer can disagree with it and recompute.

The derivation is generous twice over, and both favour the camera: it
assumes the subject is upright, fully visible and facing the lens. A seated
or walking-away person yields less.

**The result, on the cameras I hold footage for:**

| Camera | People measured | Body height p90 | Inter-ocular | Verdict |
|---|---:|---:|---:|---|
| cam09 New Bypass, Junagadh | 139 | 370 px | **9.8 px** | face-unusable |
| cam16 Visat P2, Ahmedabad | 310 | 166 px | **4.4 px** | face-unusable |
| cam05 Visat Teen Rasta | 343 | 113 px | **3.0 px** | face-unusable |
| cam01 Chimanbhai Bridge | 945 | 101 px | **2.7 px** | face-unusable |

**The best camera on this grid is short by a factor of four.** The worst is
short by fifteen. At 2.7 px there is no face present — there is a smudge
that resembles one, and a recogniser pointed at it does not fail quietly.
It returns a name.

**So no camera on this grid is assigned facial recognition,** and the
system will not run it. That is the registry doing its job, not a gap in
the build: the same control plane that declines to schedule ANPR on 27 of
30 cameras declines to schedule face matching on all of them. Onboard a
camera that clears 40 px — a ticket hall, an entrance lobby, a checkpoint
at head height — and it is assigned automatically, with a ceiling that
caps any face match at *probable* and never *confirmed*, because a
registration is a registered fact whereas a face match is a similarity.

**Two positions I hold deliberately**, and will defend:

- **Person detection and tracking, yes; biometric identification, no — on
  this grid.** The system tracks people, measures crowd motion and reads
  limb geometry. None of that identifies anybody. The distinction matters
  legally as well as technically: tracking an unidentified person through
  a scene and matching a face against a gallery are different intrusions
  and attract different scrutiny.
- **No expression or emotion analysis anywhere, at any resolution.**
  Inferring intent from a face is contested science. Applied by a police
  system to the public it is a liability rather than a feature, and it is
  the first thing a defence would attack.

*Scope of this measurement, stated plainly: four cameras, not thirty. The
gateway returned HTTP 521 during the full-grid run, so the remaining
twenty-six are measured on the next successful pass —
`tools/survey_faces.py --grid` produces the table above. The conclusion is
not in doubt at this margin, but four is what has been measured and four
is what is claimed.*

---

## 6. Matching, confidence and alert budget

Alert fatigue is what actually kills these deployments. Eighty thousand
cameras matched against a national watchlist generate more alerts in an
hour than a control room reads in a week, and once officers begin
dismissing alerts reflexively the system is worse than useless — worse,
because everyone still believes it is working.

A match therefore returns a **tier**, never a boolean:

| Tier | Meaning | Behaviour |
|---|---|---|
| confirmed | capable camera, clean read | raises an alert |
| probable | marginal camera, or inexact plate | raises a lower-priority alert, with the doubt shown |
| corroborating | too weak to stand alone | strengthens an existing track; never notifies |

Three things demote a match, and each is recorded so the officer sees why:
an inexact plate, a camera not measured capable of reading plates, and a
camera whose own clock is not trusted.

**Plate matching is confusion-aware.** OCR reliably confuses `0/O`, `5/S`,
`8/B`. A degraded read of `GJ05MN8899` returned `GJOSMNS899` — every wrong
character a shape collision. Ordinary matching calls that a different
vehicle and the stolen car is missed, on precisely the cameras where a
miss matters most. Those substitutions carry reduced cost, so the vehicle
surfaces as an inexact match, demoted a tier, rather than vanishing.

**The alert budget is explicit** — a stated ceiling per control room per
shift. When it is spent, lower tiers stop notifying and become
corroboration. Confirmed alerts always pass.

---

## 7. Scaling to 80,000 cameras

*The full treatment -- hardware and software requirements, hot/warm/cold
retention, load balancing and horizontal scaling, high availability and
disaster recovery, estimated implementation and operational costs, the
statewide rollout plan and the department-wise information requirements --
is in the companion document `scalability-plan.md`. This section carries
the arithmetic the architecture rests on.*


The arithmetic below uses figures measured on this project, not vendor
claims.

**Analytics.** An NVIDIA A100 sustains 40-60 simultaneous 1080p/30fps
detection streams in published reference architectures. Running continuous
analytics on all 80,000 cameras therefore needs on the order of **1,600
datacentre GPUs** -- expensive rather than impossible, but an order of
magnitude beyond what a state video programme budgets for compute.

*(An earlier draft of this document put the figure at 264,000 GPUs. That
was extrapolated from a laptop GPU at 0.33s per frame and overstated the
requirement by roughly 150x. It is corrected here rather than quietly
dropped, because a number nobody can reproduce is worth less than no
number.)*

What real deployments do instead is exactly what this design proposes:
they do not run heavy analytics everywhere.

*These figures are indicative, drawn from vendor and press descriptions of
each programme rather than from published operating data, and I have not
been able to source primary references for them. They are included to show
that selective assignment is ordinary practice, not to carry the argument
— the argument rests on the measured share of this grid, below, which is
reproducible from this repository.*

| Deployment | Cameras | Share running full analytics |
|---|---:|---|
| Port of Rotterdam | 5,000 | 15 % |
| Rio de Janeiro (COR) | 10,000 | 40 % |
| Aptibit, India | 10,000 | 30-40 % |
| Changi Airport | 2,000 | 70 % (high-risk site) |
| Beijing Traffic Bureau | 8,000 | 20 % real-time, rest batched overnight |

Selective assignment is not a novel idea; it is universal practice. **What
differs here is the basis for the selection.** Those deployments tier
cameras by human judgement -- importance of the location, risk of the zone,
time of day. This design tiers them by *measurement*: what the camera can
physically resolve, established from its own video. A camera at a critical
junction that delivers 45px of plate will be given plate reading by a
zone-based policy and will never produce a usable read. Mine will not be
asked.

On the measured distribution of this grid -- 3 cameras in 30 able to
support plate reading -- the heavy tier lands at 10 %, which is below every
figure in the table above and is a measured number rather than an
assumption. Including the two occasional-yield cameras raises it to 17 %,
still below all but one.

**Bandwidth.** A 1080p H.264 stream is 2-5 Mbps. Centralising all 80,000 is
160-400 Gbps sustained. Published deployments avoid this by sending
metadata rather than video to the centre, a 95-98 % reduction, with video
held in a rolling buffer at the edge and pulled on demand for an
investigation. The registry makes that tractable: it says which cameras
justify an edge inference node at all.

**Storage.** 4 Mbps continuous is about 43 GB per camera per day -- **3.4 PB
per day** statewide, 103 PB at 30-day retention. Full retention of raw
video is not viable. The design retains metadata and sightings
indefinitely, evidence clips per case, and raw video only for a short
rolling window, per department policy.

**The organisers' own constraint proves the point.** Their Resources page
states that each connected client receives its own copy of the stream and
instructs entrants to open only the cameras they are actively processing.
That is the 80,000-camera problem stated in miniature, in their own
documentation. Something must choose which cameras to open. Mine chooses
on measured evidence.

---

## 8. Integration strategy: heterogeneous cameras, NVRs and VMS

Twenty-six departments bought at different times from different vendors.
The integration layer therefore has to assume nothing about what it is
talking to, and has to keep working when what it is talking to changes --
which it did, mid-project, when the sandbox catalogue moved endpoint and
lost every field except the camera id.

### The adapter ladder

Each camera is onboarded by the highest rung it supports. Nothing is
required to support more than rung 4.

| Rung | Interface | What it gives | Used when |
|---|---|---|---|
| 1 | **Vendor VMS API / SDK** | Streams, PTZ, events, recorded playback | Department runs a supported VMS and will share an API account |
| 2 | **ONVIF Profile S / T / G** | Stream discovery, PTZ, recording playback | Camera or NVR is ONVIF-conformant |
| 3 | **RTSP (TCP) or HLS direct** | Live stream only | Endpoint reachable; no management API |
| 4 | **NVR file pull** | Recorded segments only | No live path; forensic use only |

The connector framework is a plugin interface: a new vendor is a new
adapter implementing discover / open / close, not a change to the
platform. Departments keep operational control of their own VMS
throughout -- this is federation, not migration, which is why Model 4's
rip-and-replace is rejected on cost and lock-in grounds.

### What the platform refuses to depend on

The sandbox taught this the hard way. The documented catalogue endpoint
began returning 404 mid-project; the replacement carries `{id, name}` and
nothing else -- no location, codec, resolution or stream URL. A platform
that reads camera capability from a catalogue could not have been built
against that grid at all.

So the contract is deliberately thin: **an endpoint and an owning
department.** Everything else -- resolution, delivered frame rate, codec,
what the camera can actually resolve -- is measured from the video. One
sandbox camera declares `r_frame_rate=250/1` in its container while
reporting 29.96 in another field of the same file; the measured delivery
is 29.98. Believing either header would put every sighting in the wrong
second.

### Correlating with watchlist databases

The watchlist is a store of entities -- stolen and blacklisted vehicles,
wanted and missing persons, suspect lists -- normalised on onboarding so
that comparison is confusion-aware rather than exact: a plate read as
`GJ0IAB1Z34` still matches `GJ01AB1234`, and is demoted for having needed
the recovery.

For deployment the authoritative sources are departmental rather than mine:

| Source | Carries | Integration |
|---|---|---|
| **VAHAN** | Vehicle registration, ownership, status | Pull for verification on a confirmed read; never the primary match path |
| **SARATHI** | Driving licence | Case linkage only |
| **eGujCop** | FIR, case records, officer identity | Case reference validation; **the natural home for authentication and RBAC** |
| **CCTNS** | Crime and criminal records | Watchlist population |
| **AFIS / NAFIS** | Biometric | Out of scope: this grid cannot resolve a face (§5.1) |

Correlation runs continuously against the local watchlist replica so an
alert never waits on an external system. VAHAN is consulted *after* a
match to enrich it, not during, because a national registry must not sit
in the latency path of a street-level alert.

---

## 9. Deployment and cybersecurity architecture

### Deployment tiers

Three tiers, sized in the scalability plan and summarised here.

**Edge** decodes, detects, tracks and discards. It forwards metadata and
evidence clips -- 2-5 % of raw volume -- because centralising 80,000
streams is 160-400 Gbps sustained and is not available at any budget.

**District** aggregates, runs the assigned heavy analytics, correlates
against the watchlist, raises alerts, and holds 72 hours of hot video.
Districts operate autonomously under network partition and reconcile on
restore.

**Central** holds the registry, the statewide search index, case
management, the audit chain, and warm/cold archive. It is the only tier
that must serialise writes, because the audit chain cannot fork.

### Security controls

| Control | State |
|---|---|
| TLS on every hop; RTSP forced over TCP | Built |
| Credentials environment-only, never committed | Built |
| Case-linked access, refused before a row is read | **Built and tested at the HTTP layer** |
| Tamper-evident hash chain over every query and export | Built and tested |
| Authentication against departmental identity | **Designed, not built** |
| Role-based access control | **Designed, not built** |
| Masking of uninvolved parties on export | **Designed, not built** |
| Network segmentation; camera VLANs isolated | Design |

The roles the design assumes: **officer** (search within own case),
**supervisor** (authorise, review alerts), **auditor** (read the chain,
not the footage), **administrator** (onboard cameras, never search).

Stated plainly, because a reviewer will find it: **the case-linked gate is
real and tested, but there is no authentication behind it.** It trusts the
officer name it is handed. In deployment it must sit behind eGujCop or
departmental AD with MFA on export and watchlist edits. A gate without an
identity behind it is a control on paper.

That gap was found by auditing this codebase before publication, along
with a worse one: a route returning full movement history with no
credentials at all, whose audit write was conditional on the credentials
that were missing. It is fixed, and there is now a test that enumerates
every route and fails if one is left ungated.

---

## 10. Future roadmap

| Horizon | Work | Why it is next |
|---|---|---|
| **Pre-finale** | Authentication + RBAC; masking on export; retention enforcement | The three designed-not-built controls above. All are named in the brief. |
| **Phase 1-2** | Statewide capability survey; lens-first optimisation | The 10 % plate-capable share is measured on 30 cameras. Eight of those 30 are one lens from capability. |
| **Phase 3** | Person re-identification in production; crowd-dispersal as an operator surface | Serves the wanted/missing-person watchlist categories that nothing currently serves. |
| **Phase 4** | VAHAN / CCTNS live integration; road-network routing for look-ahead | Replaces straight-line reachability with real routes. |
| **Ongoing** | Continuous re-survey and capability-drift alerting | A camera that was plate-capable and is now presence-only has been knocked or obscured. Conventional monitoring reports it healthy. |

Explicitly **not** on the roadmap: facial recognition on cameras that
cannot support it, and any affect or emotion inference at any resolution.
Both positions are argued in §5.1 rather than left as omissions.

---

## 11. Governance is architecture

Not a policy appendix — enforced in the system:

- **Case-linked access.** Every movement query carries a case reference
  and an authorising officer. Requests without them are refused before a
  row is read, and the refusal is itself logged.
- **Tamper-evident audit.** Each log entry hashes the previous one. An
  edited or deleted entry breaks the chain and is detectable. Verified by
  test: altering one row identifies exactly which.
- **No biometric face matching**, as a stated position with its reasoning,
  and enforced by measurement rather than by policy (section 5.1).

Designed, with the schema in place, but **not built** — listed here as
outstanding rather than described as if they exist:

- **Masking on export.** Uninvolved parties should be obscured in evidence
  bundles. The export path does not do this today. It is the single
  largest gap between this document and the code.

  One narrower control *is* built, and it is not the same thing. Where an
  image is published rather than exported — the stills the officer's
  console shows beside each sighting — the publication step works from an
  allow-list drawn from the sightings that survived withholding, so a row
  removed for privacy cannot leave its picture behind. That keeps one
  specific person out of a public URL. It does not obscure anybody inside
  an image a case is entitled to, which is what masking means and what is
  still missing.
- **Per-department retention.** The registry carries a `governance_class`
  per camera, and health footage is intended to attract patient
  confidentiality above ordinary privacy duty. No retention logic acts on
  that column yet, and on the sandbox grid every government camera is
  currently classed `open`.

**Licensing.** Every component is Apache-2.0 or BSD — OpenCV, Tesseract,
torchvision. Ultralytics YOLO, the default choice for work of this kind,
is AGPL, which in a state deployment is a procurement problem rather than
a footnote.

---

## 12. What is proven, and what is designed

Stated explicitly, because the next stage is production feeds in front of
the State Crime Records Bureau and anything overstated now becomes a
problem on 12 October.

**Proven, with tests** (153 passing): capability measurement, agreeing to
2.6 % against known ground truth; clock-drift detection, which caught a
deliberate 7-minute error to within 1 second; tamper-evident audit; tiered
matching and the capability ceiling; cross-camera re-identification;
look-ahead camera nomination; the officer's console; and that every route
returning movement data refuses without a case reference and an
authorising officer.

*That last one is in this list because it was false until it was audited.
`/api/officer/route` returned a citizen's full movement history to any
caller with no credentials, and the audit write beneath it was conditional
on the credentials that were missing — so the only unlogged reads were the
ones that should never have happened. Four of my own tests called it
unauthenticated and asserted on the results, which is how a suite ratifies
a hole instead of finding it. It is now gated, and swept at the HTTP layer
by a test that enumerates the application's own routes so the next one
cannot be missed the same way.*

The eight pre-submission checklist items are covered by automated tests,
including reconnect-on-feed-restart, which cannot be tested against a
government feed at all. **Those tests skip unless the local mock grid is
running**, so a default `pytest` run reports them as skipped rather than
passed. They are run with `grid/run.sh` up.

**Measured on the real grid:** camera capability and ANPR yield for all
30 cameras, no failures; transport and catalogue behaviour; per-frame
inference cost.

**Designed and demonstrated in simulation, not yet proven on real
footage:** the three person-safety layers. I have no real footage of a
crowd dispersing or a weapon being drawn, and will not claim otherwise.

---

## 13. A note on method

Five times during development, a component produced a confident wrong
answer that only revealed itself when the output was inspected rather than
trusted: a clock read as two years out; a phantom 19-second drift from a
single bad sample; a road sign measured as a 304-pixel number plate; one
vehicle counted where there were dozens; and an entire camera survey
conducted, unknowingly, on night footage.

Every one was caught by looking at pixels instead of believing a number.
That experience is the argument this submission makes about the product
itself. A platform that can state how it knows something, and under what
conditions, is worth more to a police force than one that merely answers —
because the failure mode of a confident wrong answer is an officer acting
on it.
