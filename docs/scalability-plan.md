# Sentinel — Scalability & Deployment Plan

**Integrated Video Management & Analytics Platform**
Gujarat Police Hackathon Innovation Challenge 2026
Submitted by Meherzan Turel, individual entrant

Answers Step 6 (*Plan for Scale*) and the sizing, cost and deployment
areas of Step 3 (*Expected Solution Approach*). Companion to
`high-level-design.md`, which carries the architecture itself.

---

## 0. The number this plan is built on

Every figure below is driven by one measured quantity rather than an
assumed one.

I surveyed all 30 cameras on the government sandbox grid and measured
what each can actually resolve. **Three can read a number plate.** The
median vehicle presents 27.4 px of plate against the ~120 px a read
requires. On the person axis, the closest people yield 2.7–9.8 px between
the eyes against a 40 px floor for attempting face matching.

That measurement is the input to this entire plan. It is why the heavy
analytics tier is **10 % of cameras, not 100 %**, and every compute,
bandwidth, storage and cost figure below follows from it.

A plan sized on the assumption that all 80,000 cameras need continuous
heavy inference is not conservative. It is wrong by an order of magnitude,
and it prices the programme out of existence.

### On "continuously process the CCTV feeds"

The brief asks for a solution that continuously processes the feeds. This
design does — with one qualification that is the whole architecture.

Every onboarded camera is continuously processed **by the analytic its
measured capability supports**. A plate-capable camera runs continuous
ANPR. A presence-only camera runs continuous detection, tracking and
re-identification, and continuous crowd-dispersal monitoring. Nothing is
sampled, and no camera is left dark.

What the system does *not* do is run every analytic on every camera. That
is not a limitation I accept reluctantly; it is arithmetic. The
organisers' own resources page states that each connected client receives
its own copy of the stream and instructs entrants to open only the cameras
they are actively processing — a constraint that exists in a 30-camera
sandbox and does not soften at 80,000. Running plate reading on a camera
measured at 27 px of plate does not produce weak reads; it produces
confident wrong ones, which cost a control room a dispatch each.

So: continuous on every camera, and *correctly assigned* on every camera.

---

## 1. Hardware and software requirements

### Software

| Layer | Component | Licence |
|---|---|---|
| Ingest | FFmpeg / GStreamer, RTSP-TCP, HLS, ONVIF profile S/T | LGPL / BSD |
| Inference | PyTorch + torchvision detectors | BSD-3-Clause |
| OCR | Tesseract 5 | Apache-2.0 |
| Vision | OpenCV | Apache-2.0 |
| Service | FastAPI + Uvicorn | MIT / BSD |
| Registry | PostgreSQL + PostGIS (SQLite in this submission) | PostgreSQL licence |
| Queue | Redis Streams or Kafka at district tier | BSD / Apache-2.0 |
| Object store | MinIO or department S3-compatible | AGPL / vendor |

No AGPL component sits in the processing path. Ultralytics YOLO — the
default choice for work of this kind — is AGPL-3.0, which reaches the whole
deployed system and is a procurement blocker rather than a footnote. A
permissively licensed detector was chosen for that reason.

### Hardware, by tier

| Tier | Sited at | Role |
|---|---|---|
| **Edge** | Camera site / local NVR room | Decode, detect, track, discard non-events. Sends metadata and evidence clips only. |
| **District** | 33 district HQs | Aggregation, ANPR on the assigned subset, watchlist correlation, district alerting, 72-hour hot video. |
| **Central** | State data centre + DR site | Registry, statewide correlation, long-horizon search, case management, audit chain, warm/cold archive. |

The tiering is not for elegance. It is because centralising raw video is
impossible on the bandwidth arithmetic in §3, so inference must happen
where the video already is.

---

## 2. AI processing capacity

### The measured unit cost

A datacentre GPU of A100/L40S class sustains on the order of **40–60
simultaneous 1080p streams** at detection workloads in published reference
architectures. I use 50 as the planning midpoint and state the range.

*This is a vendor-published figure, not one I measured. What I did
measure is the relative cost of the two designs, which is what the
decision turns on.*

### The two designs, side by side

| | Every camera, heavy analytics | Capability-assigned |
|---|---:|---:|
| Cameras on heavy inference | 80,000 | **8,000** |
| GPUs required | ~1,600 | **~160** |
| Cameras on cheap presence detection | 0 | 72,000 |

Presence detection — background subtraction and motion-gated tracking — is
CPU work, roughly three orders of magnitude cheaper per camera than
continuous detection inference. The 72,000 cameras in that tier are served
by district CPU capacity, not by GPUs.

*Correction, kept rather than quietly dropped: an earlier draft of my
design document put the all-camera figure at 264,000 GPUs, extrapolated
from a laptop GPU at 0.33 s per frame. That overstated the requirement by
roughly 150×. A number nobody can reproduce is worth less than no number.*

### Headroom and growth

Sizing assumes the measured 10 % heavy share. The upgrade plan in §8 shows
eight cameras on the sandbox grid are within a factor of two of plate
capability — fit longer lenses and the heavy tier grows. **Capacity
planning should therefore assume the heavy share rises toward 20–25 % as
the estate is optimised, not that it stays at 10 %.** That is the correct
direction for a programme that is improving its own cameras, and it is why
the district tier is specified with GPU expansion slots rather than fixed
appliance counts.

---

## 3. Network and bandwidth planning

### Why centralisation is not available

A 1080p H.264 stream is 2–5 Mbps. Eighty thousand of them is **160–400
Gbps sustained**, statewide, every second of every day. That is not a
procurement problem; it is a physical one, and it is the single hardest
constraint in the brief.

### What actually crosses the network

| Link | Carries | Order of magnitude |
|---|---|---|
| Camera → edge | Full stream | existing local cabling |
| Edge → district | Metadata + evidence clips | **2–5 % of raw** |
| District → central | Sightings, alerts, audit entries | kilobytes per event |
| Central → operator | On-demand pull of a specific clip | bursty, small |

Sending metadata rather than video is a 95–98 % reduction. A sighting is a
few hundred bytes: camera id, corrected timestamp, tier, confidence,
bounding box, and a reference to the clip that remains at the edge until
an officer asks for it.

### Low-bandwidth and intermittent sites

Rural Panchayat cameras will not have reliable backhaul. The design
assumes they do not:

- **Store-and-forward.** Edge nodes buffer metadata locally and reconcile
  when the link returns. Nothing is lost to an outage; it arrives late and
  is marked late.
- **Degrade to events, not to nothing.** Below a bandwidth floor the edge
  stops forwarding thumbnails and forwards only structured sightings.
- **Evidence on demand.** A clip is pulled only when an officer opens that
  sighting, not pushed on the chance it matters.
- **Clock independence.** Because the registry measures and corrects clock
  drift (I found a government camera **6 h 12 m wrong**), a late-arriving
  batch still lands in the correct place on the timeline.

---

## 4. Storage and retention strategy

### The raw arithmetic

4 Mbps continuous is ~43 GB per camera per day. Across 80,000 cameras that
is **3.4 PB per day**, and **103 PB at 30-day retention**. Full retention
of raw video statewide is not viable at any plausible budget.

### Tiered retention

| Tier | Holds | Retention | Where | Access |
|---|---|---|---|---|
| **Hot** | Raw video, rolling | **72 hours** | Edge / district NVMe + HDD | Instant scrub |
| **Warm** | Evidence clips attached to a case; alerted sightings | **1 year** | District object store | Seconds |
| **Cold** | Case-bound evidence under legal hold; archived audit chain | **7 years**, or as the case requires | Central archive / tape | Minutes to hours |
| **Indefinite** | Metadata: sightings, alerts, capability measurements, audit entries | No expiry | Central PostgreSQL | Instant, searchable |

The inversion is deliberate: **video expires, metadata does not.** A
sighting record is a few hundred bytes and remains searchable for years; the
video behind it survives 72 hours unless something made it evidence.

### Storage volume under this policy

| Tier | Calculation | Volume |
|---|---|---|
| Hot | 80,000 × 43 GB × 3 days | ~10.3 PB rolling, distributed across districts |
| Warm | assume 0.1 % of sightings become evidence clips | tens of TB per year |
| Cold | case-bound only | low TB per year |
| Metadata | ~500 bytes × est. 2 bn sightings/year | ~1 TB per year |

Hot storage is the dominant cost and it is the one that is distributed —
roughly 310 TB per district across 33 districts, which is a rack, not a
data centre.

### Per-department retention

Retention is a property of the owning department, not a global setting.
Health footage carries patient confidentiality above ordinary privacy duty.
The registry holds a `governance_class` per camera for exactly this, and
retention policy binds to it.

*Stated plainly: the `governance_class` column exists and is populated; the
retention enforcement that acts on it is designed here and **not yet
built**. It is the largest gap between this plan and the running code.*

---

## 5. Load balancing, horizontal scaling, monitoring

### Scaling shape

- **Edge and district tiers scale horizontally by camera count.** A district
  worker owns a disjoint set of cameras; adding cameras adds workers. There
  is no shared mutable state between workers, so this scales linearly.
- **Assignment is the load balancer.** The registry already decides which
  camera gets which analytic; the same decision distributes work across
  workers by cost class, so a district does not end up with all its
  plate-capable cameras on one node.
- **The central tier scales read-heavy**, with PostgreSQL read replicas for
  search and a single writer for the audit chain. The chain must remain
  serialised — a forked hash chain reports as tampered for the rest of its
  life — so audit writes are deliberately not distributed.

### Monitoring, logging, health checks

The registry already measures camera condition; monitoring extends that
rather than bolting on a separate system.

| Signal | Source | Acts on |
|---|---|---|
| Camera reachable, delivering frames | Ingest worker | Registry `last_seen_live` |
| Delivered vs declared frame rate | PTS measurement | Health score; alerts on divergence |
| PTS gaps, decoder errors | Ingest worker | Stream quality |
| **Capability drift** | Periodic re-survey | Alerts if a camera's measured plate width falls — it has been knocked, refocused or obscured |
| Inference queue depth, GPU utilisation | District workers | Autoscale / alert |
| Audit chain integrity | Scheduled verification | Security alert on break |

**Capability drift is the one worth emphasising.** A camera that was
plate-capable and is now presence-only has physically moved or been
obstructed. Conventional monitoring reports it as healthy — it is online
and delivering frames. Only a system that measures capability notices.

---

## 6. High availability, backup, disaster recovery

| Concern | Approach | Target |
|---|---|---|
| Edge node failure | Cameras re-home to a neighbouring node; degraded coverage, not lost coverage | < 5 min |
| District failure | Standby district takes assigned cameras; hot video for that district is lost, metadata is not | RTO < 1 h, RPO ~0 for metadata |
| Central failure | Warm standby at a second site; streaming replication of the registry | RTO < 4 h, RPO < 5 min |
| Network partition | Districts operate autonomously and reconcile on restore | Degraded, not down |
| Data loss | Registry backed up continuously; audit chain replicated and independently verifiable | RPO < 5 min |

**The audit chain is the one thing that must never be silently restored.**
A backup restore that rewinds the chain creates a gap that is
indistinguishable from tampering. So chain segments are replicated
append-only and verified after any restore, and a verification failure is
an incident rather than a log line.

**Graceful degradation, in order of what is given up:**
1. Face/person analytics (already unassigned on this grid)
2. Cross-camera re-identification
3. ANPR on occasional-yield cameras
4. Presence detection
5. **Last to go: alerting on confirmed plate matches from plate-capable
   cameras**, because that is the capability the programme exists for.

---

## 7. Cybersecurity architecture

| Control | Implementation | Status |
|---|---|---|
| Transport | TLS on every hop; RTSP forced over TCP | Built |
| Credential handling | Environment-only, never in source or config committed to VCS | Built |
| Case-linked access | No movement query answered without case ref + searching officer + authorising officer, refused before a row is read | **Built and tested** |
| Tamper-evident audit | SHA-256 chain; any edit breaks every subsequent hash | Built and tested |
| **Role-based access control** | Officer / supervisor / auditor / administrator, with query scope and export rights bound to role | **Designed, not built** |
| **Authentication** | Integration with departmental identity (eGujCop / AD), MFA for export and watchlist edit | **Designed, not built** |
| Network segmentation | Camera VLANs isolated from corporate; district tier in a DMZ | Design |
| Masking on export | Uninvolved parties obscured in evidence bundles | **Designed, not built** |
| Data minimisation | Metadata retained, video expired; see §4 | Partly built |

**Stated without softening: there is currently no authentication and no
RBAC.** The case-linked gate is real, tested, and refuses correctly — but
it trusts the officer name it is given. In deployment that gate must sit
behind departmental identity, and the roles above must bind to it. Shipping
the gate without the identity behind it would be security theatre, and I
would rather name it here than let a reviewer discover it.

---

## 8. Estimated implementation and operational costs

**These are order-of-magnitude planning figures for comparing designs, not
a tender response.** Hardware pricing varies with procurement route,
volume and year; every figure below should be validated against current
GeM rates before any commitment. They are included because the brief asks
for them and because the *ratio* between the two designs is robust even
where the absolute numbers are not.

**Assumptions, stated per line so each can be disputed separately.** One
*node* is one datacentre-class GPU plus its host server: an A100/L40S-class
card is roughly ₹8–13 lakh and the server around ₹8–12 lakh, so **₹20 lakh
per node** is the midpoint of a defensible range. Continuous draw 700 W per
node (GPU ~400 W, host ~300 W) at ₹8/kWh. Hot storage at **₹12,000 per TB
installed** — redundancy, chassis and controllers included, not the price
of a bare drive. District tier reuses existing department server rooms.

| Item | Every-camera design | Capability-assigned | Note |
|---|---:|---:|---|
| GPU nodes | ~1,600 | **~160** | Driven by the measured 10 % share |
| GPU capital | **₹320 cr** | **₹32 cr** | at ₹20 lakh/node |
| GPU power, per year | ₹7.8 cr | **₹0.8 cr** | 700 W continuous |
| Hot storage, 10.3 PB | ₹12.4 cr | ₹12.4 cr | **identical** — a function of camera count, not of analytic |
| Network | statewide 160–400 Gbps backhaul | district-local only | the larger hidden cost, not priced here |
| **Indicative capital** | **~₹332 cr** | **~₹44 cr** | |

**The difference is ~₹288 crore of capital, and roughly ₹7 crore a year in
power**, and it rests on one measurement that took a day: surveying 30
cameras and finding that three of them can read a plate.

Storage is deliberately shown as identical in both columns. Hot video
depends on how many cameras exist, not on which analytic runs over them, so
including it in the "saving" would inflate the comparison. The saving is the
GPU line, and only the GPU line.

*An earlier draft of this table was out by a factor of ten on every row —
₹1,600 cr against ₹160 cr, power at ₹52 cr, storage at ₹41 cr — because the
per-unit assumptions and the totals were written at different times and
never reconciled. Every one of those errors inflated the saving being
claimed. It is corrected here rather than quietly restated, and the figures
are now computed in `tools/deck_data.py` from the per-unit constants above,
so the deck and this document cannot drift apart again.*

### The cheapest capability available

The upgrade plan (`tools/plan_upgrades.py`) identifies, per camera, what
would have to change for it to do the job:

| Verdict | Cameras (of 30) |
|---|---:|
| Already plate-capable | 3 |
| **A longer lens or a different aim** | **8** |
| Move it, or add one closer | 8 |
| New camera, new site | 4 |
| No vehicles pass it — wrong question | 7 |

**Eight cameras are within a factor of two of plate capability.** Fitting
longer lenses takes the grid from 3 plate-capable cameras to 11, of the 23
that watch a road at all — for the cost of eight lenses rather than eight
cameras. Extrapolated statewide, a lens-first optimisation programme is
the highest return per rupee available to this deployment, and it is
invisible to anyone who has not measured.

---

## 9. Statewide rollout plan

| Phase | Scope | Duration | Exit criterion |
|---|---|---|---|
| **0 — Proof** | ~50 sandbox cameras | Complete | Live test, 12–13 Oct |
| **1 — District pilot** | One district, all departments, ~2,000 cameras | 3 months | Survey complete; assignment stable; alert budget holding |
| **2 — Optimise before expanding** | Lens/aim programme on pilot district | 2 months | Measured plate-capable share risen; cost per confirmed sighting falling |
| **3 — Regional** | 8 districts, ~15,000 cameras | 6 months | District tier autonomous under partition |
| **4 — Statewide** | 33 districts, ~80,000 cameras | 18 months | Federation across all 26 departments |
| **5 — Steady state** | Continuous re-survey | Ongoing | Capability drift detected and corrected |

**Phase 2 is the unusual one and it is deliberate.** Most rollouts expand
camera count; this one pauses to improve the cameras already onboarded,
because a lens is two orders of magnitude cheaper than a camera and the
measurement says which lenses. Expanding before optimising multiplies the
number of cameras that cannot answer the question being asked of them.

### Department-wise information requirements

To onboard a department I need, per camera, only what the registry cannot
measure for itself:

| Required | Why | If unavailable |
|---|---|---|
| Stream endpoint + credentials | To connect at all | **Blocking** |
| Owning department | Governance class, retention, access | **Blocking** |
| Site location (lat/lon or address) | GIS, route reconstruction, look-ahead | Degrades to non-spatial |
| Contact for physical access | Lens/aim changes in phases 2 | Blocks optimisation only |
| VMS make/model + API or ONVIF profile | Federation adapter selection | Falls back to RTSP/HLS |
| Existing retention policy | Reconciling mine with theirs | Assume department default |

Deliberately **not** required: resolution, frame rate, codec, field of
view, mounting height, focal length, or any capability claim. The system
measures all of those from the video, because on the sandbox grid the
catalogue's own declared values were wrong — one camera declares 250 fps in
its container header while reporting 29.96 in another field of the same
file. Asking 26 departments for specification data that turns out to be
unreliable would delay onboarding and then mislead the assignment layer.

**This is the single biggest reduction in onboarding friction the design
offers:** a department supplies an endpoint and an owner, and the platform
works out the rest.

---

## 10. What this plan assumes, and what would change it

| Assumption | If wrong |
|---|---|
| 40–60 1080p streams per datacentre GPU | Linear effect on GPU count; ratio between designs unchanged |
| Measured 10 % plate-capable share holds statewide | Sandbox is 30 cameras. A statewide survey is phase 1's first task, and **the share is the one number this plan is most sensitive to** |
| 0.1 % of sightings become evidence | Warm storage scales linearly; still small against hot |
| District server rooms can host the district tier | Otherwise add ~33 small facilities |
| Backhaul exists to district HQs | Phase 3 gating item; store-and-forward covers gaps, not absence |

The honest summary: the architecture is robust to most of these being
somewhat wrong, and sensitive to one — the statewide capability
distribution. Which is why the first action in phase 1 is to measure it,
not to assume that 30 cameras in a sandbox represent 80,000 across
26 departments.
