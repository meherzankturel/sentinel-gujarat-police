# Submission index — Gujarat Police Hackathon Innovation Challenge 2026

Entrant: Meherzan Turel (individual). Problem statement: Integrated Video
Management & Analytics Platform.

Status checked **27 September 2026**, the evening before submission.
Every deliverable below exists; what is left is uploading them.

---

## The five deliverables

### 1. Solution Presentation (PPT/PDF)

| | |
|---|---|
| File | `docs/deck/Sentinel-Presentation.pdf` (2.7 MB) |
| Alt | `docs/deck/Sentinel-Presentation.pptx` (3.1 MB) |
| Exists | **YES**, rebuilt 27 Sep |
| Shape | **12 slides, light.** Cover, the four things the platform brings, the registry, the measurement, the officer's console, evidence per sighting, governance, faces and persons, scale arithmetic, the six Step 6 requirements, proven-vs-designed, thank you. |
| Caution | Every figure resolves from the measurement files at build time. Re-run `python tools/deck_data.py` before rebuilding if the database has changed. |

**Do:** upload to Drive/OneDrive, sharing **Anyone with the link — Viewer**.
To rebuild: `python tools/build_deck.py --pdf` then `python tools/build_pptx.py`.

### 2. Technical Proposal — High-Level Design

| | |
|---|---|
| File | `docs/pdf/Sentinel-High-Level-Design.pdf` |
| Source | `docs/high-level-design.md` |
| Exists | **YES**, rendered 27 Sep by `python tools/build_docs_pdf.py` |

**Do:** upload to Drive/OneDrive, **Anyone with the link — Viewer**.

### 3. Demonstration on Participant's Own Feed (screen recording, max 2–3 min)

| | |
|---|---|
| File | `docs/demo/Sentinel-Own-Feed.mp4` |
| Exists | **YES** — 2 min 54 s, under the 3-minute limit |
| Re-recorded | 27 Sep, against the current console, so the film shows each sighting carrying the vehicle and the plate crop the reader worked from. The earlier take predated them. |
| Source footage | `media/own/IMG_2044.MOV`, `IMG_2045.MOV`, `IMG_2046.MOV` (142 MB total) |
| Results behind it | `audit/own_feed.json`, `tools/ingest_own_feed.py`, §3 of `docs/output-report.md` |

**Do:** confirm the file exists and is **under 3 minutes**. Upload as an
**unlisted YouTube** video. Keep the source `.MOV` files out of the
submission — they are large and are not asked for.

### 4. Live Demonstration on Government CCTV Feed (screen recording + output report)

| | |
|---|---|
| Video | `docs/demo/Sentinel-Government-Feed.mp4` (11.9 MB, **2 min 49 s**) |
| Report | `docs/pdf/Sentinel-Output-Report.pdf`, from `docs/output-report.md` |
| Exists | **YES**, both |
| Note | Both videos are genuine screen recordings made with `python tools/record_demo.py`. |

**Do:** video as **unlisted YouTube**; report as PDF on Drive/OneDrive,
**Anyone with the link — Viewer**.

### 5. Scalability Plan (Step 6)

| | |
|---|---|
| File | `docs/pdf/Sentinel-Scalability-and-Deployment-Plan.pdf` |
| Source | `docs/scalability-plan.md` |
| Exists | **YES**, rendered 27 Sep |
| Maps to the portal's six Step 6 requirements | hardware &amp; software (§1) · AI processing capacity (§2) · network &amp; bandwidth (§3) · storage &amp; retention (§4) · disaster recovery (§6) · statewide rollout (§9). Deck slide 10 lists the same six with the number behind each. |

Covers every required heading: compute tiering (§1–2), GPU sizing (§2),
bandwidth and low-bandwidth sites (§3), hot/warm/cold retention (§4), load
balancing + horizontal scaling + monitoring + logging + health checks (§5),
HA + backup + DR (§6), cybersecurity (§7), estimated implementation and
operational cost (§8).

**Do:** upload to Drive/OneDrive, **Anyone with the link — Viewer**.

---

## Optional extras

### Source repository (optional link) — PUSHED

**https://github.com/meherzankturel/sentinel-gujarat-police**

Public, 91 files, one commit, no inherited history. A clean clone runs
`python3.12 -m venv .venv && pip install -r requirements.txt && pytest`
and reports **189 passed, 21 skipped, nothing failed** — checked on 27
Sept by cloning into an empty directory and running it, on the branch a
visitor actually lands on. Every skip names its own reason: ten need the
local mock grid started, nine need government footage that is not
redistributed, one needs a populated registry, one needs the evidence
stills. Start the grid and the working tree reports **210 passed, nothing
skipped**.

Not in it, deliberately: `media/` (government CCTV of public roads, and
the entrant's own street footage), `models/*.pth` (198 MB, over GitHub's
per-file limit, and partly derived from that footage — `models/README.md`
says where the public baseline comes from), `.run/`, `CLAUDE.md`,
`sentinel.db`, `.env`, the evidence stills under `web/assets/evidence/`,
and `audit/` — the measurement files are quoted throughout the documents
but `audit/own_feed.json` carries the withheld member of the public's
registration, so the folder stays out rather than being half-published.

Rebuilt with `python tools/publish_repo.py --push`, which copies by
allow-list and refuses to push if a forbidden path reaches the export.

To republish after changes, rebuild from the working tree rather than
patching the copy, so nothing stale survives.

### Hosted URL with test credentials (optional) — DEPLOYED

| | |
|---|---|
| URL | **https://sentinel-gujaratpolice-demo.vercel.app** |
| Username | `reviewer` |
| Password | `gujarat2026` |
| Source | `deploy/` — **`python tools/build_demo_site.py` first**: it rebuilds the data, copies the consoles and vendors the light half of `sentinel/` into the deployment. Then `cd deploy && vercel deploy --prod`, **and move the alias**: `vercel alias set <new-deployment-url> sentinel-gujaratpolice-demo.vercel.app`. A production deploy does not move it on its own — the URL above is an alias, and it was left pointing at a stale deployment once already |

Give the committee all three lines. The login is on the deployment, not on
the main codebase, which still has none — see the honesty note below.

**What it serves, and what it does not.** The registry with measured
capability, the officer's search and route, the evidence bundle, the audit
chain. These are not a hosted reimplementation: the deployment imports and
mounts `sentinel.officer` itself, so a reviewer is exercising the same code
the demonstration videos show. It does not run the analytics: PyTorch and
OpenCV are ~700 MB and want a GPU, and a serverless function has 250 MB and
none. Nothing on the site infers. The login page says so in plain words and
the hunt endpoints return 501 with the reason, so a reviewer cannot mistake
it for live inference.

**The audit chain on the hosted copy is per instance.** Searches a reviewer
runs are written to the tamper-evident log exactly as an officer's would
be — but the deployment's filesystem is read-only, so the database is
copied into the instance's `/tmp` at cold start and the entries live and die
with that instance. The chain verifies; it is not a system of record, and
nothing is meant to be read back from it days later.

**Suggested wording for the submission form**, so the scope is set before
anyone clicks:

> Hosted console (read-only) — reviewer / gujarat2026. Serves the registry,
> the officer's search and route, and the audit chain over results the
> pipeline computed offline. The analytics themselves need a GPU and are
> demonstrated in the two videos.

**The brief names facial recognition, and the console answers it.** Open a
camera in the registry and read the *For people* half of its record:
measured body height, the inter-ocular distance derived from it, and the
class that follows. The best camera on this grid gives **9.8px between the
eyes against a 40px floor** — short by four times — so the control plane
assigns face matching to none of them. The person-safety layers are gated
on the same measurement, and whether a layer is *validated* is stated
separately from whether its gate opens: crowd dispersal has its false
alarms measured on real government footage and its detection on simulation;
posture and weapon-in-hand are unvalidated and run nowhere. Measured on 4
of 30 cameras — the gateway returned 521 during the full-grid run, and the
figure claimed is the one measured.

**Try this first:** sign in, open *Officer's console*, search `GJ19BE3669`
with case `FIR-0231/2026`, officer `PSI R. Chauhan`, authorised by
`DySP M. Parmar`. That is the entrant's own vehicle on the entrant's own
footage, and each sighting carries the car and the plate crop the reader
worked from. Clear any one of the three authorisation fields and search
again — it refuses and names what is missing. That is the governance
argument in one click.

**Which grid a sighting came from is labelled on every row.** The default
route on the officer's console (`GJ01AB1234`) runs on the local mock grid,
which is how the route logic is exercised end to end; the own-feed
sightings are real footage. The console says which is which rather than
leaving a reviewer to infer it from a department name.

**Two things deliberately absent from the hosted copy**

- Four sightings of one uninvolved member of the public's car,
  parked in the entrant's own street footage. Publishing a stranger's
  registration on a public URL would contradict the submission's own
  argument. Verified absent: the plate returns zero sightings, *and* the
  evidence stills cut from those reads are withheld with them — the build
  publishes stills by allow-list from the sightings that survived, so a
  withheld row cannot leave its picture behind. Both are covered by tests.
- Any footage. No video is deployed.

**What is deployed, image-wise.** Two crops per own-feed sighting of the
entrant's own vehicle: the car as that camera saw it, and the plate patch
the reader actually worked from. Both are cut from the frame the read came
from, which is deliberately *not* the arrival time printed beside them — a
track is read from its best look, and where the read was fused across
several looks the fused image is what is shown, because that is what the
OCR read. Government and mock-grid sightings say "no image retained"
rather than showing a stand-in.

**Honesty note to keep consistent with the documents.** The HLD and the
scalability plan both state that the platform has no authentication. That
remains true of the main codebase. The hosted demo adds a signed-cookie
login because a public URL without one would publish the console to
anyone who found it — it is the right weight for a screening demo and the
wrong weight for a deployment, where it belongs behind eGujCop or
departmental AD with MFA. Do not let the existence of this login be read
as closing that gap.

---

## Before you submit

- [x] `docs/demo/Sentinel-Own-Feed.mp4` exists and runs under 3 minutes (2:54)
- [x] Documents exported to PDF → `docs/pdf/`
- [x] `python -m pytest -q` → **199 passed, 10 skipped**; with the mock
      grid started (`./grid/run.sh start`) → **210 passed, nothing
      skipped**, the organisers' checklist among them
- [ ] Every Drive/OneDrive item set to **Anyone with the link — Viewer**;
      open each link in a signed-out incognito window to prove it
- [ ] Both YouTube videos set to **Unlisted**, not Private; open each
      signed-out to prove it
- [ ] No government footage (`media/real/`) in anything uploaded or pushed
- [ ] `.env` not in any upload and not in the public repo
- [ ] Nothing anywhere claims person re-identification or crowd dispersal
      as a delivered capability — both are unfinished
- [x] Deck, HLD, output report and scalability plan agree on: 30 cameras
      surveyed, 3 plate-capable, 41 cameras in the registry, 210 tests
- [ ] Hosted URL opened in a signed-out incognito window: the login page
      appears, `reviewer` / `gujarat2026` works, and `/officer` is not
      reachable without it
- [ ] Links collected in one document and each one clicked once more
