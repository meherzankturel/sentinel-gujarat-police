#!/usr/bin/env python3
"""
live_audit.py — probe the Sentinel live camera grid and produce registry seed data.

There is no file download. Every camera is a live RTSP stream, so this
connects to each one in turn, samples it for a few seconds, measures what
it can actually deliver, and disconnects.

Usage:
    python3 live_audit.py --host <host>
    python3 live_audit.py --host <host> --seconds 25 --out audit
    python3 live_audit.py --host <host> --only 1,4,7      # specific camera ids
    python3 live_audit.py --host <host> --hls             # if port 8554 is blocked

Outputs (into ./audit/):
    cameras.csv     one row per camera: the registry seed
    summary.txt     headline numbers
    frames/         sample frames, for measuring plate pixels by hand
    osd/            top and bottom strips, for reading burned-in clocks
    raw_catalogue.json

Design notes, straight from the organiser's do's and don'ts:
  - RTSP is forced over TCP.
  - Declared frame rate is recorded but never trusted. Real rate is
    measured from PTS deltas.
  - The gateway replays a buffered group-of-pictures on connect, so the
    first second or two arrives faster than real time. That burst is
    discarded before measuring rate, otherwise every camera looks fast.
  - Inter-frame gaps are recorded, not treated as disconnects.
  - Decoder warnings at join are expected and non-fatal.
  - Cameras are opened one at a time and closed immediately. Each client
    gets its own copy of the stream, so we do not hold 30 open at once.
"""

import os
# Must be set before cv2 is imported, or it has no effect.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000")

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Frames arriving in the join burst are not representative of real cadence.
JOIN_BURST_SECONDS = 2.5
# A PTS jump larger than this counts as a coverage gap worth recording.
GAP_THRESHOLD_MS = 500


def fetch_catalogue(host, timeout=20):
    """The catalogue is the contract. Never hard-code stream URLs."""
    url = host.rstrip("/") + "/api/ingest"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace")), url


def extract_cameras(catalogue):
    """
    The exact shape of /api/ingest is not documented, so accept the
    common variants rather than assuming one. raw_catalogue.json is
    written out so you can check what actually came back.
    """
    if isinstance(catalogue, list):
        return catalogue
    if isinstance(catalogue, dict):
        for key in ("cameras", "streams", "items", "data", "results", "ingest"):
            val = catalogue.get(key)
            if isinstance(val, list):
                return val
        # A dict keyed by camera id
        if all(isinstance(v, dict) for v in catalogue.values()) and catalogue:
            out = []
            for k, v in catalogue.items():
                v = dict(v)
                v.setdefault("id", k)
                out.append(v)
            return out
    return []


def pick(d, *names, default=""):
    """Pull the first present key from a camera record, case-insensitively."""
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] not in (None, ""):
            return low[n.lower()]
    return default


def stream_url(cam, host, prefer_hls=False):
    """Prefer a URL the catalogue gave us; fall back to the documented pattern."""
    if prefer_hls:
        u = pick(cam, "hls", "hls_url", "m3u8")
        if u:
            return u
        return f"http://{host}/live/stream/{pick(cam, 'id', 'camera_id', 'name')}/index.m3u8"
    u = pick(cam, "rtsp", "rtsp_url", "url", "stream_url")
    if u and str(u).startswith("rtsp://"):
        return u
    return f"rtsp://{host}:8554/stream/{pick(cam, 'id', 'camera_id', 'name')}"


def measure_stream(url, seconds, frames_dir, osd_dir, stem):
    """
    Connect, sample, disconnect. All timing comes from PTS
    (CAP_PROP_POS_MSEC), never from arrival time.
    """
    result = {
        "connected": False,
        "time_to_first_frame_s": None,
        "frames_received": 0,
        "measured_fps": None,
        "declared_fps": None,
        "width": None, "height": None,
        "pts_span_s": None,
        "gap_count": None, "gap_max_ms": None,
        "read_failures": 0,
        "brightness_mean": None, "dark_frame_share": None,
        "sharpness_mean": None, "sharpness_min": None,
        "error": "",
    }

    t_open = time.monotonic()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        result["error"] = "could not open"
        return result

    result["connected"] = True
    declared = cap.get(cv2.CAP_PROP_FPS)
    result["declared_fps"] = round(declared, 3) if declared and declared > 0 else None

    pts_list = []
    brightness, sharpness = [], []
    saved = 0
    first_frame_at = None
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            # An inter-frame gap is not a disconnect. Keep waiting.
            result["read_failures"] += 1
            if result["read_failures"] > 240:
                result["error"] = "stream stalled"
                break
            time.sleep(0.02)
            continue

        if first_frame_at is None:
            first_frame_at = time.monotonic()
            result["time_to_first_frame_s"] = round(first_frame_at - t_open, 2)
            result["height"], result["width"] = frame.shape[0], frame.shape[1]

        pts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pts and pts > 0:
            pts_list.append(float(pts))
        result["frames_received"] += 1

        # Measure every 5th frame; decoding is the cost, analysis is cheap.
        if result["frames_received"] % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            if saved < 3 and result["frames_received"] > 25:
                cv2.imwrite(str(frames_dir / f"{stem}_s{saved}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                h = gray.shape[0]
                band = max(24, h // 12)
                for edge, strip in (("top", frame[:band, :]),
                                    ("bottom", frame[-band:, :])):
                    cv2.imwrite(str(osd_dir / f"{stem}_osd{saved}_{edge}.jpg"), strip)
                saved += 1

    cap.release()

    # --- rate and continuity, from PTS only ---
    if len(pts_list) > 10:
        pts_sorted = sorted(pts_list)
        origin = pts_sorted[0]
        # Drop the join burst: the gateway replays buffered frames on connect,
        # so the opening seconds are not real-time cadence.
        usable = [p for p in pts_sorted if p - origin >= JOIN_BURST_SECONDS * 1000]
        if len(usable) > 10:
            span_ms = usable[-1] - usable[0]
            if span_ms > 0:
                result["measured_fps"] = round((len(usable) - 1) / (span_ms / 1000.0), 2)
                result["pts_span_s"] = round(span_ms / 1000.0, 1)

            deltas = [b - a for a, b in zip(usable, usable[1:])]
            gaps = [d for d in deltas if d > GAP_THRESHOLD_MS]
            result["gap_count"] = len(gaps)
            result["gap_max_ms"] = round(max(gaps), 1) if gaps else 0

    if brightness:
        result["brightness_mean"] = round(statistics.fmean(brightness), 1)
        result["dark_frame_share"] = round(
            sum(1 for b in brightness if b < 45) / len(brightness), 3)
    if sharpness:
        result["sharpness_mean"] = round(statistics.fmean(sharpness), 1)
        result["sharpness_min"] = round(min(sharpness), 1)

    return result


def classify_capability(width, height, sharpness_mean, dark_share):
    """
    Provisional only. Resolution cannot tell you whether a plate is
    readable, because that depends on how far the camera sits from the
    road. Confirm by measuring plate width on the saved frames:
    roughly 100-150 px of plate width to read reliably, roughly 80 px
    between the eyes for face work.
    """
    if not width or not height:
        return "unknown"

    pixels = width * height
    if pixels < 640 * 480:
        cls = "presence-only"
    elif pixels < 1280 * 720:
        cls = "presence-likely"
    elif pixels < 1920 * 1080:
        cls = "plate-possible"
    else:
        cls = "plate-likely"

    notes = []
    if sharpness_mean is not None and sharpness_mean < 40:
        notes.append("soft")
        if cls in ("plate-likely", "plate-possible"):
            cls = "presence-likely"
    if dark_share is not None and dark_share > 0.5:
        notes.append("mostly-dark")

    return cls + (" (" + ",".join(notes) + ")" if notes else "")


def audit_camera(cam, host, args, frames_dir, osd_dir):
    cam_id = str(pick(cam, "id", "camera_id", "name", default="unknown"))
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in cam_id)[:60]
    url = stream_url(cam, host, prefer_hls=args.hls)

    m = measure_stream(url, args.seconds, frames_dir, osd_dir, stem)

    fps_gap = None
    if m["declared_fps"] and m["measured_fps"]:
        fps_gap = round(m["declared_fps"] - m["measured_fps"], 2)

    return {
        "camera_id": cam_id,
        "url": url,
        "catalogue_live": pick(cam, "live", "status", "state", default=""),
        "location": pick(cam, "location", "place", "area", "site", default=""),
        "latitude": pick(cam, "lat", "latitude", default=""),
        "longitude": pick(cam, "lon", "lng", "longitude", default=""),
        "department": pick(cam, "department", "dept", "owner", default=""),
        "codec_catalogue": pick(cam, "codec", "video_codec", default=""),
        "connected": m["connected"],
        "error": m["error"],
        "time_to_first_frame_s": m["time_to_first_frame_s"],
        "width": m["width"],
        "height": m["height"],
        "declared_fps": m["declared_fps"],
        "measured_fps": m["measured_fps"],
        "fps_gap": fps_gap,
        "frames_received": m["frames_received"],
        "pts_span_s": m["pts_span_s"],
        "gap_count": m["gap_count"],
        "gap_max_ms": m["gap_max_ms"],
        "read_failures": m["read_failures"],
        "brightness_mean": m["brightness_mean"],
        "dark_frame_share": m["dark_frame_share"],
        "sharpness_mean": m["sharpness_mean"],
        "sharpness_min": m["sharpness_min"],
        "capability_provisional": classify_capability(
            m["width"], m["height"], m["sharpness_mean"], m["dark_frame_share"]),
        # Filled in by hand after looking at the saved images:
        "plate_px_measured": "",
        "capability_confirmed": "",
        "osd_time_read": "",
        "wallclock_at_read_utc": "",
        "time_confidence": "",
        "notes": "",
    }


def write_summary(rows, out_dir):
    n = len(rows)
    L = [f"Cameras in catalogue: {n}"]

    ok = [r for r in rows if r["connected"] and r["frames_received"] > 0]
    L.append(f"Connected and delivering frames: {len(ok)} of {n}")
    dead = [r for r in rows if not r["connected"] or r["frames_received"] == 0]
    for r in dead[:15]:
        L.append(f"   down: {r['camera_id']:<24} {r['error'] or 'no frames'}")

    res = {}
    for r in ok:
        if r["width"]:
            k = f"{r['width']}x{r['height']}"
            res[k] = res.get(k, 0) + 1
    L += ["", "Resolutions actually delivered:"]
    for k, v in sorted(res.items(), key=lambda x: -x[1]):
        L.append(f"   {k:<12} {v}")

    L += ["", "Declared vs measured frame rate:"]
    liars = [r for r in ok if r["fps_gap"] is not None and abs(r["fps_gap"]) >= 2]
    L.append(f"   cameras off by 2 fps or more: {len(liars)} of {len(ok)}")
    for r in sorted(liars, key=lambda x: -abs(x["fps_gap"]))[:12]:
        L.append(f"   {r['camera_id']:<24} declared {r['declared_fps']}"
                 f"  measured {r['measured_fps']}  ({r['fps_gap']:+})")

    gappy = [r for r in ok if r["gap_count"]]
    L += ["", f"Cameras with inter-frame gaps over {GAP_THRESHOLD_MS} ms: "
              f"{len(gappy)} of {len(ok)}"]
    for r in sorted(gappy, key=lambda x: -(x["gap_max_ms"] or 0))[:12]:
        L.append(f"   {r['camera_id']:<24} {r['gap_count']} gaps, "
                 f"worst {r['gap_max_ms']} ms")

    L += ["", "Provisional capability (confirm by measuring plate pixels):"]
    caps = {}
    for r in ok:
        caps[r["capability_provisional"]] = caps.get(r["capability_provisional"], 0) + 1
    for k, v in sorted(caps.items(), key=lambda x: -x[1]):
        L.append(f"   {k:<28} {v}")

    soft = [r for r in ok if r["sharpness_mean"] is not None and r["sharpness_mean"] < 40]
    dark = [r for r in ok if r["dark_frame_share"] is not None and r["dark_frame_share"] > 0.5]
    L += ["", f"Soft or obstructed imagery: {len(soft)} of {len(ok)}",
          f"Dark in most sampled frames: {len(dark)} of {len(ok)}"]

    L += ["", "Next, by hand:",
          "  1. Open audit/frames/ and measure plate width in pixels per camera.",
          "  2. Open audit/osd/ and read each burned-in clock. Compare against the",
          "     UTC time you ran this and record the difference in time_confidence.",
          "  3. Fill capability_confirmed. That column is the registry."]

    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text)
    return text


def preflight(test_url):
    """
    Verify this machine's OpenCV can actually open a network stream before
    we walk 30 cameras. Some OpenCV builds ship without network protocol
    support, which fails identically to a dead camera and wastes an hour.
    """
    import subprocess
    print(f"Preflight against {test_url}")
    cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    ok = False
    if opened:
        ok, _ = cap.read()
    cap.release()
    if opened and ok:
        print("  OpenCV can open and read this stream. Proceeding.\n")
        return True

    print("  OpenCV could NOT read this stream. Diagnosing...")
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-show_entries", "stream=codec_name,width,height",
             "-of", "csv=p=0", test_url],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            print(f"  ffprobe CAN read it: {r.stdout.strip()}")
            print("  So the network and the camera are fine, and the problem is")
            print("  your OpenCV build lacking network protocol support.")
            print("  Fix:  pip uninstall opencv-python-headless")
            print("        pip install --force-reinstall opencv-python")
        else:
            print(f"  ffprobe also failed: {(r.stderr or '').strip()[:300]}")
            print("  So this is network or camera side. Check that you can reach")
            print("  the host, that port 8554 is not blocked (try --hls), and")
            print("  that the camera shows as live in /api/ingest.")
    except Exception as e:
        print(f"  ffprobe could not be run: {e}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="",
                    help="gateway host, e.g. 10.0.0.5 or sandbox.example.gov.in")
    ap.add_argument("--seconds", type=int, default=20,
                    help="sampling window per camera (default 20)")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--only", default="",
                    help="comma-separated camera ids to probe")
    ap.add_argument("--hls", action="store_true",
                    help="use HLS instead of RTSP (if port 8554 is blocked)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="probe every camera even if the first one fails")
    ap.add_argument("--selftest", default="",
                    help="run the measurement logic against a local video file")
    args = ap.parse_args()
    if not args.host and not args.selftest:
        ap.error("--host is required (or use --selftest FILE)")

    if args.selftest:
        out = Path(args.out); f, o = out/"frames", out/"osd"
        for d in (out, f, o): d.mkdir(parents=True, exist_ok=True)
        r = measure_stream(args.selftest, args.seconds, f, o, "selftest")
        for k, v in r.items(): print(f"  {k:<24} {v}")
        print("\ncapability:", classify_capability(
            r["width"], r["height"], r["sharpness_mean"], r["dark_frame_share"]))
        return

    host = args.host.replace("http://", "").replace("https://", "").rstrip("/")

    out_dir = Path(args.out)
    frames_dir, osd_dir = out_dir / "frames", out_dir / "osd"
    for d in (out_dir, frames_dir, osd_dir):
        d.mkdir(parents=True, exist_ok=True)

    try:
        catalogue, url = fetch_catalogue(host)
    except Exception as e:
        sys.exit(f"Could not read the catalogue at {host}/api/ingest — {e}\n"
                 f"Check the host, and that you are on a network permitted to reach it.")

    (out_dir / "raw_catalogue.json").write_text(json.dumps(catalogue, indent=2))
    cameras = extract_cameras(catalogue)
    if not cameras:
        sys.exit(f"Catalogue read from {url} but no camera list recognised.\n"
                 f"Look at {out_dir}/raw_catalogue.json and adjust extract_cameras().")

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cameras = [c for c in cameras
                   if str(pick(c, "id", "camera_id", "name")) in wanted]

    print(f"Catalogue: {len(cameras)} cameras. "
          f"Sampling {args.seconds}s each, one at a time.\n")

    if not args.skip_preflight and cameras:
        if not preflight(stream_url(cameras[0], host, prefer_hls=args.hls)):
            sys.exit("\nStopping before the full run. Fix the above, or pass "
                     "--skip-preflight to probe anyway and record the failures.")

    rows = []
    for i, cam in enumerate(cameras, 1):
        cam_id = pick(cam, "id", "camera_id", "name", default="?")
        print(f"[{i}/{len(cameras)}] {cam_id} ", end="", flush=True)
        try:
            row = audit_camera(cam, host, args, frames_dir, osd_dir)
            rows.append(row)
            if row["frames_received"]:
                print(f"ok  {row['width']}x{row['height']}  "
                      f"declared {row['declared_fps']}  measured {row['measured_fps']}")
            else:
                print(f"NO FRAMES  {row['error']}")
        except Exception as e:
            print(f"failed: {e}")
        # Be a good citizen: the brief says pace your load.
        time.sleep(0.5)

    if not rows:
        sys.exit("No cameras could be probed.")

    csv_path = out_dir / "cameras.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 62)
    print(write_summary(rows, out_dir))
    print("=" * 62)
    print(f"\nRegistry seed: {csv_path}")
    print(f"Ran at (UTC):  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}"
          f"   <- compare the OSD clocks against this")


if __name__ == "__main__":
    main()
