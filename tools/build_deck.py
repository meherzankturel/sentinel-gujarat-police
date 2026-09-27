#!/usr/bin/env python3
"""
Render the submission deck from the measurement files.

docs/deck/template.html carries the design and the words. Every number in
it is a `data-n` reference resolved at build time from what was actually
measured, so the deck cannot drift away from the system it describes.

Images are embedded as data URIs: the deck must survive being emailed,
uploaded to a portal, or printed on a machine with no network.

    python tools/build_deck.py            # -> docs/deck/index.html
    python tools/build_deck.py --pdf      # also render the PDF
"""
import base64
import json
import mimetypes
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECK = ROOT / "docs" / "deck"
TEMPLATE = DECK / "template.html"
OUT = DECK / "index.html"
PDF = DECK / "Sentinel-Presentation.pdf"

# Slide images, by the placeholder they replace.
SHOTS = {
    # Screenshots of the hosted console as a reviewer sees it, and the two
    # evidence crops the officer's screen shows beside a sighting. The
    # crops are the entrant's own vehicle on the entrant's own footage;
    # nothing here belongs to a member of the public.
    "SHOT_GIS": ROOT / ".run" / "deck" / "shot_registry.png",
    "SHOT_OFFICER": ROOT / ".run" / "deck" / "shot_officer.png",
    "SHOT_BUNDLE": ROOT / ".run" / "deck" / "shot_bundle.png",
    "SHOT_PEOPLE": ROOT / ".run" / "deck" / "shot_people.png",
    "SHOT_CAR": ROOT / "web" / "assets" / "evidence" / "own-b-16425-vehicle.jpg",
    "SHOT_PLATE": ROOT / "web" / "assets" / "evidence" / "own-b-16425-plate.jpg",
}

CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def data_uri(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def test_counts() -> dict:
    """How many tests actually pass. Claimed on the honesty slide."""
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
        m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", r.stdout)
        if m:
            return {"passed": int(m.group(1)),
                    "skipped": int(m.group(2) or 0)}
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! could not count tests ({type(e).__name__}); leaving blank")
    return {}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from tools.deck_data import build as build_data

    data = build_data()
    data["tests"] = test_counts()

    html = TEMPLATE.read_text()

    # Numbers. The replacement goes in through a lambda: measured strings
    # contain degree signs and the like, which json.dumps escapes as °,
    # and re.sub would try to read those as regex escapes and fail.
    payload = json.dumps(data, separators=(",", ":"))
    html = re.sub(r"/\*DECK_DATA\*/.*?/\*END\*/",
                  lambda _m: payload, html, flags=re.S)

    # Images.
    for token, path in SHOTS.items():
        if not path.exists():
            print(f"  ! missing image {path.relative_to(ROOT)}; slide will be blank")
            continue
        html = html.replace(token, data_uri(path))

    OUT.write_text(html)
    size = len(html.encode()) / 1_000_000
    g = data["grid"]
    print(f"wrote {OUT.relative_to(ROOT)}  ({size:.1f} MB)")
    print(f"  grid      {g['surveyed']}/{g['total']} surveyed, "
          f"1 in {g['one_in']} plate-capable, median {g['median_plate_px']}px")
    print(f"  tests     {data['tests'].get('passed','?')} passed")

    if "--pdf" in sys.argv:
        chrome = next((c for c in CHROME if pathlib.Path(c).exists()), None)
        if not chrome:
            print("  ! no Chrome/Chromium found; skipping PDF")
            return 0
        subprocess.run([chrome, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", OUT.as_uri()],
                       capture_output=True, timeout=300)
        if PDF.exists():
            print(f"wrote {PDF.relative_to(ROOT)}  "
                  f"({PDF.stat().st_size/1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
