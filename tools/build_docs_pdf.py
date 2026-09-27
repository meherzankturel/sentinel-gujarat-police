#!/usr/bin/env python3
"""
Render the written deliverables as PDFs a panel can actually open.

The design document, the output report and the scalability plan are
written in Markdown because that is the sane way to keep prose and
measured tables under version control. A screening committee should not
be handed a `.md` file, so this renders each one through the same light
identity the console and the deck use — one system, three surfaces.

Nothing here rewrites content. The Markdown is the source; this is
typesetting, and it is repeatable, so a corrected document is one command
away from a corrected PDF.

    python tools/build_docs_pdf.py
"""
import pathlib
import re
import subprocess
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "pdf"

# Rendered in the order a reviewer meets them.
DOCS = [
    ("high-level-design.md", "Sentinel — High-Level Design",
     "Technical proposal · solution architecture"),
    ("output-report.md", "Sentinel — Output Report",
     "Results of the demonstration runs, with their sources"),
    ("scalability-plan.md", "Sentinel — Scalability & Deployment Plan",
     "Step 6 · plan for scale to ~80,000 cameras"),
]

CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

# The console's palette, set for paper. Long documents are read at length,
# so the measure is narrow, the leading is generous, and tables carry
# hairlines rather than fills — a police reviewer prints these.
CSS = """
:root{
  --paper:#FFFFFF; --panel:#FBFAF7; --line:#E4E1D9;
  --ink:#15181D; --mist:#4E5763; --dim:#8A9098;
  --teal:#0E9E83; --amber:#B8780F; --alarm:#C62A3C;
  --sans:"Archivo",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--sans);font-size:10.5pt;line-height:1.62;
  -webkit-font-smoothing:antialiased}
main{max-width:none}

/* ── the cover page ─────────────────────────────────────────────────── */
.cover{height:247mm;display:flex;flex-direction:column;
  page-break-after:always;break-after:page}
.cover .mark{font-weight:800;font-size:11pt;letter-spacing:.3em;
  text-transform:uppercase}
.cover .mark i{font-style:normal;color:var(--teal)}
.cover h1{margin:34mm 0 0;font-size:34pt;line-height:1.06;font-weight:800;
  letter-spacing:-.03em;max-width:16ch}
.cover .sub{margin-top:14px;font-size:13pt;color:var(--mist);max-width:44ch}
.cover .meta{margin-top:auto;border-top:1px solid var(--line);padding-top:16px;
  display:grid;grid-template-columns:1fr 1fr;gap:12px 24px}
.cover .meta span{display:block;font-size:7.5pt;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim)}
.cover .meta b{display:block;margin-top:4px;font-size:10.5pt;font-weight:600}

/* ── headings ───────────────────────────────────────────────────────── */
h1,h2,h3,h4{line-height:1.2;font-weight:700;letter-spacing:-.015em}
main > h1{font-size:19pt;margin:0 0 4px;padding-bottom:0}
h2{font-size:15pt;margin:26px 0 10px;padding-top:14px;
  border-top:1px solid var(--line);page-break-after:avoid;break-after:avoid}
h3{font-size:12pt;margin:18px 0 6px;page-break-after:avoid;break-after:avoid}
h4{font-size:10.5pt;margin:14px 0 4px;color:var(--mist)}
p{margin:0 0 10px}
strong{font-weight:600}
a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--line)}

ul,ol{margin:0 0 12px;padding-left:20px}
li{margin:0 0 5px}
li::marker{color:var(--dim)}

blockquote{margin:12px 0;padding:10px 16px;background:var(--panel);
  border-left:2px solid var(--teal);color:var(--mist)}
blockquote p:last-child{margin-bottom:0}

hr{border:0;border-top:1px solid var(--line);margin:20px 0}

code{font-family:var(--mono);font-size:9pt;background:var(--panel);
  padding:1px 4px;border-radius:2px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:3px;
  padding:12px 14px;overflow:hidden;margin:0 0 12px;
  page-break-inside:avoid;break-inside:avoid}
pre code{background:none;padding:0;font-size:8.5pt;line-height:1.5;
  white-space:pre-wrap;word-break:break-word}

/* ── tables: the measured half of every one of these documents ─────── */
table{width:100%;border-collapse:collapse;margin:12px 0 16px;font-size:9.5pt;
  page-break-inside:avoid;break-inside:avoid}
th{text-align:left;font-size:7.5pt;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);font-weight:600;padding:0 10px 7px 0;
  border-bottom:1px solid var(--line);vertical-align:bottom}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--line);
  vertical-align:top;line-height:1.5}
tr:last-child td{border-bottom:0}
td code,th code{font-size:8.5pt}

img{max-width:100%}

@page{size:A4;margin:22mm 20mm 20mm}
@media print{
  h2,h3{page-break-after:avoid;break-after:avoid}
  p,li{orphans:3;widows:3}
}
"""


def render(md_name: str, title: str, subtitle: str) -> pathlib.Path:
    src = ROOT / "docs" / md_name
    text = src.read_text()

    # The Markdown carries its own H1; the cover states it instead, so the
    # first line is dropped rather than printed twice.
    text = re.sub(r"\A#\s+.*\n", "", text, count=1)

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body>
<section class="cover">
  <div class="mark">Sentinel<i>.</i></div>
  <h1>{title.split('—')[1].strip()}</h1>
  <div class="sub">{subtitle}</div>
  <div class="meta">
    <div><span>Problem statement</span><b>Integrated Video Management
      &amp; Analytics Platform</b></div>
    <div><span>Submitted by</span><b>Meherzan Turel · individual entrant</b></div>
    <div><span>For</span><b>Gujarat Police Hackathon Innovation Challenge 2026 · State Crime
      Records Bureau</b></div>
    <div><span>Source</span><b>docs/{md_name}</b></div>
  </div>
</section>
<main>
{html_body}
</main>
</body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    stem = title.split("—")[1].strip().replace(" ", "-").replace("&", "and")
    html_path = OUT / f"{stem}.html"
    pdf_path = OUT / f"Sentinel-{stem}.pdf"
    html_path.write_text(page)

    chrome = next((c for c in CHROME if pathlib.Path(c).exists()), None)
    if not chrome:
        sys.exit("no Chrome/Chromium found; cannot render PDFs")
    subprocess.run([chrome, "--headless", "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   capture_output=True, timeout=300)
    html_path.unlink()
    return pdf_path


def main() -> int:
    for md_name, title, subtitle in DOCS:
        pdf = render(md_name, title, subtitle)
        if not pdf.exists():
            print(f"  ! {md_name} did not render")
            continue
        print(f"wrote {pdf.relative_to(ROOT)}  "
              f"({pdf.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
