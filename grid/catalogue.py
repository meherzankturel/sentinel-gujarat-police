#!/usr/bin/env python3
"""
catalogue.py -- local stand-in for GET /api/ingest.

Mirrors the documented contract: "returns every camera with its id,
location, codec, live status, stream properties, and all three URLs."

Live status is not hard-coded. It is read from MediaMTX's own API, so a
camera reports live only while something is genuinely publishing to it.
That means killing a publisher is visible here exactly as it would be on
the government grid -- which is what lets us test reconnect behaviour.

    python3 grid/catalogue.py            # serves on :8080
    curl -s http://127.0.0.1:8080/api/ingest | jq .
"""

import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MTX_API = "http://127.0.0.1:9997/v3/paths/list"

_defs = json.loads((ROOT / "cameras.json").read_text())


def live_paths():
    """Ask MediaMTX which paths currently have a publisher."""
    try:
        with urllib.request.urlopen(MTX_API, timeout=3) as r:
            data = json.loads(r.read().decode())
        out = {}
        for item in data.get("items", []):
            name = item.get("name", "")
            if name.startswith("stream/"):
                out[name[len("stream/"):]] = bool(item.get("ready"))
        return out
    except Exception:
        return {}


def build(host, port_rtsp=8554, port_whep=8889, port_hls=8888):
    ready = live_paths()
    cams = []
    for c in _defs["cameras"]:
        cid = c["id"]
        cams.append({
            "id": cid,
            "name": c["location"],
            "location": c["location"],
            "department": c["department"],
            "governance_class": c["governance"],
            "lat": c["lat"],
            "lon": c["lon"],
            "live": ready.get(cid, False),
            "codec": c["codec"],
            "width": c["width"],
            "height": c["height"],
            "fps": c["fps"],
            "rtsp_url": f"rtsp://{host}:{port_rtsp}/stream/{cid}",
            "whep_url": f"http://{host}:{port_whep}/stream/{cid}/whep",
            "hls_url":  f"http://{host}:{port_hls}/live/stream/{cid}/index.m3u8",
            "_synthetic_truth": {
                "plate_px": c["plate_px"],
                "plate": c["plate"],
                "clock_offset_s": c.get("clock_offset_s", 0),
                "expect": c["expect"],
            },
        })
    return {"cameras": cams, "count": len(cams),
            "corridor": _defs.get("corridor", ""), "source": "local-grid"}


class Handler(BaseHTTPRequestHandler):
    host = "127.0.0.1"

    def do_GET(self):
        if self.path.rstrip("/") in ("/api/ingest", "/api/ingest"):
            body = json.dumps(build(self.host), indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "only /api/ingest is served here")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    Handler.host = args.host
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"catalogue on http://{args.host}:{args.port}/api/ingest")
    srv.serve_forever()


if __name__ == "__main__":
    main()
