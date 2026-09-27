#!/usr/bin/env python3
"""
Put the government grid into the registry, with what we measured about it.

Their catalogue supplies an id and a name. Every other column here came from
watching the camera's own video: what resolution it delivers, how large
vehicles appear, and therefore what share of passing vehicles could have
their plate read at all. That last figure is the one that decides which
analytics each camera is given.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json
from sentinel.registry import init, connect, upsert_camera, record_capability, audit

# Approximate coordinates for the sites named in their catalogue. Their
# catalogue carries no location at all, so a map cannot be drawn from it --
# these are recovered from the camera names and marked as such.
PLACES = {
    "cam01": (23.0265, 72.5490, "Chimanbhai Bridge, Ahmedabad", "Police"),
    "cam02": (23.0330, 72.5580, "Janpath, Ahmedabad", "Police"),
    "cam03": (23.0225, 72.5300, "O.N.G.C. Office, Ahmedabad", "Police"),
    "cam04": (23.0120, 72.5590, "Paldi Circle, Ahmedabad", "Municipal"),
    "cam05": (23.0810, 72.5590, "Visat Teen Rasta, Ahmedabad", "Police"),
    "cam06": (21.5220, 70.4579, "Timbavadi Gate, Junagadh", "Municipal"),
    "cam07": (20.9000, 70.3667, "Hero Showroom, Gir Somnath", "Municipal"),
    "cam08": (21.5170, 70.4600, "Majewadi Gate, Junagadh", "Municipal"),
    "cam09": (21.5000, 70.4500, "New Bypass Circle, Junagadh", "Police"),
    "cam10": (21.5150, 70.4650, "Char Chowk Road, Junagadh", "Municipal"),
    "cam11": (21.4900, 70.4400, "Dolatpara, Junagadh", "Panchayat"),
    "cam12": (23.1660, 72.5790, "Tri Mandir Adalaj Tollnaka", "GSRTC"),
    "cam13": (23.0300, 72.5480, "CN Vidhyalaya, Ahmedabad", "Municipal"),
    "cam14": (23.0400, 72.5600, "Delight RLVD, Ahmedabad", "Police"),
    "cam15": (23.0500, 72.5700, "Suvidha Park, Ahmedabad", "Municipal"),
    "cam16": (23.0820, 72.5600, "Visat P2, Ahmedabad", "Police"),
    "cam17": (22.3039, 70.8022, "Rajkot Bus Port", "GSRTC"),
    "cam18": (22.3080, 70.8000, "Rajkot City", "Police"),
    "cam19": (20.8000, 72.9500, "Khaparia Gram Panchayat, Navsari", "Panchayat"),
    "cam20": (23.0600, 72.5400, "Mohanpura", "Panchayat"),
    "cam21": (23.8500, 72.1200, "Patan Dethali Char Rasta", "Municipal"),
    "cam22": (23.7800, 72.1500, "BK Mervada Tran Rasta", "Panchayat"),
    "cam23": (22.9000, 72.4000, "Kheram", "Panchayat"),
    "cam24": (23.1700, 72.8200, "Dehgam", "Municipal"),
    "cam25": (22.8500, 72.3000, "Dhanori", "Panchayat"),
    "cam26": (21.7500, 72.1500, "Tankal", "Panchayat"),
    "cam27": (20.7700, 72.9600, "Bilimora", "Municipal"),
    "cam28": (20.7720, 72.9580, "Bilimora 2", "Municipal"),
    "cam29": (20.7740, 72.9560, "Bilimora 3", "Municipal"),
    "cam30": (23.0800, 70.1300, "Gandhidham Rambaugh", "GSRTC"),
}

survey = {r["id"]: r for r in json.loads(
    pathlib.Path("audit/grid_survey.json").read_text())}

init()
with connect() as con:
    n = 0
    for cid, (lat, lon, name, dept) in PLACES.items():
        s = survey.get(cid, {})
        upsert_camera(con, {
            "id": cid, "catalogue_id": cid, "department": dept,
            "location_name": name, "lat": lat, "lon": lon,
            "hls_url": f"https://cctv.corp8.cloud/{cid}/index.m3u8",
            "width": s.get("width"), "height": s.get("height"),
            "codec": "h264",
            "governance_class": "sensitive" if dept == "Health" else "open",
            "last_seen_live": None,
        })
        if s and not s.get("error"):
            y = s.get("anpr_yield")
            note = (f"ANPR yield {y*100:.1f}% of passing vehicles"
                    if y is not None else "not measured")
            record_capability(
                con, cid,
                capability_class=s.get("capability", "unknown"),
                plate_px=int(s["plate_p90"]) if s.get("plate_p90") else None,
                basis="measured from delivered video (vehicle scale)",
                note=note,
                declared_fps=s.get("declared_fps"),
                brightness=s.get("brightness"))
        n += 1
    audit(con, "system", "load_real_grid", result_count=n)

with connect() as con:
    print(f"{'camera':<8} {'dept':<10} {'capability':<18} {'plate':>6}  location")
    print("-" * 78)
    for r in con.execute("SELECT * FROM camera WHERE id LIKE 'cam__' ORDER BY id"):
        print(f"{r['id']:<8} {r['department']:<10} "
              f"{str(r['capability_class'] or '-'):<18} "
              f"{str(r['plate_px_measured'] or '-'):>6}  {r['location_name'][:38]}")
