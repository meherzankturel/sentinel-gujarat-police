#!/usr/bin/env python3
"""
sentinel.matching -- turn a plate read into an alert, or deliberately not.

Alert fatigue is what actually kills these deployments. Eighty thousand
cameras matched against a national watchlist will generate more alerts in
an hour than a control room can look at in a week, and once officers start
dismissing alerts reflexively the system is worse than useless -- it is
worse because everyone believes it is working.

So matching returns a tier, never a boolean, and only the top tiers are
allowed to interrupt a human:

    confirmed      a capable camera read it cleanly. Notify.
    probable       read is good but not certain, or the camera is marginal.
                   Notify at lower priority, with the doubt shown.
    corroborating  too weak to raise on its own. It may only strengthen a
                   track that already exists. It never notifies.

That last tier is the important one. It is how a camera that cannot read a
plate still contributes evidence without adding noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .registry import normalise_plate

TIER_ORDER = ["corroborating", "probable", "confirmed"]

# A camera's measured capability caps what its reads are allowed to claim,
# however confident the OCR happens to feel on a given frame.
# Must list every class the survey can write. plate-occasional and
# no-vehicles-observed were absent, so they hit the "corroborating"
# default here while sentinel.officer.tier_ceiling fell through to
# "probable" for the same camera -- two ceilings, same row, opposite
# answers, and the more permissive one was the one on screen.
CAPABILITY_CEILING = {
    "plate-capable":        "confirmed",
    "plate-occasional":     "probable",
    "plate-marginal":       "probable",
    "presence-only":        "corroborating",
    "no-vehicles-observed": "corroborating",
    "unusable":             "corroborating",
    "unknown":              "corroborating",
}


# Appearance similarity a candidate must clear before it is worth an
# officer's attention at all, and the mark above which a measured
# appearance match is allowed to read as probable. Measured, not guessed:
# on real footage the median same-vehicle score is well above the floor and
# the median different-vehicle score is well below it.
#
# They sit beside the ceilings above because they are the same kind of
# statement -- what a piece of evidence may claim -- and because the
# officer's console has to explain an appearance match in those terms
# without importing the vision stack to do it. sentinel.hunt re-exports
# them for the code that does the matching.
APPEARANCE_FLOOR = 0.55
APPEARANCE_STRONG = 0.72


def _demote(tier: str, steps: int = 1) -> str:
    i = TIER_ORDER.index(tier)
    return TIER_ORDER[max(0, i - steps)]


def _cap_to(tier: str, ceiling: str) -> str:
    return TIER_ORDER[min(TIER_ORDER.index(tier), TIER_ORDER.index(ceiling))]


# Character pairs an OCR routinely confuses on a plate. These are shape
# collisions, not random errors: a degraded read of GJ05MN8899 came back as
# GJOSMNS899 -- every wrong character is one of these pairs. Treating them
# as ordinary substitutions means a genuine watchlist vehicle is missed on
# exactly the cameras where a miss matters most.
#
# They are given reduced cost rather than zero. Two confusions still add up
# to an inexact match, which is demoted a tier and shown to the officer as
# inexact -- so the system recovers the vehicle without ever claiming the
# read was clean.
CONFUSIONS = {
    frozenset("0O"), frozenset("0D"), frozenset("0Q"), frozenset("1I"),
    frozenset("1L"), frozenset("2Z"), frozenset("5S"), frozenset("6G"),
    frozenset("8B"), frozenset("8S"), frozenset("4A"), frozenset("7T"),
    frozenset("VY"), frozenset("MN"),
    # Digit pairs that collide once the plate is small: on a real read of
    # GJ03KH8510 the 8 came back as 0. These are only cheap, not free, so a
    # match that needs one is still shown to the officer as inexact.
    frozenset("08"), frozenset("68"), frozenset("56"), frozenset("39"),
    frozenset("13"), frozenset("79"), frozenset("09"),
}
CONFUSION_COST = 0.5


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return CONFUSION_COST if frozenset((a, b)) in CONFUSIONS else 1.0


def plate_distance(a: str, b: str) -> float:
    """
    Edit distance that knows which characters an OCR mixes up.

    Used for watchlist comparison. A plain distance treats S-for-5 as a
    different vehicle; this treats it as a smudged reading of the same one,
    at a cost, so it surfaces as a lower-confidence match rather than
    vanishing.
    """
    if abs(len(a) - len(b)) > 2:
        return 99.0
    prev = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0,
                           prev[j - 1] + sub_cost(ca, cb)))
        prev = cur
    return prev[-1]


def edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Match:
    watchlist_id: int
    plate_on_list: str
    plate_read: str
    reason: str
    priority: int
    case_ref: Optional[str]
    exact: bool
    distance: float
    tier: str
    confidence: float
    why: str

    def notifies(self) -> bool:
        """Corroborating sightings never interrupt anyone."""
        return self.tier in ("confirmed", "probable")


def match_plate(con, plate_text: str, read_confidence: float, read_tier: str,
                camera_capability: str, camera_time_confidence: Optional[str] = None
                ) -> List[Match]:
    """
    Compare a read against the watchlist and decide what it is allowed to do.

    Three things can pull a match down a tier and each is recorded so the
    officer sees why: an inexact plate, a camera not measured capable of
    reading plates at all, and a camera whose own clock we do not trust.
    """
    norm = normalise_plate(plate_text)
    if not norm:
        return []

    rows = con.execute(
        "SELECT id, plate_number, plate_normalised, reason, priority, case_ref "
        "FROM watchlist_entry WHERE entity_type='vehicle'").fetchall()

    out: List[Match] = []
    ceiling = CAPABILITY_CEILING.get(camera_capability, "corroborating")

    for r in rows:
        target = r["plate_normalised"] or normalise_plate(r["plate_number"])
        if not target:
            continue
        d = 0.0 if target == norm else plate_distance(norm, target)
        if d > 1.0:
            continue

        why = []
        tier = read_tier

        capped = _cap_to(tier, ceiling)
        if capped != tier:
            why.append(f"camera measured {camera_capability}")
            tier = capped

        if d > 0:
            tier = _demote(tier)
            why.append("read differs from the listed plate by "
                       f"{d:g} (OCR-confusable characters)")

        # The registry stores the verdict with its reasoning attached
        # ("unreliable: OSD clock read 6/6 samples; ..."), so comparing the
        # whole string against a bare status meant a camera with a known
        # bad clock was never demoted. Take the verdict off the front.
        tc = (camera_time_confidence or "").split(":")[0].strip().lower()
        if tc in ("unreliable", "low-agreement"):
            # If we cannot trust when it happened, we should not treat it as
            # a confirmed sighting -- a route built on it would be wrong.
            tier = _demote(tier)
            why.append(f"camera clock {camera_time_confidence}")

        out.append(Match(
            watchlist_id=r["id"], plate_on_list=r["plate_number"],
            plate_read=plate_text, reason=r["reason"], priority=r["priority"],
            case_ref=r["case_ref"], exact=(d == 0), distance=d, tier=tier,
            confidence=round(read_confidence * (1.0 if d == 0 else 0.7), 3),
            why="; ".join(why) if why else "clean read on a capable camera"))

    out.sort(key=lambda m: (-TIER_ORDER.index(m.tier), m.priority, m.distance))
    return out


# --------------------------------------------------------------- budget


@dataclass
class AlertBudget:
    """
    An explicit ceiling on how many alerts a control room is asked to look
    at per shift, stated up front rather than discovered when officers stop
    reading them. When the budget is spent, lower-tier alerts stop
    notifying and become corroboration; confirmed alerts always get through.
    """
    per_shift: int = 40
    spent: int = 0

    def admit(self, m: Match) -> bool:
        if not m.notifies():
            return False
        if m.tier == "confirmed":
            self.spent += 1
            return True
        if self.spent < self.per_shift:
            self.spent += 1
            return True
        return False

    def remaining(self) -> int:
        return max(0, self.per_shift - self.spent)
