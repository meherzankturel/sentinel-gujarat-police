#!/usr/bin/env python3
"""
Demonstrate that the audit log detects tampering -- on demand, not by
shipping a database that permanently reports itself as tampered with.

A deliberately altered row was left in sentinel.db on the first day of this
project to prove the chain works. It does work, and it has been reporting
`chain_intact: false` to everyone who has looked since. A control that is
always alarming is a control nobody reads, which is the same failure the
alert budget exists to prevent.

So the demonstration lives here. Run it, watch the chain break, and it puts
the log back exactly as it found it.

    python3 tools/demo_tamper.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sentinel.registry import connect, verify_audit_chain

with connect() as con:
    row = con.execute("SELECT id, actor, action, result_count FROM audit_log "
                      "WHERE result_count IS NOT NULL ORDER BY id LIMIT 1").fetchone()
    if row is None:
        sys.exit("no audit entries to demonstrate against")
    rid, before = row["id"], row["result_count"]
    intact, broken = verify_audit_chain(con)
    print(f"chain before      : intact={intact} broken_at={broken}")

    con.execute("UPDATE audit_log SET result_count=? WHERE id=?", (99, rid))
    print(f"quietly changed   : entry #{rid} result_count {before} -> 99 "
          f"(actor {row['actor']}, action {row['action']})")

with connect() as con:
    intact, broken = verify_audit_chain(con)
    print(f"chain after       : intact={intact} broken_at={broken}   <- detected")

with connect() as con:
    con.execute("UPDATE audit_log SET result_count=? WHERE id=?", (before, rid))

with connect() as con:
    intact, broken = verify_audit_chain(con)
    print(f"chain restored    : intact={intact} broken_at={broken}")
