#!/usr/bin/env bash
# Control the local camera grid.
#   ./grid/run.sh start | stop | status | kill <cam-id> | restart <cam-id>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=.run; mkdir -p $RUN
PY=./.venv/bin/python

case "${1:-status}" in
  start)
    pgrep -qf "mediamtx grid/mediamtx.yml" || { mediamtx grid/mediamtx.yml > $RUN/mediamtx.log 2>&1 & sleep 2; }
    pgrep -qf "grid/catalogue.py"          || { $PY grid/catalogue.py          > $RUN/catalogue.log 2>&1 & sleep 1; }
    pgrep -qf "grid/publish.py"            || { $PY grid/publish.py            > $RUN/publish.log 2>&1 & sleep 6; }
    echo "grid up.  catalogue: http://127.0.0.1:8080/api/ingest"
    ;;
  stop)
    pkill -f "grid/publish.py"; pkill -f "grid/catalogue.py"
    pkill -f "mediamtx grid/mediamtx.yml"; echo "grid down."
    ;;
  kill)      # drop ONE camera, to test reconnect-with-backoff
    pkill -f "stream/${2:?need a camera id}\$" && echo "killed publisher for $2"
    ;;
  restart)   # bring that camera back
    CID="${2:?need a camera id}"
    # Kill any previous stand-in for this camera first. Two publishers on one
    # path fight over it, which corrupts the stream and looks like a decoder bug.
    pkill -f "publish.py --only $CID" 2>/dev/null; sleep 1
    $PY grid/publish.py --only "$CID" >> $RUN/publish.log 2>&1 &
    echo "restarted $CID"
    ;;
  status)
    curl -s --max-time 5 http://127.0.0.1:8080/api/ingest \
      | $PY -c "import json,sys;d=json.load(sys.stdin);[print(f\"  {c['id']}  live={str(c['live']):<5} {c['width']}x{c['height']:<5} {c['codec']:<5} {c['department']}\") for c in d['cameras']];print(f\"  {sum(1 for c in d['cameras'] if c['live'])}/{d['count']} live\")" \
      2>/dev/null || echo "grid not running (./grid/run.sh start)"
    ;;
  *) echo "usage: $0 {start|stop|status|kill <id>|restart <id>}"; exit 1;;
esac
