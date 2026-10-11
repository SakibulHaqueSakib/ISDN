#!/usr/bin/env bash
# ./run.sh [dual_arm_sim.py args]     e.g. ./run.sh --shape cube
# ./run.sh test [pytest args]         the tests in tests/
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${VENV:-$HERE/.venv}/bin/python"
[ -x "$PY" ] || { echo "no venv at $PY: run 'bash setup.sh' first" >&2; exit 1; }
cd "$HERE"
if [ "${1:-}" = test ]; then shift; exec "$PY" -m pytest "$@"; fi
exec "$PY" -u dual_arm_sim.py "$@"
