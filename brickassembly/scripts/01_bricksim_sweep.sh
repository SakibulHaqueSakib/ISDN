#!/usr/bin/env bash
# WP1 batching sweep: one Kit process per environment count.
#
#   bash brickassembly/scripts/01_bricksim_sweep.sh [counts...]
#
# Each run appends a line to scripts/01_bricksim_probe.jsonl. A count that
# times out is recorded as a timeout rather than left as a silent gap -- an
# environment that cannot even be built at N envs is the answer to the probe.
set -u

BRICKSIM="${BRICKSIM:-$HOME/Documents/ISDN_Robofab/BrickSim}"
PROBE="$(cd "$(dirname "$0")" && pwd)/01_bricksim_probe.py"
OUT="$(dirname "$PROBE")/01_bricksim_probe.jsonl"
TIMEOUT="${TIMEOUT:-900}"          # per count, including Kit boot
COUNTS=("${@:-1 4 16 64 256 512}")

cd "$BRICKSIM" || exit 1
for n in ${COUNTS[@]}; do
    echo "=== $n envs (timeout ${TIMEOUT}s) ==="
    OMNI_KIT_ACCEPT_EULA=1 timeout "$TIMEOUT" \
        .venv/bin/python -u "$PROBE" --headless --num_envs "$n" --out "$OUT"
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "{\"n_envs\": $n, \"failed\": \"timeout after ${TIMEOUT}s\"}" >> "$OUT"
        echo "  TIMED OUT -- stopping; larger counts will not do better"
        break
    elif [ $rc -ne 0 ]; then
        echo "  exited $rc -- stopping"
        break
    fi
done

echo
echo "-- sweep results --"
cat "$OUT"
