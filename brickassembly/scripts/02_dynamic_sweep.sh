#!/usr/bin/env bash
# WP2b dynamic calibration sweep: one Kit process per measurement.
#
#   bash brickassembly/scripts/02_dynamic_sweep.sh
#
# Each point runs in a fresh scene, so a released brick cannot contaminate the
# next measurement. Results append to scripts/02_dynamic_calibration.jsonl;
# a point that times out or fails to release is recorded, not silently skipped.
set -u

BRICKSIM="${BRICKSIM:-$HOME/Documents/ISDN_Robofab/BrickSim}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBE="$HERE/02_dynamic_calibration.py"
OUT="${OUT:-$HERE/02_dynamic_calibration.jsonl}"
TIMEOUT="${TIMEOUT:-900}"
RAMP="${RAMP:-0.1}"

# 2x4 carries the fit (2x2 was non-monotonic in earlier runs); 2x6 tests
# whether per-stud capacity holds across article size, which a contaminated
# run had suggested it did not.
PRELOADS="${PRELOADS:-10 20 35 50 60 70}"
ARTICLES="${ARTICLES:-4 6}"
REPEATS="${REPEATS:-1}"

cd "$BRICKSIM" || exit 1
for ny in $ARTICLES; do
    for preload in $PRELOADS; do
        for _ in $(seq 1 "$REPEATS"); do
            echo "=== 2x${ny} preload ${preload} ramp ${RAMP} ==="
            OMNI_KIT_ACCEPT_EULA=1 timeout "$TIMEOUT" \
                .venv/bin/python -u "$PROBE" --headless \
                --ny "$ny" --preload "$preload" --ramp "$RAMP" --out "$OUT" \
                2>&1 | grep -E "^RESULT|Error:" || true
            rc=${PIPESTATUS[0]}
            if [ "$rc" -eq 124 ]; then
                echo "{\"brick\": \"2x${ny}\", \"preload_n\": ${preload}, \
\"ramp_n_per_step\": ${RAMP}, \"release_n\": null, \
\"failed\": \"timeout after ${TIMEOUT}s\"}" >> "$OUT"
                echo "  TIMED OUT"
            fi
        done
    done
done

echo
echo "-- sweep complete --"
wc -l "$OUT"
