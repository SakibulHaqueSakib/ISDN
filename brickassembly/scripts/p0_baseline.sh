#!/usr/bin/env bash
# P0 (plan_v4): the prototype at HEAD, 1x, fixed slots, on cube/arch/hollow_box/S3
# x {grasp_lp, lever_press} x 3 repeats, with the sec 2.4 contact monitor.
# One JSONL row per run in results/v4/p0_baseline.jsonl; insert recordings in
# results/v4/p0_inserts/<structure>_<model>_r<k>.npz. Sequential (~1.5 min per
# cube run); OUT=... redirects the rows, INS=... the insert recordings, REPEATS=n
# changes the repeat count. P0' (r4, stiff fingers, the default now):
#   OUT=results/v4/p0_stiff.jsonl INS=results/v4/p0_stiff_inserts bash scripts/p0_baseline.sh
set -u
cd "$(dirname "$0")/.."
OUT="${OUT:-results/v4/p0_baseline.jsonl}"
INS="${INS:-$(dirname "$OUT")/p0_inserts}"
for src in "--shape cube" "--shape arch" "--shape hollow_box" "--structure S3"; do
  for model in grasp_lp lever_press; do
    for k in $(seq 0 $(( ${REPEATS:-3} - 1 ))); do
      name="${src##* }"
      echo "== $name $model r$k"
      bash scripts/run.sh dual_arm_sim.py $src --brace-model "$model" --viewer null --test \
        --num-frames 30000 --quiet --repeat "$k" --p0-out "$OUT" \
        --record-inserts "$INS/${name}_${model}_r$k.npz" || echo "run failed: $name $model r$k"
    done
  done
done
