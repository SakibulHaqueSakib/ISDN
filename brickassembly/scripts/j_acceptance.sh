#!/usr/bin/env bash
# J acceptance (plan_v4 r5, unbraced now): cube, arch, hollow_box, S3 x REPEATS (default 3),
# --strategy none (the default), headless, ground truth. Rows in $DIR/rows.jsonl, per-frame records
# in $DIR/<name>_r<k>.npz; then `bash scripts/run.sh experiments/j_tables.py $DIR`.
# Sequential. REPEATS=n, DIR=... (default results/v4/j/acc), SHAPES="cube arch" filter the batch.
# Backs up and restores plans/sim_*.json and results/proto_episodes.jsonl, which every run rewrites.
set -u
cd "$(dirname "$0")/.."
DIR="${DIR:-results/v4/j/acc}"
BK="$(mktemp -d)"
cp plans/sim_*.json results/proto_episodes.jsonl "$BK"/ 2>/dev/null
ls plans/sim_*.json > "$BK/plans.list"
restore() {
  for f in plans/sim_*.json; do grep -qxF "$f" "$BK/plans.list" || rm -f "$f"; done
  cp "$BK"/sim_*.json plans/ 2>/dev/null
  cp "$BK/proto_episodes.jsonl" results/proto_episodes.jsonl
  rm -rf "$BK"
}
trap restore EXIT
for name in ${SHAPES:-cube arch hollow_box S3}; do
  case "$name" in S3) src="--structure S3" ;; *) src="--shape $name" ;; esac
  for k in $(seq 0 $(( ${REPEATS:-3} - 1 ))); do
    echo "== $name r$k"
    bash scripts/run.sh dual_arm_sim.py $src --viewer null --test --num-frames 40000 --quiet \
      --repeat "$k" --out "$DIR/rows.jsonl" --record-all "$DIR/${name}_r$k.npz" || echo "run failed: $name r$k"
  done
done
