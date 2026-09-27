# How to check progress

Everything below reads what has actually been measured and written down.
Nothing recomputes, so none of it can report progress that did not happen.

## v3.1 (the MuJoCo twin, CPU)

```bash
cd brickassembly
python scripts/status.py                         # gates, ledger, experiment summaries
../mjenv/bin/python -m experiments.analyze       # refresh results/analysis.json + figures
```

* Results and the gate-by-gate verdicts: [`Docs/results_report.md`](../Docs/results_report.md)
* What changed from the v3.0 plan and why: `Docs/master_report.md` §0.6 and §7.7
* Raw data: `results/g3/`, `results/a6/`, `results/g6/` (trials.jsonl = one line per
  trial; episodes.jsonl = one §2.6 insertion episode per attempt), `results/wp5/`
  (training logs per run, `eval.jsonl`)
* Test evidence: `tests/test_{clutch,planner,control,env}.json`, written by the suites

**To watch the simulation, see [VIEWING.md](VIEWING.md).** Short version:
`bash scripts/run.sh dual_arm_sim.py --shape arch` runs the whole dual-arm
pipeline live (images → blueprint → plan → two arms in Newton physics) in a
window within seconds. BrickSim runs (Isaac Sim 5.1, which cannot render here)
are recorded and played back.

## Which environment runs what

Three environments, none of which can run everything. **Use the launcher and
you don't have to remember**:

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
bash scripts/run.sh orchestration/run_plan.py --headless --structure S1
bash scripts/run.sh planner.py --crosscheck
```

| environment | python | has | runs |
|---|---|---|---|
| `BrickSim/.venv` | 3.11 | **bricksim**, Isaac Sim 5.1 | `run_plan.py`, `test_joint.py`, `test_gate.py`, `02_dynamic_calibration.py` |
| `isdnenv` | 3.11 | cuRobo, Viser | `planner.py`, `run_assembly.py`, `bricksim_adapter.py`, `calibration.py`, `status.py` |
| `~/Codes/CAIRSS/Issac` | 3.12 | Isaac Sim 6.0, Isaac Lab, Newton 1.2.1 | `dual_arm_sim.py`, `viz/to_usd.py`, `viz/open_usd.py`, `00b_impedance_check.py` |

`bricksim` exists **only** in BrickSim's venv, so a physics run started with
the Isaac Sim 6.0 environment fails. Every such script now checks its
interpreter before launching Isaac and prints the correct command, so a wrong
environment costs a second rather than a five-minute startup.

## The one command

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
../isdnenv/bin/python scripts/status.py
```

Prints gate verdicts, per-work-package activity, the current calibration, the
latest test results, and anything flagged as needing your decision.

```bash
../isdnenv/bin/python scripts/status.py --open   # only blockers, failures, deviations
../isdnenv/bin/python scripts/status.py --full   # every ledger entry in order
```

## Where the truth lives

| File | What it is |
|---|---|
| `ledger.jsonl` | Append-only record of every gate, decision, deviation, failure and result. **This is the project's memory.** Later entries supersede earlier ones. |
| `joint_calibration.json` | The joint model's fitted parameters, per solver path, with limitations |
| `tests/test_joint.json` | Last G1 joint-mechanics run |
| `tests/test_gate.json` | Last gate-boundary run (criteria since cut — see ledger) |
| `scripts/02_dynamic_calibration.jsonl` | Every isolated pull-off measurement |
| `scripts/00_env_check.*.json` | WP0 platform benchmarks |
| `scripts/01_bricksim_probe.jsonl` | WP1 batching sweep |
| `README.md` | Narrative: what each result means |

Read the ledger directly when you want the reasoning, not just the verdict:

```bash
../isdnenv/bin/python -c "
import json
for l in open('ledger.jsonl'):
    r = json.loads(l)
    print(r['ts'][:16], r.get('wp','-'), r['type'], r.get('id',''))"
```

## Re-running things yourself

```bash
# planner: structures -> sequence -> bracing (fast, no GPU)
../isdnenv/bin/python planner.py --crosscheck

# images -> brick blueprint (and its self-check)
../isdnenv/bin/python blueprint.py --shape arch
../isdnenv/bin/python blueprint.py

# the dual-arm prototype, live, then headless with its pass/fail check
bash scripts/run.sh dual_arm_sim.py --shape arch
bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --num-frames 6000 --test

# the force model, with its own assertions
../isdnenv/bin/python sim/joint_model/bricksim_adapter.py
../isdnenv/bin/python sim/joint_model/calibration.py --check

# watch the arm build a structure in a browser
../isdnenv/bin/python run_assembly.py --plan plans/S3_weakest_joint.json
#   then open http://localhost:8080

# joint mechanics in the live sim (~10 min, boots Isaac Sim)
cd ../BrickSim && OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python \
    ../brickassembly/tests/test_joint.py --headless
```

## Watching a long run

Sim runs are slow because Isaac Sim's startup dominates (~5-10 min per
process). They are all launched to a log:

```bash
tail -f /tmp/<name>.log                 # live
grep -aE "^RESULT|^  \[" /tmp/<name>.log  # just the results
```

## Training (WP5)

`python scripts/status.py` prints a TRAINING section: the last line of each
`results/wp5/<run>/train.jsonl` (samples, curriculum stage, success over the
last 100 episodes, mean peak force). Each line of a train.jsonl is one PPO
update; `experiments/analyze.py` plots them in `results/figures/wp5_training.png`.
Held-out evaluations are one line per (policy, stage, joint) in
`results/wp5/eval.jsonl`.
