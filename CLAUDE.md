# ISDN_Robofab

The active project is `brickassembly/` (dual-arm LEGO assembly; plan in `Docs/master_report.md`, results in `Docs/results_report.md`). Read `brickassembly/README.md` first.

## Multi-agent workflow

The main session runs as the `orchestrator` agent (`.claude/settings.json`; `claude --agent orchestrator` does the same). It talks to the user and oversees; the agents in `.claude/agents/` do the work:

| agent | model, effort | use for |
|---|---|---|
| `orchestrator` | opus, high | main session: runs the plan task by task, verifies every output, commits |
| `planner` | opus, xhigh | the plan of record at the start (experiments, gates, what to report); revisions when it is faulty |
| `logic-reviewer` | opus, high | reviewing the plan (and revisions) and diffs before commit |
| `implementer` | sonnet, medium | executing code steps of an approved plan |
| `doc-writer` | opus, medium | docs, report prose, ledger entries, WORKLOG |
| `scout` | haiku, low | finding code, reading logs/result files - read-only facts |
| `work-reviewer` | opus, medium | independently re-checking a scout/worker output the orchestrator doubts |

- **Once, at the start of a project or phase:** planner writes the plan of record -> logic-reviewer reviews it -> user decides what it raises -> doc-writer records it (`Docs/master_report.md`, ledger).
- **Every task after that** (the orchestrator breaks the plan down itself; no planner): scout -> implementer / doc-writer -> orchestrator verifies (doubts -> work-reviewer) -> logic-reviewer (diff, when it changes code, results or conclusions) -> commit.
- **Plan found faulty:** orchestrator gives the planner the evidence -> revision -> logic-reviewer -> user if it changes a conclusion or protocol -> doc-writer records the amendment and a ledger `deviation`.
- Keep it proportional: a question is scout-only; a small fix goes straight to its owner and is still verified.
- `CHANGES REQUESTED` goes back to the owner; after two rounds, escalate to the user.
- Long jobs (experiment batches, training) run in the background from the main session; a scout digests their logs.

## Project rules (all agents)

- Interpreters: `bash brickassembly/scripts/run.sh <script>` picks the right one. The MuJoCo twin uses `mjenv/` (CPU MuJoCo 3.3.7 physics, CUDA torch for the PPO update); Isaac/Newton scripts use `~/Codes/CAIRSS/Issac`; cuRobo/viser uses `isdnenv/`.
- Physics stays on CPU MuJoCo: MuJoCo Warp drops the twin's stud-wall contacts (ledger `mjwarp_insertion_probe`). Parallelism is `--workers` (one process per physical core).
- `brickassembly/ledger.jsonl` is the record of every gate, decision, deviation, result and failure: append an entry for each, then regenerate `WORKLOG.md` with `python3 brickassembly/scripts/worklog.py`. Never hand-edit `WORKLOG.md`.
- Results in reports must be computed from the result files (`python -m experiments.analyze`, `experiments.tables`), never typed from memory.
- Running tests rewrites the tracked `brickassembly/tests/*.json`; restore them unless the change is meant to update them.
- Don't commit the `BrickSim` submodule's local edits (they live in `bricksim-local.patch`). Commit on a branch, not `main`.
