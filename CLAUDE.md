# ISDN_Robofab

The active project is `brickassembly/` (dual-arm LEGO assembly; plan in `Docs/master_report.md`, results in `Docs/results_report.md`). Read `brickassembly/README.md` first.

## Multi-agent workflow

The main session runs as the `orchestrator` agent (`.claude/settings.json`; `claude --agent orchestrator` does the same). It talks to the user and oversees; the agents in `.claude/agents/` do the work:

| agent | model, effort | use for |
|---|---|---|
| `orchestrator` | opus, high | main session: dispatches per the plan, verifies every output, commits |
| `planner` | opus, xhigh | deciding how to do a non-trivial task - steps with owners, checks, risks |
| `logic-reviewer` | opus, high | reviewing the plan before building and the diff before commit |
| `implementer` | sonnet, medium | executing code steps of an approved plan |
| `doc-writer` | opus, medium | docs, report prose, ledger entries, WORKLOG |
| `scout` | haiku, low | finding code, reading logs/result files - read-only facts |
| `work-reviewer` | opus, medium | independently re-checking a scout/worker output the orchestrator doubts |

Flow for any non-trivial task: **scout -> planner -> logic-reviewer (plan) -> implementer / doc-writer -> orchestrator verifies (doubts -> work-reviewer) -> logic-reviewer (diff) -> commit**.
- Keep it proportional: a question is scout-only; a one-line fix skips planning but is still verified.
- Anything that changes a conclusion, protocol or pinned version goes to the user before building.
- `CHANGES REQUESTED` goes back to the owner; after two rounds, escalate to the user.
- Long jobs (experiment batches, training) run in the background from the main session; a scout digests their logs.

## Project rules (all agents)

- Interpreters: `bash brickassembly/scripts/run.sh <script>` picks the right one. The MuJoCo twin uses `mjenv/` (CPU MuJoCo 3.3.7 physics, CUDA torch for the PPO update); Isaac/Newton scripts use `~/Codes/CAIRSS/Issac`; cuRobo/viser uses `isdnenv/`.
- Physics stays on CPU MuJoCo: MuJoCo Warp drops the twin's stud-wall contacts (ledger `mjwarp_insertion_probe`). Parallelism is `--workers` (one process per physical core).
- `brickassembly/ledger.jsonl` is the record of every gate, decision, deviation, result and failure: append an entry for each, then regenerate `WORKLOG.md` with `python3 brickassembly/scripts/worklog.py`. Never hand-edit `WORKLOG.md`.
- Results in reports must be computed from the result files (`python -m experiments.analyze`, `experiments.tables`), never typed from memory.
- Running tests rewrites the tracked `brickassembly/tests/*.json`; restore them unless the change is meant to update them.
- Don't commit the `BrickSim` submodule's local edits (they live in `bricksim-local.patch`). Commit on a branch, not `main`.
