#!/usr/bin/env bash
# Run a project script with the interpreter it actually needs.
#
#   bash scripts/run.sh dual_arm_sim.py --shape arch      # live, dual-arm
#   bash scripts/run.sh orchestration/run_plan.py --structure S1
#   bash scripts/run.sh planner.py --crosscheck
#   bash scripts/run.sh tests/test_joint.py
#
# Three environments, none of which can run everything:
#   BrickSim/.venv   bricksim + Isaac Sim 5.1   BrickSim physics runs
#   isdnenv          cuRobo + Viser             planner, viewer, force queries
#   Issac            Isaac Sim 6.0 + Newton     dual_arm_sim, USD viewers, WP0
#   mjenv            MuJoCo twin (v3.1; CPU     tests/test_{clutch,planner,control,env,brace_bandit}.py,
#                    physics, CUDA torch)
#                                               orchestration/twin_executor.py, experiments/*,
#                                               tasks/*, stability.py, bracing.py,
#                                               planner.py --frame twin
#                    (MJ_PY overrides the interpreter; requirements-twin.txt)
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$HERE")"
BRICKSIM_PY="$REPO/BrickSim/.venv/bin/python"
ISDN_PY="$HERE/../isdnenv/bin/python"
ISAAC_PY="$HOME/Codes/CAIRSS/Issac/bin/python"
MJ_PY="${MJ_PY:-$HERE/../mjenv/bin/python}"

[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
script="$1"; shift

# Resolve to an ABSOLUTE path before any cd: the handlers below change
# directory, which silently breaks a relative path.
if [ -f "$script" ]; then
    script="$(cd "$(dirname "$script")" && pwd)/$(basename "$script")"
elif [ -f "$HERE/$script" ]; then
    script="$(cd "$(dirname "$HERE/$script")" && pwd)/$(basename "$script")"
else
    echo "no such script: $script" >&2; exit 1
fi

# v4 (plan_v4 §2.8): Newton + cuRobo overlay, all in the Isaac interpreter.
# Matched before the twin's experiments/* rule.
case "$script" in
    */tests/test_vision.py|*/tests/test_cell.py|*/tests/test_motion.py|*/tests/test_arm_protocol.py)
        cd "$HERE"; exec "$ISAAC_PY" -m pytest "${script#$HERE/}" "$@" ;;
    */cell/*.py|*/motion/plan_curobo.py|*/vision.py|*/scripts/1[0-3]_*.py|*/scripts/fixture_check.py|*/experiments/v4*.py|*/experiments/analyze_v4.py)
        cd "$HERE"; exec "$ISAAC_PY" -u "$script" "$@" ;;
esac

# The MuJoCo twin (v3.1): run as a module from brickassembly so the package
# imports (sim.mj, orchestration, tasks, experiments) resolve.
twin=0
case "$script" in
    */tests/test_clutch.py|*/tests/test_planner.py|*/tests/test_control.py|*/tests/test_env.py|*/tests/test_brace_bandit.py) twin=2 ;;
    */twin_executor.py|*/experiments/*.py|*/tasks/*.py|*/stability.py|*/bracing.py|*/sim/mj/*.py) twin=1 ;;
    */planner.py) case " $* " in *" --frame twin "*|*" --frame=twin "*) twin=1 ;; esac ;;
esac
if [ "$twin" -ne 0 ]; then
    cd "$HERE"
    rel="${script#$HERE/}"
    if [ "$twin" -eq 2 ]; then
        exec "$MJ_PY" -m pytest "$rel" "$@"
    fi
    mod="${rel%.py}"
    exec "$MJ_PY" -u -m "${mod//\//.}" "$@"
fi

case "$script" in
    # Physics runs: need bricksim, and BrickSim as the working directory.
    *run_plan.py|*test_joint.py|*test_gate.py|*02_dynamic_calibration.py)
        # DEFAULT TO HEADLESS. Isaac Sim's RTX renderer segfaults on this
        # driver (595.84; see VIEWING.md), so a run that tries to open a
        # window dies with exit 139 before doing anything. Ask for a window
        # explicitly with --window and you get a warning instead of a
        # mysterious crash.
        want_window=0
        args=()
        for a in "$@"; do
            case "$a" in
                --window) want_window=1 ;;
                --headless) want_window=0; args+=("$a") ;;
                *) args+=("$a") ;;
            esac
        done
        if [ "$want_window" -eq 1 ]; then
            echo "[run.sh] --window: rendering enabled. The RTX Vulkan compat" >&2
            echo "         layer is applied automatically (envcheck.rtx_compat)." >&2
        else
            case " ${args[*]} " in
                *" --headless "*) : ;;
                *) args+=(--headless)
                   echo "[run.sh] adding --headless (rendering crashes on this driver)" >&2 ;;
            esac
        fi
        cd "$REPO/BrickSim"
        exec env OMNI_KIT_ACCEPT_EULA=1 "$BRICKSIM_PY" -u "$script" "${args[@]}"
        ;;
    # WP0 platform checks: Isaac Sim 6.0 / Isaac Lab, no bricksim.
    *00b_impedance_check.py)
        cd "$REPO/BrickSim"
        exec "$ISAAC_PY" -u "$script" "$@"
        ;;
    # USD export and playback, and the live dual-arm prototype (Newton +
    # its viewer): the Isaac Sim 6.0 env, the only one with newton. Run from
    # brickassembly so a relative results/ path resolves the way it reads.
    *to_usd.py|*open_usd.py|*dual_arm_sim.py)
        cd "$HERE"
        exec "$ISAAC_PY" -u "$script" "$@"
        ;;
    # Everything else is cuRobo / plain python.
    *)
        cd "$HERE"
        exec "$ISDN_PY" -u "$script" "$@"
        ;;
esac
