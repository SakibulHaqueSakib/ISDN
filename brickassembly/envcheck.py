"""Fail fast, and helpfully, when a script is run with the wrong interpreter.

This project spans three environments and none of them can run everything:

    isdnenv          py3.11  cuRobo, Viser          planner, viewer, force queries
    ~/Codes/CAIRSS/Issac  py3.12  Isaac Sim 6.0     WP0 platform checks only
    BrickSim/.venv   py3.11  bricksim, Isaac 5.1    every physics run

Without this guard a wrong interpreter costs five minutes of Isaac Sim startup
before dying on ModuleNotFoundError, which reads like a broken script rather
than a wrong command. Call require() BEFORE launching the app.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

ENVS = {
    "bricksim": {
        "python": REPO / "BrickSim/.venv/bin/python",
        "why": "bricksim and Isaac Sim 5.1 live only here",
        "prefix": "OMNI_KIT_ACCEPT_EULA=1 ",
        "cwd": REPO / "BrickSim",
    },
    "isaaclab": {
        "python": Path.home() / "Codes/CAIRSS/Issac/bin/python",
        "why": "Isaac Sim 6.0 and Isaac Lab live here",
        "prefix": "",
        "cwd": REPO / "BrickSim",
    },
    "curobo": {
        "python": REPO / "isdnenv/bin/python",
        "why": "cuRobo and Viser live here",
        "prefix": "",
        "cwd": ROOT,
    },
}


def require(env, modules=()):
    """Exit with an actionable message unless this interpreter can serve `env`.

    `modules` are checked with find_spec, which is cheap and does not import --
    importing bricksim needs a running Kit app, so a real import is not an
    option this early.
    """
    spec = ENVS[env]
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return

    script = Path(sys.argv[0]).resolve()
    try:
        shown = script.relative_to(REPO)
    except ValueError:
        shown = script

    print("", file=sys.stderr)
    print("WRONG ENVIRONMENT for %s" % shown.name, file=sys.stderr)
    print("  missing: %s" % ", ".join(missing), file=sys.stderr)
    print("  running: %s" % sys.executable, file=sys.stderr)
    print("  needs:   %s" % spec["python"], file=sys.stderr)
    print("           (%s)" % spec["why"], file=sys.stderr)
    print("", file=sys.stderr)
    print("  Run it like this:", file=sys.stderr)
    print("    cd %s" % spec["cwd"], file=sys.stderr)
    print("    %s%s %s %s"
          % (spec["prefix"], spec["python"], script,
             " ".join(sys.argv[1:])), file=sys.stderr)
    print("", file=sys.stderr)
    print("  Or let the launcher pick: bash %s"
          % (ROOT / "scripts/run.sh <script> [args]"), file=sys.stderr)
    print("", file=sys.stderr)
    sys.exit(2)


def rtx_compat():
    """Enable BrickSim's Vulkan workaround before anything initialises Vulkan.

    Isaac Sim's RTX renderer segfaults on drivers newer than 590.48.01 that
    report an unsafe maxMemoryAllocationSize. BrickSim ships the fix, but only
    its own CLI imports it -- scripts that launch through AppLauncher have to
    do it themselves, and it MUST happen before the app starts.

    Only needed when rendering. Harmless headless, so it is unconditional.
    """
    sys.path.insert(0, str(REPO / "BrickSim" / "python"))
    try:
        import isaacsim_rtx_compat  # noqa: F401  (enables itself on import)
    except Exception as exc:
        print("[env] RTX compat unavailable (%s: %s); rendering may crash"
              % (type(exc).__name__, exc), file=sys.stderr)


def banner(env):
    """One line naming the interpreter, so a log says which env produced it."""
    print("[env] %s  (%s)" % (sys.executable,
                              os.path.basename(os.path.dirname(
                                  os.path.dirname(sys.executable)))),
          flush=True)
