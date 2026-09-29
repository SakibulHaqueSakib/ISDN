"""Route 1 overlay -- plan v4 §2.4 rule 1, P3 step 1.

cuRobo 0.8 is pure Python; only a handful of modules are missing from the
Isaac interpreter (torch 2.10+cu128, warp 1.13). Rather than touch that
env, the missing modules are pip-installed with `--no-deps --target` into
`brickassembly/.vendor` (see `scripts/12_curobo_probe.py` for the exact
list and versions) and appended -- never prepended -- to `sys.path`, so
the Isaac env's own torch/numpy/scipy/trimesh/etc. win over anything the
overlay happens to also ship.

Call `add_overlay()` before `import curobo` anywhere in this project.
"""

import sys
from pathlib import Path

_BRICKASSEMBLY = Path(__file__).resolve().parent.parent
VENDOR_DIR = _BRICKASSEMBLY / ".vendor"
CUROBO_DIR = _BRICKASSEMBLY.parent / "curobo"

_added = False


def add_overlay():
    """Append the .vendor overlay and the curobo checkout to sys.path (idempotent)."""
    global _added
    if _added:
        return
    for p in (VENDOR_DIR, CUROBO_DIR):
        p = str(p)
        if p not in sys.path:
            sys.path.append(p)
    _added = True
