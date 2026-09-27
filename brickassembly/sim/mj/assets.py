"""Franka Panda model from MuJoCo Menagerie, fetched at a pinned commit.

    python -m sim.mj.assets          # fetch into .cache/ (idempotent)

Only the collision meshes are fetched by default; the 30 MB of visual meshes
are needed for rendering alone (`fetch(visual=True)`). Pinned in env.lock
[mujoco]; bump only with a ledger entry (guardrail §5.4.2).
"""

import re
import sys
import urllib.request
from pathlib import Path

MENAGERIE_SHA = "c96a32d28fb5da84da38c1da4d749e7a13212855"
BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/%s/franka_emika_panda/"
CACHE = Path(__file__).resolve().parents[2] / ".cache" / ("menagerie_" + MENAGERIE_SHA[:7])
PANDA_DIR = CACHE / "franka_emika_panda"
COLLISION = ["link0.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl",
             "link5_collision_0.obj", "link5_collision_1.obj", "link5_collision_2.obj",
             "link6.stl", "link7.stl", "hand.stl", "finger_0.obj"]


def _get(name, dest):
    if dest.exists() and dest.stat().st_size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE % MENAGERIE_SHA + name, timeout=60) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)


def fetch(visual=False):
    """Download panda.xml and its meshes; returns the path to panda.xml."""
    xml = PANDA_DIR / "panda.xml"
    _get("panda.xml", xml)
    names = list(COLLISION)
    if visual:
        names += sorted(set(re.findall(r'file="([^"]+)"', xml.read_text())) - set(names))
    for n in names:
        _get("assets/" + n, PANDA_DIR / "assets" / n)
    return xml


if __name__ == "__main__":
    path = fetch(visual="--visual" in sys.argv)
    print("panda ->", path)
