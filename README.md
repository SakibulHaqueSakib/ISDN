# ISDN Robofab

Coursework and final project for the ISDN *Robotic Fabrication with Python* course.

## Layout

| Path | Contents |
|------|----------|
| `Codes/` | Course exercises — `hello_world.py`, `robot_game.py` (Panda cube-picking game in Viser) |
| `Docs/` | Final project brief: *Watch and Build* — robotic LEGO assembly |

Not tracked: `isdnenv/` (virtualenv), `curobo/` (NVIDIA cuRobo, cloned in-tree) and `Slides/` (lecture pptx, too large for GitHub).

## Setup

```bash
python -m venv isdnenv
source isdnenv/bin/activate
git clone https://github.com/NVlabs/curobo
pip install -e curobo
python Codes/robot_game.py
```
