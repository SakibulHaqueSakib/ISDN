#!/usr/bin/env bash
# One-command environment for robofab_sim (dual_arm_sim.py).
#
#   bash setup.sh                  # .venv (Python 3.12), torch CUDA build picked from nvidia-smi
#   bash setup.sh --cuda cu128     # force a torch CUDA build (cu128 | cu130)
#   bash setup.sh --cpu            # CPU torch (no GPU: Newton/Warp then run on the CPU, slowly)
#   bash setup.sh --no-torch       # skip torch (dual_arm_sim itself does not import it)
#   bash setup.sh --check          # after installing: also run the tests and a headless smoke build
#
# Idempotent: re-run it to repair or upgrade. VENV=/path overrides the location.
# Needs internet: packages, and Newton's Franka (FR3) asset, downloaded once.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
TORCH=2.10.0                                            # cu128; cu130 builds exist too
PYVER=3.12

cuda=""; cpu=0; notorch=0; check=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1 ;;
        --no-torch) notorch=1 ;;
        --check) check=1 ;;
        --cuda) cuda="${2:?--cuda needs cu128 or cu130}"; shift ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done
die() { echo "[setup] ERROR: $*" >&2; exit 1; }
say() { echo "[setup] $*"; }

# --- torch build: --cpu, --cuda, or the newest the driver supports ----------
if [ "$notorch" -eq 0 ]; then
    if [ "$cpu" -eq 1 ]; then
        cuda=cpu
    elif [ -z "$cuda" ]; then
        command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver, or use --cpu"
        drv="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)"
        [ -n "$drv" ] || die "cannot read the CUDA version from nvidia-smi; pass --cuda cu128|cu130 or --cpu"
        set -- $drv
        if [ "$1" -ge 13 ]; then cuda=cu130
        elif [ "$1" -eq 12 ] && [ "$2" -ge 8 ]; then cuda=cu128
        else die "driver supports CUDA $1.$2; torch needs >= 12.8 (RTX 50xx) -- update the driver, or use --cpu"
        fi
    fi
    case "$cuda" in cpu|cu128|cu130) ;; *) die "--cuda must be cu128 or cu130 (got $cuda)" ;; esac
    say "torch build: $cuda"
fi

# --- venv (uv if present, else python3.12 -m venv) --------------------------
if [ ! -x "$VENV/bin/python" ]; then
    if command -v uv >/dev/null; then
        say "creating $VENV with uv (Python $PYVER)"
        uv venv "$VENV" --python "$PYVER"
    else
        command -v "python$PYVER" >/dev/null || die "need uv (https://docs.astral.sh/uv/) or python$PYVER on PATH"
        say "creating $VENV with python$PYVER -m venv"
        "python$PYVER" -m venv "$VENV"
    fi
fi
PY="$VENV/bin/python"
"$PY" -c "import sys; sys.exit(sys.version_info[:2] != tuple(map(int, '$PYVER'.split('.'))))" \
    || die "$VENV is not Python $PYVER: delete it and re-run"

if command -v uv >/dev/null; then
    pipi() { uv pip install --python "$PY" "$@"; }
    torch_args=(--torch-backend "${cuda:-auto}")
else
    "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null
    pipi() { "$PY" -m pip install "$@"; }
    torch_args=(--index-url "https://download.pytorch.org/whl/${cuda:-cpu}")
fi

if [ "$notorch" -eq 0 ]; then
    say "installing torch==$TORCH"
    pipi "torch==$TORCH" "${torch_args[@]}"
fi
say "installing requirements.txt"
pipi -r "$HERE/requirements.txt"

# --- Newton's FR3 asset (network on first use; cached afterwards) -----------
say "fetching the Newton FR3 asset (franka_emika_panda)"
"$PY" -c "import newton.utils as u; print(u.download_asset('franka_emika_panda'))" \
    || die "asset download failed (needs internet once; it is cached afterwards)"

# --- import check -----------------------------------------------------------
say "import check"
"$PY" - <<'PYEOF'
import warp as wp, newton, mujoco, mujoco_warp, numpy, scipy, cv2, yaml, trimesh, viser, pyglet, imgui_bundle
wp.config.quiet = True
print("newton", newton.__version__, "| warp", wp.__version__, "| mujoco", mujoco.__version__)
print("warp device:", wp.get_device())
try:
    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
except ImportError:
    print("torch: not installed")
PYEOF

if [ "$check" -eq 1 ]; then
    say "tests"
    bash "$HERE/run.sh" test -q
    say "headless smoke build (cube, 300 frames)"
    bash "$HERE/run.sh" --shape cube --viewer null --num-frames 300
fi
say "done. Run: ./run.sh --shape cube   (tests: ./run.sh test)"
