#!/usr/bin/env bash
# One-command environment for everything except BrickSim / Isaac Sim 5.1. See SETUP.md.
#
#   bash setup_env.sh                  # .venv, torch CUDA build picked from nvidia-smi
#   bash setup_env.sh --cuda cu128     # force a torch CUDA build (cu128 | cu130)
#   bash setup_env.sh --cpu            # CPU torch (no GPU: Newton/Warp then run on the CPU, slowly)
#   bash setup_env.sh --with-curobo    # also clone + build cuRobo (NVlabs/curobo @ pinned sha)
#
# Idempotent: re-run it to repair or upgrade. VENV=/path and CUROBO_DIR=/path override the locations.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$REPO/.venv}"
CUROBO_DIR="${CUROBO_DIR:-$REPO/curobo}"
CUROBO_SHA=78fd485fa82d9b9a063fb4985e371814587e666a     # v0.8.0-43-g78fd485, env.lock [v4]
TORCH=2.10.0                                            # the v4 record (cu128); cu130 builds exist too
PYVER=3.12

cuda=""; cpu=0; curobo=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1 ;;
        --cuda) cuda="${2:?--cuda needs cu128 or cu130}"; shift ;;
        --with-curobo) curobo=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done
die() { echo "[setup] ERROR: $*" >&2; exit 1; }
say() { echo "[setup] $*"; }

# --- torch build: --cpu, --cuda, or the newest the driver supports ----------
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
    torch_args=(--torch-backend "$cuda")
else
    "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null
    pipi() { "$PY" -m pip install "$@"; }
    torch_args=(--index-url "https://download.pytorch.org/whl/$cuda")
fi

say "installing torch==$TORCH"
pipi "torch==$TORCH" "${torch_args[@]}"
say "installing requirements.txt"
pipi -r "$REPO/requirements.txt"

# --- cuRobo (optional; compiles/needs the CUDA toolchain at first use) ------
if [ "$curobo" -eq 1 ]; then
    [ "$cuda" != cpu ] || die "cuRobo needs a CUDA torch: drop --cpu"
    command -v git >/dev/null || die "git is required for --with-curobo"
    if [ ! -d "$CUROBO_DIR/.git" ]; then
        say "cloning NVlabs/curobo into $CUROBO_DIR"
        git clone https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
    fi
    git -C "$CUROBO_DIR" cat-file -e "$CUROBO_SHA^{commit}" 2>/dev/null || git -C "$CUROBO_DIR" fetch origin
    [ "$(git -C "$CUROBO_DIR" rev-parse HEAD)" = "$CUROBO_SHA" ] || git -C "$CUROBO_DIR" checkout --quiet "$CUROBO_SHA" \
        || die "cannot check out $CUROBO_SHA in $CUROBO_DIR (local edits?)"
    say "building cuRobo @ ${CUROBO_SHA:0:7}"
    pipi -r "$REPO/requirements-curobo.txt"
    ( cd "$CUROBO_DIR" && pipi -e . --no-build-isolation --no-deps )
fi

# --- Newton's FR3 asset (network on first use; cached afterwards) -----------
say "fetching the Newton FR3 asset (franka_emika_panda)"
"$PY" -c "import newton.utils as u; print(u.download_asset('franka_emika_panda'))" \
    || die "asset download failed (network? see SETUP.md 'No network')"

# --- self-check -------------------------------------------------------------
say "self-check"
"$PY" - <<'PYEOF'
import torch, warp as wp, newton, mujoco, mujoco_warp, numpy, scipy, cv2, trimesh, viser, pxr, pyglet, imgui_bundle, py_trees, yourdfpy, quaternion
wp.config.quiet = True
print("newton", newton.__version__, "| warp", wp.__version__, "| mujoco", mujoco.__version__, "| torch", torch.__version__)
print("warp device:", wp.get_device(), "| torch cuda:", torch.cuda.is_available())
PYEOF
if [ "$curobo" -eq 1 ]; then
    # cd away: a ./curobo checkout in the cwd would shadow the installed package as a namespace dir
    ( cd "$VENV" && "$PY" -c "import curobo; print('curobo', curobo.__file__)" ) || die "cuRobo import failed"
fi
( cd "$REPO/brickassembly" && ROBOFAB_PY="$PY" bash scripts/run.sh tests/test_aim.py -q -p no:cacheprovider ) \
    || die "self-check test failed"
say "done. Run scripts with: bash brickassembly/scripts/run.sh <script>  (uses <repo>/.venv, or ROBOFAB_PY=$PY)"
