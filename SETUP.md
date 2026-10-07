# Setting up the environment

One Python 3.12 virtualenv (`.venv` at the repo root) runs everything in the repo **except**
the BrickSim / Isaac Sim 5.1 physics runs (see "Not covered").

## Prerequisites

- Ubuntu 22.04 or 24.04 (x86_64); other Linux distributions probably work, untested.
- An NVIDIA GPU and driver supporting CUDA >= 12.8 (RTX 50xx needs it; `nvidia-smi` prints it).
  Without a GPU use `--cpu` (Newton/Warp run on the CPU, slowly; cuRobo is unavailable).
- `git`, and either [`uv`](https://docs.astral.sh/uv/) (recommended; it also fetches Python 3.12)
  or `python3.12` with `venv` (`sudo apt install python3.12-venv`).
- A desktop session with OpenGL (X11 or Wayland) for the live Newton viewer.
  Headless runs (`--viewer null`) need none.
- Network access for the first install: PyPI, the PyTorch wheel index, and GitHub
  (the Newton FR3 asset, and cuRobo with `--with-curobo`).
- For `--with-curobo`: nothing beyond the above. cuRobo compiles its kernels at first use through
  `cuda-core` and pip-installed CUDA wheels; no system CUDA toolkit or `nvcc` is needed (tested without).

## One command

```bash
git clone <this repo> && cd ISDN_Robofab
bash setup_env.sh                  # torch build picked from nvidia-smi
bash setup_env.sh --with-curobo    # + cuRobo (clones NVlabs/curobo at the pinned sha into ./curobo)
```

Options: `--cuda cu128|cu130` forces the torch build, `--cpu` installs CPU torch. `VENV=/path` and
`CUROBO_DIR=/path` relocate the venv and the cuRobo checkout. The script is idempotent: re-run it to
repair or upgrade. Time is dominated by the torch/CUDA download (about 3 GB): about 30 s to 1 min with a warm cache (measured), a few minutes on a cold one; the venv
is about 6 GB (torch/CUDA wheels), plus about 0.5 GB for the Newton asset cache and 0.3 GB for cuRobo.

After it, `bash brickassembly/scripts/run.sh <script>` automatically uses `.venv` for every script
except the BrickSim physics runs. `ROBOFAB_PY=/path/to/python` points it at another interpreter instead.
Without a `.venv` (and without `ROBOFAB_PY`) `run.sh` falls back to the old multi-environment routing.

## What the script does (manual steps)

```bash
uv venv .venv --python 3.12                       # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python torch==2.10.0 --torch-backend cu130     # cu128 | cpu
#   pip equivalent: .venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
# optional cuRobo:
git clone https://github.com/NVlabs/curobo.git curobo && git -C curobo checkout 78fd485fa82d9b9a063fb4985e371814587e666a
uv pip install --python .venv/bin/python -r requirements-curobo.txt
(cd curobo && uv pip install --python ../.venv/bin/python -e . --no-build-isolation --no-deps)
.venv/bin/python -c "import newton.utils as u; u.download_asset('franka_emika_panda')"   # FR3 model, cached in ~/.cache/newton
```

Pins live in `requirements.txt` (and `requirements-curobo.txt`); torch is installed separately because
its CUDA build depends on your driver. Key versions: Python 3.12, torch 2.10.0, newton 1.2.1,
warp-lang 1.13.0 (newton 1.2.1 requires `>=1.13,<1.14`), mujoco 3.8.0 and mujoco-warp 3.8.0.3, numpy 2.5.2.

## Verify

```bash
cd brickassembly
bash scripts/run.sh tests/test_aim.py                       # 18 passed; also test_vision, test_v4_vision, test_arm_protocol, test_manual
bash scripts/run.sh tests/test_brace_bandit.py              # 11 passed
bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 40000                    # 8 / 8 placed
bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 40000 --look --aim fk_vision   # 8 / 8 placed
bash scripts/run.sh experiments/v4_vision.py e1 --dry-run
bash scripts/run.sh scripts/status.py
bash scripts/run.sh scripts/12_curobo_probe.py --step 1     # with --with-curobo
bash scripts/run.sh dual_arm_sim.py --shape arch            # the live viewer (needs a desktop)
```

The test suite and `dual_arm_sim.py` runs rewrite tracked files (`tests/*.json`, `results/proto_episodes.jsonl`,
`results/v4/p3/step1.json`); `git checkout` them unless you mean to update them.

**MuJoCo version.** The frozen v3.1 twin was recorded with mujoco 3.3.7; the single env has 3.8.0
(`mujoco-warp` needs `~=3.8.0`). The twin tests `test_control` (9) and `test_env` (4) pass, `test_clutch`
passes 9 of 11: `test_scripted_insertion_from_2mm` fails exactly as in the recorded result (2/10
insertions, also on 3.3.7) and `test_energy_non_increasing` fails on 3.8.0 only, by a 5e-11 energy
step against a 1e-12 tolerance. To reproduce the twin's recorded numbers exactly use the legacy
`requirements-twin.txt` environment (`brickassembly/README.md`, "Legacy multi-environment setup").

## Not covered

- **BrickSim / Isaac Sim 5.1** (`run_plan.py`, `test_joint.py`, `test_gate.py`,
  `scripts/02_dynamic_calibration.py`, the historical v1-v3 physics runs). It pins its own Python 3.11,
  Isaac Sim 5.1 and bricksim and cannot share an environment. Set it up from `BrickSim/README.md`
  ("Install from Source": `cd BrickSim && uv sync --locked`); `run.sh` keeps routing those scripts to
  `BrickSim/.venv`.
- `scripts/00b_impedance_check.py` (WP0) needs Isaac Sim 6.0 / Isaac Lab and still routes to the Isaac
  interpreter in the legacy setup. The v4 code does not use Isaac Sim, only standalone Newton.
- Jupyter: the `Codes/*.ipynb` notebooks need `uv pip install --python .venv/bin/python ipykernel`.

## Troubleshooting

- **`driver supports CUDA x.y; torch needs >= 12.8`**: update the NVIDIA driver, or `--cpu`.
- **`torch.cuda.is_available()` is False / `CUDA error`**: `nvidia-smi` must work; do not mix a cu130 torch with a
  driver that reports CUDA 12.x. Re-run `bash setup_env.sh --cuda cu128` (reinstall torch: `rm -rf .venv` first).
- **Viewer: `pyglet`/GLFW "no display", `GLX`/`EGL` errors**: you need a desktop session (`echo $DISPLAY`); over SSH use
  `ssh -X` or `--viewer null`. On Wayland-only sessions install `xwayland`. `sudo apt install libgl1 libglu1-mesa` if
  `libGL.so` is missing. Hybrid-graphics laptops: `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`.
- **No network / asset download fails**: the FR3 model comes from `newton.utils.download_asset("franka_emika_panda")`
  and is cached under `~/.cache/newton`. Copy that directory from a connected machine to the same path; behind a proxy export `HTTPS_PROXY`.
  PyPI/PyTorch installs can use a mirror via `UV_INDEX_URL` / `PIP_INDEX_URL`.
- **First Warp run is slow**: kernels compile once into `~/.cache/warp`; later runs print `(cached)`.
- **`import curobo` resolves to `None` / a namespace package**: you are in a directory that contains a `curobo/` folder
  (e.g. the repo root). Run from elsewhere or from `brickassembly/`.
- **Python 3.12 not found without uv**: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv`
  (22.04), or install uv.
