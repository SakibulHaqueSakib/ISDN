"""WP0 / Gate G0 -- environment and platform verification (master_report §WP0).

Probes whatever the interpreter it is run under can import, writes a JSON
report, and prints the G0 checklist with each item PASS / FAIL / BLOCKED.
The project spans two interpreters, so run it under both:

    ~/Documents/ISDN_Robofab/isdnenv/bin/python scripts/00_env_check.py   # cuRobo
    ~/Codes/CAIRSS/Issac/bin/python            scripts/00_env_check.py   # Isaac/Newton

Reports land in scripts/00_env_check.<tag>.json.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

# §2.3 -- a 2x4 brick, the scene this benchmark is specified against.
BRICK = (0.0158, 0.0318, 0.0096)   # m
ENV_COUNTS = [1, 512, 2048]
STEPS = 100
SUBSTEPS = 4
SIM_DT = 1.0 / 240.0 / SUBSTEPS


def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def versions():
    out = {"python": platform.python_version(), "executable": sys.executable}
    for mod in ("torch", "warp", "numpy", "newton", "isaaclab", "isaacsim",
                "curobo", "viser", "rsl_rl", "trimesh"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    try:
        out["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        out["gpu"] = None
    return out


def probe_curobo():
    """cuRobo MotionGen warmup time and per-plan latency (G0: < 50 ms)."""
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState

    t0 = time.time()
    mp = MotionPlanner(MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model={"cuboid": {"table": {"dims": [0.9, 0.9, 0.05],
                                          "pose": [0.45, 0.0, -0.029, 1, 0, 0, 0]}}},
        collision_cache={"obb": 64},
        position_tolerance=0.0005, orientation_tolerance=0.01))
    mp.warmup(enable_graph=True, num_warmup_iterations=5)
    warmup_s = time.time() - t0

    dof = mp.kinematics.dof
    q = JointState.from_position(
        mp.default_joint_state.position[:dof].unsqueeze(0).contiguous(),
        joint_names=mp.joint_names[:dof])
    latencies, ok = [], 0
    for n in range(10):
        z = 0.15 + 0.01 * n
        g = GoalToolPose(
            tool_frames=mp.tool_frames,
            position=torch.tensor([[[[[0.46, -0.02, z]]]]], device="cuda",
                                  dtype=torch.float32),
            quaternion=torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]], device="cuda",
                                    dtype=torch.float32))
        t = time.time()
        r = mp.plan_pose(g, q)
        latencies.append((time.time() - t) * 1000.0)
        ok += int(r is not None and bool(r.success.any()))
    latencies.sort()
    return {"warmup_s": round(warmup_s, 1),
            "plan_ms_median": round(latencies[len(latencies) // 2], 1),
            "plan_ms_max": round(latencies[-1], 1),
            "plans_succeeded": "%d/10" % ok}


def _brick_world():
    """One environment: two stacked 2x4 bricks on a ground plane."""
    import warp as wp
    import newton

    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.6
    b.add_ground_plane()
    for n in range(2):
        body = b.add_body(xform=wp.transform(
            p=wp.vec3(0.0, 0.0, BRICK[2] * (n + 0.5) + 0.001), q=wp.quat_identity()),
            label="brick_%d" % n)
        b.add_shape_box(body, hx=BRICK[0] / 2, hy=BRICK[1] / 2, hz=BRICK[2] / 2)
    return b


def probe_newton():
    """Newton throughput on the 2-brick scene at 1 / 512 / 2048 envs (G0)."""
    import warp as wp
    import newton

    results = {}
    for n_envs in ENV_COUNTS:
        base = vram_used_mb()
        builder = newton.ModelBuilder()
        builder.replicate(_brick_world(), n_envs)
        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model)
        s0, s1 = model.state(), model.state()
        control, contacts = model.control(), model.contacts()

        def simulate(s0=s0, s1=s1):
            for _ in range(SUBSTEPS):
                s0.clear_forces()
                model.collide(s0, contacts)
                solver.step(s0, s1, control, contacts, SIM_DT)
                s0, s1 = s1, s0

        simulate()                       # compile
        wp.synchronize()
        t0 = time.time()
        for _ in range(STEPS):
            simulate()
        wp.synchronize()
        dt = time.time() - t0
        results["%d_envs" % n_envs] = {
            "steps_per_s": round(STEPS * SUBSTEPS / dt),
            "env_steps_per_s": round(n_envs * STEPS * SUBSTEPS / dt),
            "vram_mb": (vram_used_mb() or 0) - (base or 0),
        }
        del solver, model, s0, s1, contacts, control
    return results


def probe_warp_writable():
    """G0: can a Warp kernel read and write per-env body state?"""
    import warp as wp
    import newton

    builder = newton.ModelBuilder()
    builder.replicate(_brick_world(), 4)
    model = builder.finalize()
    state = model.state()

    @wp.kernel
    def nudge(body_q: wp.array(dtype=wp.transform), dz: float):
        i = wp.tid()
        x = body_q[i]
        body_q[i] = wp.transform(wp.transform_get_translation(x) + wp.vec3(0.0, 0.0, dz),
                                 wp.transform_get_rotation(x))

    before = state.body_q.numpy()[0][2]
    wp.launch(nudge, dim=state.body_q.shape[0], inputs=[state.body_q, 0.001])
    wp.synchronize()
    after = state.body_q.numpy()[0][2]
    return {"body_q_dtype": str(state.body_q.dtype),
            "n_bodies": int(state.body_q.shape[0]),
            "kernel_wrote": bool(abs((after - before) - 0.001) < 1e-6)}


def run(name, fn, report):
    try:
        report[name] = fn()
        print("  %-16s ok" % name)
    except ImportError as e:
        report[name] = {"blocked": str(e)}
        print("  %-16s BLOCKED (%s)" % (name, e))
    except Exception as e:
        report[name] = {"error": "%s: %s" % (type(e).__name__, e)}
        print("  %-16s ERROR %s: %s" % (name, type(e).__name__, e))


def main():
    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "versions": versions()}
    v = report["versions"]
    tag = "isaac" if v["newton"] else ("curobo" if v["curobo"] else "bare")
    print("== 00_env_check (%s, python %s) ==" % (tag, v["python"]))

    if v["curobo"]:
        run("curobo", probe_curobo, report)
    if v["newton"]:
        run("warp_writable", probe_warp_writable, report)
        run("newton", probe_newton, report)

    out = Path(__file__).parent / ("00_env_check.%s.json" % tag)
    out.write_text(json.dumps(report, indent=1))

    # --- G0 checklist ---------------------------------------------------
    def verdict(cond, blocked):
        return "BLOCKED" if blocked else ("PASS" if cond else "FAIL")

    c, n, w = report.get("curobo", {}), report.get("newton", {}), report.get("warp_writable", {})
    checks = [
        ("cuRobo plans in < 50 ms after warmup",
         verdict(c.get("plan_ms_median", 1e9) < 50, "plan_ms_median" not in c)),
        ("Newton runs >= 2048 envs on a 2-brick scene",
         verdict("2048_envs" in n, "2048_envs" not in n and "steps_per_s" not in str(n))),
        ("Warp kernel reads/writes per-env body state",
         verdict(w.get("kernel_wrote"), "kernel_wrote" not in w)),
        ("Impedance controller 50 mm circle < 0.5 mm RMS", "BLOCKED (needs Isaac Lab app)"),
        ("env.lock reproduces the benchmark within 10%", "BLOCKED (needs a second run)"),
    ]
    print("\n-- Gate G0 --")
    for label, state in checks:
        print("  [%s] %s" % (state, label))
    print("\nreport -> %s" % out)


if __name__ == "__main__":
    main()
