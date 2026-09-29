"""P3 -- cuRobo in-process, FR3 model, executed tracking, threaded soak (plan_v4 §3 P3, Gate P3).

    bash scripts/run.sh scripts/12_curobo_probe.py --step 1            # one step (repeatable)
    bash scripts/run.sh scripts/12_curobo_probe.py --step 1 --step 2 --step 3
    bash scripts/run.sh scripts/12_curobo_probe.py --all               # 1..8 in order
    bash scripts/run.sh scripts/12_curobo_probe.py --summary           # Gate P3 items from the step files

Steps (each writes results/v4/p3/stepN.json):
  1  import cuRobo under the Isaac interpreter (torch 2.10+cu128, warp 1.13) next to Newton, plan on Franka
  2  fit collision spheres to the FR3 collision meshes -> motion/fr3.yml (every vertex within 2 mm of the sphere
     union, 0.5 mm on the finger links; achieved maxima per link in step2.json)
  3  warm-up: capture every cuRobo graph, collision cache pre-sized, guard counts graph inits afterwards
  4  100 plan queries for B against A's spheres in random static configurations, a plate and 10 cuboids
  5  execute plans in Newton at time scales {1.0, 0.5, 0.3} with velocity feed-forward
  6  GPU memory per process (Newton cell + two planners) -> worker count for WP6
  7  threaded soak: plans from a ThreadPoolExecutor(1) worker while the cell steps at 60 fps, renders cameras
  8  contact-segment false positives on P0's insert recordings (results/v4/p0_inserts/*.npz)

Route 1 overlay (plan §2.4): `pip install --no-deps --target brickassembly/.vendor` into the Isaac interpreter's
pip, appended to sys.path by motion/vendor.py -- the Isaac env itself is untouched. Exact list and versions
(also in env.lock [v4]); cuRobo is the ../curobo checkout at 78fd485 (v0.8.0-43-g78fd485):

    cuda-core 1.2.1   numpy-quaternion 2024.0.13   pydot 4.0.1   py_trees 2.6.0   setuptools-scm 9.2.2   yourdfpy 0.0.60

ponytail: one file. WP2's motion/plan_curobo.py lifts the pieces it keeps (fr3_cfg, make_planner, the world builder).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from motion.vendor import add_overlay  # noqa: E402

add_overlay()
os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # noqa: E402
import warp as wp  # noqa: E402
import yaml  # noqa: E402

import newton  # noqa: E402

import dual_arm_sim as D  # noqa: E402  (arm builder, base poses, HOME_Q; not edited)
import planner as P  # noqa: E402
from cell.contacts import merge_events  # noqa: E402

OUT = HERE / "results" / "v4" / "p3"
FR3_YML = HERE / "motion" / "fr3.yml"           # the full model: fingers to 0.5 mm (contact-segment check, step 8)
FR3_PLAN_YML = HERE / "motion" / "fr3_plan.yml"  # the coarse planner model (sphere count sets plan latency, step 4)
P0_DIR = HERE / "results" / "v4"
MM = 1e-3
BASE = {"A": D.ARM_A, "B": D.ARM_B}            # (xyz, yaw) of each arm's base in the cell's world frame
PLATE = (0.0, 0.0, 0.0032, 0.40)                # cell.plate: centre xy, top z, side (m); a fixed slab for the probe
FINGER_LOCK = 0.0075                            # cuRobo locks the fingers: half-opening while carrying a 14.8 mm brick
# sphere-fit profiles (tolerance = every vertex and surface sample within tol of the union; prot = max protrusion), mm
PROFILES = {"full": dict(arm=(2.0, 8.0), finger=(0.5, 0.5), yml=FR3_YML),
            "plan": dict(arm=(2.0, 12.0), finger=(2.0, 3.0), yml=FR3_PLAN_YML),
            "mixed": dict(arm=(2.0, 12.0), finger=(0.5, 0.5), yml=OUT / "fr3_mixed.yml"),
            "lite": dict(arm=(2.0, 20.0), finger=(2.0, 4.0), yml=OUT / "fr3_lite.yml")}
N_CUBOID = 24                                   # collision-cache size = the run's largest world: table, plate, 10 cuboids, 10 other-arm links
                                                # (the kernels loop over the cache, not the active count: 64 slots cost +25 ms/plan, step 4)
TCP = D.EE                                      # fr3_hand_tcp body index in one arm


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, q):
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else float("nan")


def asset_dir():
    return newton.utils.download_asset("franka_emika_panda")


# --- frames: the planner works in each arm's base frame, the cell in the world frame -----------------------
def yaw_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def to_base(arm, pts):
    """World points (n, 3) -> the arm's base frame."""
    (bx, by, bz), yaw = BASE[arm]
    return (np.asarray(pts, float) - [bx, by, bz]) @ yaw_R(yaw)


def from_base(arm, pts):
    (bx, by, bz), yaw = BASE[arm]
    return np.asarray(pts, float) @ yaw_R(yaw).T + [bx, by, bz]


def cuboid_pose_base(arm, centre, yaw, R=None):
    """[x, y, z, qw, qx, qy, qz] of a box (z-yawed, or with world rotation R), world -> arm base frame."""
    p = to_base(arm, [centre])[0]
    Rw = Rot.from_euler("z", yaw).as_matrix() if R is None else np.asarray(R)
    x, y, z, w = Rot.from_matrix(yaw_R(-BASE[arm][1]) @ Rw).as_quat()
    return [*p, w, x, y, z]


# --- FR3 model ---------------------------------------------------------------------------------------------
def fr3_cfg(model="plan", pad_mm=0.0):
    """motion/fr3[_plan].yml with its URDF paths pointed at the copy Newton loads on this machine and the
    planner padding (cuRobo's collision_sphere_buffer: every robot sphere grows by it) set to pad_mm."""
    cfg = yaml.safe_load(PROFILES[model]["yml"].read_text())
    k = cfg["robot_cfg"]["kinematics"]
    k["collision_sphere_buffer"] = pad_mm * MM
    # the padding is for the world, not between the robot's own links: their ignore matrix was built for the unpadded
    # spheres, so self-collision keeps the unpadded radii (a negative self buffer cancels the sphere buffer)
    # pinned to curobo @ 78fd485: kinematics_loader.py:866-911 applies the sphere buffer to self-collision too and
    # passes the unmodified self_collision_buffer on; if cuRobo changes that, this would double-subtract
    k["self_collision_buffer"] = {l: float(v) - pad_mm * MM for l, v in k["self_collision_buffer"].items()}
    k["asset_root_path"] = str(asset_dir().parent)
    k["urdf_path"] = str(asset_dir() / "urdf/fr3_franka_hand.urdf")
    return cfg


class GraphGuard:
    """Counts cuRobo GraphExecutor initialisations and torch CUDA-graph captures (plan §2.4 discipline, item 3)."""

    def __init__(self):
        from curobo._src.util.cuda_graph_util import GraphExecutor
        self.inits, self.torch_captures, self.log = 0, 0, []
        guard = self
        for name in ("_initialize_cuda_graph", "_initialize_direct"):
            orig = getattr(GraphExecutor, name)

            def wrap(this, inputs, _orig=orig, _name=name):
                guard.inits += 1
                guard.log.append([threading.current_thread().name, _name,
                                  getattr(this._capture_fn, "__qualname__", str(this._capture_fn))])
                return _orig(this, inputs)
            setattr(GraphExecutor, name, wrap)
        orig_begin = torch.cuda.CUDAGraph.capture_begin

        def begin(this, *a, **k):
            guard.torch_captures += 1
            return orig_begin(this, *a, **k)
        torch.cuda.CUDAGraph.capture_begin = begin
        self.mark_inits = self.mark_torch = 0

    def mark(self):
        self.mark_inits, self.mark_torch = self.inits, self.torch_captures

    def since_mark(self):
        return {"graph_inits": self.inits - self.mark_inits, "torch_captures": self.torch_captures - self.mark_torch,
                "new": self.log[self.mark_inits:]}


_GUARD = None


def guard():
    global _GUARD
    if _GUARD is None:
        _GUARD = GraphGuard()
    return _GUARD


def make_planner(collision_cache=None, use_cuda_graph=True, scene=None, model="plan", pad_mm=0.0):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    guard()
    cfg = MotionPlannerCfg.create(robot=fr3_cfg(model, pad_mm), scene_model=scene,
                                  collision_cache=collision_cache or {"cuboid": N_CUBOID},
                                  use_cuda_graph=use_cuda_graph)
    return MotionPlanner(cfg)


def sync():
    torch.cuda.synchronize()
    wp.synchronize()


def gpu_used_mb():
    """This process's GPU memory from nvidia-smi (includes CUDA context, warp and torch pools)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            pid, mb = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mb)
    except Exception:
        pass
    return None


# ============================================================================================================
# step 1 -- import
# ============================================================================================================
def step1(args):
    t0 = time.time()
    import curobo
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: F401
    from curobo.types import GoalToolPose, JointState
    import cuda.core
    info = {"torch": torch.__version__, "warp": wp.__version__, "newton": newton.__version__ if hasattr(newton, "__version__") else "1.2.1",
            "curobo": getattr(curobo, "__version__", "?"), "curobo_file": curobo.__file__,
            "cuda_core": getattr(cuda.core, "__version__", "?"), "python": sys.version.split()[0],
            "device": torch.cuda.get_device_name(0), "import_s": round(time.time() - t0, 2)}
    # Newton and warp live in this process: build a tiny model and step it, then plan with cuRobo in the same process.
    b = newton.ModelBuilder()
    b.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
    m = b.finalize()
    s0, s1 = m.state(), m.state()
    solver = newton.solvers.SolverXPBD(m)
    solver.step(s0, s1, m.control(), None, 1 / 60)
    wp.synchronize()
    info["newton_step_ok"] = bool(np.isfinite(s1.body_q.numpy()).all())
    guard()
    planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml"))
    t = time.time()
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    info["franka_warmup_s"] = round(time.time() - t, 2)
    q0 = JointState.from_position(planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names)
    goal = GoalToolPose(tool_frames=planner.tool_frames,
                        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32))
    lat = []
    for _ in range(5):
        t = time.time()
        r = planner.plan_pose(goal, q0)
        sync()
        lat.append((time.time() - t) * 1000)
    info.update(franka_plan_ok=bool(r.success.any()), franka_plan_ms=[round(x, 1) for x in lat])
    info["ok"] = info["newton_step_ok"] and info["franka_plan_ok"]
    print(json.dumps(info, indent=1))
    save("step1.json", info)
    planner.destroy()
    return info


# ============================================================================================================
# step 2 -- sphere fit
# ============================================================================================================
def _fib(n):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


class _Union:
    """Signed distance to a union of watertight components (finger = four boxes); + outside, - inside."""

    def __init__(self, comps, dev):
        from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
        self.q = [WarpMeshQuery(m, dev) for m in comps]

    def sdf(self, pts):
        pts = pts.contiguous()
        return torch.stack([q.query_sdf(pts)[0] for q in self.q]).min(0).values


def fit_link(comps, tol, prot, dev, n_surf=6000, max_sph=600):
    """cuRobo's VOXEL fit (inscribed spheres) as the base, then a greedy patch until every mesh vertex and every
    surface sample is within `tol` of the sphere union, no sphere protruding more than `prot` from the mesh.

    Each patch is centred inward along the outward normal of the worst-covered point, the candidate (depth,
    radius) with the most newly covered points winning. Redundant spheres are pruned last. Distances in m.
    """
    import trimesh
    from curobo.sphere_fit import SphereFitType, estimate_sphere_count, fit_spheres_to_mesh
    from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery  # noqa: F401
    un = _Union(comps, dev)
    unit = torch.tensor(_fib(400), dtype=torch.float32, device=dev)
    V = np.concatenate([np.asarray(m.vertices) for m in comps])
    VN = np.concatenate([np.asarray(m.vertex_normals) for m in comps])
    area = np.array([m.area for m in comps])
    S, SN = [], []
    for m, a in zip(comps, area):
        p, f = trimesh.sample.sample_surface(m, max(int(n_surf * a / area.sum()), 1), seed=0)
        S.append(np.asarray(p))
        SN.append(np.asarray(m.face_normals)[f])
    S, SN = np.concatenate(S), np.concatenate(SN)

    def exposed(pts):   # drop points inside another component (the boxes of a finger overlap)
        return un.sdf(torch.tensor(pts, dtype=torch.float32, device=dev)).cpu().numpy() > -1e-4
    ev, es = exposed(V), exposed(S)
    V, VN, S, SN = V[ev], VN[ev], S[es], SN[es]
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    Pt, Nt = T(np.concatenate([V, S])), T(np.concatenate([VN, SN]))

    def gap(pts, C, R):
        return (torch.cdist(pts, C) - R[None]).min(1).values.clamp(min=0) if len(C) else torch.full((len(pts),), 1e3, device=dev)

    def protrusion(C, R, n=400):
        pts = (C[:, None, :] + R[:, None, None] * unit[None, :n]).reshape(-1, 3)
        return un.sdf(pts).max().item()

    mesh = trimesh.util.concatenate(comps) if len(comps) > 1 else comps[0]
    base = fit_spheres_to_mesh(mesh, num_spheres=estimate_sphere_count(mesh), fit_type=SphereFitType.VOXEL)
    C, R = base.centers.to(dev).float(), base.radii.to(dev).float()
    keep = [i for i in range(len(C)) if protrusion(C[i:i + 1], R[i:i + 1]) <= prot]
    C, R = C[keep], R[keep]
    n_base = len(C)
    te = 0.9 * tol
    hs = torch.tensor(np.geomspace(0.0004, 0.15, 24), dtype=torch.float32, device=dev)
    kap = torch.tensor([-te, 0.0, 0.45 * prot, 0.9 * prot], dtype=torch.float32, device=dev)
    d = gap(Pt, C, R)
    while d.max().item() > tol and len(C) < max_sph:
        i = int(d.argmax())
        u = -Nt[i] / Nt[i].norm()
        h = hs[:, None].expand(-1, len(kap)).reshape(-1)
        rr = h + kap[None, :].expand(len(hs), -1).reshape(-1)
        m = rr > 1e-4
        h, rr = h[m], rr[m]
        cc = Pt[i][None] + u[None] * h[:, None]
        pr = un.sdf((cc[:, None, :] + rr[:, None, None] * unit[None]).reshape(-1, 3)).reshape(len(rr), -1).max(1).values
        newd = torch.cdist(cc, Pt) - rr[:, None]
        gain = ((newd <= tol) & (d[None] > tol)).sum(1).float()
        gain[pr > 0.9 * prot] = -1
        gain[newd[:, i] > tol] = -2
        j = int(gain.argmax())
        C, R = torch.cat([C, cc[j:j + 1]]), torch.cat([R, rr[j:j + 1]])
        d = torch.minimum(d, newd[j]).clamp(min=0)
    D_ = torch.cdist(Pt, C) - R[None]
    alive = torch.ones(len(C), dtype=torch.bool, device=dev)
    for i in R.argsort().tolist():                       # prune, smallest first
        alive[i] = False
        if (D_[:, alive].min(1).values.clamp(min=0) > tol).any():
            alive[i] = True
    C, R = C[alive], R[alive]
    return dict(C=C, R=R, n_base=n_base, n=len(C),
                vertex_max_mm=gap(T(V), C, R).max().item() * 1000, surface_max_mm=gap(T(S), C, R).max().item() * 1000,
                protrusion_max_mm=protrusion(C, R) * 1000)


def step2(args):
    res = {}
    for prof in args.profiles:
        print("--- profile %s ---" % prof)
        res[prof] = _fit_profile(prof, args)
    res["ok"] = all(r["ok"] for r in res.values() if isinstance(r, dict))
    save("step2.json", {**(load("step2.json") or {}), **res})
    return res


def _fit_profile(prof, args):
    from curobo.robot_builder import RobotBuilder
    pr = PROFILES[prof]
    (atol, aprot), (ftol, fprot) = pr["arm"], pr["finger"]
    dev = torch.device("cuda", 0)
    adir = asset_dir()
    b = RobotBuilder(str(adir / "urdf/fr3_franka_hand.urdf"), str(adir.parent))
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("_step2_fit_%s.json" % prof)
    spheres, table = {}, {}
    reuse = args.reuse_fit and cache.exists()      # skip the fit + matrix (minutes); only rebuild the yml
    if reuse:
        c = json.loads(cache.read_text())
        spheres, table = c["spheres"], c["table"]
        b._collision_spheres, b._self_collision_ignore = spheres, c["ignore"]
    for link in ([] if reuse else b._mesh_link_names):
        comps = [g.get_trimesh_mesh(transform_with_pose=True) for g in b._parser.get_link_geometry(link, use_collision_mesh=True)]
        finger = "finger" in link
        tol, prot = (ftol, fprot) if finger else (atol, aprot)
        f = fit_link(comps, tol * MM, prot * MM, dev)
        spheres[link] = [{"center": [round(x, 6) for x in c], "radius": round(r, 6)}
                         for c, r in zip(f["C"].cpu().tolist(), f["R"].cpu().tolist())]
        table[link] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k not in ("C", "R")}
        table[link].update(limit_mm=tol, protrusion_limit_mm=prot, ok=bool(f["vertex_max_mm"] <= tol + 1e-3))
    for link, t in table.items():
        print("%-15s n=%3d (curobo base %2d)  vertex max %.2f mm  surface max %.2f mm  protrusion max %.2f mm  (limit %.1f / %.1f)  %s" % (
            link, t["n"], t["n_base"], t["vertex_max_mm"], t["surface_max_mm"], t["protrusion_max_mm"], t["limit_mm"],
            t["protrusion_limit_mm"], "OK" if t["ok"] else "FAIL"))
    if not reuse:
        b._collision_spheres = spheres      # ponytail: RobotBuilder has no public setter for externally patched spheres
        t0 = time.time()
        b.compute_collision_matrix(prune_collisions=True, num_samples=args.collision_samples, batch_size=100)
        print("self-collision matrix: %.0f s" % (time.time() - t0))
        cache.write_text(json.dumps({"spheres": spheres, "table": table, "ignore": b._self_collision_ignore}))
    tmp = OUT / "_fr3_builder.yml"
    b.save(b.build(), str(tmp))
    k = yaml.safe_load(tmp.read_text())["kinematics"]
    tmp.unlink()
    # the planner's robot: 7 arm joints, fingers locked, one tool frame (the TCP), a slot for a held brick
    k["tool_frames"] = ["fr3_hand_tcp"]
    k["lock_joints"] = {"fr3_finger_joint1": FINGER_LOCK}      # joint2 mimics joint1
    k["cspace"]["joint_names"] = [f"fr3_joint{i}" for i in range(1, 8)]
    for key in list(k["cspace"]):
        if isinstance(k["cspace"][key], list) and len(k["cspace"][key]) == 8:
            k["cspace"][key] = k["cspace"][key][:7]
    k["cspace"]["default_joint_position"] = [float(x) for x in D.HOME_Q]
    k["cspace"]["max_acceleration"] = [10.0] * 7
    k["cspace"]["max_jerk"] = [500.0] * 7
    k["extra_links"] = {"attached_object": {"fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint",
                                            "joint_type": "FIXED", "link_name": "attached_object",
                                            "parent_link_name": "fr3_hand_tcp"}}
    k["extra_collision_spheres"] = {"attached_object": 16}
    # fr3_link0 never moves and stands on the table (its spheres protrude 7 mm below z = 0): not a planner collision link
    k["collision_link_names"] = [l for l in k["collision_link_names"] if l != "fr3_link0"] + ["attached_object"]
    k["grasp_contact_link_names"] = ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "attached_object"]
    k["self_collision_ignore"]["attached_object"] = ["fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger",
                                                     "fr3_link6", "fr3_link7"]
    k["self_collision_buffer"]["attached_object"] = 0.0
    header = ("# FR3 + hand collision spheres, profile '%s', generated by scripts/12_curobo_probe.py --step 2 (do not hand-edit).\n"
              "# Links: the collision meshes of fr3_franka_hand.urdf, the file Newton loads. Fit: cuRobo VOXEL spheres, then a\n"
              "# greedy patch to vertices/surface within %.1f mm (arm) / %.1f mm (fingers), protrusion <= %.1f / %.1f mm.\n"
              "# urdf_path / asset_root_path are rewritten at load time (fr3_cfg): they point into Newton's asset cache.\n"
              "# Fingers are locked at %.4f m each (carrying a brick); the planner has 7 dof.\n" % (
                  prof, atol, ftol, aprot, fprot, FINGER_LOCK))
    pr["yml"].write_text(header + yaml.safe_dump({"robot_cfg": {"kinematics": k}, "load_dynamics": False}, sort_keys=False))
    total = sum(len(v) for l, v in spheres.items() if l != "fr3_link0")
    out = {"links": table, "total_spheres": total, "yml": str(pr["yml"].relative_to(HERE)), "finger_lock_m": FINGER_LOCK,
           "ok": all(t["ok"] for t in table.values())}
    out["fk"] = _fk_check(prof)      # the model is the same robot: cuRobo's TCP against Newton's
    out["ok"] = out["ok"] and out["fk"]["max_err_mm"] < 0.1
    print(json.dumps(out["fk"]), "total spheres", total)
    return out


def _fk_check(model="full"):
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    arm = D.build_arm()
    m = arm.finalize()
    st = m.state()
    rng = np.random.default_rng(0)
    lo, hi = m.joint_limit_lower.numpy()[:7], m.joint_limit_upper.numpy()[:7]
    errs = []
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(fr3_cfg(model)))
    for _ in range(8):
        q = np.clip(np.array(D.HOME_Q) + rng.normal(0, 0.4, 7), lo * 0.9, hi * 0.9)
        jq = m.joint_q.numpy().copy()
        jq[:7] = q
        jq[7:9] = FINGER_LOCK
        m.joint_q.assign(jq)
        newton.eval_fk(m, m.joint_q, m.joint_qd, st)
        newton_p = st.body_q.numpy()[TCP][:3]
        js = JointState.from_position(torch.tensor(q[None], dtype=torch.float32, device="cuda"), joint_names=kin.joint_names)
        cur = kin.compute_kinematics(js).tool_poses.get_link_pose("fr3_hand_tcp").position[0].cpu().numpy()
        errs.append(float(np.linalg.norm(newton_p - cur)) * 1000)
    return {"max_err_mm": round(max(errs), 4), "n": len(errs)}


# ============================================================================================================
# world model, queries, warm-up (steps 3-7 share these)
# ============================================================================================================
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

BRICK_DIMS = (0.0318, 0.0158, 0.0096)           # the 2x4 carried in the probe's brick-attached queries (m)
BRICK_IN_TCP = (0.0, 0.0, D.GRASP_DZ - BRICK_DIMS[2] / 2)   # brick centre in the TCP frame (tool z points at the fingertips)
TABLE = dict(name="table", dims=(2.0, 2.0, 0.1), c=(0.0, 0.0, -0.05), yaw=0.0)     # the real one: top at z = 0


def base_boxes():
    """Table and plate cuboids at the registered pose (world frame)."""
    return [TABLE, dict(name="plate", dims=(PLATE[3], PLATE[3], PLATE[2]), c=(PLATE[0], PLATE[1], PLATE[2] / 2), yaw=0.0)]


def scene_for(arm, boxes, mesh=None):
    """cuRobo scene for planning `arm`: the world's cuboids (+ optionally the other arm's sphere mesh) in its base frame."""
    from curobo.scene import Cuboid, Scene
    cub = [Cuboid(name=b["name"], dims=list(b["dims"]), pose=cuboid_pose_base(arm, b["c"], b.get("yaw", 0.0), b.get("R")))
           for b in boxes]
    return Scene(cuboid=cub)


_KIN = {}


def kin(model="plan"):
    """A stand-alone cuRobo kinematics of the FR3 (sphere poses for any configuration, both arms)."""
    if model not in _KIN:
        from curobo.kinematics import Kinematics, KinematicsCfg
        _KIN[model] = Kinematics(KinematicsCfg.from_robot_yaml_file(fr3_cfg(model)))
    return _KIN[model]


def arm_spheres(q7, model="plan"):
    """(N, 4) collision spheres [x y z r] of one arm at q7, in the arm's base frame (held-brick slots dropped)."""
    from curobo.types import JointState
    k = kin(model)
    js = JointState.from_position(torch.as_tensor(np.asarray(q7, np.float32)[None, None], device="cuda"), joint_names=k.joint_names)
    sp = k.compute_kinematics(js).robot_spheres.reshape(-1, 4).cpu().numpy()
    return sp[sp[:, 3] > 0]


def link_slices(model="full"):
    """{link: index array} into the rows of arm_spheres(q, model) (attached_object slots, radius < 0, are dropped)."""
    from curobo.types import JointState
    k = kin(model)
    kp = k.config.kinematics_config
    r = k.compute_kinematics(JointState.from_position(torch.zeros(1, 1, 7, device="cuda"), joint_names=k.joint_names)
                             ).robot_spheres.reshape(-1, 4)[:, 3].cpu().numpy()
    full = r > 0
    remap = np.cumsum(full) - 1
    out = {}
    for link in kp.link_name_to_idx_map:
        idx = kp.get_sphere_index_from_link_name(link).cpu().numpy()
        idx = idx[full[idx]]
        if len(idx):
            out[link] = remap[idx]
    return out


def other_arm_boxes(other, q_other):
    """The other arm as one oriented cuboid per link (world frame), from its plan-model spheres.

    Measured (step 4): the same arm as ONE refitted sphere mesh costs 4 s per plan (mesh SDF queries for ~450
    robot spheres); as ~11 cuboids the plan takes ~75 ms. Each cuboid is the PCA-oriented (of the centres) box that
    contains every sphere of the link (asserted): an over-approximation, tighter than a sphere-per-cube.
    """
    sp = arm_spheres(q_other)
    out = []
    for link, idx in _plan_links().items():
        c = from_base_pts(other, sp[idx, :3])
        r = sp[idx, 3]
        mu = c.mean(0)
        axes = np.linalg.svd(c - mu, full_matrices=False)[2] if len(c) > 2 else np.eye(3)     # PCA of the centres
        if np.linalg.det(axes) < 0:
            axes[2] *= -1
        loc = (c - mu) @ axes.T
        lo, hi = (loc - r[:, None]).min(0), (loc + r[:, None]).max(0)       # every sphere's own extent along the box axes
        box = dict(name="other_" + link, dims=tuple(hi - lo), c=tuple(mu + ((lo + hi) / 2) @ axes), R=axes.T)
        assert (obb_dist(c, box) + r <= 1e-6).all(), "other_arm_boxes: a sphere of %s sticks out of its box" % link
        out.append(box)
    return out


_PLAN_LINKS = None


def _plan_links():
    global _PLAN_LINKS
    if _PLAN_LINKS is None:
        _PLAN_LINKS = link_slices("plan")
    return _PLAN_LINKS


def from_base_pts(arm, pts):
    return from_base(arm, pts)


def goal_from(planner, pos, quat):
    from curobo.types import GoalToolPose
    return GoalToolPose(tool_frames=planner.tool_frames,
                        position=torch.tensor(np.asarray(pos, np.float32).reshape(1, 1, 1, 1, 3), device="cuda"),
                        quaternion=torch.tensor(np.asarray(quat, np.float32).reshape(1, 1, 1, 1, 4), device="cuda"))


def tool_down_quat_base(arm, yaw):
    """wxyz, in `arm`'s base frame, of the gripper pointing down and rotated `yaw` about the world z."""
    R = yaw_R(-BASE[arm][1]) @ Rot.from_euler("z", yaw).as_matrix() @ Rot.from_quat(D.Q_DOWN).as_matrix()
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def plan_ok(r):
    """cuRobo's plan_pose returns None when every attempt failed before trajectory optimisation."""
    return r is not None and bool(r.success.any())


def js7(q):
    from curobo.types import JointState
    return JointState.from_position(torch.as_tensor(np.asarray(q, np.float32).reshape(1, 7), device="cuda"), joint_names=kin().joint_names)


class Rig:
    """One MotionPlanner per arm with the collision cache pre-sized and a brick slot."""

    def __init__(self, arms=("B",), use_cuda_graph=True, model="plan", pad_mm=0.0):
        self.pl = {a: make_planner(use_cuda_graph=use_cuda_graph, scene=scene_for(a, base_boxes()), model=model, pad_mm=pad_mm)
                   for a in arms}
        self.brick_spheres = None

    def set_world(self, arm, boxes, other_q=None):
        """Reload the cuboids (in place, the cache never grows): the world plus the other arm at other_q."""
        other = "A" if arm == "B" else "B"
        self.pl[arm].update_world(scene_for(arm, list(boxes) + other_arm_boxes(other, D.HOME_Q if other_q is None else other_q)))

    def attach_brick(self, arm, q7):
        """Held-brick spheres in the TCP-frame slot `attached_object` (fitted once, then written in place)."""
        from curobo.scene import Cuboid
        pl = self.pl[arm]
        am = pl.attachment_manager
        if self.brick_spheres is None:
            self.brick_spheres = am.fit_spheres([Cuboid(name="held", dims=list(BRICK_DIMS), pose=[*BRICK_IN_TCP, 1, 0, 0, 0])], num_spheres=12)
        am.update(self.brick_spheres, js7(q7))

    def detach_brick(self, arm):
        self.pl[arm].attachment_manager.detach()


def ik_free(pl, pos, quat):
    """A collision-free IK solution (7,) for the TCP pose in the planner's base frame, or None."""
    r = pl.ik_solver.solve_pose(goal_from(pl, pos, quat), return_seeds=4, current_state=pl.default_joint_state.clone().unsqueeze(0))
    ok = r.success.reshape(-1)
    if not bool(ok.any()):
        return None
    return r.solution.reshape(-1, 7)[int(ok.nonzero()[0])].cpu().numpy()


def random_boxes(rng, n=10):
    out = []
    for i in range(n):
        d = rng.uniform(0.02, 0.09, 3)
        d[2] = rng.uniform(0.02, 0.15)
        c = (rng.uniform(-0.05, 0.40), rng.uniform(-0.45, 0.15), d[2] / 2)
        out.append(dict(name="obs%d" % i, dims=tuple(d), c=c, yaw=float(rng.uniform(0, math.pi))))
    return out


def random_pose(rng, arm="B"):
    pos_w = np.array([rng.uniform(-0.05, 0.40), rng.uniform(-0.45, 0.15), rng.uniform(0.04, 0.32)])
    return to_base(arm, [pos_w])[0], tool_down_quat_base(arm, rng.uniform(-math.pi / 2, math.pi / 2)), pos_w


def random_other_q(rng):
    """A's random static configuration: HOME_Q perturbed inside the joint limits."""
    lo = np.array([-2.74, -1.78, -2.90, -3.04, -2.80, 0.54, -3.01]) * 0.9
    hi = np.array([2.74, 1.78, 2.90, -0.15, 2.80, 4.51, 3.01]) * 0.9
    return np.clip(np.array(D.HOME_Q) + rng.normal(0, 0.5, 7), lo, hi)


def warm(rig, log=print):
    """Plan-of-record §2.4 discipline: capture every cuRobo graph BEFORE the loop starts.

    Per arm: the planner's own warm-up (IK, trajopt, graph planner), then free-space and brick-attached plan_pose
    with the full world loaded (cuboids and the other-arm mesh, so both collision branches are inside the graphs),
    a plan that fails its first attempt (a wall between start and goal) to reach the graph-seeded attempt, and a
    plan_cspace. Every one at the fixed batch (1) and goalset (1) the run uses. Returns the guard's count.
    """
    g = guard()
    rng = np.random.default_rng(1)
    t0 = time.time()
    phase = {}
    for a, pl in rig.pl.items():
        rig.set_world(a, base_boxes() + random_boxes(rng, 10))
        t1 = time.time()
        pl.warmup(enable_graph=True, num_warmup_iterations=5)
        sync()
        phase["%s_planner_warmup_s" % a] = round(time.time() - t1, 2)
        t1 = time.time()
        for attached in (False, True):
            if attached:
                rig.attach_brick(a, D.HOME_Q)
            for _ in range(3):
                p0, q0_, _ = random_pose(rng, a)
                p1, q1_, _ = random_pose(rng, a)
                s, e = ik_free(pl, p0, q0_), ik_free(pl, p1, q1_)
                if s is not None and e is not None:
                    pl.plan_pose(goal_from(pl, p1, q1_), js7(s), max_attempts=3)
            if attached:
                rig.detach_brick(a)
        wall = [dict(name="wall", dims=(0.02, 0.5, 0.5), c=(0.15, -0.15, 0.25), yaw=0.0)]
        rig.set_world(a, base_boxes() + wall)
        s = ik_free(pl, *tool_pose_w(a, (0.35, -0.15, 0.12), 0.0))
        e = ik_free(pl, *tool_pose_w(a, (-0.05, -0.15, 0.12), 0.0))
        if s is not None and e is not None:
            pl.plan_pose(goal_from(pl, *tool_pose_w(a, (-0.05, -0.15, 0.12), 0.0)), js7(s), max_attempts=5)
            pl.plan_cspace(js7(e), js7(s), max_attempts=3)
        rig.set_world(a, base_boxes())
        pl.reset_seed()
        sync()
        phase["%s_query_types_s" % a] = round(time.time() - t1, 2)
    sync()
    g.mark()
    return {"graph_inits": g.inits, "torch_captures": g.torch_captures, "warm_s": round(time.time() - t0, 1), "phase_s": phase,
            "captured": [x[2] for x in g.log]}


def tool_pose_w(arm, pos_w, yaw):
    return to_base(arm, [pos_w])[0], tool_down_quat_base(arm, yaw)


# ============================================================================================================
# step 3 -- warm-up
# ============================================================================================================
def step3(args):
    rig = Rig(("A", "B"))
    info = warm(rig)
    # the guard: fresh worlds, attached brick, retries -- nothing may initialise a graph now
    rng = np.random.default_rng(7)
    n_ok = 0
    for i in range(args.n_check):
        for a, pl in rig.pl.items():
            rig.set_world(a, base_boxes() + random_boxes(rng, 10), random_other_q(rng))
            if i % 2:
                rig.attach_brick(a, D.HOME_Q)
            p0, q0_, _ = random_pose(rng, a)
            p1, q1_, _ = random_pose(rng, a)
            s, e = ik_free(pl, p0, q0_), ik_free(pl, p1, q1_)
            if s is not None and e is not None:
                r = pl.plan_pose(goal_from(pl, p1, q1_), js7(s), max_attempts=5)
                n_ok += int(bool(r.success.any()))
            if i % 2:
                rig.detach_brick(a)
    sync()
    after = guard().since_mark()
    info.update(post_warmup=after, checked_plans=n_ok, total_spheres=int(rig.pl["B"].kinematics.total_spheres),
                ok=after["graph_inits"] == 0 and after["torch_captures"] == 0)
    print(json.dumps({k: v for k, v in info.items() if k != "captured"}, indent=1, default=str))
    print("captured during warm-up (%d):" % len(info["captured"]), info["captured"])
    save("step3.json", info)
    return info


# ============================================================================================================
# geometry helpers: sphere-vs-cuboid clearances (numpy), spheres of an arm in the world frame
# ============================================================================================================
def box_frame(b):
    """(centre, R world<-box, half extents) of a box dict."""
    R = np.asarray(b["R"]) if b.get("R") is not None else yaw_R(b.get("yaw", 0.0))
    return np.asarray(b["c"], float), R, np.asarray(b["dims"], float) / 2


def obb_dist(pts, b):
    """Signed distance from points (n, 3) to the surface of box b (outside +)."""
    c, R, h = box_frame(b)
    q = np.abs((np.asarray(pts) - c) @ R) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def clearance_boxes(sph, boxes):
    """Min over spheres (n, 4) and boxes of (surface distance - radius), and the box it is against."""
    best, who = np.inf, None
    for b in boxes:
        m = float((obb_dist(sph[:, :3], b) - sph[:, 3]).min())
        if m < best:
            best, who = m, b["name"]
    return best, who


def clearance_spheres(sa, sb):
    """Min gap between two sphere sets (n, 4), (m, 4)."""
    d = np.linalg.norm(sa[:, None, :3] - sb[None, :, :3], axis=2) - sa[:, None, 3] - sb[None, :, 3]
    return float(d.min())


def arm_spheres_batch(Q, model="full"):
    """(T, N, 4) spheres of one arm over T configurations Q (T, 7), base frame, held-brick slots dropped."""
    from curobo.types import JointState
    k = kin(model)
    out = []
    for i in range(0, len(Q), 256):
        js = JointState.from_position(torch.as_tensor(np.asarray(Q[i:i + 256], np.float32)[:, None, :], device="cuda"), joint_names=k.joint_names)
        sp = k.compute_kinematics(js).robot_spheres[:, 0].cpu().numpy()
        out.append(sp[:, sp[0, :, 3] > 0])
    return np.concatenate(out)


def arm_world_spheres(arm, Q, model="full"):
    """(T, N, 4) world-frame spheres of `arm` (fr3_link0 is not in the model: it never moves and stands on the table)."""
    sp = arm_spheres_batch(Q, model)
    sp[..., :3] = from_base(arm, sp[..., :3].reshape(-1, 3)).reshape(sp[..., :3].shape)
    return sp


def tcp_poses(arm, Q):
    """World TCP positions (T, 3) and rotations (T, 3, 3) of `arm`."""
    from curobo.types import JointState
    k = kin("full")
    js = JointState.from_position(torch.as_tensor(np.asarray(Q, np.float32)[:, None, :], device="cuda"), joint_names=k.joint_names)
    pose = k.compute_kinematics(js).tool_poses.get_link_pose("fr3_hand_tcp")
    pos = from_base(arm, pose.position.reshape(-1, 3).cpu().numpy())
    w, x, y, z = pose.quaternion.reshape(-1, 4).cpu().numpy().T
    R = Rot.from_quat(np.stack([x, y, z, w], 1)).as_matrix()
    return pos, yaw_R(BASE[arm][1]) @ R


# ============================================================================================================
# the Newton cell: dual_arm_sim's scene, stepped by joint targets (no behaviour tree), ContactMonitor attached
# ============================================================================================================
class Cell:
    def __init__(self, viewer="null", shape="cube"):
        import argparse as ap
        sim = OUT / "_sim"                  # make_plan writes plans/sim_*.json under D.HERE: keep tracked files untouched
        (sim / "plans").mkdir(parents=True, exist_ok=True)
        (sim / "results").mkdir(parents=True, exist_ok=True)
        D.HERE = sim
        if viewer == "gl":
            v = newton.viewer.ViewerGL(headless=False)
        else:
            v = newton.viewer.ViewerNull(num_frames=10 ** 9)
        self.viewer_kind = viewer
        # Newton's MuJoCo solver only reads joint_target_vel for POSITION_VELOCITY joints; the URDF import makes them
        # POSITION (measured: feed-forward had no effect). dual_arm_sim.build_arm is wrapped, not edited: the 7 arm
        # dofs get a velocity actuator with kd = joint_target_kd (the same damping as before at zero velocity target).
        if not getattr(D.build_arm, "_p3_ff", False):
            _orig = D.build_arm

            def build_arm_ff(*a, **k):        # dual_arm_sim.build_arm(substeps, finger_ke, finger_kd)
                b = _orig(*a, **k)
                b.joint_target_mode[:7] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 7
                return b
            build_arm_ff._p3_ff = True
            D.build_arm = build_arm_ff
        args = ap.Namespace(front=None, side=None, top=None, width=None, depth=None, name=None, shape=shape, structure=None,
                            strategy="weakest_joint", brace_model="grasp_lp", viewer=viewer)
        self.ex = D.Example(v, args)
        from cell.contacts import ContactMonitor
        if getattr(self.ex, "monitor", None) is None:
            self.ex.monitor = ContactMonitor(self.ex)     # dual_arm_sim's own step() is not used here
        self.mon = self.ex.monitor
        self.nb = self.ex.n_arm_bodies
        self.st_plan = self.ex.model.state()
        self.have_vel = getattr(self.ex.control, "joint_target_vel", None) is not None

    # -- state ---------------------------------------------------------------------------------------------
    def q18(self):
        return self.ex.state_0.joint_q.numpy()[:18].copy()

    def reset(self, qA, qB, settle=45):
        ex = self.ex
        jq = ex.state_0.joint_q.numpy()
        jq[:18] = [*qA, FINGER_LOCK, FINGER_LOCK, *qB, FINGER_LOCK, FINGER_LOCK]
        ex.state_0.joint_q.assign(jq)
        jqd = ex.state_0.joint_qd.numpy()
        jqd[:18] = 0
        ex.state_0.joint_qd.assign(jqd)
        newton.eval_fk(ex.model, ex.state_0.joint_q, ex.state_0.joint_qd, ex.state_0)
        for _ in range(settle):
            self.frame(qA, qB)

    def frame(self, qA, qB, vA=None, vB=None):
        """One 60 fps frame: joint position (+ velocity feed-forward) targets for both arms, 16 substeps, monitor."""
        ex = self.ex
        tgt = ex.control.joint_target_pos.numpy()
        tgt[:18] = [*qA, FINGER_LOCK, FINGER_LOCK, *qB, FINGER_LOCK, FINGER_LOCK]
        ex.control.joint_target_pos.assign(tgt)
        if self.have_vel:
            tv = ex.control.joint_target_vel.numpy()
            tv[:18] = [*(np.zeros(7) if vA is None else vA), 0, 0, *(np.zeros(7) if vB is None else vB), 0, 0]
            ex.control.joint_target_vel.assign(tv)
        wp.capture_launch(ex.graph) if ex.graph else ex.simulate()
        ex.sim_time += ex.frame_dt
        ex.frame = getattr(ex, "frame", 0) + 1
        t = time.perf_counter()
        if self.mon:
            self.mon.step(ex)
        self.t_mon = (time.perf_counter() - t) * 1000

    def body_pos(self, state=None):
        return (state or self.ex.state_0).body_q.numpy()[:, :3]

    def fk_bodies(self, q18):
        """World positions of all bodies with the arms at q18 (planned configuration), via Newton's own FK."""
        jq = self.st_plan.joint_q.numpy() if self.st_plan.joint_q is not None else self.ex.model.joint_q.numpy()
        jq = self.ex.state_0.joint_q.numpy()
        jq[:18] = q18
        self.st_plan.joint_q.assign(jq)
        newton.eval_fk(self.ex.model, self.st_plan.joint_q, self.st_plan.joint_qd, self.st_plan)
        return self.st_plan.body_q.numpy()[:, :3]

    def tray_boxes(self):
        """Bricks waiting in B's feeder as cuboids (studs included), world frame."""
        ex, out = self.ex, []
        for bid, xyz in ex.feeder.items():
            L, W = ex.dims[bid]
            h = 0.0096 + 0.0017
            out.append(dict(name="tray_" + bid, dims=(L, W, h), c=(xyz[0], xyz[1], 0.0005 + h / 2), yaw=ex.target[bid][1]))
        return out


def resample(traj, scale, fps=60):
    """cuRobo trajectory (positions q (T,7), velocities v, dt) -> per-frame q_ref and time-scaled feed-forward qd_ref."""
    q, v, dt = traj
    t = np.arange(len(q)) * dt
    tau = np.arange(0, t[-1] * 1.0, scale / fps)        # planned-time of each frame: slower execution = smaller step
    qr = np.stack([np.interp(tau, t, q[:, j]) for j in range(7)], 1)
    vr = np.stack([np.interp(tau, t, v[:, j]) for j in range(7)], 1) * scale
    return np.vstack([qr, q[-1:]]), np.vstack([vr, np.zeros((1, 7))])


def plan_traj(res, pl):
    """(q, v, dt) numpy, 7 joints, from a cuRobo plan result (its buffer's static tail and the locked finger column dropped)."""
    js = res.get_interpolated_plan()
    q = js.position.reshape(-1, js.position.shape[-1])[:, :7].cpu().numpy()
    v = js.velocity.reshape(-1, js.velocity.shape[-1])[:, :7].cpu().numpy() if js.velocity is not None else np.gradient(q, axis=0)
    moving = np.nonzero(np.abs(np.diff(q, axis=0)).max(1) > 1e-6)[0]
    end = int(moving[-1]) + 2 if len(moving) else 2
    return q[:end], v[:end], float(pl.trajopt_solver.config.interpolation_dt)


# ============================================================================================================
# step 4 -- 100 plan queries
# ============================================================================================================
def feasible_query(pl, rig, rng, arm="B", tries=40):
    """A random pair (start q, goal pose) that both have a collision-free IK solution in the loaded world."""
    skipped = 0
    for _ in range(tries):
        p0, q0_, _ = random_pose(rng, arm)
        p1, q1_, _ = random_pose(rng, arm)
        s, e = ik_free(pl, p0, q0_), ik_free(pl, p1, q1_)
        if s is not None and e is not None:
            return s, (p1, q1_), e, skipped
        skipped += 1
    return None, None, None, skipped


def run_queries(model, pad_mm, n, seed):
    """n random plan queries for B (plate + 10 cuboids + A's link boxes in a random static configuration; half of them
    with a held brick) at planner padding pad_mm: latency and success on the feasible ones."""
    rig = Rig(("B",), model=model, pad_mm=pad_mm)
    pl = rig.pl["B"]
    warm_info = warm(rig)
    rng = np.random.default_rng(seed)
    rows, skipped_total = [], 0
    for i in range(n):
        attached = bool(i % 2)
        t = time.perf_counter()
        rig.set_world("B", base_boxes() + random_boxes(rng, 10), random_other_q(rng))
        sync()
        t_world = (time.perf_counter() - t) * 1000
        if attached:
            rig.attach_brick("B", D.HOME_Q)
        s, goal, _, sk = feasible_query(pl, rig, rng)
        skipped_total += sk
        if s is None:
            rows.append(dict(i=i, feasible=False))
            continue
        t = time.perf_counter()
        r = pl.plan_pose(goal_from(pl, *goal), js7(s), max_attempts=5)
        sync()
        ms = (time.perf_counter() - t) * 1000
        ok = plan_ok(r)
        rows.append(dict(i=i, feasible=True, ok=ok, plan_ms=ms, world_ms=t_world, attached=attached,
                         traj_s=float(len(plan_traj(r, pl)[0]) * pl.trajopt_solver.config.interpolation_dt) if ok else None))
        if attached:
            rig.detach_brick("B")
        print("  q%3d %s %s %6.1f ms (world %.1f ms)" % (i, "brick" if attached else "free ", "ok  " if ok else "FAIL", ms, t_world), flush=True)
    feas = [r for r in rows if r["feasible"]]
    lat = [r["plan_ms"] for r in feas]
    out = {"model": model, "padding_mm": pad_mm, "total_spheres": int(pl.kinematics.total_spheres), "queries": len(rows), "feasible": len(feas),
           "skipped_ik_infeasible_pairs": skipped_total, "success": sum(r["ok"] for r in feas),
           "success_rate": (sum(r["ok"] for r in feas) / len(feas)) if feas else None,
           "plan_ms": {"median": pct(lat, 50), "p95": pct(lat, 95), "max": max(lat) if lat else None,
                       "median_success_only": pct([r["plan_ms"] for r in feas if r["ok"]], 50)},
           "world_update_ms_median": pct([r["world_ms"] for r in feas], 50),
           "attached_median_ms": pct([r["plan_ms"] for r in feas if r["attached"]], 50),
           "free_median_ms": pct([r["plan_ms"] for r in feas if not r["attached"]], 50),
           "post_warmup_captures": guard().since_mark(), "warm": {k: v for k, v in warm_info.items() if k != "captured"}, "rows": rows}
    out["gate"] = bool((out["success_rate"] or 0) >= 0.95 and out["plan_ms"]["median"] <= 100 and out["post_warmup_captures"]["graph_inits"] == 0)
    del rig
    torch.cuda.empty_cache()
    return out


def step4(args):
    res = {}
    for model in args.models:
        out = run_queries(model, args.padding_mm, args.n_queries, args.seed)
        out["sample_ok"] = out["queries"] >= 100                    # the plan's sample size
        out["ok"] = out["gate"] and out["sample_ok"]
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
        res[model] = out
    save("step4.json", {**(load("step4.json") or {}), **res})
    return res


# ============================================================================================================
# step 5 -- execute plans in Newton
# ============================================================================================================
TABLE_TRUE = TABLE          # what Newton's ground plane is
MOVING = range(2, 14)          # local body indices of an arm that move: link1 .. link8, hand, tcp, both fingers


def true_boxes(extra=()):
    return [TABLE_TRUE, base_boxes()[1], *extra]


def gap_over_time(spheres_T, boxes, other=None):
    """Per-frame min clearance (m) of spheres (T, N, 4) to boxes (and to a static sphere set `other`)."""
    T, N, _ = spheres_T.shape
    pts = spheres_T[..., :3].reshape(-1, 3)
    d = np.full((T, N), np.inf)
    for b in boxes:
        d = np.minimum(d, obb_dist(pts, b).reshape(T, N) - spheres_T[..., 3])
    gap = d.min(1)
    if other is not None:
        so = torch.as_tensor(other[:, :3], dtype=torch.float32, device="cuda")
        ro = torch.as_tensor(other[:, 3], dtype=torch.float32, device="cuda")
        g2 = []
        for i in range(T):
            sb = torch.as_tensor(spheres_T[i], dtype=torch.float32, device="cuda")
            g2.append((torch.cdist(sb[:, :3], so) - sb[:, 3:4] - ro[None]).min().item())
        gap = np.minimum(gap, np.array(g2))
    return gap


def execute(cell, qA, traj, scale, ff=True, tail=30):
    """Run one plan in Newton at a time scale: targets at 60 fps, velocity feed-forward when ff. Returns metrics."""
    qref, vref = resample(traj, scale)
    q0 = qref[0]
    cell.reset(qA, q0)
    mon = cell.mon
    n0 = len(mon.raw)
    body_exec, q_exec = [], []
    n = len(qref) + tail
    for f in range(n):
        k = min(f, len(qref) - 1)
        cell.frame(qA, qref[k], None, vref[k] if (ff and f < len(qref) - 1) else None)
        body_exec.append(cell.body_pos()[cell.nb + 2: 2 * cell.nb].copy())
        q_exec.append(cell.q18()[9:16])
    mon._close()
    events = merge_events(mon.raw[n0:])          # the monitor fills .events/.counts only in finalize()
    body_plan = np.stack([cell.fk_bodies(np.r_[qA, FINGER_LOCK, FINGER_LOCK, qref[min(f, len(qref) - 1)], FINGER_LOCK, FINGER_LOCK])[cell.nb + 2: 2 * cell.nb]
                          for f in range(n)])
    dev = np.linalg.norm(np.stack(body_exec) - body_plan, axis=2) * 1000          # (frames, links) mm
    q_exec = np.stack(q_exec)
    return {"scale": scale, "ff": ff, "frames": n, "duration_s": len(qref) / 60,
            "dev_mm": dev[:, [i - 2 for i in MOVING if i - 2 < dev.shape[1]]], "q_exec": q_exec, "qref": qref,
            "events": events, "final_err_rad": float(np.abs(q_exec[-1] - qref[-1]).max())}


VEL_LIMIT = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])      # rad/s, the URDF's joint limits (what cuRobo plans to)
ACC_LIMIT = 10.0                                                       # rad/s^2, max_acceleration in fr3*.yml


def limit_fractions(traj, scale):
    """Peak velocity and acceleration of a plan executed at `scale`, as fractions of the planner's limits.

    scale 1.0 executes the plan as cuRobo timed it (its velocity limits are the URDF's, its acceleration limit is
    max_acceleration); scale s stretches time by 1/s: velocities x s, accelerations x s^2."""
    q, v, dt = traj
    a = np.gradient(v, dt, axis=0)
    return float((np.abs(v) / VEL_LIMIT).max() * scale), float(np.abs(a).max() / ACC_LIMIT * scale ** 2)


def arm_arm_control(cell, rig, qA):
    """Positive control for the monitor: B is driven straight (joint interpolation) onto A's hand; an arm_arm event
    must appear, otherwise a zero elsewhere means nothing."""
    pl = rig.pl["B"]
    tcpA = tcp_poses("A", qA[None])[0][0]
    pl.update_world(scene_for("B", []))
    qB0 = cell.q18()[9:16]
    goal = ik_free(pl, *tool_pose_w("B", tcpA + [0.0, 0.0, 0.02], 0.0))
    assert goal is not None, "arm_arm_control: no IK for B at A's hand"
    n, dt = 60, 0.05
    q = np.linspace(qB0, goal, n)
    v = np.gradient(q, dt, axis=0)
    m = execute(cell, qA, (q, v, dt), 1.0, True)
    counts = {}
    for e in m["events"]:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    return {"events": counts, "detected": counts.get("arm_arm", 0) > 0}


def step5(args):
    cell = Cell("null")               # Newton's graphs are captured first, as in the run
    rig = Rig(("B",), pad_mm=args.padding_mm)
    warm_info = warm(rig)
    pl = rig.pl["B"]
    rng = np.random.default_rng(args.seed)
    tray = cell.tray_boxes()
    plans, tries = [], 0
    while len(plans) < args.n_exec and tries < 10 * args.n_exec:
        tries += 1
        qA = random_other_q(rng)
        spA = arm_world_spheres("A", qA[None], "full")[0]
        if clearance_boxes(spA, true_boxes(tray))[0] < 0.03 or spA[:, 2].min() < 0.0:     # A must stand clear of the cell
            continue
        rig.set_world("B", base_boxes() + tray, qA)
        s, goal, _, _ = feasible_query(pl, rig, rng)
        if s is None:
            continue
        r = pl.plan_pose(goal_from(pl, *goal), js7(s), max_attempts=5)
        if plan_ok(r):
            plans.append(dict(qA=qA, traj=plan_traj(r, pl)))
    print("%d plans in %d tries" % (len(plans), tries), flush=True)
    rows = []
    pool = {sc: [] for sc in args.scales}
    for i, pl_ in enumerate(plans):
        for sc in args.scales:
            for ff in ([True, False] if (i == 0 and sc == 1.0) else [True]):
                m = execute(cell, pl_["qA"], pl_["traj"], sc, ff)
                Bsp = arm_world_spheres("B", m["q_exec"], "full")
                Bref = arm_world_spheres("B", m["qref"], "full")
                Aw = arm_world_spheres("A", pl_["qA"][None], "full")[0]
                clr_e = gap_over_time(Bsp, true_boxes(tray), Aw).min()
                clr_p = gap_over_time(Bref, true_boxes(tray), Aw).min()
                d = m["dev_mm"]
                ev = {}
                for e in m["events"]:
                    ev[e["class"]] = ev.get(e["class"], 0) + 1
                vf, af = limit_fractions(pl_["traj"], sc)
                row = {"plan": i, "scale": sc, "ff": ff, "duration_s": round(m["duration_s"], 2), "dev_p99_mm": pct(d.ravel(), 99),
                       "dev_max_mm": float(d.max()), "dev_tcp_p99_mm": pct(d[:, 9], 99), "final_err_rad": m["final_err_rad"],
                       "clearance_exec_mm": clr_e * 1000, "clearance_plan_mm": clr_p * 1000, "events": ev,
                       "arm_arm_events": ev.get("arm_arm", 0), "peak_vel_frac": vf, "peak_acc_frac": af}
                rows.append(row)
                if ff:
                    pool[sc].append(d.ravel())
                print("  plan %2d scale %.1f ff=%d  dev p99 %.2f mm max %.2f  clearance exec %.1f / plan %.1f mm  events %s" % (
                    i, sc, ff, row["dev_p99_mm"], row["dev_max_mm"], row["clearance_exec_mm"], row["clearance_plan_mm"], ev), flush=True)
    control = arm_arm_control(cell, rig, plans[0]["qA"] if plans else cell.q18()[:7])
    print("positive control (B driven into A):", control, flush=True)
    per_scale = {}
    for sc in args.scales:
        allv = np.concatenate(pool[sc]) if pool[sc] else np.array([])
        p99 = pct(allv, 99)
        rs = [r for r in rows if r["scale"] == sc and r["ff"]]
        per_scale[str(sc)] = {"dev_p99_mm": p99, "dev_max_mm": float(allv.max()) if len(allv) else None,
                              "padding_needed_mm": max(5.0, p99 + 2.0),
                              "min_clearance_exec_mm": min([r["clearance_exec_mm"] for r in rs], default=None),
                              "peak_vel_frac": max([r["peak_vel_frac"] for r in rs], default=None),
                              "peak_acc_frac": max([r["peak_acc_frac"] for r in rs], default=None),
                              "arm_arm_events": sum(r["arm_arm_events"] for r in rs)}
    # (time scale, padding) pairs: at the padding each scale needs, do the plans still meet latency and success?
    pairs = []
    for sc in args.scales:
        need = math.ceil(per_scale[str(sc)]["padding_needed_mm"])
        q = run_queries("plan", float(need), args.n_pad_queries, args.seed + 1)
        pairs.append({"scale": sc, "padding_mm": need, "dev_p99_mm": per_scale[str(sc)]["dev_p99_mm"], "queries": q["queries"],
                      "feasible": q["feasible"], "success_rate": q["success_rate"], "plan_ms_median": q["plan_ms"]["median"],
                      "plan_ms_p95": q["plan_ms"]["p95"], "meets_latency_success": q["gate"]})
        print("  pair scale %.1f padding %d mm: median %.1f ms, success %s -> %s" % (
            sc, need, q["plan_ms"]["median"], q["success_rate"], "OK" if q["gate"] else "FAIL"), flush=True)
    good = [p_["scale"] for p_ in pairs if p_["meets_latency_success"]]
    ff_cmp = [r for r in rows if r["plan"] == 0 and r["scale"] == 1.0]
    out = {"plans": len(plans), "executions": len(rows), "planning_padding_mm": args.padding_mm, "per_scale": per_scale,
           "scale_padding_pairs": pairs, "chosen_scale": max(good) if good else None, "ff_vs_no_ff_plan0_scale1": ff_cmp,
           "time_scale_definition": "execution time = planned time / scale; velocity x scale, acceleration x scale^2; scale 1.0 = "
                                    "the plan as cuRobo timed it (URDF velocity limits, max_acceleration %.0f rad/s^2)" % ACC_LIMIT,
           "velocity_target_supported": cell.have_vel, "positive_control": control,
           "warm": {k: v for k, v in warm_info.items() if k != "captured"},
           "arm_arm_events_total": sum(r["arm_arm_events"] for r in rows), "rows": rows}
    out["sample_ok"] = len(plans) >= 20 and set(args.scales) >= {1.0, 0.5, 0.3}
    out["ok"] = (out["chosen_scale"] is not None and out["arm_arm_events_total"] == 0 and control["detected"] and out["sample_ok"])
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    save("step5.json", out)
    assert control["detected"], "the ContactMonitor did not report the deliberate arm-arm collision: step 5's zero is vacuous"
    return out


# ============================================================================================================
# step 6 -- GPU memory per process
# ============================================================================================================
def total_vram_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], text=True)
    tot, used = [int(x) for x in out.strip().split(",")]
    return tot, used


def step6(args):
    stages = {"context": gpu_used_mb() or 0}
    cell = Cell("null")
    sync()
    stages["newton_cell"] = gpu_used_mb()
    rig = Rig(("A", "B"))
    warm(rig)
    sync()
    stages["two_planners_warm"] = gpu_used_mb()
    rng = np.random.default_rng(args.seed)
    for _ in range(10):
        rig.set_world("B", base_boxes() + random_boxes(rng, 10), random_other_q(rng))
        s, goal, _, _ = feasible_query(rig.pl["B"], rig, rng)
        if s is not None:
            rig.pl["B"].plan_pose(goal_from(rig.pl["B"], *goal), js7(s), max_attempts=3)
    sync()
    stages["after_plans"] = gpu_used_mb()
    tot, used = total_vram_mb()
    per = stages["after_plans"] or 0
    ncpu = os.cpu_count() or 1
    out = {"stages_mb": stages, "torch_reserved_mb": torch.cuda.memory_reserved() / 2 ** 20,
           "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20, "gpu_total_mb": tot,
           "gpu_used_now_mb": used, "other_processes_mb": used - per, "cpu_cores": ncpu,
           "workers_by_vram": int(0.85 * tot // per) if per else None,
           "workers_by_cpu": ncpu // 2,          # a worker = one MuJoCo-Warp step thread + one planner thread + Python
           "note": "a memory UPPER bound: workers = min(by_vram, by_cpu) from one process's footprint; camera/BVH memory, compute "
                   "contention between workers and other GPU jobs (other_processes_mb) are not included. Measure throughput before using it."}
    out["workers"] = min(v for v in (out["workers_by_vram"], out["workers_by_cpu"]) if v)
    out["ok"] = per > 0
    print(json.dumps(out, indent=1))
    save("step6.json", out)
    return out


# ============================================================================================================
# step 7 -- threaded soak
# ============================================================================================================
CAMS = {"top": dict(w=1280, h=960, vfov=60.0), "wrist_B": dict(w=640, h=480, vfov=55.0)}     # as scripts/11_camera_probe.py
TOP_POS = np.array([0.10, -0.20, 1.10])
WRIST_TILT, WRIST_DX, WRIST_DZ = math.radians(25.0), 0.045, 0.060


class Cameras:
    """SensorTiledCamera renders of `top` and `wrist_B` (depth + HDR) from the cell's model, main thread only."""

    def __init__(self, cell):
        from newton.sensors import SensorTiledCamera
        import newton.geometry
        self.cell = cell
        m = cell.ex.model
        newton.geometry.build_bvh_shape(m, cell.ex.state_0)
        self.sensor = SensorTiledCamera(m, config=SensorTiledCamera.RenderConfig(enable_shadows=True))
        u = self.sensor.utils
        self.c = {}
        for k, c in CAMS.items():
            rays = u.compute_pinhole_camera_rays(c["w"], c["h"], math.radians(c["vfov"]))
            self.c[k] = dict(rays=rays, depth=u.create_depth_image_output(c["w"], c["h"]), hdr=u.create_hdr_color_image_output(c["w"], c["h"]))

    def render(self, name):
        import newton.geometry
        ex = self.cell.ex
        if name == "top":
            pos, quat = TOP_POS, np.array([0.0, 0.0, 0.0, 1.0])
        else:
            bq = ex.state_0.body_q.numpy()[self.cell.nb + TCP]
            hR = Rot.from_quat(bq[3:])
            f = np.array([-math.sin(WRIST_TILT), 0, math.cos(WRIST_TILT)])
            up = np.array([math.cos(WRIST_TILT), 0, math.sin(WRIST_TILT)])
            quat = (hR * Rot.from_matrix(np.stack([[0, 1, 0], up, -f], 1))).as_quat()
            pos = bq[:3] + hR.apply([WRIST_DX, 0, -WRIST_DZ])
        newton.geometry.refit_bvh_shape(ex.model, ex.state_0)
        xf = wp.array([[wp.transformf(wp.vec3f(*pos), wp.quatf(*quat))]], dtype=wp.transformf)
        c = self.c[name]
        self.sensor.update(ex.state_0, xf, c["rays"], depth_image=c["depth"], hdr_color_image=c["hdr"])
        return c["depth"]


def soak_job(rig, i, seed):
    """One plan on the worker thread: fresh world, other arm, IK for a feasible pair, plan_pose. Never raises."""
    rng = np.random.default_rng(seed + i)
    arm = "B" if i % 2 == 0 else "A"
    pl = rig.pl[arm]
    row = {"i": i, "arm": arm, "thread": threading.current_thread().name}
    try:
        t = time.perf_counter()
        rig.set_world(arm, base_boxes() + random_boxes(rng, 10), random_other_q(rng))
        s, goal, _, _ = feasible_query(pl, rig, rng)
        if s is None:
            return dict(row, feasible=False)
        r = pl.plan_pose(goal_from(pl, *goal), js7(s), max_attempts=3)
        torch.cuda.current_stream().synchronize()
        return dict(row, feasible=True, ok=plan_ok(r), total_ms=(time.perf_counter() - t) * 1000)
    except Exception as e:      # noqa: BLE001 -- the soak counts, it does not stop
        return dict(row, error="%s: %s" % (type(e).__name__, str(e)[:300]))


def step7(args):
    viewer_used, note = args.viewer, None
    try:
        cell = Cell(args.viewer)
    except Exception as e:      # noqa: BLE001
        viewer_used, note = "null", "ViewerGL failed (%s: %s); ran headless" % (type(e).__name__, str(e)[:200])
        print(note)
        cell = Cell("null")
    cams = Cameras(cell) if not args.no_cameras else None
    rig = Rig(("A", "B"))
    warm_info = warm(rig)            # the cuRobo graphs are captured after Newton's and before the loop, as in the run
    ex = cell.ex
    q0 = cell.q18()
    qA, qB = q0[:7], q0[9:16]            # the cell's own parked start (HOME_Q would interlock the two hands)
    cell.reset(qA, qB, settle=30)
    t = time.perf_counter()
    if cams is not None:
        for name in CAMS:
            cams.render(name)
    first_render_ms = (time.perf_counter() - t) * 1000       # includes the renderer's kernel compile, excluded below
    parts = {"sim_monitor": [], "viewer": [], "cameras": [], "monitor": []}

    def one_frame(with_cams):
        t = time.perf_counter()
        cell.frame(qA, qB)
        t1 = time.perf_counter()
        if cell.viewer_kind == "gl":
            ex.render()
        t2 = time.perf_counter()
        if with_cams and cams is not None and cell.ex.frame % args.cam_every == 0:
            for name in ("top", "wrist_B"):
                cams.render(name)
            wp.synchronize()
        t3 = time.perf_counter()
        parts["sim_monitor"].append((t1 - t) * 1000)
        parts["monitor"].append(cell.t_mon)
        parts["viewer"].append((t2 - t1) * 1000)
        if with_cams and cams is not None and cell.ex.frame % args.cam_every == 0:
            parts["cameras"].append((t3 - t2) * 1000)
        return (t3 - t) * 1000

    def paced(n, with_cams, jobs=None):
        """n frames at 60 fps (sleeping off the rest of each 16.7 ms); returns work times and loop periods."""
        work, period, nxt, last = [], [], time.perf_counter(), time.perf_counter()
        f = 0
        while (f < n) if jobs is None else (not jobs["done"]()):
            work.append(one_frame(with_cams))
            nxt += 1 / 60
            slack = nxt - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                nxt = time.perf_counter()        # missed the deadline: do not try to catch up
            now = time.perf_counter()
            period.append((now - last) * 1000)
            last = now
            f += 1
            if jobs is not None:
                jobs["poll"]()
        return work, period

    base_w, base_p = paced(args.baseline_frames, True)          # the loop with no planning going on
    ex_pool = ThreadPoolExecutor(1, thread_name_prefix="planner")
    state = {"next": 0, "fut": None, "rows": [], "completed": 0}

    def finished():           # n_soak plans COMPLETED (an infeasible random pair is retried, capped at 3x the jobs)
        return state["fut"] is None and (state["completed"] >= args.n_soak or state["next"] >= 3 * args.n_soak)

    def poll():
        if state["fut"] is not None and state["fut"].done():
            r = state["fut"].result()
            state["rows"].append(r)
            state["completed"] += bool(r.get("feasible"))
            state["fut"] = None
        if state["fut"] is None and not finished():
            state["fut"] = ex_pool.submit(soak_job, rig, state["next"], args.seed)
            state["next"] += 1

    jobs = {"poll": poll, "done": finished}
    poll()
    t0 = time.time()
    soak_w, soak_p = paced(0, True, jobs)
    dur = time.time() - t0
    ex_pool.shutdown()
    cuda_err = []
    try:
        sync()
    except Exception as e:      # noqa: BLE001
        cuda_err.append("final sync: %s" % e)
    rows = state["rows"]
    errs = [r["error"] for r in rows if "error" in r]
    cuda_err += [e for e in errs if "cuda" in e.lower() or "cublas" in e.lower() or "warp" in e.lower()]
    done = [r for r in rows if r.get("feasible")]
    since = guard().since_mark()
    cell.mon.finalize()          # .events/.counts/.summary() are only filled here
    mon = {c: v["count"] for c, v in cell.mon.summary().items()}
    out = {"viewer": viewer_used, "viewer_note": note, "cameras": None if cams is None else {"names": list(CAMS), "every_n_frames": args.cam_every},
           "plans_requested": args.n_soak, "plans_completed": len(done), "plans_ok": sum(r["ok"] for r in done),
           "infeasible_pairs": sum(1 for r in rows if r.get("feasible") is False), "exceptions": len(errs), "exception_samples": errs[:5],
           "cuda_errors": len(cuda_err), "cuda_error_samples": cuda_err[:5],
           "post_warmup_graph_inits": since["graph_inits"], "post_warmup_torch_captures": since["torch_captures"], "new_inits": since["new"],
           "planner_thread_names": sorted({r["thread"] for r in rows}), "soak_s": round(dur, 1), "frames": len(soak_w),
           "plan_ms": {"median": pct([r["total_ms"] for r in done], 50), "p95": pct([r["total_ms"] for r in done], 95),
                       "max": max([r["total_ms"] for r in done], default=None)},
           "frame_work_ms": {"baseline_p50": pct(base_w, 50), "baseline_p99": pct(base_w, 99),
                             "soak_p50": pct(soak_w, 50), "soak_p99": pct(soak_w, 99), "soak_max": max(soak_w)},
           "frame_period_ms": {"baseline_p99": pct(base_p, 99), "soak_p99": pct(soak_p, 99), "soak_max": max(soak_p)},
           "first_camera_render_ms": first_render_ms,
           "frame_parts_ms_mean": {k: float(np.mean(v)) if v else None for k, v in parts.items()},
           "frames_over_33ms": int(sum(1 for x in soak_p if x > 33.4)), "monitor_events": mon,
           "warm": {k: v for k, v in warm_info.items() if k != "captured"}}
    out["monitor_note"] = "the arms hold their parked pose during the soak, so the monitor is exercised for cost only; its positive control is step 5"
    out["sample_ok"] = args.n_soak >= 200 and out["plans_completed"] >= args.n_soak
    out["gate"] = out["cuda_errors"] == 0 and out["post_warmup_graph_inits"] == 0 and out["post_warmup_torch_captures"] == 0 and out["exceptions"] == 0
    out["ok"] = out["gate"] and out["sample_ok"]
    print(json.dumps(out, indent=1, default=str))
    save("step7.json", out)
    return out


# ============================================================================================================
# step 8 -- contact-segment false positives (plan §2.3 per-link check on P0's insert recordings)
# ============================================================================================================
import re  # noqa: E402

TAG_RE = re.compile(r"^(?P<structure>.+?)_(?P<model>grasp_lp|lever_press)_r(?P<k>\d+)$")
STUD_H = 0.0017                                  # stud height at 1x (the placed-brick cuboids include the studs)
EVENT_CLASSES_IGNORED = {"neighbour_rub"}        # permitted contact, logged only (plan §2.4)


def world_for_plan(structure, model):
    """dual_arm_sim's plan for the structure (in memory, nothing written under the repo), its brick dims/poses, neighbours."""
    import argparse as ap
    from cell.contacts import _same_course_neighbours
    sim = OUT / "_sim"
    (sim / "plans").mkdir(parents=True, exist_ok=True)
    D.HERE = sim
    plan, _ = D.make_plan(ap.Namespace(front=None, side=None, top=None, width=None, depth=None, name=None, shape=structure,
                                       structure=None, strategy="weakest_joint", brace_model=model))
    bricks = {b["id"]: b for b in plan["bricks"]}
    info = {}
    for st in plan["sequence"]:
        b = bricks[st["brick_id"]]
        nx, ny = P.footprint(b["type"], b["yaw_index"])
        L, W = max(nx, ny) * P.PITCH, min(nx, ny) * P.PITCH
        yaw = math.pi / 2 if ny > nx else 0.0
        info[st["brick_id"]] = dict(L=L, W=W, yaw=yaw, bottom=np.array(st["target_pose"][:3], float))
    return plan, info, _same_course_neighbours(plan["bricks"])


def brick_box(bid, info, studs=True, name=None):
    b = info[bid]
    h = P.BRICK_H + (STUD_H if studs else 0.0)
    return dict(name=name or "brick_" + bid, dims=(b["L"], b["W"], h), c=tuple(b["bottom"] + [0, 0, h / 2]), yaw=b["yaw"])


def insert_segments(z):
    """[(row_start, row_end)] of contiguous frames with B.phase == insert and one b_step in an insert recording."""
    f, ph, st = z["frame"], z["b_phase"], z["b_step"]
    seg, start = [], None
    for i in range(len(f)):
        ok = ph[i] == "insert"
        if ok and start is None:
            start = i
        broke = start is not None and (not ok or (i > 0 and (f[i] - f[i - 1] > 1 or st[i] != st[i - 1])))
        if broke:
            seg.append((start, i - 1))
            start = i if ok else None
    if start is not None:
        seg.append((start, len(f) - 1))
    return seg


def resample_2mm(pos, step=0.002):
    """Fractional row indices along a TCP path such that consecutive samples are at most `step` apart (arc length)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.array([0.0, float(len(pos) - 1)])
    n = int(math.ceil(d[-1] / step)) + 1
    return np.interp(np.linspace(0, d[-1], n), d, np.arange(len(pos)))


def finger_shifted_spheres(arm, Q, G):
    """World spheres (T, N, 4) of `arm` (full model, base link dropped) with the fingers moved from the locked
    opening to the recorded one: left +d, right -d along the hand's y axis, d = g - FINGER_LOCK."""
    sp = arm_spheres_batch(Q, "full")
    sp[..., :3] = from_base(arm, sp[..., :3].reshape(-1, 3)).reshape(sp[..., :3].shape)
    _, R = tcp_poses(arm, Q)
    y = R[:, :, 1]
    sl = link_slices("full")
    sp[:, sl["fr3_leftfinger"], :3] += ((G - FINGER_LOCK)[:, None] * y)[:, None, :]
    sp[:, sl["fr3_rightfinger"], :3] -= ((G - FINGER_LOCK)[:, None] * y)[:, None, :]
    return sp, sl


RELEASE_FRAMES = 30                             # step 8: events up to 0.5 s into Release belong to the segment


def obb_obb_gap(b1, b2):
    """Separating-axis gap (m) between two oriented boxes (15 axes): > 0 they are apart by at least that much,
    < 0 they overlap (the smallest overlap). A lower bound of the true distance near edges/corners."""
    c1, R1, h1 = box_frame(b1)
    c2, R2, h2 = box_frame(b2)
    t = c2 - c1
    best = -np.inf
    for a in [*R1.T, *R2.T, *[np.cross(u, v) for u in R1.T for v in R2.T]]:
        n = np.linalg.norm(a)
        if n < 1e-9:
            continue
        a = a / n
        best = max(best, abs(t @ a) - float(h1 @ np.abs(R1.T @ a)) - float(h2 @ np.abs(R2.T @ a)))
    return best


_FBOX = None


def finger_box_frames():
    """The finger links' collision boxes from the URDF Newton loads: {'left'|'right': [(R_box, xyz, size)]} in the link frame."""
    global _FBOX
    if _FBOX is None:
        import xml.etree.ElementTree as ET
        root = ET.parse(asset_dir() / "urdf/fr3_franka_hand.urdf").getroot()
        _FBOX = {}
        for link, side in (("fr3_leftfinger", "left"), ("fr3_rightfinger", "right")):
            boxes = []
            for col in root.find("./link[@name='%s']" % link).findall("collision"):
                o = col.find("origin")
                xyz = np.array([float(x) for x in o.get("xyz").split()])
                rpy = [float(x) for x in o.get("rpy").split()]
                size = np.array([float(x) for x in col.find("geometry/box").get("size").split()])
                boxes.append((Rot.from_euler("xyz", rpy).as_matrix(), xyz, size))
            _FBOX[side] = boxes
    return _FBOX


def finger_boxes(tcp, R, g):
    """World boxes of both fingers (the exact collision geometry Newton uses) for TCP position/rotation and opening g each."""
    hand = tcp - R @ np.array([0, 0, 0.1034])          # fr3_hand_tcp is 0.1034 m along z from the hand
    out = []
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        Rj = R if side == "left" else R @ Rot.from_euler("z", math.pi).as_matrix()
        pj = hand + R @ np.array([0, 0, 0.0584]) + Rj @ np.array([0, g, 0])
        for Rb, xyz, size in finger_box_frames()[side]:
            out.append(dict(name="finger_" + side, dims=tuple(size), c=tuple(pj + Rj @ xyz), R=Rj @ Rb))
    return out


def step8(args):
    # self-test of the box tests: two unit cubes 0.1 m apart along x, and a turned one that overlaps
    u = dict(name="u", dims=(1, 1, 1), c=(0, 0, 0), yaw=0.0)
    assert abs(obb_obb_gap(u, dict(u, c=(1.1, 0, 0))) - 0.1) < 1e-9
    assert obb_obb_gap(u, dict(u, c=(0.9, 0, 0), yaw=0.7)) < 0
    jsonl = {}
    p_jsonl = Path(args.p0_jsonl)
    if p_jsonl.exists():
        for line in p_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                jsonl[r["tag"]] = r
    files = sorted(Path(args.p0_inserts).glob("%s_*.npz" % args.structure))
    if args.synthetic:
        files, jsonl = _synthetic_recordings(args)
    if not files:
        print("no insert recordings matching %s_*.npz in %s" % (args.structure, args.p0_inserts))
        save("step8.json", {"ok": None, "note": "no recordings", "p0_inserts": str(args.p0_inserts)})
        return None
    pads = [float(x) for x in args.pads_mm]
    segs_out, cache = [], {}
    for f in files:
        m = TAG_RE.match(f.stem)
        if not m:
            print("skip %s (name is not <structure>_<model>_r<k>)" % f.name)
            continue
        key = (m["structure"], m["model"])
        if key not in cache:
            cache[key] = world_for_plan(*key)
        plan, info, neigh = cache[key]
        z = np.load(f)
        events = [e for e in jsonl.get(f.stem, {}).get("events", []) if e["class"] not in EVENT_CLASSES_IGNORED]
        have_events = f.stem in jsonl
        seq = plan["sequence"]
        for a, b in insert_segments(z):
            rows = np.arange(a, b + 1)
            bstep = int(z["b_step"][a])
            bid = seq[bstep]["brick_id"]
            qB, qA, G = z["q"][rows, 9:16].astype(float), z["q"][rows, :7].astype(float), z["q"][rows, 16].astype(float)
            tcp, _ = tcp_poses("B", qB)
            idx = resample_2mm(tcp)
            lo = np.minimum(np.floor(idx).astype(int), len(rows) - 2) if len(rows) > 1 else np.zeros(len(idx), int)
            w = (idx - lo)[:, None] if len(rows) > 1 else np.zeros((len(idx), 1))
            nxt = np.minimum(lo + 1, len(rows) - 1)
            iq = lambda X: X[lo] * (1 - w) + X[nxt] * w                                   # linear in joint space between frames
            qBs, qAs, Gs = iq(qB), iq(qA), iq(G[:, None])[:, 0]
            tcp_s, _ = tcp_poses("B", qBs)
            # world: table, plate, bricks placed before this step (not the one being grasped)
            placed = [seq[j]["brick_id"] for j in range(bstep)]
            static = [TABLE_TRUE, base_boxes()[1]] + [brick_box(x, info) for x in placed]
            Bsp, links = finger_shifted_spheres("B", qBs, Gs)
            Asp = arm_world_spheres("A", qAs, "full")
            fin = np.concatenate([links["fr3_leftfinger"], links["fr3_rightfinger"]])
            arm_idx = np.setdiff1d(np.arange(Bsp.shape[1]), fin)
            g_fin = gap_over_time(Bsp[:, fin], static)
            g_arm = gap_over_time(Bsp[:, arm_idx], static)
            g_aa = np.array([clearance_spheres(Bsp[i], Asp[i]) for i in range(len(qBs))])
            # diagnostic: the fingers' exact collision boxes (URDF) against the same static world, separating-axis gap
            _, Rs = tcp_poses("B", qBs)
            g_fbox = np.array([min(obb_obb_gap(fb, x) for fb in finger_boxes(tcp_s[i], Rs[i], Gs[i]) for x in static) for i in range(len(qBs))])
            # the held brick: nominal grasp (TCP xy, bottom GRASP_DZ below the TCP), yaw of its target; world minus
            # supports and same-course neighbours; A's spheres count as obstacles too
            b0 = info[bid]
            supports = {m_[0] for m_ in seq[bstep]["mating_studs"]} or {None}     # None = the baseplate
            skip = {x for x in supports if x} | neigh.get(bid, set())
            hw = [x for x in static if x["name"] not in ("table",)]
            hw = [x for x in hw if not (x["name"] == "plate" and None in supports)
                  and not any(x["name"] == "brick_" + s_ for s_ in skip)]
            held = [dict(name="held", dims=(b0["L"], b0["W"], P.BRICK_H), yaw=b0["yaw"],
                         c=(tcp_s[i, 0], tcp_s[i, 1], tcp_s[i, 2] - D.GRASP_DZ + P.BRICK_H / 2)) for i in range(len(tcp_s))]
            g_held = np.full(len(held), np.inf)
            for i, hb in enumerate(held):
                # held brick OBB vs the world's cuboids: separating-axis test; vs A's spheres: exact OBB-sphere distance
                d = min([obb_obb_gap(hb, x) for x in hw] or [np.inf])
                d = min(d, float((obb_dist(Asp[i][:, :3], hb) - Asp[i][:, 3]).min()))
                g_held[i] = d
            # the segment's contacts include Release: the fingers open from the end-of-insert pose, and P0' shows the
            # finger-neighbour contacts starting 3-4 frames into release (the npz holds insert frames only), so the
            # event window runs RELEASE_FRAMES (the monitor's 0.5 s merge gap) past the segment's last frame
            seg_ev = [e for e in events if e["end_frame"] >= int(z["frame"][a])
                      and e["start_frame"] <= int(z["frame"][b]) + RELEASE_FRAMES]
            mins = {"finger": g_fin.min() * 1000, "arm": g_arm.min() * 1000, "arm_arm": g_aa.min() * 1000, "held_brick": g_held.min() * 1000}
            row = {"file": f.name, "b_step": bstep, "brick": bid, "frames": [int(z["frame"][a]), int(z["frame"][b])],
                   "samples_2mm": int(len(qBs)), "tcp_path_mm": float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum() * 1000),
                   "min_clearance_mm": {k: float(v) for k, v in mins.items()}, "finger_box_min_mm": float(g_fbox.min() * 1000),
                   "monitor_events": [e["class"] for e in seg_ev],
                   "monitor_events_known": have_events}
            for pad in pads:
                fl = {k: bool(v < pad) for k, v in mins.items()}
                row["flag_pad%g" % pad] = {**fl, "any": any(fl.values())}
            segs_out.append(row)
    n = len(segs_out)
    summ = {"segments": n, "files": len(files), "pads_mm": pads}
    for pad in pads:
        k = "flag_pad%g" % pad
        summ["pad%g" % pad] = {"flagged": sum(r[k]["any"] for r in segs_out),
                              "false_positives": sum(r[k]["any"] and not r["monitor_events"] for r in segs_out),
                              "true_positives": sum(r[k]["any"] and bool(r["monitor_events"]) for r in segs_out),
                              "missed_events": sum((not r[k]["any"]) and bool(r["monitor_events"]) for r in segs_out)}
    summ["segments_with_monitor_events"] = sum(bool(r["monitor_events"]) for r in segs_out)
    clean = [pad for pad in pads if summ["pad%g" % pad]["false_positives"] == 0 and summ["pad%g" % pad]["missed_events"] == 0]
    summ["max_padding_without_false_positive_mm"] = max(clean) if clean else None      # the gate allows 0-1 mm padding
    # margin: how far the smallest clearance of any monitor-clean segment is above the padding that is used
    cleanseg = [min(r["min_clearance_mm"].values()) for r in segs_out if not r["monitor_events"]]
    summ["min_clean_segment_clearance_mm"] = min(cleanseg) if cleanseg else None
    summ["margin_mm"] = (min(cleanseg) - max(clean)) if (cleanseg and clean) else None
    evseg = [min(r["min_clearance_mm"].values()) for r in segs_out if r["monitor_events"]]
    summ["max_event_segment_clearance_mm"] = max(evseg) if evseg else None            # a segment with a real event must be below the padding
    fb_clean = [r["finger_box_min_mm"] for r in segs_out if not r["monitor_events"]]
    fb_event = [r["finger_box_min_mm"] for r in segs_out if r["monitor_events"]]
    summ["finger_box_diagnostic_mm"] = {"clean_segments_min": min(fb_clean) if fb_clean else None,
                                        "clean_segments_p5": pct(fb_clean, 5) if fb_clean else None,
                                        "event_segments_max": max(fb_event) if fb_event else None}
    summ["ok"] = bool(n) and bool(clean) and all(r["monitor_events_known"] for r in segs_out)
    summ["rows"] = segs_out
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))
    save("step8%s.json" % ("_synthetic" if args.synthetic else ""), summ)
    return summ


def _synthetic_recordings(args):
    """A recording in P0's npz/jsonl format built from the cube plan: B descends straight onto the first brick's target
    (IK per 3 mm waypoint, A parked): clean -> must not flag; then the same with the hand pushed 25 mm lower -> must flag."""
    d = OUT / "_synthetic8"
    d.mkdir(parents=True, exist_ok=True)
    plan, info, _ = world_for_plan("cube", "grasp_lp")
    bid = plan["sequence"][0]["brick_id"]
    tgt = info[bid]["bottom"]
    rig = Rig(("B",))
    pl = rig.pl["B"]
    pl.update_world(scene_for("B", []))       # IK without obstacles: the descent is meant to touch the plate
    yaw = info[bid]["yaw"]
    qA = np.array(rig_park_q())
    files, jsonl = [], {}
    for label, extra in (("clean", 0.0), ("crash", -0.025)):
        zs = np.linspace(tgt[2] + D.GRASP_DZ + 0.025, tgt[2] + D.GRASP_DZ - D.PRESS + extra, 30)
        qs, prev = [], None
        for z_ in zs:
            p, qt = tool_pose_w("B", (tgt[0], tgt[1], z_), yaw)
            r = pl.ik_solver.solve_pose(goal_from(pl, p, qt), return_seeds=8, current_state=(js7(prev) if prev is not None else pl.default_joint_state.clone().unsqueeze(0)))
            sols = r.solution.reshape(-1, 7)[r.success.reshape(-1)].cpu().numpy()
            assert len(sols), "IK failed in the synthetic path"
            prev = sols[np.argmin(np.linalg.norm(sols - prev, axis=1))] if prev is not None else sols[0]
            qs.append(prev)
        qs = np.array(qs)
        q = np.zeros((len(qs), 18), np.float32)
        q[:, :7], q[:, 9:16], q[:, 7:9], q[:, 16:18] = qA, qs, 0.0, 0.0073
        tag = "cube_grasp_lp_r%d" % (0 if label == "clean" else 1)
        np.savez(d / (tag + ".npz"), frame=np.arange(100, 100 + len(qs)), q=q, a_phase=np.array(["park"] * len(qs)),
                 b_phase=np.array(["insert"] * len(qs)), b_step=np.zeros(len(qs), int))
        files.append(d / (tag + ".npz"))
        jsonl[tag] = {"tag": tag, "events": [] if label == "clean" else [{"class": "unintended", "start_frame": 100, "end_frame": 120}]}
    args.structure = "cube"
    return files, jsonl


def rig_park_q():
    """Arm A's parked joint vector as dual_arm_sim solves it (IK to PARK, hand down)."""
    arm = D.build_arm()
    ex = type("E", (), {})()
    ex.model_ik = arm.finalize()
    return D.Example.solve_park(ex)[:7]


def summary(args):
    """Gate P3 items, from the step files (results/v4/p3/stepN.json)."""
    def mark(ok):
        return "PASS" if ok else ("n/a " if ok is None else "FAIL")
    s1, s2, s3, s4, s5, s6, s7, s8 = (load("step%d.json" % i) for i in range(1, 9))
    items = []
    items.append(("runs alongside Newton and warp 1.13 in one process", None if not s1 else bool(s1.get("ok")),
                  "" if not s1 else "torch %s, warp %s, newton %s, cuda-core %s; Franka plan %.0f ms" % (
                      s1["torch"], s1["warp"], s1["newton"], s1["cuda_core"], np.median(s1["franka_plan_ms"]))))
    if s2:
        worst = {m: max((t["vertex_max_mm"] for k, t in v["links"].items() if "finger" not in k), default=None) for m, v in s2.items() if isinstance(v, dict) and "links" in v}
        fin = {m: max((t["vertex_max_mm"] for k, t in v["links"].items() if "finger" in k), default=None) for m, v in s2.items() if isinstance(v, dict) and "links" in v}
        items.append(("sphere fit: vertices within 2 mm (arm), 0.5 mm (fingers) [profile full]", bool(s2.get("full", {}).get("ok")),
                      "arm max %.2f mm, finger max %.2f mm, %d spheres" % (worst["full"], fin["full"], s2["full"]["total_spheres"])))
    else:
        items.append(("sphere fit", None, ""))
    if s4:
        for m, r in s4.items():
            n_ok = r["sample_ok"]
            items.append(("median plan <= 100 ms warm [model %s, %d spheres, padding %g mm]" % (m, r["total_spheres"], r["padding_mm"]),
                          (r["plan_ms"]["median"] <= 100) and n_ok,
                          "median %.1f ms, p95 %.1f ms%s" % (r["plan_ms"]["median"], r["plan_ms"]["p95"], "" if n_ok else "  (%d queries < 100: smoke)" % r["queries"])))
            items.append(("success >= 95%% on feasible queries [model %s]" % m, ((r["success_rate"] or 0) >= 0.95) and n_ok,
                          "%d/%d" % (r["success"], r["feasible"])))
    else:
        items += [("median plan <= 100 ms warm", None, ""), ("success >= 95% on feasible queries", None, "")]
    if s5:
        items.append(("(time scale, padding) pairs: plans at the needed padding meet latency/success", s5["chosen_scale"] is not None and s5["sample_ok"],
                      "chosen scale %s; %s%s" % (s5["chosen_scale"], [(p_["scale"], p_["padding_mm"], round(p_["plan_ms_median"]), p_["meets_latency_success"]) for p_ in s5["scale_padding_pairs"]],
                                                "" if s5["sample_ok"] else "  (%d plans < 20: smoke)" % s5["plans"])))
        items.append(("0 arm-arm events (single frame) on the executed plans, monitor positive control detected",
                      s5["arm_arm_events_total"] == 0 and s5["positive_control"]["detected"],
                      "%d events over %d executions; control %s" % (s5["arm_arm_events_total"], s5["executions"], s5["positive_control"]["events"])))
    else:
        items += [("deviation p99 + 2 mm <= padding", None, ""), ("0 arm-arm events on executed plans", None, "")]
    if s7:
        items.append(("soak: >= 200 plans, 0 CUDA errors, 0 post-warm-up captures", bool(s7["ok"]),
                      "%d plans, viewer %s, %d cuda errors, %d captures, frame p99 %.1f ms (baseline %.1f)%s" % (
                          s7["plans_completed"], s7["viewer"], s7["cuda_errors"], s7["post_warmup_graph_inits"] + s7["post_warmup_torch_captures"],
                          s7["frame_work_ms"]["soak_p99"], s7["frame_work_ms"]["baseline_p99"], "" if s7["sample_ok"] else "  (smoke size)")))
    else:
        items.append(("soak: 0 CUDA errors, 0 post-warm-up captures", None, ""))
    items.append(("step 3 guard: 0 graph inits after warm-up", None if not s3 else bool(s3["ok"]), "" if not s3 else "%d captured in warm-up" % s3["graph_inits"]))
    if s8:
        items.append(("0 contact-segment false positives on the cube's inserts", s8.get("ok"),
                      "%s segments (%s with monitor events); max padding with 0 FP and 0 missed: %s mm; margin over it %s mm" % (
                          s8.get("segments"), s8.get("segments_with_monitor_events"), s8.get("max_padding_without_false_positive_mm"),
                          None if s8.get("margin_mm") is None else round(s8["margin_mm"], 2))))
    else:
        items.append(("0 contact-segment false positives on the cube's inserts", None, ""))
    if s6:
        items.append(("(info) GPU memory / worker count for WP6", None, "%s MB per process -> %s workers" % (s6["stages_mb"]["after_plans"], s6["workers"])))
    print("Gate P3")
    for name, ok, note in items:
        print("  [%s] %s  %s" % (mark(ok), name, note))


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", type=int, action="append", choices=range(1, 9), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--reuse-fit", action="store_true", help="step 2: rebuild the yml(s) from the cached fit")
    ap.add_argument("--profiles", nargs="+", default=["full", "plan"], choices=list(PROFILES), help="step 2: sphere-fit profiles")
    ap.add_argument("--n-queries", type=int, default=100, help="step 4: plan queries per model")
    ap.add_argument("--models", nargs="+", default=["plan"], choices=list(PROFILES), help="steps 4: planner sphere models")
    ap.add_argument("--n-exec", type=int, default=20, help="step 5: plans executed at every time scale")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.3], help="step 5: time scales")
    ap.add_argument("--padding-mm", type=float, default=5.0, help="steps 4, 5: planner padding (collision_sphere_buffer, mm); the plan's floor is 5")
    ap.add_argument("--n-pad-queries", type=int, default=20, help="step 5: queries re-measured at each scale's needed padding")
    ap.add_argument("--n-soak", type=int, default=200, help="step 7: plans from the worker thread")
    ap.add_argument("--viewer", choices=["gl", "null"], default="gl", help="step 7: ViewerGL window or headless (falls back to null if GL fails)")
    ap.add_argument("--no-cameras", action="store_true", help="step 7: skip the top/wrist_B renders")
    ap.add_argument("--cam-every", type=int, default=6, help="step 7: render both cameras every N frames (60/N Hz)")
    ap.add_argument("--baseline-frames", type=int, default=180, help="step 7: frames without planning, for the frame-time reference")
    ap.add_argument("--p0-jsonl", default=str(P0_DIR / "p0_stiff.jsonl"), help="step 8: P0' rows (tag, events), stiff finger contacts")
    ap.add_argument("--p0-inserts", default=str(P0_DIR / "p0_stiff_inserts"), help="step 8: directory of <structure>_<model>_r<k>.npz")
    ap.add_argument("--structure", default="cube", help="step 8: which structure's recordings to check")
    ap.add_argument("--pads-mm", nargs="+", default=["0", "0.25", "0.5", "1"], help="step 8: finger/link padding(s) tried")
    ap.add_argument("--synthetic", action="store_true", help="step 8: self-test on a synthetic recording (clean descent + crash)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-check", type=int, default=20, help="step 3: post-warm-up guard queries per arm")
    ap.add_argument("--collision-samples", type=int, default=1000, help="step 2: random configs for the self-collision matrix")
    args = ap.parse_args()
    if args.summary:
        return summary(args)
    from curobo.logging import setup_logger
    setup_logger("error")       # cuRobo's "Mesh already in cache, reusing" warning fires on every world update
    for n in (range(1, 9) if args.all else sorted(set(args.step))):
        print("\n=== step %d ===" % n)
        STEPS[n](args)


if __name__ == "__main__":
    main()
