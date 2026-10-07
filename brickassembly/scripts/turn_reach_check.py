"""Dry IK check of the ACC insertion yaws (plan_v4_perception §2.2a): can arm B reach every step's look, pre-insert and insert pose at the hand yaw of
look 0 and at the TURNED hand yaw (look 1's: look 0's + (pi - 1e-6) toward the wrist's middle)? The IK is dual_arm_sim's (IKSolver, position + rotation +
joint-limit objectives, FR3 + hand), each pose solved from the park configuration; a pose is reachable when the FK of the solution is within 0.2 mm and
0.05 deg of it and every joint is inside its limits. Prints every unreachable (structure, step, yaw, pose) and the smallest wrist-joint margin.

    ~/Codes/CAIRSS/Issac/bin/python scripts/turn_reach_check.py [--margins] [cube arch hollow_box S3 S5p13]
"""
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as Rot

import newton
import newton.ik as ik

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402
import planner as P  # noqa: E402
from experiments import v4_vision as X  # noqa: E402

POSES = (("look", D.ex.STUD_HEIGHT + D.H_LOOK + D.GRASP_DZ), ("pre-insert", 0.025 + D.GRASP_DZ), ("insert", D.GRASP_DZ - D.PRESS))


VERBOSE = "--margins" in sys.argv


def main(names):
    model = D.build_arm().finalize()
    lo, hi = model.joint_limit_lower.numpy()[:7], model.joint_limit_upper.numpy()[:7]
    state = model.state()
    q0 = np.array([D.HOME_Q + [0.01, 0.01]], np.float32)
    bad, n, margin = [], 0, np.inf
    base, byaw = D.ARM_B
    for name in names:
        bricks = X.bricks_of(name)
        P.STRUCTURES[name] = bricks
        NI, NJ = X.nij(bricks)
        P.VOXEL_ORIGIN = (-NI * P.PITCH / 2, -NJ * P.PITCH / 2, D.Z0)
        plan = P.build_plan(name, "none")
        for st in plan["sequence"]:
            tgt = np.array(st["target_pose"][:3])
            yaw = math.radians(st["grasp"]["yaw_offset_deg"])
            shift = round((byaw + D.wrap(yaw - byaw, math.pi) - yaw) / math.pi) * math.pi
            gy = yaw + shift
            turn = (-1 if gy - byaw > 0 else 1) * (math.pi - 1e-6)
            for label, hy in (("look0", gy), ("turned", gy + turn)):
                pos = [D.rotate(D.qz(-byaw), tgt + [0, 0, dz] - np.array(base)) for _, dz in POSES]
                rot = [D.qmul(D.qz(-byaw), D.qmul(D.qz(hy), D.Q_DOWN)) for _ in POSES]
                q = wp.array(np.tile(q0, (len(POSES), 1)), dtype=wp.float32)
                solver = ik.IKSolver(model=model, n_problems=len(POSES), lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
                                     objectives=[ik.IKObjectivePosition(link_index=D.EE, link_offset=wp.vec3(0, 0, 0), target_positions=wp.array(np.array(pos, np.float32), dtype=wp.vec3)),
                                                 ik.IKObjectiveRotation(link_index=D.EE, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array(np.array(rot, np.float32), dtype=wp.vec4)),
                                                 ik.IKObjectiveJointLimit(joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper)])
                for _ in range(40):
                    solver.step(q, q, iterations=24)
                qs = q.numpy()
                for i, (pn, _) in enumerate(POSES):
                    newton.eval_fk(model, wp.array(qs[i], dtype=wp.float32), wp.zeros(9, dtype=wp.float32), state)
                    t = state.body_q.numpy()[D.EE]
                    perr = np.linalg.norm(t[:3] - pos[i]) * 1e3
                    aerr = math.degrees((Rot.from_quat(t[3:]) * Rot.from_quat(rot[i]).inv()).magnitude())
                    over = max(0.0, float((lo - qs[i, :7]).max()), float((qs[i, :7] - hi).max()))
                    mg = float(np.minimum(qs[i, :7] - lo, hi - qs[i, :7]).min())
                    n += 1
                    if label == "turned":
                        margin = min(margin, mg)
                    if VERBOSE and pn == "insert":
                        print("  %s step %2d %-6s insert: tightest joint %d, margin %5.1f deg; q7 = %7.1f deg" % (name, st["step"], label, int(np.argmin(np.minimum(qs[i, :7] - lo, hi - qs[i, :7]))) + 1,
                                                                                                                math.degrees(mg), math.degrees(qs[i, 6])))
                    if perr > 0.2 or aerr > 0.05 or over > 1e-3:
                        bad.append((name, st["step"], label, pn, round(perr, 3), round(aerr, 3), round(over, 4)))
    print("poses checked:", n, "; unreachable:", len(bad), "; smallest joint-limit margin at the turned yaw: %.1f deg" % math.degrees(margin))
    for b in bad:
        print("  UNREACHABLE", b)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or ["cube", "arch", "hollow_box", "S3", "S5p13"])
