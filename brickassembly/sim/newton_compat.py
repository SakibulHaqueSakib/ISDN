"""SUPERSEDED 2026-09-20 -- WP1 adopted BrickSim, which is PhysX-only, and D2
was reverted from Newton to PhysX (ledger: wp1_platform). Nothing imports this.
Kept only until the platform decision has settled; then delete it.

Thin adapter over the Newton field names this project touches (§2.4).

Newton is a beta whose field names move between releases. Everything the rest
of the project needs from Newton is named exactly once, here, so a rename costs
one file. Verified against the versions in env.lock:

    newton 1.2.1, warp 1.13.0, mujoco_warp 3.8.0.3

    python sim/newton_compat.py     # self-check: every name below still exists

Run under the Isaac interpreter (~/Codes/CAIRSS/Issac/bin/python).
"""

import warp as wp
import newton

# --- construction -----------------------------------------------------------
ModelBuilder = newton.ModelBuilder
#: Replicate a single-environment builder into n worlds: replicate(b, n, spacing)
replicate = ModelBuilder.replicate
#: builder.add_body(xform=wp.transform(p, q), label=...) -> body index
#: builder.add_shape_box(body, hx=, hy=, hz=)
#: builder.finalize() -> Model

# --- solver -----------------------------------------------------------------
#: D2: Newton (MuJoCo-Warp) primary. Swap this one name for the PhysX path.
Solver = newton.solvers.SolverMuJoCo
SolverNotifyFlags = newton.solvers.SolverNotifyFlags

# --- per-step state ---------------------------------------------------------
#: Live Warp arrays, writable from a kernel -- this is the hook the brick-joint
#: model runs through, with no per-env python loop (§2.4).
BODY_XFORM = "body_q"      # wp.array(dtype=wp.transform), one per body per world
BODY_VEL = "body_qd"       # wp.array(dtype=wp.spatial_vector)
BODY_FORCE = "body_f"      # wp.array(dtype=wp.spatial_vector), cleared each substep


def make_state(model):
    """The (state_0, state_1, control, contacts) tuple a step needs."""
    return model.state(), model.state(), model.control(), model.contacts()


def step(model, solver, s0, s1, control, contacts, dt, substeps):
    """One control step. Returns the state that holds the result.

    Newton double-buffers: the solver writes s1 from s0, and the caller keeps
    the swap. Getting this wrong silently halves the timestep.
    """
    for _ in range(substeps):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    return s0, s1


def body_array(state, field=BODY_XFORM):
    """The live Warp array for a state field, by the name pinned above."""
    return getattr(state, field)


def _self_check():
    b = ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05),
                                         q=wp.quat_identity()), label="brick")
    b.add_shape_box(body, hx=0.0079, hy=0.0159, hz=0.0048)

    world = ModelBuilder()
    world.replicate(b, 4)
    model = world.finalize()
    solver = Solver(model)
    s0, s1, control, contacts = make_state(model)

    for field in (BODY_XFORM, BODY_VEL, BODY_FORCE):
        arr = body_array(s0, field)
        assert isinstance(arr, wp.array), "%s is not a Warp array" % field
        assert arr.shape[0] == 4, "%s: expected one entry per world" % field

    # a kernel must be able to write per-world body state
    @wp.kernel
    def _lift(q: wp.array(dtype=wp.transform), dz: float):
        i = wp.tid()
        x = q[i]
        q[i] = wp.transform(wp.transform_get_translation(x) + wp.vec3(0.0, 0.0, dz),
                            wp.transform_get_rotation(x))

    before = body_array(s0).numpy()[0][2]
    wp.launch(_lift, dim=4, inputs=[body_array(s0), 0.001])
    wp.synchronize()
    assert abs(body_array(s0).numpy()[0][2] - before - 0.001) < 1e-6, "kernel write lost"

    z0 = body_array(s0).numpy()[0][2]
    s0, s1 = step(model, solver, s0, s1, control, contacts, 1.0 / 960.0, 40)
    assert body_array(s0).numpy()[0][2] < z0, "brick did not fall: step() is a no-op"

    assert hasattr(SolverNotifyFlags, "__members__") or SolverNotifyFlags is not None
    print("newton_compat self-check OK (newton %s, warp %s)"
          % (newton.__version__, wp.config.version))


if __name__ == "__main__":
    _self_check()
