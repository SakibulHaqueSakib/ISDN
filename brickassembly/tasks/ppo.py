"""PPO for the WP5 insertion task -- runs R1-R4 of master_report §5.3.

    python -m tasks.ppo --run R1 --samples 150000 --workers 2
    python -m tasks.ppo --eval results/wp5/R1/policy.pt --stage 3 --episodes 100

A small, dependency-free PPO (the report names rsl_rl; its Isaac Lab runner
is not available on this CPU box):
  * tanh-squashed Gaussian actor over the 121-D observation (§5.2)
  * asymmetric critic on actor obs + 26 privileged dims [InformedAAC]
  * a success head on the actor trunk (§5.6, [FORGE]): P(episode succeeds |
    state), trained with BCE on the finished episodes' outcomes
  * running observation normalisation, GAE(0.95), clip 0.2
  * the §5.4 curriculum: advance when success over the last 100 episodes
    > 80%, regress below 50%

Runs (§5.3): R1 residual + sparse; R2 residual + dense; R3 end-to-end + dense;
R4 = R1 with the vision group zeroed (ablation A3).
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp5"

RUNS = {
    "R1": dict(mode="residual", reward="sparse", vision=True),
    "R2": dict(mode="residual", reward="dense", vision=True),
    "R3": dict(mode="e2e", reward="dense", vision=True),
    "R4": dict(mode="residual", reward="sparse", vision=False),
    # ablation A10: R1 trained on the hand-tuned joint, evaluated on the calibrated one
    "R1ht": dict(mode="residual", reward="sparse", vision=True, joint="handtuned"),
    # A10 with the force budget and base press held at the calibrated seating
    # force: R1ht's budget, scaled by the hand-tuned 3 N per stud, sat below
    # the touchdown transient and the run collapsed to never touching
    "R1hc": dict(mode="residual", reward="sparse", vision=True, joint="handtuned",
                 budget_from="calibrated"),
}


# --- vectorised environments ------------------------------------------------------

def _worker(remote, n, kwargs, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from tasks.insertion_env import InsertionEnv
    envs = [InsertionEnv(seed=seed + k, **kwargs) for k in range(n)]
    obs = [e.reset() for e in envs]
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            out = []
            for k, (e, a) in enumerate(zip(envs, data)):
                o, r, d, info = e.step(a)
                if d:
                    info = dict(info)
                    o = e.reset()
                out.append((o, r, d, info if d else None))
            remote.send(out)
        elif cmd == "obs":
            remote.send([e._obs() for e in envs])
        elif cmd == "stage":
            for e in envs:
                e.stage = data
            remote.send(True)
        elif cmd == "close":
            remote.close()
            return


class VecEnv:
    def __init__(self, workers, per_worker, kwargs, seed=0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, per_worker, kwargs, seed + 1000 * w),
                            daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.per = per_worker
        self.n = workers * per_worker

    def obs(self):
        for r in self.remotes:
            r.send(("obs", None))
        return [o for r in self.remotes for o in r.recv()]

    def step(self, actions):
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[w * self.per:(w + 1) * self.per]))
        return [x for r in self.remotes for x in r.recv()]

    def set_stage(self, s):
        for r in self.remotes:
            r.send(("stage", s))
        for r in self.remotes:
            r.recv()

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


# --- networks ------------------------------------------------------------------------

class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, float)
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = np.array(s["mean"]), np.array(s["var"]), s["count"]


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, init_std=0.4):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU())
        self.mu = nn.Linear(256, act_dim)
        self.succ = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        self.critic = mlp(obs_dim + priv_dim, 1)
        nn.init.zeros_(self.mu.weight)          # residual: start at the scripted base
        nn.init.zeros_(self.mu.bias)

    def dist(self, obs):
        h = self.trunk(obs)
        return torch.distributions.Normal(self.mu(h), self.log_std.exp()), h

    def value(self, cobs):
        return self.critic(cobs).squeeze(-1)

    def success_logit(self, obs):
        return self.succ(self.trunk(obs)).squeeze(-1)


def squash_logp(dist, u):
    return (dist.log_prob(u) - torch.log(1 - torch.tanh(u) ** 2 + 1e-6)).sum(-1)


# --- training --------------------------------------------------------------------------

def train(run, samples, workers=2, per_worker=4, horizon=128, seed=0, stage0=0, out=None,
          resume=False):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    from tasks.insertion_env import ACT_DIM, OBS_DIM, PRIV_DIM
    cfg = RUNS[run]
    out = Path(out or OUT / run)
    out.mkdir(parents=True, exist_ok=True)
    # a residual starts at the scripted base: exploration noise of 0.4 (0.8 mm
    # per step) against 0.1 mm stud clearance halved its success rate
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM, init_std=0.15 if cfg["mode"] == "residual" else 0.4)
    an, cn = RunningNorm(OBS_DIM), RunningNorm(OBS_DIM + PRIV_DIM)
    stage, done0 = stage0, 0
    if resume and (out / "policy.pt").exists():
        ck = torch.load(out / "policy.pt", weights_only=False)
        net.load_state_dict(ck["net"])
        an.load(ck["an"])
        cn.load(ck["cn"])
        stage, done0 = ck["stage"], ck["samples"]
        print("resumed %s at %d samples, stage %d" % (run, done0, stage), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    venv = VecEnv(workers, per_worker, dict(stage=stage, **cfg), seed=seed * 7919 + done0)
    n = venv.n
    window = deque(maxlen=100)
    log = open(out / "train.jsonl", "a")
    obs = venv.obs()
    ep_states = [[] for _ in range(n)]            # actor obs per running episode (success head)
    succ_x, succ_y = [], []
    t0 = time.time()
    done_samples, update = done0, done0 // (horizon * n)
    while done_samples < samples:
        A = np.stack([o["actor"] for o in obs])
        Cb = np.stack([o["critic"] for o in obs])
        buf = {k: [] for k in ("a", "c", "u", "logp", "r", "d", "v")}
        infos = []
        for t in range(horizon):
            an.update(A)
            cn.update(Cb)
            a_t, c_t = torch.as_tensor(an(A)), torch.as_tensor(cn(Cb))
            with torch.no_grad():
                dist, _ = net.dist(a_t)
                u = dist.sample()
                logp = squash_logp(dist, u)
                v = net.value(c_t)
            res = venv.step(torch.tanh(u).numpy())
            for k in range(n):
                ep_states[k].append(an(A[k]))
            buf["a"].append(a_t)
            buf["c"].append(c_t)
            buf["u"].append(u)
            buf["logp"].append(logp)
            buf["v"].append(v)
            buf["r"].append(torch.tensor([x[1] for x in res], dtype=torch.float32))
            buf["d"].append(torch.tensor([float(x[2]) for x in res]))
            obs = [x[0] for x in res]
            A = np.stack([o["actor"] for o in obs])
            Cb = np.stack([o["critic"] for o in obs])
            for k, x in enumerate(res):
                if x[2]:
                    info = x[3]
                    infos.append(info)
                    window.append(float(info["success"]))
                    lab = float(info["success"])
                    # a sample of the episode's states, labelled with its outcome
                    st = ep_states[k]
                    idx = np.linspace(0, len(st) - 1, min(len(st), 8)).astype(int)
                    succ_x += [st[i] for i in idx]
                    succ_y += [lab] * len(idx)
                    ep_states[k] = []
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(cn(Cb)))
        # GAE; a timeout is treated as terminal (the episode length is part of the task)
        T = horizon
        R = torch.stack(buf["r"])
        D = torch.stack(buf["d"])
        V = torch.stack(buf["v"])
        adv = torch.zeros(T, n)
        gae = torch.zeros(n)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + 0.99 * nv * (1 - D[t]) - V[t]
            gae = delta + 0.99 * 0.95 * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        flat = lambda x: x.reshape(T * n, *x.shape[2:])
        Ab, Cbb, Ub, Lb = flat(torch.stack(buf["a"])), flat(torch.stack(buf["c"])), \
            flat(torch.stack(buf["u"])), flat(torch.stack(buf["logp"]))
        advb, retb = flat(adv), flat(ret)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        sx = torch.as_tensor(np.array(succ_x[-4096:])) if succ_x else None
        sy = torch.as_tensor(np.array(succ_y[-4096:]), dtype=torch.float32) if succ_y else None
        stats = []
        for epoch in range(5):
            perm = torch.randperm(T * n)
            for mb in perm.chunk(4):
                dist, _ = net.dist(Ab[mb])
                lp = squash_logp(dist, Ub[mb])
                ratio = (lp - Lb[mb]).exp()
                pg = -torch.min(ratio * advb[mb], ratio.clamp(0.8, 1.2) * advb[mb]).mean()
                vl = (net.value(Cbb[mb]) - retb[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + 0.5 * vl - 0.001 * ent
                if sx is not None and len(sx) > 32:
                    j = torch.randint(len(sx), (256,))
                    loss = loss + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                        net.success_logit(sx[j]), sy[j])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item(), (ratio - 1).abs().mean().item()))
        done_samples += T * n
        update += 1
        sr = float(np.mean(window)) if window else 0.0
        # curriculum (§5.4)
        if len(window) >= 100 and sr > 0.8 and stage < 3:
            stage += 1
            venv.set_stage(stage)
            window.clear()
        elif len(window) >= 100 and sr < 0.5 and stage > stage0:
            stage -= 1
            venv.set_stage(stage)
            window.clear()
        s = np.mean(stats, axis=0)
        row = {"run": run, "update": update, "samples": done_samples, "stage": stage,
               "success_100": round(sr, 3), "episodes": len(infos),
               "ep_success": round(float(np.mean([i["success"] for i in infos])), 3) if infos else None,
               "peak_force_mean": round(float(np.mean([i["peak_force_N"] for i in infos])), 1) if infos else None,
               "reasons": {r: sum(i["reason"] == r for i in infos) for r in
                           {i["reason"] for i in infos}},
               "pg": round(s[0], 4), "vl": round(s[1], 4), "ent": round(s[2], 3),
               "std": [round(v, 3) for v in net.log_std.exp().tolist()],
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if update % 10 == 0 or done_samples >= samples:
            torch.save({"net": net.state_dict(), "an": an.state(), "cn": cn.state(),
                        "run": run, "cfg": cfg, "stage": stage, "samples": done_samples},
                       out / "policy.pt")
    venv.close()
    return out / "policy.pt"


def load(path):
    from tasks.insertion_env import ACT_DIM, OBS_DIM, PRIV_DIM
    ck = torch.load(path, weights_only=False)
    net = ActorCritic(OBS_DIM, PRIV_DIM, ACT_DIM)
    net.load_state_dict(ck["net"])
    an = RunningNorm(OBS_DIM)
    an.load(ck["an"])
    return net, an, ck


class Policy:
    """Deterministic policy (mean action) + success probability, for eval and
    for the WP7 executor."""

    def __init__(self, path):
        self.net, self.an, self.ck = load(path)
        self.cfg = self.ck["cfg"]

    def __call__(self, actor_obs):
        with torch.no_grad():
            x = torch.as_tensor(self.an(np.asarray(actor_obs)))[None]
            dist, h = self.net.dist(x)
            return torch.tanh(dist.mean)[0].numpy(), float(torch.sigmoid(self.net.succ(h))[0])


def evaluate(path, stage, episodes, seed=12345, vision=None, scripted=False, joint="calibrated"):
    """Success rate, peak force, and the success head's AUC on held-out
    episodes -- always on the calibrated joint unless told otherwise."""
    from tasks.insertion_env import InsertionEnv
    pol = None if scripted else Policy(path)
    cfg = dict(RUNS["R1"]) if scripted else dict(pol.cfg)
    if vision is not None:
        cfg["vision"] = vision
    cfg["joint"] = joint
    env = InsertionEnv(stage=stage, seed=seed, **cfg)
    rows, probs, labels = [], [], []
    for ep in range(episodes):
        o = env.reset()
        done = False
        p_first = None
        while not done:
            if pol is None:
                a, p = np.zeros(7), None
            else:
                a, p = pol(o["actor"])
            if p_first is None and env.steps >= 20:
                p_first = p
            o, r, done, info = env.step(a)
        rows.append(info)
        if p_first is not None:
            probs.append(p_first)
            labels.append(float(info["success"]))
    succ = [float(i["success"]) for i in rows]
    out = {"policy": "scripted" if scripted else str(path), "stage": stage, "episodes": episodes,
           "success_rate": float(np.mean(succ)),
           "success_ci95": 1.96 * float(np.std(succ)) / math.sqrt(len(succ)),
           "peak_force_mean_N": float(np.mean([i["peak_force_N"] for i in rows])),
           "peak_force_success_mean_N": float(np.mean([i["peak_force_N"] for i in rows
                                                       if i["success"]] or [float("nan")])),
           "peak_over_nominal_press": float(np.mean([i["peak_force_N"] / (i["n_studs"] * 8.9)
                                                     for i in rows])),
           "duration_steps_mean": float(np.mean([i["steps"] for i in rows])),
           "reasons": {r: sum(i["reason"] == r for i in rows) for r in {i["reason"] for i in rows}},
           "vision": cfg["vision"], "joint": joint}
    if probs and 0 < sum(labels) < len(labels):
        out["success_head_auc"] = _auc(np.array(probs), np.array(labels))
    return out


def _auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def eval_all(episodes=100, stages=(0, 3), only=None, joint="calibrated"):
    """Every trained run (and the scripted base) on the CALIBRATED joint (or
    the one named), one row per (policy, stage) in results/wp5/eval.jsonl."""
    out = OUT / "eval.jsonl"
    names = ["scripted"] + [r for r in RUNS if (OUT / r / "policy.pt").exists()]
    for name in names:
        if only and name not in only:
            continue
        for st in stages:
            r = evaluate(None if name == "scripted" else OUT / name / "policy.pt", st, episodes,
                         scripted=name == "scripted", joint=joint)
            r["name"] = name
            with open(out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(name, st, round(r["success_rate"], 3), round(r["peak_force_mean_N"], 1),
                  r.get("success_head_auc"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--samples", type=int, default=150000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-worker", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--scripted", action="store_true", help="evaluate the scripted base alone")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--eval-all", nargs="*", default=None, help="evaluate these runs (all if empty)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--joint", default="calibrated", help="--eval-all: the joint to evaluate on")
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 3], help="--eval-all: stages")
    a = ap.parse_args()
    if a.eval_all is not None:
        eval_all(a.episodes, stages=a.stages, only=a.eval_all or None, joint=a.joint)
    elif a.eval or a.scripted:
        r = evaluate(a.eval, a.stage, a.episodes, vision=False if a.no_vision else None,
                     scripted=a.scripted)
        print(json.dumps(r, indent=1))
    else:
        train(a.run, a.samples, a.workers, a.per_worker, seed=a.seed, resume=a.resume)
