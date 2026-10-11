"""top_detect over many worlds, for rl_batched.BatchedReachEnv: detect -> remember -> track_held per world, the worlds split over worker processes.

top_detect.detect is ~5 ms per 640x480 image and python-bound (a thread pool measured 5.5 -> 4.3 ms per world: the GIL), so past a few worlds it
is the cost of a tick. Each worker owns a fixed slice of the worlds and their `det` state (remember and track_held read the previous result), reads the
images from shared memory and sends back the slimmed result. The workers import no CUDA, newton or torch (they run `_loop`; torch is only the
parent's view of the shared memory).

    pool = Detector(n_worlds, shape, intr, pos, R, procs)     procs = 0: in this process, the same code, no workers
    dets = pool.detect(hands)                                 hands[w]: the two hands of world w (top_detect.track_held's argument)
    pool.reset(worlds); pool.close()
    pool.rgb_t, pool.depth_t                                  the (n_worlds, H, W[, 3]) shared tensors (uint8, float32) to fill before detect()
"""

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

import top_detect as TD


def detect_world(state, w, rgb, depth, intr, pos, R, hands):
    """One world: detect on its image, remember / track_held against state[w]; returns the result without the big arrays (height, camera rays), copied."""
    res = TD.detect(rgb, depth, intr, pos, R)
    det = state[w] = TD.track_held(TD.remember(state.get(w), res), state.get(w), hands)
    return dict({k: v for k, v in det.items() if k not in ("height", "cam")}, bricks=[dict(b) for b in det["bricks"]], held={k: dict(b) for k, b in det["held"].items()})


def _loop(conn, names, shapes, intr, pos, R):
    import cv2
    cv2.setNumThreads(1)                                   # one process per core: cv2's own thread pool would oversubscribe
    shm = [shared_memory.SharedMemory(name=n) for n in names]
    rgb, depth = np.ndarray(shapes[0], np.uint8, shm[0].buf), np.ndarray(shapes[1], np.float32, shm[1].buf)
    state = {}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind, payload = msg
        if kind == "reset":
            for w in payload:
                state.pop(w, None)
        else:
            conn.send({w: detect_world(state, w, rgb[w], depth[w], intr, pos, R, hands) for w, hands in payload.items()})
    for s in shm:
        s.close()


class Detector:
    def __init__(self, n, shape, intr, pos, R, procs=0):
        """n worlds; shape (H, W) of the images detect reads; intr, pos, R: top_detect.detect's camera arguments (the same for every world)."""
        self.n, self.intr, self.pos, self.R, self.procs = n, intr, np.asarray(pos, float), R, procs
        self.shm = [shared_memory.SharedMemory(create=True, size=int(np.prod(s)) * b) for s, b in (((n, *shape, 3), 1), ((n, *shape), 4))]
        self.rgb = np.ndarray((n, *shape, 3), np.uint8, self.shm[0].buf)
        self.depth = np.ndarray((n, *shape), np.float32, self.shm[1].buf)
        import torch                                           # the parent's only: the workers never import it
        self.rgb_t, self.depth_t = torch.from_numpy(self.rgb), torch.from_numpy(self.depth)       # the same memory as torch tensors (a GPU tensor copies into it in one go)
        self.state, self.conns, self.workers = {}, [], []
        ctx = mp.get_context("spawn")                          # never fork a process that holds CUDA
        for j in range(procs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_loop, args=(b, [s.name for s in self.shm], [self.rgb.shape, self.depth.shape], intr, self.pos, R), daemon=True)
            p.start()
            self.conns.append(a)
            self.workers.append(p)
        self.owner = [w * max(procs, 1) // n for w in range(n)]     # world -> worker (contiguous slices)

    def detect(self, hands):
        if not self.procs:
            return [detect_world(self.state, w, self.rgb[w], self.depth[w], self.intr, self.pos, self.R, hands[w]) for w in range(self.n)]
        for j, c in enumerate(self.conns):
            c.send(("detect", {w: hands[w] for w in range(self.n) if self.owner[w] == j}))
        out = {}
        for c in self.conns:
            out.update(c.recv())
        return [out[w] for w in range(self.n)]

    def reset(self, worlds):
        if not self.procs:
            for w in worlds:
                self.state.pop(int(w), None)
            return
        for j, c in enumerate(self.conns):
            c.send(("reset", [int(w) for w in worlds if self.owner[w] == j]))

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.workers:
            p.join(5)
        self.conns, self.workers = [], []
        self.rgb = self.depth = self.rgb_t = self.depth_t = None     # release the views before closing the buffers
        for s in self.shm:
            s.close()
            s.unlink()
