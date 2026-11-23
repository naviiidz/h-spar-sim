"""VF-RRT* and SVF-RRT* as described by Zhang et al., "SVF-RRT*: A Stream-Based VF-RRT*
for USVs Path Planning Considering Ocean Currents", IEEE RA-L 8(4), 2023
(doi 10.1109/LRA.2023.3245409). Equation and algorithm numbers refer to that paper.

Shared (VF-RRT*, Sec. II-B and Alg. 1 without lines 13-15):
  - cost: upstream criterion C(q) = ∫ (||f|| - <f, q'>) ds                       (Eq. 1)
  - GetNewDirection: VF-RRT direction from the random direction and the field,
    controlled by lambda (Ko et al., VF-RRT)
  - Extend: a full step delta along the new direction; NULL only on collision
  - a node is inefficient if its distance to the nearest other tree node is below
    delta; it is kept, and only counted for E_ineff = N_ineff / (N_ineff + N_eff)
    over each cycle of n iterations                                              (Eq. 3)
  - lambda = lambda (1 - E_ineff + E_s) every n iterations                        (Eq. 2)
  - Near: R_rewire = min(delta, gamma (log|V| / |V|)^(1/d)), gamma = 1.1 gamma_min  (Eqs. 4, 5)
  - choose parent and rewire as in RRT*
  - GeneratePath: the cheapest tree path ending in the goal ball (radius r)
  - K iterations; the best path found is returned

SVF-RRT* adds (Alg. 1 lines 12-15, Algs. 2-4, Eqs. 6-13), each time a cheaper path q is found:
  - epsilon = C(q) / C_max(q), C_max = ∫ 2 ||f|| ds                               (Eqs. 6, 7)
  - interval Xi = [Phi_min - k_e dPhi, Phi_max + k_e dPhi] from the stream values Phi of
    the waypoints of q, k_e = eps / (1 - eps), dPhi = Phi_max - Phi_min           (Alg. 2, Eqs. 10, 11)
  - TreePruning: remove nodes with cost-to-come > C_min, and nodes with an ancestor
    (or themselves) outside Xi                                                    (Alg. 4)
  - BiasedSampling: a sample outside Xi is kept with probability epsilon         (Alg. 3)
  - lambda update y = lambda (1 - E_ineff + E_s) / eps; lambda = y if no path yet or
    y <= 1, else ln(y) + 1                                                        (Eqs. 12, 13)

The only addition is the optional length_weight (cost per metre added to Eq. 1), which
is 0 for the paper's method.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Params:
    iterations: int = 2000        # K
    delta: float = 30.0           # step / maximum edge length (m); paper value (Table I)
    goal_radius: float = 30.0     # r; paper value (Table I)
    E_s: float = 0.3
    lambda0: float = 3.0
    n_cycle: int = 100            # n: lambda update period (iterations)
    gamma_factor: float = 1.1     # gamma = gamma_factor * gamma_min
    sample_checks: int = 30       # points per edge for collision and cost integrals
    length_weight: float = 0.0    # not in the paper: cost per metre added to Eq. 1


@dataclass
class Result:
    path: list
    reached: bool
    cost: float = math.nan            # planning cost of the path (Eq. 1 + length term)
    upstream: float = math.nan        # Eq. 1 only
    first_iteration: int = 0          # iteration of the first feasible path
    improvements: int = 0             # number of times C_min decreased
    nodes: int = 0
    pruned: int = 0
    runtime_s: float = 0.0
    epsilon: float = 1.0
    log: list = field(default_factory=list)
    trace: list = field(default_factory=list)  # (iteration, C_min, epsilon, Xi low, Xi high) at each improvement


class Map:
    """Occupancy grid (100 = land), mean-current lookup and stream-function grid."""

    def __init__(self, folder: str, clearance: float = 0.0):
        z = np.load(f"{folder}/occupancy_grid.npz")
        self.origin = np.array([float(z["origin_x"]), float(z["origin_y"])])
        self.res = float(z["resolution"])
        self.land = z["grid"]
        self.clearance = clearance
        if clearance > 0:
            from planner_args import inflate
            self.grid = inflate(self.land, self.res, clearance)
        else:
            self.grid = self.land
        gx, gy = np.load(f"{folder}/grid_x.npy"), np.load(f"{folder}/grid_y.npy")
        u, v = np.load(f"{folder}/average_velocity_u_lookup.npy"), np.load(f"{folder}/average_velocity_v_lookup.npy")
        self.uv = RegularGridInterpolator((gx[:, 0], gy[0, :]), np.stack([u, v], axis=-1),
                                          method="linear", bounds_error=False, fill_value=np.nan)
        self.psi = np.load(f"{folder}/stream_function.npy")
        if self.psi.shape != self.grid.shape:
            raise ValueError(f"stream_function.npy {self.psi.shape} does not match the occupancy grid {self.grid.shape}")
        h, w = self.grid.shape
        self.lo = self.origin
        self.hi = self.origin + np.array([w, h]) * self.res
        self.free_cells = np.argwhere(self.grid <= 50)
        self.free_area = len(self.free_cells) * self.res ** 2

    def cell(self, pts):
        g = ((np.atleast_2d(pts) - self.origin) / self.res).astype(int)  # truncation
        return g[:, 0], g[:, 1]

    def free(self, pts) -> np.ndarray:
        gx, gy = self.cell(pts)
        h, w = self.grid.shape
        ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        out = np.zeros(len(gx), dtype=bool)
        out[ok] = self.grid[gy[ok], gx[ok]] <= 50
        return out

    def stream(self, p):
        gx, gy = self.cell(p)
        h, w = self.grid.shape
        if not (0 <= gx[0] < w and 0 <= gy[0] < h):
            return None
        val = self.psi[gy[0], gx[0]]
        return float(val) if np.isfinite(val) else None

    def flow(self, pts) -> np.ndarray:
        return np.nan_to_num(self.uv(np.atleast_2d(pts)), nan=0.0)

    def mean_speed(self) -> float:
        c = self.origin + (self.free_cells[:, ::-1] + 0.5) * self.res
        return float(np.mean(np.linalg.norm(self.flow(c), axis=1)))


class VFRRTStar:
    """VF-RRT* (svf=False) or SVF-RRT* (svf=True)."""

    def __init__(self, m: Map, p: Params, svf: bool, rng: np.random.Generator | None = None):
        self.m, self.p, self.svf = m, p, svf
        self.rng = rng or np.random  # np.random is seeded by planner_args --seed
        d = 2
        gamma_min = 2.0 * (1.0 + 1.0 / d) ** (1.0 / d) * (m.free_area / math.pi) ** (1.0 / d)
        self.gamma = p.gamma_factor * gamma_min

    # ---------------------------------------------------------------- geometry
    def _pts(self, a, b, include_start):
        t = np.arange(0 if include_start else 1, self.p.sample_checks + 1) / self.p.sample_checks
        return a + t[:, None] * (b - a)

    def collision_free(self, a, b) -> bool:
        return bool(np.all(self.m.free(self._pts(a, b, True))))

    def edge_integrals(self, a, b):
        """(Eq. 1 integral, Eq. 6 C_max integral) along the straight edge a -> b."""
        L = float(np.hypot(*(b - a)))
        if L < 1e-12:
            return 0.0, 0.0
        f = self.m.flow(self._pts(a, b, False))
        s = np.linalg.norm(f, axis=1)
        ds = L / self.p.sample_checks
        return float(np.sum(s - f @ ((b - a) / L)) * ds), float(np.sum(2.0 * s) * ds)

    def edge_cost(self, a, b) -> float:
        return self.edge_integrals(a, b)[0] + self.p.length_weight * float(np.hypot(*(b - a)))

    # ---------------------------------------------------------------- sampling
    def random_free(self):
        for _ in range(1000):
            q = self.rng.uniform(self.m.lo, self.m.hi)
            if self.m.free(q)[0]:
                return q
        gy, gx = self.m.free_cells[self.rng.randint(len(self.m.free_cells))]
        return self.m.origin + (np.array([gx, gy]) + 0.5) * self.m.res

    def biased_sampling(self):
        """Alg. 3 (with Xi = (-inf, inf) and epsilon = 1 it is uniform sampling)."""
        q = self.random_free()
        if not self.svf:
            return q
        phi = self.m.stream(q)
        lo, hi = self.xi
        if phi is None or lo <= phi <= hi:
            return q
        return q if self.rng.uniform(0.0, 1.0) >= 1.0 - self.eps else None

    def new_direction(self, q_from, q_to):
        """VF-RRT GetNewDirection: blend of the field direction and the random direction,
        with the deviation weight drawn from a truncated exponential with rate lambda."""
        v_rand = q_to - q_from
        v_rand = v_rand / np.linalg.norm(v_rand)
        f = self.m.flow(q_from)[0]
        nf = np.linalg.norm(f)
        if nf < 1e-12:
            return v_rand
        v_field = f / nf
        diff = v_rand - v_field
        dn = float(np.linalg.norm(diff))
        if dn < 1e-12:
            return v_field
        lam = max(1e-9, self.lam)
        f_lam = lam / (1.0 - math.exp(-2.0 * lam))           # normalisation on z in [0, 2]
        z_max = dn ** 2 / 2.0
        s_max = (f_lam / lam) * (1.0 - math.exp(-lam * z_max))  # CDF at z_max
        s = float(self.rng.uniform(0.0, s_max))
        z = -math.log(1.0 - s * lam / f_lam) / lam             # inverse CDF
        alpha = min(1.0, math.sqrt(max(0.0, 2.0 * z)) / dn)
        v = v_field + alpha * diff
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-12 else v_rand

    # ---------------------------------------------------------------- tree
    def _alive_idx(self):
        return np.flatnonzero(self.alive[:self.n])

    def _dist(self, q):
        idx = self._alive_idx()
        return idx, np.hypot(*(self.X[idx] - q).T)

    def _path_to(self, i):
        out = []
        while i is not None and i >= 0:
            out.append(i)
            i = int(self.parent[i]) if self.parent[i] >= 0 else None
        return out[::-1]

    def _propagate(self, root):
        stack = [root]
        while stack:
            u = stack.pop()
            for c in self.children[u]:
                self.cost[c] = self.cost[u] + self.edge_c[c]
                stack.append(c)

    def _remove(self, i):
        par = self.parent[i]
        if par >= 0:
            self.children[par].discard(i)
        self.alive[i] = False
        self.children[i] = set()

    def tree_pruning(self, c_min):
        """Alg. 4: nodes with cost-to-come > C_min, and nodes with themselves or an ancestor
        outside Xi, are removed (with them their whole subtrees)."""
        lo, hi = self.xi
        prune = []
        for i in self._alive_idx():
            if i == 0:
                continue
            if self.cost[i] > c_min + 1e-9:
                prune.append(i)
                continue
            j = i
            while j > 0:
                phi = self.node_phi[j]
                if phi is not None and not (lo <= phi <= hi):
                    prune.append(i)
                    break
                j = int(self.parent[j])
        for i in prune:
            self._remove(i)
        return len(prune)

    # ---------------------------------------------------------------- plan
    def plan(self, start, goal) -> Result:
        p = self.p
        K = p.iterations
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        self.X = np.zeros((K + 2, 2))
        self.parent = np.full(K + 2, -1, dtype=int)
        self.cost = np.zeros(K + 2)
        self.edge_c = np.zeros(K + 2)
        self.alive = np.zeros(K + 2, dtype=bool)
        self.children = [set() for _ in range(K + 2)]
        self.node_phi = [None] * (K + 2)
        self.X[0], self.alive[0], self.node_phi[0], self.n = start, True, self.m.stream(start), 1
        self.lam = p.lambda0
        self.eps, self.xi = 1.0, (-math.inf, math.inf)
        c_min, best_q = math.inf, None
        n_eff = n_ineff = 0
        res = Result(path=[], reached=False)
        t0 = time.time()

        for k in range(K):
            q_rand = self.biased_sampling()
            if q_rand is not None:
                idx, d = self._dist(q_rand)
                nearest = int(idx[np.argmin(d)])
                if np.min(d) > 1e-12:
                    v = self.new_direction(self.X[nearest], q_rand)
                    q_new = self.X[nearest] + p.delta * v                       # Extend
                    if self.m.free(q_new)[0] and self.collision_free(self.X[nearest], q_new):
                        idx, d = self._dist(q_new)
                        if np.min(d) < p.delta - 1e-9:                          # Eq. 3 bookkeeping
                            n_ineff += 1
                        else:
                            n_eff += 1
                        nv = max(len(idx) + 1, 2)
                        r_rew = min(p.delta, self.gamma * math.sqrt(math.log(nv) / nv))   # Eq. 4
                        near = idx[d <= r_rew + 1e-9]
                        # choose parent (nearest is always a candidate: it is collision free)
                        best_par = nearest
                        best_ec = self.edge_cost(self.X[nearest], q_new)
                        best_c = self.cost[nearest] + best_ec
                        for j in near:
                            if j == nearest:
                                continue
                            ec = self.edge_cost(self.X[j], q_new)
                            if self.cost[j] + ec < best_c and self.collision_free(self.X[j], q_new):
                                best_par, best_ec, best_c = int(j), ec, self.cost[j] + ec
                        i = self.n
                        self.n += 1
                        self.X[i], self.alive[i], self.node_phi[i] = q_new, True, self.m.stream(q_new)
                        self.parent[i], self.edge_c[i], self.cost[i] = best_par, best_ec, best_c
                        self.children[best_par].add(i)
                        # rewire
                        for j in near:
                            if j == best_par or j == 0:
                                continue
                            ec = self.edge_cost(q_new, self.X[j])
                            if self.cost[i] + ec + 1e-12 < self.cost[j] and self.collision_free(q_new, self.X[j]):
                                self.children[self.parent[j]].discard(int(j))
                                self.parent[j], self.edge_c[j] = i, ec
                                self.children[i].add(int(j))
                                self.cost[j] = self.cost[i] + ec
                                self._propagate(int(j))
                        # GeneratePath: cheapest tree node in the goal ball
                        idx = self._alive_idx()
                        in_goal = idx[np.hypot(*(self.X[idx] - goal).T) < p.goal_radius]
                        if len(in_goal):
                            g = int(in_goal[np.argmin(self.cost[in_goal])])
                            if self.cost[g] < c_min - 1e-12:                    # Alg. 1 line 12
                                if best_q is None:
                                    res.first_iteration = k + 1
                                c_min, best_q = float(self.cost[g]), g
                                res.improvements += 1
                                if self.svf:
                                    self._update_interval(g)
                                    res.pruned += self.tree_pruning(c_min)
                                res.trace.append((k + 1, c_min, self.eps, *self.xi))
            # lambda update every n iterations (Eq. 2 / Eqs. 12-13)
            if (k + 1) % p.n_cycle == 0:
                e_ineff = n_ineff / (n_ineff + n_eff) if (n_ineff + n_eff) else 0.0
                if self.svf:
                    y = self.lam if self.eps == 0 else self.lam * (1.0 - e_ineff + p.E_s) / self.eps
                    self.lam = y if (best_q is None or y <= 1.0) else math.log(y) + 1.0
                else:
                    self.lam = self.lam * (1.0 - e_ineff + p.E_s)
                res.log.append((k + 1, round(self.lam, 4), round(e_ineff, 3), round(self.eps, 3)))
                n_eff = n_ineff = 0

        res.runtime_s = time.time() - t0
        res.nodes = int(self.alive[:self.n].sum())
        res.epsilon = self.eps
        if best_q is None:
            idx, d = self._dist(goal)
            res.path = [tuple(self.X[i]) for i in self._path_to(int(idx[np.argmin(d)]))]
            return res
        ids = self._path_to(best_q)
        pts = [self.X[i] for i in ids]
        if np.hypot(*(pts[-1] - goal)) > 1e-9 and self.collision_free(pts[-1], goal):
            pts.append(goal)  # the mission needs the goal point itself (outside the cost comparison)
        res.path = [tuple(map(float, q)) for q in pts]
        res.reached = True
        res.cost = float(sum(self.edge_cost(a, b) for a, b in zip(pts[:-1], pts[1:])))
        res.upstream = float(sum(self.edge_integrals(a, b)[0] for a, b in zip(pts[:-1], pts[1:])))
        return res

    def _update_interval(self, g):
        """Eq. 7 and Alg. 2 for the path ending at node g."""
        ids = self._path_to(g)
        up = cmax = 0.0
        for a, b in zip(ids[:-1], ids[1:]):
            u, c = self.edge_integrals(self.X[a], self.X[b])
            up, cmax = up + u, cmax + c
        self.eps = 1.0 if cmax == 0 else up / cmax
        phis = [self.node_phi[i] for i in ids if self.node_phi[i] is not None]
        if self.eps >= 1.0 or not phis:
            self.xi = (-math.inf, math.inf)
            return
        k_e = self.eps / (1.0 - self.eps)
        lo, hi = min(phis), max(phis)
        self.xi = (lo - k_e * (hi - lo), hi + k_e * (hi - lo))


def main(svf: bool):
    """Plan once with the command-line options of planner_args and write the outputs."""
    import planner_args
    lookup, out_dir, suffix = planner_args.parse()
    import matplotlib.pyplot as plt
    name = "svf" if svf else "vf"
    label = "SVF-RRT*" if svf else "VF-RRT*"
    m = Map(lookup, clearance=planner_args.clearance())
    p = Params(iterations=planner_args.max_iter(2000))
    k = planner_args.length_factor(0.0)
    if k:
        p.length_weight = k * m.mean_speed()
    start, goal = planner_args.START, planner_args.GOAL
    print(f"{label}: map {lookup}, {p.iterations} iterations, delta {p.delta} m, r {p.goal_radius} m, "
          f"E_s {p.E_s}, lambda0 {p.lambda0}, n {p.n_cycle}, length weight {p.length_weight:.4f}, "
          f"clearance {m.clearance:g} m")
    planner = VFRRTStar(m, p, svf=svf)
    r = planner.plan(start, goal)
    L = float(sum(math.dist(a, b) for a, b in zip(r.path[:-1], r.path[1:])))
    if r.reached:
        print(f"✓ Goal reached: first path at iteration {r.first_iteration}, {r.improvements} improvements, "
              f"{r.nodes} nodes ({r.pruned} pruned), epsilon {r.epsilon:.3f}, {r.runtime_s:.1f}s")
    else:
        print("⚠ Goal not reached")
        print(f"  Closest node distance to goal: {math.dist(r.path[-1], goal):.1f} m")
    print(f"  Path length: {L:.1f} m")
    print(f"  Upstream cost: {r.upstream:.3f}")
    print(f"  Goal reached: {r.reached}")
    with open(f"{out_dir}/{name}_waypoints{suffix}.csv", "w") as fh:
        fh.write("x,y\n")
        for x, y in r.path:
            fh.write(f"{x:.6f},{y:.6f}\n")
    # figure: tree and path over the mean speed
    fig, ax = plt.subplots(figsize=(12, 5))
    ext = [m.lo[0], m.hi[0], m.lo[1], m.hi[1]]
    xs = m.origin[0] + (np.arange(m.grid.shape[1]) + 0.5) * m.res
    ys = m.origin[1] + (np.arange(m.grid.shape[0]) + 0.5) * m.res
    xx, yy = np.meshgrid(xs, ys)
    sp = np.linalg.norm(m.flow(np.column_stack([xx.ravel(), yy.ravel()])), axis=1).reshape(xx.shape)
    ax.imshow(np.where(m.grid <= 50, sp, np.nan), origin="lower", extent=ext, cmap="viridis")
    ax.imshow(np.where(m.grid > 50, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)
    for i in planner._alive_idx():
        if planner.parent[i] >= 0:
            a, b = planner.X[planner.parent[i]], planner.X[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=0.5, alpha=0.5)
    pa = np.array(r.path)
    ax.plot(pa[:, 0], pa[:, 1], "r-", lw=2.5)
    ax.plot(*start, "go", ms=10)
    ax.plot(*goal, "y*", ms=15, mec="k")
    ax.set_title(f"{label}{suffix}: {L:.0f} m, upstream {r.upstream:.1f}")
    ax.set_aspect("equal")
    fig.savefig(f"{out_dir}/{name}_rrt_star_path{suffix}.png", dpi=120, bbox_inches="tight")
    if planner_args.show():
        plt.show()
    return r
