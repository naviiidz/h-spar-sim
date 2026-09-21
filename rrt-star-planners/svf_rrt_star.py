"""
VF-RRT* (paper-faithful style, refactored)

Single-file / single-cell friendly script:
- keeps existing occupancy-grid NPZ loading pattern
- keeps existing velocity lookup loading via RegularGridInterpolator
- uses modular planner methods for easier verification
- pruning not implemented yet (but can be added by modifying rewire and propagate_subtree_costs)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import euclidean


# -----------------------------------------------------------------------------
# IO helpers (same loading pattern)
# -----------------------------------------------------------------------------
LOOKUP_FOLDER = "./lookup_tables/sydney_regatta"
OUTPUT_DIR = "./lookup_tables/sydney_regatta/output"
DATA_DIR = LOOKUP_FOLDER

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_velocity_interpolators(folder: str) -> Tuple[Tuple[RegularGridInterpolator, RegularGridInterpolator], Tuple[float, float, float, float]]:
    """Load (u, v) lookup tables and return interpolators + bounds."""
    grid_x = np.load(f"{folder}/grid_x.npy")
    grid_y = np.load(f"{folder}/grid_y.npy")
    u_lookup = np.load(f"{folder}/average_velocity_u_lookup.npy")
    v_lookup = np.load(f"{folder}/average_velocity_v_lookup.npy")

    x_coords = grid_x[:, 0]
    y_coords = grid_y[0, :]

    u_interp = RegularGridInterpolator(
        (x_coords, y_coords),
        u_lookup,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    v_interp = RegularGridInterpolator(
        (x_coords, y_coords),
        v_lookup,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    bounds = (float(x_coords.min()), float(x_coords.max()), float(y_coords.min()), float(y_coords.max()))
    return (u_interp, v_interp), bounds


def load_occupancy_from_npz(data_dir: str) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Load occupancy grid, origin and resolution from NPZ."""
    data = np.load(f"{data_dir}/occupancy_grid.npz")
    grid = data["grid"]
    origin = (float(data["origin_x"]), float(data["origin_y"]))
    resolution = float(data["resolution"])
    return grid, origin, resolution


# -----------------------------------------------------------------------------
# Phase 1: clean skeleton / layout
# -----------------------------------------------------------------------------
class VFRRTStarSkeleton:
    """Method layout only (for readability and verification)."""

    def sample_state(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        raise NotImplementedError

    def query_flow(self, q: Tuple[float, float]) -> np.ndarray:
        raise NotImplementedError

    def generate_vf_direction(self, q_from: Tuple[float, float], q_to: Tuple[float, float]) -> np.ndarray:
        
        raise NotImplementedError

    def collision_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        raise NotImplementedError

    def edge_cost(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        raise NotImplementedError

    def select_parent(self, new_state: Tuple[float, float], neighbor_ids: List[int], nearest_id: int) -> Tuple[int, float]:
        raise NotImplementedError

    def rewire(self, new_id: int, neighbor_ids: List[int]) -> None:
        raise NotImplementedError

    def propagate_subtree_costs(self, root_id: int) -> None:
        raise NotImplementedError

    def extract_path(self, end_id: int) -> List[Tuple[float, float]]:
        raise NotImplementedError

    def log_progress(self, iteration: int, start_time: float) -> None:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Phase 2: filled implementation
# -----------------------------------------------------------------------------
@dataclass
class PlannerStats:
    iterations: int = 0
    reached_goal: bool = False
    ineffective_extensions: int = 0
    runtime_sec: float = 0.0
    recent_total: int = 0
    recent_ineff: int = 0
    latest_inefficiency: float = 0.0


class VFRRTStar(VFRRTStarSkeleton):
    """Refactored VF-RRT* planner using upstream flow cost functional."""

    def __init__(
        self,
        occupancy_grid: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        interpolators: Tuple[RegularGridInterpolator, RegularGridInterpolator],
        max_iter: int = 500,
        step_size: float = 20.0,
        goal_sample_rate: float = 0.5,
        sample_checks: int = 80,
        rewire_base_radius: float = 80.0,
        goal_tolerance: float = 30.0,
        lambda_gain: float = 3.0,
        ineff_window_size: int = 100,
        lambda_update_every: int = 50,
        ineff_setpoint: float = 0.4,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
        use_field_magnitude_scaling: bool = False,
        field_scale_m: float = 1.0,
        debug_step_by_step: bool = False,
        debug_pause_sec: float = 0.0,
        es: float = 0.1,
        debug_live_plot: bool = False,
        log_every: int = 100,
    ) -> None:
        self.grid = occupancy_grid
        self.origin = origin
        self.resolution = resolution
        self.u_interp, self.v_interp = interpolators

        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.sample_checks = sample_checks
        self.rewire_base_radius = rewire_base_radius
        self.goal_tolerance = goal_tolerance
        self.lambda_gain = float(lambda_gain)
        self.ineff_window_size = max(1, int(ineff_window_size))
        self.lambda_update_every = max(1, int(lambda_update_every))
        self.ineff_setpoint = float(ineff_setpoint)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.use_field_magnitude_scaling = bool(use_field_magnitude_scaling)
        self.field_scale_m = float(field_scale_m)
        self.debug_step_by_step = bool(debug_step_by_step)
        self.debug_pause_sec = float(debug_pause_sec)
        self.es = float(es)
        self.debug_live_plot = bool(debug_live_plot)
        self.log_every = log_every

        self.heuristic_interval = (float(-np.inf), float(np.inf)) #svf
        self.epsilon = 1.0 #svf upstream_coefficient
        self.psi: Optional[np.ndarray] = None
        self.stream_grid: Optional[np.ndarray] = None
        stream_path = f"{OUTPUT_DIR}/stream_function.npy"
        try:
            sg = np.load(stream_path)
            if sg.shape == self.grid.shape:
                self.psi = sg
                self.stream_grid = sg
            else:
                print(f"Warning: stream grid shape {sg.shape} != occupancy shape {self.grid.shape}; stream lookup disabled.")
        except Exception:
            pass
        self.grid_h, self.grid_w = self.grid.shape
        self.world_x_min = origin[0]
        self.world_y_min = origin[1]
        self.world_x_max = origin[0] + self.grid_w * resolution
        self.world_y_max = origin[1] + self.grid_h * resolution

        # tree containers
        self.nodes: List[Tuple[float, float]] = []
        self.parent: Dict[int, Optional[int]] = {}
        self.children: Dict[int, Set[int]] = {}
        self.cost: List[float] = []
        self.final_path_node_id: Optional[int] = None

        self.path: List[Tuple[float, float]] = []
        self.current_path: List[Tuple[float, float]] = []
        self.path_psi_values: List[float] = []
        self.node_psi: Dict[int, float] = {}
        self.stats = PlannerStats()
        self.recent_attempt_flags: Deque[int] = deque(maxlen=self.ineff_window_size)

    # ---------- low-level helpers ----------
    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.array([0.0, 0.0], dtype=float)
        return v / n

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def is_free_state(self, q: Tuple[float, float]) -> bool:
        gx, gy = self.world_to_grid(q[0], q[1])
        if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
            return False
        return self.grid[gy, gx] <= 50

    def nearest_id(self, q: Tuple[float, float]) -> int:
        return min(range(len(self.nodes)), key=lambda i: euclidean(self.nodes[i], q))

    def near_ids(self, q: Tuple[float, float]) -> List[int]:
        n = max(2, len(self.nodes))
        radius = max(30.0, self.rewire_base_radius * math.sqrt(math.log(n + 1) / (n + 1)))
        return [i for i, node in enumerate(self.nodes) if euclidean(node, q) <= radius]

    def _attach_child(self, child: int, parent: Optional[int]) -> None:
        self.parent[child] = parent
        if child not in self.children:
            self.children[child] = set()
        if parent is not None:
            if parent not in self.children:
                self.children[parent] = set()
            self.children[parent].add(child)

    def _detach_child(self, child: int) -> None:
        p = self.parent.get(child)
        if p is not None and p in self.children:
            self.children[p].discard(child)

    def _record_expansion_attempt(self, inefficient: bool) -> None:
        """Track expansion attempts in a fixed-size sliding window."""
        self.recent_attempt_flags.append(1 if inefficient else 0)
        self.stats.recent_total = len(self.recent_attempt_flags)
        self.stats.recent_ineff = int(sum(self.recent_attempt_flags))
        self.stats.latest_inefficiency = self.compute_inefficiency()
        if inefficient:
            self.stats.ineffective_extensions += 1

    def compute_inefficiency(self) -> float:
        """E_ineff = N_ineff / N_total over recent expansion attempts."""
        n_total = len(self.recent_attempt_flags)
        if n_total == 0:
            return 0.0
        n_ineff = int(sum(self.recent_attempt_flags))
        return float(n_ineff / n_total)

    def compute_window_inefficiency(self) -> float:
        """Sliding-window inefficiency: sum(I_ineff)/W over recent window size W."""
        if self.ineff_window_size <= 0:
            return 0.0
        n_ineff = int(sum(self.recent_attempt_flags))
        return float(n_ineff / self.ineff_window_size)

    def inefficiency_interpretation(self, e_ineff: float) -> str:
        if e_ineff < self.ineff_setpoint:
            return "E_ineff < E_s: effective exploration, lambda increases (follow field more)"
        if e_ineff > self.ineff_setpoint:
            return "E_ineff > E_s: wasted attempts, lambda decreases (explore more)"
        return "E_ineff ~= E_s: balanced exploration-following behavior"

    def update_lambda_from_inefficiency(self) -> float:
        """Update lambda: lambda_new = lambda_old * (1 - E_ineff + E_s)."""
        e_ineff = self.compute_window_inefficiency()
        scale = 1.0 - e_ineff + self.ineff_setpoint
        self.lambda_gain = self.lambda_gain *float(1-e_ineff+self.es)#float(np.clip(self.lambda_gain * scale, self.lambda_min, self.lambda_max))
        self.stats.latest_inefficiency = e_ineff
        self.stats.recent_total = len(self.recent_attempt_flags)
        self.stats.recent_ineff = int(sum(self.recent_attempt_flags))
        return e_ineff

    # ---------- required modular methods ----------
    def sample_state(self, goal: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Goal-biased sampling in the world bounding box."""
        if np.random.random() < self.goal_sample_rate:
            s_goal = self.lookup_stream(goal)
            if self.debug_step_by_step:
                print(f"  sampled goal, psi={s_goal}")
            return self._apply_heuristic_interval(goal, s_goal)

        # Rejection-sample until the point lies inside free occupancy-grid space.
        # This keeps samples inside the occupancy polygon rather than only inside the
        # rectangular world bounding box.
        for _ in range(500):
            q = (
                float(np.random.uniform(-600, 500)),
                float(np.random.uniform(150, 300)),
            )
            if self.is_free_state(q):
                s_q = self.lookup_stream(q)
                if self.debug_step_by_step:
                    print(f"  sampled free point {q}, psi={s_q}")
                return self._apply_heuristic_interval(q, s_q)

        # Fallback: choose the center of a random free cell.
        free_cells = np.argwhere(self.grid <= 50)
        if len(free_cells) == 0:
            return (
                float(np.random.uniform(self.world_x_min, self.world_x_max)),
                float(np.random.uniform(self.world_y_min, self.world_y_max)),
            )
        gy, gx = free_cells[np.random.randint(len(free_cells))]
        q_fb = (
            self.origin[0] + (float(gx) + 0.5) * self.resolution,
            self.origin[1] + (float(gy) + 0.5) * self.resolution,
        )
        s_fb = self.lookup_stream(q_fb)
        if self.debug_step_by_step:
            print(f"  fallback sampled free cell center {q_fb}, psi={s_fb}") #svf
        return self._apply_heuristic_interval(q_fb, s_fb)

    def _apply_heuristic_interval(
        self,
        q: Tuple[float, float],
        psi_value: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        """Accept q if psi is inside heuristic interval; otherwise accept with probability epsilon."""
        if psi_value is None:
            return q

        lo, hi = self.heuristic_interval
        if lo <= psi_value <= hi:
            return q

        pp = float(np.random.uniform(0.0, 1.0))
        if pp >= 1.0 - self.epsilon:
            return q
        return None

    def lookup_stream(self, q: Tuple[float, float]) -> Optional[float]:
        """Lookup stream-function value at a state using nearest occupancy cell."""
        if self.psi is None:
            return None
        gx, gy = self.world_to_grid(q[0], q[1])
        if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
            return None
        val = self.psi[gy, gx]
        if not np.isfinite(val):
            return None
        return float(val)

    def get_interval(self, new_id: int) -> List[float]:
        """Get current root-to-node path and attribute the node's psi to its node id."""
        self.current_path = self.extract_path(new_id)
        psi_new = self.lookup_stream(self.nodes[new_id])
        if psi_new is not None:
            psi_new = float(psi_new)
            self.path_psi_values.append(psi_new)
            self.node_psi[new_id] = psi_new
        if new_id in self.node_psi:
            print(f"node {new_id} -> psi {self.node_psi[new_id]}")
        if not self.path_psi_values:
            self.heuristic_interval = (-np.inf, np.inf)
            return self.path_psi_values
        if self.epsilon==1:
            self.heuristic_interval=(-np.inf, np.inf)
        else:
            ke=self.epsilon/(1-self.epsilon)
            delta_psi=abs(self.path_psi_values[-1]-self.path_psi_values[0])
            self.heuristic_interval = (min(self.path_psi_values)-ke*delta_psi, max(self.path_psi_values)+ke*delta_psi)
            print(self.node_psi)
        return self.path_psi_values

    def query_flow(self, q: Tuple[float, float]) -> np.ndarray:
        """Return flow vector f(q)=[u,v], with NaNs replaced by 0."""
        u = self.u_interp([[q[0], q[1]]])[0]
        v = self.v_interp([[q[0], q[1]]])[0]
        if not np.isfinite(u):
            u = 0.0
        if not np.isfinite(v):
            v = 0.0
        return np.array([float(u), float(v)], dtype=float)

    def generate_vf_direction(self, q_from: Tuple[float, float], q_to: Tuple[float, float]) -> np.ndarray:
        """VF-style direction: λ-based sampling between flow and sample directions."""
        v_rand = self._unit(np.array([q_to[0] - q_from[0], q_to[1] - q_from[1]], dtype=float))
        f = self.query_flow(q_from)
        v_field = self._unit(f)

        if np.linalg.norm(v_field) < 1e-12:
            return v_rand

        lam = max(1e-9, self.lambda_gain)
        if self.use_field_magnitude_scaling:
            f_norm = float(np.linalg.norm(f))
            if f_norm > 1e-12:
                lam = max(1e-9, lam * (self.field_scale_m / f_norm))

        # 1) normalizing factor f(lambda)
        f_lam = lam / (1.0 - np.exp(-2.0 * lam))

        # 2) geometric limit z_max (v_field and v_rand are unit vectors)
        diff = v_rand - v_field
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm < 1e-12:
            return v_field
        z_max = (diff_norm ** 2) / 2.0

        # 3) sampling limit s_max
        s_max = (f_lam / lam) * (1.0 - np.exp(-lam * z_max))

        # 4) sample s in [0, s_max)
        s = float(np.random.uniform(0.0, s_max))

        # 5) corrected inverse CDF for z
        z = -(np.log(1.0 - (s * lam / f_lam))) / lam

        # 6) convert to control weight v
        v = float(np.sqrt(max(0.0, 2.0 * z)))

        # build sampled direction and normalize
        alpha = float(np.clip(v / diff_norm, 0.0, 1.0))
        v_new = self._unit(v_field + alpha * diff)
        v_hat = v_new
        if np.linalg.norm(v_hat) < 1e-12:
            return v_rand
        return v_hat

    def collision_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Segment collision checking against occupancy grid."""
        for i in range(self.sample_checks + 1):
            t = i / self.sample_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            gx, gy = self.world_to_grid(x, y)
            if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
                return False
            if self.grid[gy, gx] > 50:
                return False
        return True

    def edge_cost(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Paper-style upstream cost: ∫ (||f|| - <f, q'>) ds."""
        if euclidean(p1, p2) < 1e-12:
            return 0.0

        total = 0.0
        prev = p1
        c_max = 0.0
        for i in range(1, self.sample_checks + 1):
            t = i / self.sample_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            cur = (x, y)

            ds_vec = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=float)
            ds = float(np.linalg.norm(ds_vec))
            if ds < 1e-12:
                prev = cur
                continue

            q_prime = ds_vec / ds
            f = self.query_flow(cur)
            integrand = float(np.linalg.norm(f) - np.dot(f, q_prime))

            c_max+= float(np.linalg.norm(f)) * 2.0
            total += integrand * ds
            prev = cur

        if c_max < 1e-12 or euclidean(p1, p2) < 1e-12:
            self.epsilon= 1.0
        else:
            self.epsilon= total/c_max

        return total

    def select_parent(self, new_state: Tuple[float, float], neighbor_ids: List[int], nearest_id: int) -> Tuple[int, float]:
        """Choose parent that minimizes cost-to-come + edge cost."""
        best_parent = nearest_id
        best_cost = self.cost[nearest_id] + self.edge_cost(self.nodes[nearest_id], new_state)

        for idx in neighbor_ids:
            if idx >= len(self.cost):
                continue
            if not self.collision_free(self.nodes[idx], new_state):
                continue
            c = self.cost[idx] + self.edge_cost(self.nodes[idx], new_state)
            if c < best_cost:
                best_parent = idx
                best_cost = c

        return best_parent, best_cost

    def rewire(self, new_id: int, neighbor_ids: List[int]) -> None:
        """RRT* rewiring with subtree cost propagation."""
        for idx in neighbor_ids:
            if idx == new_id or idx >= len(self.cost):
                continue
            if not self.collision_free(self.nodes[new_id], self.nodes[idx]):
                continue

            candidate = self.cost[new_id] + self.edge_cost(self.nodes[new_id], self.nodes[idx])
            if candidate + 1e-12 < self.cost[idx]:
                # re-parent idx under new_id
                self._detach_child(idx)
                self._attach_child(idx, new_id)
                self.cost[idx] = candidate
                self.propagate_subtree_costs(idx)

    def propagate_subtree_costs(self, root_id: int) -> None:
        """Update descendants after a rewire (BFS)."""
        queue = [root_id]
        while queue:
            u = queue.pop(0)
            for ch in self.children.get(u, set()):
                self.cost[ch] = self.cost[u] + self.edge_cost(self.nodes[u], self.nodes[ch])
                queue.append(ch)

    def extract_path(self, end_id: int) -> List[Tuple[float, float]]:
        """Backtrack parent links from end node to root."""
        path: List[Tuple[float, float]] = []
        cur: Optional[int] = end_id
        while cur is not None:
            path.append(self.nodes[cur])
            cur = self.parent.get(cur)
        path.reverse()
        return path

    def log_progress(self, iteration: int, start_time: float) -> None:
        if (iteration + 1) % self.log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Iteration {iteration + 1}/{self.max_iter}, nodes: {len(self.nodes)}, time: {elapsed:.1f}s")

    def _debug_step(self, iteration: int, message: str) -> None:
        if self.debug_step_by_step:
            print(f"[step {iteration + 1}] {message}")
            if self.debug_pause_sec > 0:
                time.sleep(self.debug_pause_sec)

    def _init_live_plot(self):
        if not self.debug_live_plot:
            return None, None

        fig, ax = plt.subplots(figsize=(16, 8))
        origin = self.origin
        grid = self.grid
        res = self.resolution
        extent = [
            origin[0],
            origin[0] + grid.shape[1] * res,
            origin[1],
            origin[1] + grid.shape[0] * res,
        ]
        ax.imshow(grid <= 50, cmap="gray", origin="lower", extent=extent, alpha=0.25)
        #ax.set_title("SVF-RRT* Live Tree Growth")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)
        plt.ion()
        plt.show(block=False)
        return fig, ax

    def _update_live_plot(
        self,
        fig,
        ax,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        q_near: Optional[Tuple[float, float]] = None,
        q_rand: Optional[Tuple[float, float]] = None,
        direction: Optional[np.ndarray] = None,
        step: Optional[float] = None,
    ) -> None:
        if fig is None or ax is None:
            return

        ax.clear()
        origin = self.origin
        grid = self.grid
        res = self.resolution
        extent = [
            origin[0],
            origin[0] + grid.shape[1] * res,
            origin[1],
            origin[1] + grid.shape[0] * res,
        ]
        ax.imshow(grid <= 50, cmap="gray", origin="lower", extent=extent, alpha=0.25)

        for i, p in self.parent.items():
            if p is None:
                continue
            a = self.nodes[p]
            b = self.nodes[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", alpha=0.7, linewidth=1.0)

        if q_near is not None and direction is not None and step is not None:
            qn = np.array(q_near, dtype=float)
            d = np.array(direction, dtype=float)
            q_end = qn + step * d
            ax.arrow(
                qn[0],
                qn[1],
                q_end[0] - qn[0],
                q_end[1] - qn[1],
                length_includes_head=True,
                head_width=4.0,
                head_length=6.0,
                fc="cyan",
                ec="cyan",
                alpha=0.9,
                linewidth=1.8,
            )

        if q_rand is not None:
            ax.scatter(q_rand[0], q_rand[1], c="magenta", s=35, marker="x", label="q_rand")
        if q_near is not None:
            ax.scatter(q_near[0], q_near[1], c="dodgerblue", s=45, marker="o", label="q_near")

        ax.scatter(*start, c="green", s=200, marker="o", edgecolors="black", linewidth=1.5, label="Start")
        ax.scatter(*goal, c="gold", s=200, marker="s", edgecolors="black", linewidth=1.5, label="Goal")
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        ax.set_title(f"SVF-RRT* Live Tree Growth | nodes={len(self.nodes)}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()
        plt.pause(0.001)

    # ---------- orchestration ----------
    def extend(self, from_id: int, direction: np.ndarray, step: float) -> Optional[int]:
        """Create a new node by extension if collision-free."""
        q1 = self.nodes[from_id]
        q_new = (q1[0] + step * direction[0], q1[1] + step * direction[1])

        # inefficiency condition 1: collision at node or along edge
        in_collision = (not self.is_free_state(q_new)) or (not self.collision_free(q1, q_new))
        if in_collision:
            self._record_expansion_attempt(inefficient=True)
            return None

        # inefficiency condition 2: no meaningful expansion (< step_size from nearest node)
        nearest_existing_dist = min(euclidean(node, q_new) for node in self.nodes)
        if nearest_existing_dist + 1e-12 < self.step_size:
            self._record_expansion_attempt(inefficient=True)
            return None

        # efficient candidate node
        self._record_expansion_attempt(inefficient=False)

        new_id = len(self.nodes)
        self.nodes.append(q_new)
        self.parent[new_id] = None
        self.children[new_id] = set()
        return new_id

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> bool:
        # reset
        self.nodes = [start]
        self.parent = {0: None}
        self.children = {0: set()}
        self.cost = [0.0]
        self.path = []
        self.current_path = []
        self.path_psi_values = []
        self.node_psi = {}
        self.stats = PlannerStats()
        self.recent_attempt_flags.clear()

        print("\n" + "=" * 60)
        print("VF-RRT* Path Planning")
        print("=" * 60)
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"World bounds: x=[{self.world_x_min:.1f}, {self.world_x_max:.1f}], y=[{self.world_y_min:.1f}, {self.world_y_max:.1f}]")
        print(f"Iterations: {self.max_iter}, Step size: {self.step_size}")
        print("=" * 60)

        start_time = time.time()
        goal_id: Optional[int] = None
        live_fig, live_ax = self._init_live_plot()

        start_psi = self.lookup_stream(start)
        if start_psi is not None:
            self.node_psi[0] = float(start_psi)
            self.path_psi_values.append(float(start_psi))
            print(f"node 0 -> psi {self.node_psi[0]}")

        for k in range(self.max_iter):
            q_rand = self.sample_state(goal)
            if q_rand is None:
                self._debug_step(k, "sample rejected by heuristic interval")
                continue
            self._debug_step(k, f"sampled q_rand={q_rand}")
            nearest = self.nearest_id(q_rand)
            q_near = self.nodes[nearest]
            self._debug_step(k, f"nearest_id={nearest}, q_near={q_near}")

            d = euclidean(q_near, q_rand)
            if d < 1e-12:
                self._debug_step(k, "skipped: q_rand too close to q_near")
                continue

            v_hat = self.generate_vf_direction(q_near, q_rand)
            step = min(self.step_size, d)
            self._debug_step(k, f"direction={v_hat}, step={step:.3f}")
            new_id = self.extend(nearest, v_hat, step)
            if new_id is None:
                self._debug_step(k, "rejected: collision or invalid expansion")
                if (k + 1) % self.lambda_update_every == 0 and len(self.recent_attempt_flags) > 0:
                    e_ineff = self.update_lambda_from_inefficiency()
                    meaning = self.inefficiency_interpretation(e_ineff)
                    print(
                        f"  λ update @ iter {k + 1}: λ={self.lambda_gain:.3f}, "
                        f"E_ineff={e_ineff:.3f} ({self.stats.recent_ineff}/{self.stats.recent_total}), {meaning}"
                    )
                self.log_progress(k, start_time)
                self._update_live_plot(live_fig, live_ax, start, goal, q_near=q_near, q_rand=q_rand, direction=v_hat, step=step)
                continue

            q_new = self.nodes[new_id]
            self.get_interval(new_id)
            self._debug_step(k, f"accepted q_new={q_new} (new_id={new_id})")
            neighbors = [i for i in self.near_ids(q_new) if i != new_id]
            self._debug_step(k, f"neighbor_ids={neighbors}")

            # parent selection
            best_parent, best_cost = self.select_parent(q_new, neighbors, nearest)
            self._attach_child(new_id, best_parent)
            self.cost.append(best_cost)
            self._debug_step(k, f"parent={best_parent}, cost={best_cost:.3f}")

            # rewiring + subtree cost propagation
            self.rewire(new_id, neighbors)
            self._debug_step(k, "rewire check complete")

            # goal check
            if euclidean(q_new, goal) < self.goal_tolerance:
                if self.collision_free(q_new, goal):
                    goal_id = len(self.nodes)
                    self.nodes.append(goal)
                    self.parent[goal_id] = new_id
                    self.children[goal_id] = set()
                    self.children[new_id].add(goal_id)
                    self.cost.append(self.cost[new_id] + self.edge_cost(q_new, goal))
                else:
                    goal_id = new_id
                self.stats.reached_goal = True
                self.stats.iterations = k + 1
                self.stats.runtime_sec = time.time() - start_time
                print(f"✓ Goal reached at iteration {k + 1} ({self.stats.runtime_sec:.2f}s)")
                self._update_live_plot(live_fig, live_ax, start, goal, q_near=q_near, q_rand=q_rand, direction=v_hat, step=step)
                break

            # periodic lambda update from recent expansion inefficiency
            if (k + 1) % self.lambda_update_every == 0 and len(self.recent_attempt_flags) > 0:
                e_ineff = self.update_lambda_from_inefficiency()
                meaning = self.inefficiency_interpretation(e_ineff)
                print(
                    f"  λ update @ iter {k + 1}: λ={self.lambda_gain:.3f}, "
                    f"E_ineff={e_ineff:.3f} ({self.stats.recent_ineff}/{self.stats.recent_total}), {meaning}"
                )

            self.log_progress(k, start_time)
            self._update_live_plot(live_fig, live_ax, start, goal, q_near=q_near, q_rand=q_rand, direction=v_hat, step=step)

        # fallback if not reached
        if goal_id is None:
            closest = min(range(len(self.nodes)), key=lambda i: euclidean(self.nodes[i], goal))
            closest_dist = euclidean(self.nodes[closest], goal)
            can_connect_goal = self.collision_free(self.nodes[closest], goal)
            self.stats.iterations = self.max_iter
            self.stats.runtime_sec = time.time() - start_time
            if closest_dist < self.goal_tolerance:
                if can_connect_goal:
                    goal_id = len(self.nodes)
                    self.nodes.append(goal)
                    self.parent[goal_id] = closest
                    self.children[goal_id] = set()
                    self.children[closest].add(goal_id)
                    self.cost.append(self.cost[closest] + self.edge_cost(self.nodes[closest], goal))
                else:
                    goal_id = closest
                self.stats.reached_goal = True
                print(f"✓ Goal reached within tolerance ({self.stats.runtime_sec:.2f}s)")
            else:
                goal_id = closest
                print("⚠ Goal not reached")
                print(f"  Closest node distance to goal: {closest_dist:.1f} m")

        self.final_path_node_id = goal_id
        self.path = self.extract_path(goal_id)

        if live_fig is not None:
            plt.ioff()
        return self.stats.reached_goal


def visualize_solution(
    planner: VFRRTStar,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    output_dir: str,
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
) -> None:
    """Keep visualization style close to existing script."""
    fig, ax = plt.subplots(figsize=(16, 8))
    from matplotlib.colors import ListedColormap

    origin = planner.origin
    grid = planner.grid
    res = planner.resolution
    extent = [
        origin[0],
        origin[0] + grid.shape[1] * res,
        origin[1],
        origin[1] + grid.shape[0] * res,
    ]

    xs = origin[0] + (np.arange(grid.shape[1]) + 0.5) * res
    ys = origin[1] + (np.arange(grid.shape[0]) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    q = np.column_stack([xx.ravel(), yy.ravel()])
    u = u_interp(q).reshape(grid.shape)
    v = v_interp(q).reshape(grid.shape)
    mag = np.hypot(u, v)
    mag = np.where(np.isfinite(mag), mag, np.nan)

    free_mask = grid <= 50
    mag_masked = np.where(free_mask, mag, np.nan)

    im = ax.imshow(mag_masked, cmap="viridis", origin="lower", extent=extent)
    plt.colorbar(im, ax=ax, label="Velocity magnitude")

    occ_cmap = ListedColormap(["black"])
    occ_mask = np.where(~free_mask, 1.0, np.nan)
    ax.imshow(occ_mask, cmap=occ_cmap, origin="lower", extent=extent, alpha=1.0)

    # tree
    for i, p in planner.parent.items():
        if p is None:
            continue
        a = planner.nodes[p]
        b = planner.nodes[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", alpha=0.75, linewidth=1.3)

    # path
    if len(planner.path) > 1:
        pa = np.array(planner.path)
        ax.plot(pa[:, 0], pa[:, 1], "r-", linewidth=3, label="Final path")
        ax.scatter(pa[:, 0], pa[:, 1], c="red", s=30)

    ax.scatter(*start, c="green", s=300, marker="o", edgecolors="black", linewidth=2, label="Start")
    ax.scatter(*goal, c="gold", s=300, marker="s", edgecolors="black", linewidth=2, label="Goal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    #ax.set_title("SVF-RRT* Path Planning (Refactored)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/svf_rrt_star_path_refactored.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Visualization saved: {output_dir}/svf_rrt_star_path_refactored.png")
    plt.show()


def main() -> None:
    # same loading pattern
    print("Loading occupancy grid from NPZ...")
    grid, origin, resolution = load_occupancy_from_npz(DATA_DIR)
    print(f"✓ Loaded occupancy grid: {grid.shape}")
    print(f"  Origin: ({origin[0]:.1f}, {origin[1]:.1f})")
    print(f"  Resolution: {resolution} m/cell")

    print("Loading velocity lookup...")
    (u_interp, v_interp), bounds = load_velocity_interpolators(LOOKUP_FOLDER)
    print(f"✓ Velocity lookup bounds: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}]")

    start = (-400, 200)
    goal = (-100, 280)

    planner = VFRRTStar(
        occupancy_grid=grid,
        origin=origin,
        resolution=resolution,
        interpolators=(u_interp, v_interp),
        # core planner
        max_iter=200,
        step_size=10,
        goal_sample_rate=0.05,
        sample_checks=30,
        rewire_base_radius=200.0,
        goal_tolerance=5.0,
        log_every=100,
        # adaptive lambda tuning
        lambda_gain=1.0,
        ineff_window_size=50,
        lambda_update_every=50,
        ineff_setpoint=0.5,
        lambda_min=0.1,
        lambda_max=5.0,
        es=0.6,
        # optional non-unit field scaling
        use_field_magnitude_scaling=False,
        field_scale_m=1.0,
        # step-by-step debug
        debug_step_by_step=False,
        debug_pause_sec=0.0,
        debug_live_plot=True,
    )

    success = planner.plan(start, goal)

    if len(planner.path) > 1:
        path_length = sum(euclidean(planner.path[i], planner.path[i + 1]) for i in range(len(planner.path) - 1))
        final_upstream_cost = planner.cost[planner.final_path_node_id] if planner.final_path_node_id is not None else float("nan")
        print("\n" + "=" * 60)
        print("SVF-RRT* Results")
        print("=" * 60)
        print(f"✓ Path found with {len(planner.path)} waypoints")
        print(f"  Path length: {path_length:.1f} m")
        print(f"  Final upstream cost: {final_upstream_cost:.3f}")
        print(f"  Tree nodes explored: {len(planner.nodes)}")
        print(f"  Ineffective extensions: {planner.stats.ineffective_extensions}")
        print(f"  Goal reached: {success}")
        print("=" * 60)
    else:
        print("✗ No path found")

    visualize_solution(planner, start, goal, OUTPUT_DIR, u_interp, v_interp)

    # Export waypoints to CSV
    if len(planner.path) > 0:
        waypoints_path = f"{OUTPUT_DIR}/svf_waypoints.csv"
        with open(waypoints_path, 'w') as f:
            f.write("x,y\n")
            for x, y in planner.path:
                f.write(f"{x:.6f},{y:.6f}\n")
        print(f"✓ Waypoints exported: {waypoints_path}")

    # requested short summary
    print("\nSummary:")
    print("- Refactored into modular methods: sampling, flow query, VF direction, collision, edge cost,")
    print("  parent selection, rewiring, subtree propagation, path extraction, diagnostics.")
    print("- Preserved occupancy-grid NPZ and RegularGridInterpolator loading pattern.")
    print("- Still approximate vs paper: VF direction blending and neighborhood radius/gain are heuristic;")
    print("  no exact closed-form control law or formal informed-sampling proof is enforced here.")


if __name__ == "__main__":
    main()
 
