"""
RRT* Path Planning using occupancy grid loaded from NPZ
Clean implementation with easy loading pattern
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import RegularGridInterpolator
import time
import math
import os

class RRTStar:
    """RRT* path planning algorithm"""
    
    def __init__(self, occupancy_grid, origin, resolution, max_iter=5000, 
                 step_size=15, goal_sample_rate=0.15):
        """
        occupancy_grid: 2D numpy array (0=free, 100=occupied)
        origin: (x_min, y_min) in world coordinates
        resolution: grid cell size
        """
        self.grid = occupancy_grid
        self.origin = origin
        self.resolution = resolution
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.grid_height, self.grid_width = occupancy_grid.shape
        
        # World bounds
        self.world_x_min = origin[0]
        self.world_y_min = origin[1]
        self.world_x_max = origin[0] + self.grid_width * resolution
        self.world_y_max = origin[1] + self.grid_height * resolution
        
        self.path = []
        self.nodes = []
        self.edges = {}
        self.costs = []
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def is_collision_free(self, x1, y1, x2, y2, num_checks=10):
        """Check if line segment is collision-free using linear interpolation"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            gx, gy = self.world_to_grid(x, y)
            
            # Check bounds
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return False
            
            # Check occupancy: free space is 0, occupied is 100
            if self.grid[gy, gx] > 50:  # threshold at 50
                return False
        
        return True
    
    def plan(self, start, goal):
        """Plan path using RRT*"""
        # Initialize tree with start node
        self.nodes = [start]
        self.edges = {0: None}
        self.costs = [0.0]
        
        print(f"\n{'='*60}")
        print(f"RRT* Path Planning")
        print(f"{'='*60}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"World bounds: x=[{self.world_x_min:.1f}, {self.world_x_max:.1f}], "
              f"y=[{self.world_y_min:.1f}, {self.world_y_max:.1f}]")
        print(f"Iterations: {self.max_iter}, Step size: {self.step_size}")
        print(f"{'='*60}")
        
        goal_reached = False
        goal_idx = None
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # Sample random point or goal
            if np.random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                random_point = (
                float(np.random.uniform(-600, 500)),
                float(np.random.uniform(150, 300))
                )
            
            # Find nearest node
            nearest_idx = min(range(len(self.nodes)),
                            key=lambda i: euclidean(self.nodes[i], random_point))
            nearest_node = self.nodes[nearest_idx]
            
            # Steer towards random point
            direction = np.array(random_point) - np.array(nearest_node)
            dist = euclidean(nearest_node, random_point)
            
            if dist < 1e-6:
                new_point = random_point
            else:
                new_point = tuple(np.array(nearest_node) + 
                                 (direction / dist) * min(self.step_size, dist))
            
            # Check collision
            if not self.is_collision_free(nearest_node[0], nearest_node[1], 
                                         new_point[0], new_point[1]):
                continue
            
            # RRT* parent selection among neighbors
            neighbor_radius = self._neighbor_radius(len(self.nodes) + 1)
            neighbors = self._near_indices(new_point, neighbor_radius)

            best_parent = nearest_idx
            best_cost = self.costs[nearest_idx] + euclidean(nearest_node, new_point)

            for idx in neighbors:
                if self.is_collision_free(self.nodes[idx][0], self.nodes[idx][1],
                                          new_point[0], new_point[1]):
                    candidate_cost = self.costs[idx] + euclidean(self.nodes[idx], new_point)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_parent = idx

            # Add new node
            self.nodes.append(new_point)
            new_idx = len(self.nodes) - 1
            self.edges[new_idx] = best_parent
            self.costs.append(best_cost)

            # Rewire neighbors
            for idx in neighbors:
                if idx == best_parent:
                    continue
                new_cost = self.costs[new_idx] + euclidean(self.nodes[idx], new_point)
                if new_cost < self.costs[idx]:
                    if self.is_collision_free(self.nodes[idx][0], self.nodes[idx][1],
                                              new_point[0], new_point[1]):
                        self.edges[idx] = new_idx
                        self.costs[idx] = new_cost
            
            # Check if goal reached
            if euclidean(new_point, goal) < 30:
                self.nodes.append(goal)
                goal_idx = len(self.nodes) - 1
                self.edges[goal_idx] = new_idx
                goal_reached = True
                elapsed = time.time() - start_time
                print(f"✓ Goal reached at iteration {iteration + 1} ({elapsed:.2f}s)")
                break
            
            if (iteration + 1) % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  Iteration {iteration + 1}/{self.max_iter}, "
                      f"nodes: {len(self.nodes)}, time: {elapsed:.1f}s")
        
        if goal_reached and goal_idx is not None:
            self.path = self._extract_path(goal_idx)
            return True
        else:
            # Return path to closest node to goal
            closest_idx = min(range(len(self.nodes)),
                            key=lambda i: euclidean(self.nodes[i], goal))
            self.path = self._extract_path(closest_idx)
            dist_to_goal = euclidean(self.nodes[closest_idx], goal)
            print(f"⚠ Goal not reached exactly")
            print(f"  Closest node distance to goal: {dist_to_goal:.1f} m")
            return False
    
    def _extract_path(self, end_idx):
        """Extract path from root to end_idx"""
        path = []
        current_idx = end_idx
        
        while current_idx is not None:
            path.append(self.nodes[current_idx])
            if current_idx not in self.edges or self.edges[current_idx] is None:
                break
            current_idx = self.edges[current_idx]
        
        path.reverse()
        return path

    def _neighbor_radius(self, n):
        """Adaptive neighbor radius for RRT*"""
        return max(30.0, 50.0 * math.sqrt(math.log(n + 1) / (n + 1)))

    def _near_indices(self, point, radius):
        """Find nodes within a given radius"""
        return [i for i, node in enumerate(self.nodes) if euclidean(node, point) <= radius]

# ============================================================
# MAIN: Load occupancy grid and run RRT*
# ============================================================

output_dir = './lookup_tables/sydney_regatta/output'
LOOKUP_FOLDER = './lookup_tables/sydney_regatta'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


def load_velocity_interpolators(folder: str):
    """Load velocity lookup and return interpolators and bounds."""
    grid_x = np.load(f"{folder}/grid_x.npy")
    grid_y = np.load(f"{folder}/grid_y.npy")
    u_lookup = np.load(f"{folder}/average_velocity_u_lookup.npy")
    v_lookup = np.load(f"{folder}/average_velocity_v_lookup.npy")

    x_coords = grid_x[:, 0]
    y_coords = grid_y[0, :]

    u_interp = RegularGridInterpolator(
        (x_coords, y_coords), u_lookup, method="linear", bounds_error=False, fill_value=np.nan
    )
    v_interp = RegularGridInterpolator(
        (x_coords, y_coords), v_lookup, method="linear", bounds_error=False, fill_value=np.nan
    )
    bounds = (float(x_coords.min()), float(x_coords.max()), float(y_coords.min()), float(y_coords.max()))
    return (u_interp, v_interp), bounds


def upstream_edge_cost(p1, p2, u_interp, v_interp, sample_checks=100):
    """Compute upstream cost for an edge: \n+    integral (||f|| - <f, q'>) ds along segment p1->p2.\n+    This is for diagnostics only and does not affect planning."""
    if euclidean(p1, p2) < 1e-12:
        return 0.0

    total = 0.0
    prev = p1
    for i in range(1, sample_checks + 1):
        t = i / sample_checks
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        cur = (x, y)

        ds_vec = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=float)
        ds = float(np.linalg.norm(ds_vec))
        if ds < 1e-12:
            prev = cur
            continue

        q_prime = ds_vec / ds
        u = u_interp([[cur[0], cur[1]]])[0]
        v = v_interp([[cur[0], cur[1]]])[0]
        if not np.isfinite(u):
            u = 0.0
        if not np.isfinite(v):
            v = 0.0
        f = np.array([float(u), float(v)], dtype=float)
        integrand = float(np.linalg.norm(f) - np.dot(f, q_prime))
        total += integrand * ds
        prev = cur

    return float(total)


def save_trajectory_csv(path_points, output_path):
    """Save a path as a two-column CSV with x,y waypoints."""
    with open(output_path, "w") as f:
        f.write("x,y\n")
        for x, y in path_points:
            f.write(f"{x:.6f},{y:.6f}\n")

# Load occupancy grid from NPZ
print("Loading occupancy grid from NPZ...")
data = np.load(f'{output_dir}/occupancy_grid.npz')
grid = data['grid']
origin_x = float(data['origin_x'])
origin_y = float(data['origin_y'])
resolution = float(data['resolution'])
origin = (origin_x, origin_y)

print(f"✓ Loaded occupancy grid: {grid.shape}")
print(f"  Origin: ({origin_x:.1f}, {origin_y:.1f})")
print(f"  Resolution: {resolution} m/cell")

# Load velocity lookup (optional, used for upstream cost diagnostics and visualization)
have_lookup = False
try:
    (u_interp, v_interp), bounds = load_velocity_interpolators(LOOKUP_FOLDER)
    print(f"✓ Velocity lookup bounds: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}]")
    have_lookup = True
except Exception:
    print("⚠ Velocity lookup not available — skipping flow diagnostics/visualization")

# Do NOT flip - keep grid as saved
# Flip grid for proper coordinate alignment
# grid = np.flipud(grid)

# Define start and goal in free space
start = (-400, 200)
goal = (300, 200)


# Run RRT*
rrt = RRTStar(grid, origin, resolution, max_iter=5000, step_size=20, goal_sample_rate=0.005)
success = rrt.plan(start, goal)

# Print results
if len(rrt.path) > 1:
    path_length = sum(euclidean(rrt.path[i], rrt.path[i+1]) for i in range(len(rrt.path)-1))
    print(f"\n{'='*60}")
    print(f"Path Planning Results")
    print(f"{'='*60}")
    print(f"✓ Path found with {len(rrt.path)} waypoints")
    print(f"  Path length: {path_length:.1f} m")
    print(f"  Tree nodes explored: {len(rrt.nodes)}")
    print(f"  Goal reached: {success}")
    print(f"{'='*60}")
else:
    print("✗ No path found")

if len(rrt.path) > 0:
    trajectory_path = f"{output_dir}/rrt_waypoints.csv"
    save_trajectory_csv(rrt.path, trajectory_path)
    print(f"✓ Trajectory saved: {trajectory_path}")

# Visualization
fig, ax = plt.subplots(figsize=(16, 8))


# Overlay velocity magnitude (viridis) masked to free space, then occupancy mask
from matplotlib.colors import ListedColormap, Normalize
colors_map = ['white', 'black']
cmap = ListedColormap(colors_map)
norm = Normalize(vmin=0, vmax=100)

# try to load velocity lookup and plot magnitude background similar to SVF-RRT*
try:
    (u_interp, v_interp), bounds = load_velocity_interpolators(LOOKUP_FOLDER)
    xs = origin[0] + (np.arange(grid.shape[1]) + 0.5) * resolution
    ys = origin[1] + (np.arange(grid.shape[0]) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    q = np.column_stack([xx.ravel(), yy.ravel()])
    u = u_interp(q).reshape(grid.shape)
    v = v_interp(q).reshape(grid.shape)
    mag = np.hypot(u, v)
    mag = np.where(np.isfinite(mag), mag, np.nan)

    free_mask = grid <= 50
    mag_masked = np.where(free_mask, mag, np.nan)

    im = ax.imshow(mag_masked, cmap='viridis', origin='lower', extent=[origin[0], origin[0] + grid.shape[1] * resolution, origin[1], origin[1] + grid.shape[0] * resolution])
    plt.colorbar(im, ax=ax, label='Velocity magnitude')

    # (no quiver/arrows: heatmap only — arrows intentionally omitted)
except Exception:
    # fallback: plain occupancy grid if lookup not available
    ax.imshow(grid, cmap=cmap, norm=norm, origin='lower', extent=[origin[0], origin[0] + grid.shape[1] * resolution, origin[1], origin[1] + grid.shape[0] * resolution])

# occupancy mask overlay (dark)
occ_cmap = ListedColormap(["black"])
occ_mask = np.where(~(grid <= 50), 1.0, np.nan)
ax.imshow(occ_mask, cmap=occ_cmap, origin='lower', extent=[origin[0], origin[0] + grid.shape[1] * resolution, origin[1], origin[1] + grid.shape[0] * resolution], alpha=1.0)

# Draw tree (all branches) — use same style as SVF-RRT* (black lines)
if len(rrt.nodes) > 1:
    for i, parent_idx in rrt.edges.items():
        if parent_idx is not None:
            node = rrt.nodes[i]
            parent = rrt.nodes[parent_idx]
            ax.plot([parent[0], node[0]], [parent[1], node[1]], color='black', alpha=0.7, linewidth=1.0)

# (node markers removed — tree shown as lines only)

# Draw final path
if len(rrt.path) > 1:
    path_array = np.array(rrt.path)
    ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=3, label='Final path', zorder=5)
    ax.scatter(path_array[:, 0], path_array[:, 1], c='red', s=50, zorder=5)

    # If velocity lookup is available, compute upstream cost per path segment (diagnostic only)
    if 'have_lookup' in globals() and have_lookup:
        total_upstream = 0.0
        costs = []
        for i in range(len(rrt.path) - 1):
            p1 = rrt.path[i]
            p2 = rrt.path[i + 1]
            c = upstream_edge_cost(p1, p2, u_interp, v_interp, sample_checks=100)
            costs.append((p1[0], p1[1], p2[0], p2[1], c))
            total_upstream += c
        print(f"Total upstream path cost: {total_upstream:.3f}")
        # save CSV
        try:
            with open(f"{output_dir}/upstream_edge_costs.csv", 'w') as f:
                f.write("x1,y1,x2,y2,cost\n")
                for r in costs:
                    f.write(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n")
            print(f"✓ Upstream edge costs saved: {output_dir}/upstream_edge_costs.csv")
        except Exception:
            pass

# Draw start and goal
ax.scatter(*start, c='green', s=300, marker='o', edgecolors='black', linewidth=2, 
          label='Start', zorder=10)
ax.scatter(*goal, c='gold', s=300, marker='s', edgecolors='black', linewidth=2, 
          label='Goal', zorder=10)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('RRT* Path Planning - Loaded from Occupancy Grid NPZ', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.2)
ax.set_xlim([origin[0], origin[0] + grid.shape[1] * resolution])
ax.set_ylim([origin[1], origin[1] + grid.shape[0] * resolution])

plt.tight_layout()
plt.savefig(f'{output_dir}/rrt_star_path_from_npz.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_dir}/rrt_star_path_from_npz.png")

plt.show()
