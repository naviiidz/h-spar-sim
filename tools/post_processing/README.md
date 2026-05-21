# Post-processing Tools

This folder contains scripts for plotting planner outputs and computing trajectory metrics after ROS/planner runs.

## Expected Data Layout

Place planner logs under:

```text
tools/post_processing/data/
  rrt/
    robot_trajectory_*.csv
    goal_sequence_*.csv or *_waypoints.csv
    execution_log_*.csv
  vf-rrt/
    robot_trajectory_*.csv
    goal_sequence_*.csv
    execution_log_*.csv
  svf-rrt/
    robot_trajectory_*.csv
    goal_sequence_*.csv
    execution_log_*.csv
  velocity_vectors.pkl
```

## `plot_velocity_vectors.py`

Builds the background velocity-vector figure used by the trajectory plots. The script reads VTU velocity field files, extracts point coordinates and 2D velocity vectors, averages all valid VTU files, optionally removes occupied cells using an occupancy grid, and saves both a PNG preview and a pickled Matplotlib figure.

Default input:

```text
velocity_fields/sydney_regatta/raw/Velocity2d/*.vtu
```

Default outputs:

```text
tools/post_processing/outputs/velocity_vectors.png
tools/post_processing/data/velocity_vectors.pkl
```

Usage:

```bash
python3 tools/post_processing/plot_velocity_vectors.py
```

Useful options:

- `--arrow-fraction`: fraction of averaged vectors to draw, for example `0.05` draws more arrows than the default `0.03`.
- `--xmin`, `--xmax`, `--ymin`, `--ymax`: crop the plotted region.
- `--output-png`: choose a PNG output path.
- `--output-pkl`: choose a pickle output path.
- `--occupancy`: overlay and filter vectors using a `.npz` or `.npy` occupancy grid.

Example with a cropped region:

```bash
python3 tools/post_processing/plot_velocity_vectors.py \
  --xmin -420 --xmax -80 --ymin 130 --ymax 300 \
  --arrow-fraction 0.05
```

## `plot_trajectories.py`

Overlays planner waypoints and executed robot trajectories on the saved velocity-vector figure.

Outputs:

```text
tools/post_processing/outputs/waypoints_over_velocity_vectors.svg
tools/post_processing/outputs/trajectories_over_velocity_vectors.svg
```

Usage:

```bash
python3 tools/post_processing/plot_trajectories.py
```

The script looks for `tools/post_processing/data/velocity_vectors.pkl` first. Generate that file with `plot_velocity_vectors.py`. If it is missing, `plot_trajectories.py` falls back to the older location:

```text
tools/vtu_converter/output/velocity_vectors.pkl
```

## `trajectory_metrics.py`

Computes metrics for one planner run from a robot trajectory CSV and a goal sequence CSV.

Metrics:

- `Trajectory length (m)`: total length of the executed trajectory.
- `Execution time (s)`: time span of the trajectory log.
- `Mean tracking error (m)`: mean distance from each trajectory point to its current active goal.
- `Maximum tracking error (m)`: maximum distance to the current active goal.
- `Path deviation (m)`: mean distance from trajectory points to the planned goal-sequence polyline.
- `Executed upstream cost`: integral of `||f|| - dot(f, q')` along the executed trajectory.

Usage with a planner directory:

```bash
python3 tools/post_processing/trajectory_metrics.py tools/post_processing/data/svf-rrt
```

Usage with explicit files:

```bash
python3 tools/post_processing/trajectory_metrics.py \
  python3 trajectory_metrics.py  data/svf-rrt
```

The upstream cost uses velocity lookup files from:

```text
rrt-star-planners/lookup_tables/sydney_regatta/
```

Override that location if needed:

```bash
python3 tools/post_processing/trajectory_metrics.py tools/post_processing/data/svf-rrt \
  --lookup-folder rrt-star-planners/lookup_tables/sydney_regatta
```
