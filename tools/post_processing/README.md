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

The script looks for `tools/post_processing/data/velocity_vectors.pkl` first. If it is missing, it falls back to:

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
