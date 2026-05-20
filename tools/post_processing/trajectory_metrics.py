"""
Compute trajectory tracking metrics against a goal sequence.

Metrics:
- Mean tracking error: mean distance from each trajectory point to its current goal.
- Maximum tracking error: maximum distance from each trajectory point to its current goal.
- Path deviation: mean distance from each trajectory point to the goal-sequence polyline.
- Executed upstream cost: integral of (||f|| - <f, q'>) ds along the executed trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LOOKUP_FOLDER = REPO_ROOT / "rrt-star-planners" / "lookup_tables" / "sydney_regatta"


def load_velocity_interpolators(folder: Path) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
    grid_x = np.load(folder / "grid_x.npy")
    grid_y = np.load(folder / "grid_y.npy")
    u_lookup = np.load(folder / "average_velocity_u_lookup.npy")
    v_lookup = np.load(folder / "average_velocity_v_lookup.npy")

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
    return u_interp, v_interp


def xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {"x_ned", "y_ned"}.issubset(df.columns):
        return "x_ned", "y_ned"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise ValueError("CSV must contain x_ned/y_ned or x/y columns.")


def load_xy(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    x_col, y_col = xy_columns(df)
    return df[[x_col, y_col]].to_numpy(dtype=float)


def latest_matching_csv(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(folder.glob(pattern))
    return matches[-1] if matches else None


def resolve_input_paths(input_path: Path) -> Tuple[Path, Path, Optional[Path]]:
    if input_path.is_dir():
        trajectory_csv = latest_matching_csv(input_path, "robot_trajectory_*.csv")
        goal_sequence_csv = latest_matching_csv(input_path, "goal_sequence_*.csv")
        execution_log_csv = latest_matching_csv(input_path, "execution_log_*.csv")
        if trajectory_csv is None:
            raise FileNotFoundError(f"No robot_trajectory_*.csv found in {input_path}")
        if goal_sequence_csv is None:
            raise FileNotFoundError(f"No goal_sequence_*.csv found in {input_path}")
        return trajectory_csv, goal_sequence_csv, execution_log_csv

    trajectory_csv = input_path
    goal_sequence_csv = latest_matching_csv(input_path.parent, "goal_sequence_*.csv")
    execution_log_csv = latest_matching_csv(input_path.parent, "execution_log_*.csv")
    if goal_sequence_csv is None:
        raise FileNotFoundError(f"No goal_sequence_*.csv found in {input_path.parent}")
    return trajectory_csv, goal_sequence_csv, execution_log_csv


def tracking_errors(trajectory_df: pd.DataFrame, goals_df: pd.DataFrame) -> np.ndarray:
    if "current_goal_number" not in trajectory_df.columns:
        raise ValueError("Trajectory CSV must contain current_goal_number for tracking error.")
    if "goal_number" not in goals_df.columns:
        raise ValueError("Goal sequence CSV must contain goal_number for tracking error.")

    traj_x, traj_y = xy_columns(trajectory_df)
    goal_x, goal_y = xy_columns(goals_df)

    goal_lookup = {
        int(row.goal_number): np.array([float(getattr(row, goal_x)), float(getattr(row, goal_y))])
        for row in goals_df.itertuples(index=False)
    }

    errors = []
    for row in trajectory_df.itertuples(index=False):
        goal_number = int(getattr(row, "current_goal_number"))
        goal = goal_lookup.get(goal_number)
        if goal is None:
            continue
        point = np.array([float(getattr(row, traj_x)), float(getattr(row, traj_y))])
        errors.append(float(np.linalg.norm(point - goal)))

    if not errors:
        raise ValueError("No trajectory rows matched goal_sequence goal_number values.")
    return np.asarray(errors, dtype=float)


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def path_deviation(trajectory_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    if len(goal_xy) == 0:
        raise ValueError("Goal sequence is empty.")
    if len(goal_xy) == 1:
        distances = np.linalg.norm(trajectory_xy - goal_xy[0], axis=1)
        return float(np.mean(distances))

    deviations = []
    for point in trajectory_xy:
        distances = [
            point_to_segment_distance(point, goal_xy[i], goal_xy[i + 1])
            for i in range(len(goal_xy) - 1)
        ]
        deviations.append(min(distances))
    return float(np.mean(deviations))


def trajectory_length(trajectory_xy: np.ndarray) -> float:
    if len(trajectory_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1)))


def execution_time_seconds(trajectory_df: pd.DataFrame) -> float:
    if "mission_time_sec" in trajectory_df.columns:
        times = trajectory_df["mission_time_sec"].to_numpy(dtype=float)
        finite_times = times[np.isfinite(times)]
        if len(finite_times) >= 2:
            return float(np.max(finite_times) - np.min(finite_times))

    if "timestamp" in trajectory_df.columns:
        timestamps = pd.to_datetime(trajectory_df["timestamp"], errors="coerce")
        timestamps = timestamps.dropna()
        if len(timestamps) >= 2:
            return float((timestamps.max() - timestamps.min()).total_seconds())

    return 0.0


def execution_cost(execution_log_csv: Path) -> float:
    df = pd.read_csv(execution_log_csv)
    if "duration_sec" not in df.columns:
        raise ValueError(f"Execution log must contain duration_sec: {execution_log_csv}")
    return float(df["duration_sec"].sum())


def upstream_edge_cost(
    p1: np.ndarray,
    p2: np.ndarray,
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    sample_checks: int,
) -> float:
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-12:
        return 0.0

    total = 0.0
    prev = p1
    for i in range(1, sample_checks + 1):
        t = i / sample_checks
        cur = p1 + t * (p2 - p1)
        ds_vec = cur - prev
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

        flow = np.array([float(u), float(v)], dtype=float)
        integrand = float(np.linalg.norm(flow) - np.dot(flow, q_prime))
        total += integrand * ds
        prev = cur

    return float(total)


def executed_upstream_cost(
    trajectory_xy: np.ndarray,
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    sample_checks: int,
) -> float:
    if len(trajectory_xy) < 2:
        return 0.0

    p1 = trajectory_xy[:-1]
    p2 = trajectory_xy[1:]
    segment = p2 - p1
    lengths = np.linalg.norm(segment, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        return 0.0

    p1 = p1[valid]
    segment = segment[valid]
    lengths = lengths[valid]
    q_prime = segment / lengths[:, None]
    ds = lengths / float(sample_checks)

    t = (np.arange(1, sample_checks + 1, dtype=float) / float(sample_checks))[None, :, None]
    sample_points = p1[:, None, :] + t * segment[:, None, :]
    flat_points = sample_points.reshape(-1, 2)

    u = u_interp(flat_points).reshape(len(p1), sample_checks)
    v = v_interp(flat_points).reshape(len(p1), sample_checks)
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)

    flow = np.stack((u, v), axis=2)
    flow_norm = np.linalg.norm(flow, axis=2)
    flow_along_path = np.einsum("ijk,ik->ij", flow, q_prime)
    integrand = flow_norm - flow_along_path
    return float(np.sum(integrand * ds[:, None]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute trajectory tracking and upstream-cost metrics.")
    parser.add_argument(
        "input",
        type=Path,
        help="Directory containing robot_trajectory_*.csv and goal_sequence_*.csv, or a robot trajectory CSV.",
    )
    parser.add_argument(
        "goal_sequence_csv",
        type=Path,
        nargs="?",
        help="Optional goal sequence CSV. If omitted, goal_sequence_*.csv is discovered next to the trajectory.",
    )
    parser.add_argument(
        "--lookup-folder",
        type=Path,
        default=LOOKUP_FOLDER,
        help="Folder containing grid_x.npy, grid_y.npy, and average velocity lookup arrays.",
    )
    parser.add_argument("--sample-checks", type=int, default=100, help="Samples per trajectory segment for upstream cost.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.goal_sequence_csv is None:
        trajectory_csv, goal_sequence_csv, _ = resolve_input_paths(args.input)
    else:
        trajectory_csv = args.input
        goal_sequence_csv = args.goal_sequence_csv

    trajectory_df = pd.read_csv(trajectory_csv)
    goals_df = pd.read_csv(goal_sequence_csv)

    traj_x, traj_y = xy_columns(trajectory_df)
    goal_x, goal_y = xy_columns(goals_df)
    trajectory_xy = trajectory_df[[traj_x, traj_y]].to_numpy(dtype=float)
    goal_xy = goals_df[[goal_x, goal_y]].to_numpy(dtype=float)

    errors = tracking_errors(trajectory_df, goals_df)
    length = trajectory_length(trajectory_xy)
    execution_time = execution_time_seconds(trajectory_df)
    deviation = path_deviation(trajectory_xy, goal_xy)
    u_interp, v_interp = load_velocity_interpolators(args.lookup_folder)
    upstream_cost = executed_upstream_cost(trajectory_xy, u_interp, v_interp, args.sample_checks)
    print(f"Trajectory CSV: {trajectory_csv}")
    print(f"Goal sequence CSV: {goal_sequence_csv}")
    print(f"Trajectory length (m): {length:.3f}")
    print(f"Execution time (s): {execution_time:.3f}")
    print(f"Mean tracking error (m): {np.mean(errors):.3f}")
    print(f"Maximum tracking error (m): {np.max(errors):.3f}")
    print(f"Path deviation (m): {deviation:.3f}")
    print(f"Executed upstream cost: {upstream_cost:.3f}")




if __name__ == "__main__":
    main()
