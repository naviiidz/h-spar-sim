from __future__ import annotations

import itertools
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.figure as mfig
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import cbook
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_VELOCITY_PICKLE = DATA_DIR / "velocity_vectors.pkl"
FALLBACK_VELOCITY_PICKLE = SCRIPT_DIR.parent / "vtu_converter" / "output" / "velocity_vectors.pkl"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_TRAJECTORIES_PATH = OUTPUT_DIR / "trajectories_over_velocity_vectors.svg"
OUTPUT_WAYPOINTS_PATH = OUTPUT_DIR / "waypoints_over_velocity_vectors.svg"
ROI_X_LIMITS = (-420.0, -80.0)
ROI_Y_LIMITS = (130.0, 300.0)


@dataclass(frozen=True)
class PlannerDataset:
	name: str
	label: str
	color: str
	folder: Path
	waypoint_file: Optional[Path]
	trajectory_file: Optional[Path]
	execution_file: Optional[Path]


PLANNERS: Sequence[PlannerDataset] = (
	PlannerDataset(
		name="rrt",
		label="RRT*",
		color="#d1495b",
		folder=DATA_DIR / "rrt",
		waypoint_file=None,
		trajectory_file=None,
		execution_file=None,
	),
	PlannerDataset(
		name="vf-rrt",
		label="VF-RRT*",
		color="#2a9d8f",
		folder=DATA_DIR / "vf-rrt",
		waypoint_file=None,
		trajectory_file=None,
		execution_file=None,
	),
	PlannerDataset(
		name="svf-rrt",
		label="SVF-RRT*",
		color="#457b9d",
		folder=DATA_DIR / "svf-rrt",
		waypoint_file=None,
		trajectory_file=None,
		execution_file=None,
	),
)


def patch_matplotlib_pickle_compatibility() -> None:
	"""Install a tiny shim so the saved Matplotlib figure can be unpickled."""
	if "matplotlib.colorizer" not in sys.modules:
		module = types.ModuleType("matplotlib.colorizer")
		module.Colorizer = getattr(mcolors, "Normalize", object)
		module.ColorizingArtist = object
		sys.modules["matplotlib.colorizer"] = module

	if not hasattr(mgridspec, "SubplotParams"):
		mgridspec.SubplotParams = mfig.SubplotParams

	original_connect = cbook.CallbackRegistry.connect
	if getattr(original_connect, "__name__", "") != "patched_connect":
		def patched_connect(self, signal, func):
			if isinstance(getattr(self, "_cid_gen", None), int):
				self._cid_gen = itertools.count(self._cid_gen)
			return original_connect(self, signal, func)

		patched_connect.__name__ = "patched_connect"
		cbook.CallbackRegistry.connect = patched_connect


def load_stream_figure(pickle_path: Path):
	patch_matplotlib_pickle_compatibility()
	with pickle_path.open("rb") as handle:
		return pickle.load(handle)


def extract_stream_background(figure) -> tuple[np.ndarray, Optional[np.ndarray], tuple[float, float, float, float]]:
	if not figure.axes:
		raise ValueError("Stream-function pickle did not contain any axes.")

	main_axis = next((axis for axis in figure.axes if axis.images), figure.axes[0])
	if not main_axis.images:
		raise ValueError("Stream-function pickle does not contain an image layer to reuse.")

	psi_image = np.asarray(main_axis.images[0].get_array(), dtype=float)
	obstacle_image = None
	if len(main_axis.images) > 1:
		obstacle_image = np.ma.array(main_axis.images[1].get_array(), copy=True)

	x_min, x_max = main_axis.get_xlim()
	y_min, y_max = main_axis.get_ylim()
	extent = (float(min(x_min, x_max)), float(max(x_min, x_max)), float(min(y_min, y_max)), float(max(y_min, y_max)))
	return psi_image, obstacle_image, extent


def load_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing expected input file: {path}")
	return pd.read_csv(path)


def latest_matching_csv(folder: Path, prefix: str) -> Optional[Path]:
	matches = sorted(folder.glob(f"{prefix}*.csv"))
	return matches[-1] if matches else None


def build_planned_path(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	if {"x", "y"}.issubset(df.columns):
		return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)

	if {"x_ned", "y_ned"}.issubset(df.columns):
		return df["x_ned"].to_numpy(dtype=float), df["y_ned"].to_numpy(dtype=float)

	raise ValueError("Planned-path CSV does not contain x/y coordinates.")


def build_trajectory(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	if {"x_ned", "y_ned"}.issubset(df.columns):
		return df["x_ned"].to_numpy(dtype=float), df["y_ned"].to_numpy(dtype=float)

	if {"x", "y"}.issubset(df.columns):
		return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)

	raise ValueError("Trajectory CSV does not contain x/y coordinates.")


def draw_background(ax, psi: np.ndarray, obstacle: Optional[np.ndarray], extent: tuple[float, float, float, float]):
	masked_psi = np.ma.masked_invalid(psi)
	bg = ax.imshow(masked_psi, origin="lower", extent=extent, cmap="turbo", aspect="equal")

	if obstacle is not None:
		ax.imshow(obstacle, origin="lower", extent=extent, cmap="gray", alpha=0.9, aspect="equal")

	if int(np.count_nonzero(np.isfinite(masked_psi))) > 10:
		xs = np.linspace(extent[0], extent[1], psi.shape[1])
		ys = np.linspace(extent[2], extent[3], psi.shape[0])
		xx, yy = np.meshgrid(xs, ys)
		try:
			ax.contour(xx, yy, masked_psi, levels=24, colors="k", linewidths=0.35, alpha=0.5)
		except Exception:
			pass

	ax.set_xlim(*ROI_X_LIMITS)
	ax.set_ylim(*ROI_Y_LIMITS)
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.12)
	return bg


def plot_goal_path(ax, df: pd.DataFrame, color: str, label: str) -> None:
	x, y = build_planned_path(df)
	markevery = max(1, len(x) // 18)
	(line,) = ax.plot(
		x,
		y,
		linestyle="-",
		linewidth=4.0,
		color=color,
		alpha=1.0,
		label=label,
		zorder=10,
	)
	line.set_path_effects(
		[
			patheffects.Stroke(linewidth=6.8, foreground="white", alpha=0.95),
			patheffects.Normal(),
		]
	)

	# Draw waypoint circles separately so they don't inherit the white halo.
	ax.scatter(
		x[::markevery],
		y[::markevery],
		s=38,
		facecolors=color,
		edgecolors=color,
		linewidths=0.0,
		marker="o",
		zorder=11,
	)

	# Unique markers for start/end
	ax.scatter(x[0], y[0], color=color, marker="*", s=600, edgecolors=color, linewidths=0.0, zorder=12)
	ax.scatter(x[-1], y[-1], color=color, marker="X", s=600, edgecolors=color, linewidths=0.0, zorder=12)


def plot_trajectory(ax, df: pd.DataFrame, color: str, label: str) -> None:
	x, y = build_trajectory(df)
	(line,) = ax.plot(
		x,
		y,
		linestyle="-",
		linewidth=5,
		color=color,
		alpha=1.0,
		label=label,
		zorder=10,
	)
	line.set_path_effects(
		[
			patheffects.Stroke(linewidth=6.5, foreground="white", alpha=0.95),
			patheffects.Normal(),
		]
	)

	# Unique markers for start/end
	ax.scatter(x[0], y[0], color=color, marker="*", s=600, edgecolors="white", linewidths=1.2, zorder=11)
	ax.scatter(x[-1], y[-1], color=color, marker="X", s=600, edgecolors="white", linewidths=1.2, zorder=11)


def annotate_panel(ax, title: str) -> None:
	ax.set_title(title)
	ax.set_xlabel("X (m)", fontsize=20)
	ax.set_ylabel("Y (m)", fontsize=20)


def configure_velocity_overlay_axis(axis: plt.Axes) -> None:
	"""Configure axis limits + font sizes for the velocity overlay plots."""
	axis.set_xlim(*ROI_X_LIMITS)
	axis.set_ylim(*ROI_Y_LIMITS)
	axis.set_aspect("equal", adjustable="box")
	axis.set_xlabel("X (m)", fontsize=20)
	axis.set_ylabel("Y (m)", fontsize=20)
	axis.tick_params(axis="both", which="both", labelsize=20)
	axis.xaxis.get_offset_text().set_fontsize(20)
	axis.yaxis.get_offset_text().set_fontsize(20)


def resolve_velocity_pickle() -> Path:
	velocity_pickle = DEFAULT_VELOCITY_PICKLE if DEFAULT_VELOCITY_PICKLE.exists() else FALLBACK_VELOCITY_PICKLE
	if not velocity_pickle.exists():
		raise FileNotFoundError(
			"Could not find a velocity vector figure pickle. Expected one of:\n"
			f"- {DEFAULT_VELOCITY_PICKLE}\n"
			f"- {FALLBACK_VELOCITY_PICKLE}"
		)
	return velocity_pickle


def plot_trajectories_over_velocity(velocity_pickle: Path) -> None:
	figure = load_stream_figure(velocity_pickle)

	FIGURE_SIZE_INCHES = (8, 8)   # width, height
	OUTPUT_DPI = 300

	# Add this
	figure.set_size_inches(*FIGURE_SIZE_INCHES)


	if not getattr(figure, "axes", None):
		raise ValueError("Velocity-vector pickle did not contain any axes.")
	axis = figure.axes[0]
	configure_velocity_overlay_axis(axis)

	for planner in PLANNERS:
		trajectory_path = planner.trajectory_file or latest_matching_csv(planner.folder, "robot_trajectory_")
		if trajectory_path is None:
			raise FileNotFoundError(f"No trajectory log found for {planner.name}")
		trajectory_df = load_csv(trajectory_path)
		plot_trajectory(axis, trajectory_df, planner.color, planner.label)

	axis.legend(loc="upper left", frameon=True, fontsize=20, framealpha=0.9)
	figure.savefig(OUTPUT_TRAJECTORIES_PATH, dpi=180, bbox_inches="tight")
	plt.close(figure)
	print(f"Saved trajectories over velocity vectors to {OUTPUT_TRAJECTORIES_PATH}")


def plot_waypoints_over_velocity(velocity_pickle: Path) -> None:
	figure = load_stream_figure(velocity_pickle)

	FIGURE_SIZE_INCHES = (8, 8)   # width, height
	OUTPUT_DPI = 300

	# Add this
	figure.set_size_inches(*FIGURE_SIZE_INCHES)

	if not getattr(figure, "axes", None):
		raise ValueError("Velocity-vector pickle did not contain any axes.")
	axis = figure.axes[0]
	configure_velocity_overlay_axis(axis)

	for planner in PLANNERS:
		waypoint_path = planner.waypoint_file or latest_matching_csv(planner.folder, "goal_sequence_") or latest_matching_csv(planner.folder, "*_waypoints")
		if waypoint_path is None:
			waypoint_path = planner.execution_file or latest_matching_csv(planner.folder, "execution_log_")
		if waypoint_path is None:
			raise FileNotFoundError(f"No waypoint source found for {planner.name}")
		waypoint_df = load_csv(waypoint_path)
		plot_goal_path(axis, waypoint_df, planner.color, planner.label)

	axis.legend(loc="upper left", frameon=True, fontsize=20, framealpha=0.9)
	figure.savefig(OUTPUT_WAYPOINTS_PATH, dpi=180, bbox_inches="tight")
	plt.close(figure)
	print(f"Saved waypoints over velocity vectors to {OUTPUT_WAYPOINTS_PATH}")


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	velocity_pickle = resolve_velocity_pickle()
	plot_waypoints_over_velocity(velocity_pickle)
	plot_trajectories_over_velocity(velocity_pickle)


if __name__ == "__main__":
	main()
