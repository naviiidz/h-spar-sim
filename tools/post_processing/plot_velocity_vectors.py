#!/usr/bin/env python3
"""
Plot velocity vectors from VTU files as a figure.

This reuses the VTU parsing logic from convert_to_lookup.py and produces a
quiver plot of the 2D velocity field for a chosen timestamp.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.path import Path as MplPath


DEFAULT_ARROW_FRACTION = 0.03
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_PNG = OUTPUT_DIR / "velocity_vectors.png"
DEFAULT_OUTPUT_PKL = DATA_DIR / "velocity_vectors.pkl"
DEFAULT_OCCUPANCY_PATH = (
	REPO_ROOT
	/ "rrt-star-planners"
	/ "lookup_tables"
	/ "sydney_regatta_results"
	/ "occupancy_grid.npz"
)


def ensure_output_path(path: Optional[str], output_dir: str) -> Optional[str]:
	"""Return a path under output_dir when a bare filename is provided.

	- Absolute paths are kept as-is.
	- Relative paths that already include a directory are kept as-is.
	- Bare filenames are placed under output_dir.
	- Ensures parent directory exists.
	"""
	if path is None:
		return None
	if str(path).strip() == "":
		return None

	p = Path(path)
	if p.is_absolute() or p.parent != Path("."):
		out_path = p
	else:
		out_path = Path(output_dir) / p

	out_path.parent.mkdir(parents=True, exist_ok=True)
	return str(out_path)


def extract_xml_header(filepath: str) -> str:
	"""Extract XML header from VTU file with appended binary data."""
	with open(filepath, "rb") as f:
		content = f.read()

	appended_marker = b"<AppendedData encoding=\"raw\">"
	idx = content.find(appended_marker)

	if idx == -1:
		return content.decode("utf-8", errors="ignore")

	xml_part = content[:idx].decode("utf-8", errors="ignore")
	return xml_part + "</VTKFile>"


def read_binary_data(filepath: str, offset: int, dtype: str, count: int) -> Optional[np.ndarray]:
	"""Read binary data from appended section of VTU file."""
	with open(filepath, "rb") as f:
		content = f.read()

	appended_marker = b"<AppendedData encoding=\"raw\">\n_"
	idx = content.find(appended_marker)

	if idx == -1:
		return None

	binary_start = idx + len(appended_marker)
	data_offset = binary_start + offset

	if dtype == "Float64":
		fmt = "<d"
	elif dtype == "Float32":
		fmt = "<f"
	elif dtype == "Int32":
		fmt = "<i"
	elif dtype == "UInt8":
		fmt = "B"
	else:
		return None

	size_bytes = content[data_offset : data_offset + 4]
	if len(size_bytes) < 4:
		return None
	size = struct.unpack("<I", size_bytes)[0]

	data_start = data_offset + 4
	data_bytes = content[data_start : data_start + size]

	if len(data_bytes) < size:
		return None

	values = np.frombuffer(data_bytes, dtype=np.dtype(fmt), count=count)
	return values


def extract_timestamp(filepath: str) -> Optional[int]:
	"""Extract timestamp from filename (e.g., Velocity2d_123.vtu -> 123)."""
	basename = os.path.basename(filepath)
	parts = basename.replace(".vtu", "").split("_")
	if len(parts) >= 2:
		try:
			return int(parts[-1])
		except ValueError:
			return None
	return None


def read_vtu_data(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
	"""Read coordinates and velocity from a single VTU file."""
	try:
		xml_str = extract_xml_header(filepath)
		root = ET.fromstring(xml_str)

		piece = root.find(".//Piece")
		if piece is None:
			return None

		n_points = int(piece.get("NumberOfPoints", 0))
		coords_data = None
		velocity_data = None

		for data_array in root.findall(".//DataArray"):
			name = data_array.get("Name")
			if name in ("firedrake_default_coordinates", "Coordinates"):
				coords_data = data_array
			elif name in ("Velocity2d", "Depth averaged velocity", "velocity"):
				velocity_data = data_array

		if coords_data is None or velocity_data is None:
			return None

		coords_dtype = coords_data.get("type")
		coords_offset = int(coords_data.get("offset", 0))
		coords_n_comp = int(coords_data.get("NumberOfComponents", 3))
		coords = read_binary_data(filepath, coords_offset, coords_dtype, n_points * coords_n_comp)
		if coords is None:
			return None
		coords = coords.reshape(-1, coords_n_comp)

		vel_dtype = velocity_data.get("type")
		vel_offset = int(velocity_data.get("offset", 0))
		vel_n_comp = int(velocity_data.get("NumberOfComponents", 2))
		velocity = read_binary_data(filepath, vel_offset, vel_dtype, n_points * vel_n_comp)
		if velocity is None:
			return None
		velocity = velocity.reshape(-1, vel_n_comp)

		return coords, velocity

	except Exception as e:
		print(f"Error reading {filepath}: {e}", file=sys.stderr)
		return None


def choose_plot_sample(coords: np.ndarray, max_vectors: int) -> np.ndarray:
	"""Select a roughly even subsample of vectors for the quiver plot."""
	n_points = len(coords)
	if n_points <= max_vectors:
		return np.arange(n_points)
	step = max(1, n_points // max_vectors)
	return np.arange(0, n_points, step)


def default_search_path() -> str:
	"""Return the repository-relative default VTU glob."""
	return str(REPO_ROOT / "velocity_fields" / "sydney_regatta" / "raw" / "Velocity2d" / "*.vtu")


def load_average_velocity_field(files: list[str]) -> Tuple[np.ndarray, np.ndarray, int]:
	"""Load all VTUs and return the averaged coordinates, velocity, and count."""
	coords_ref: Optional[np.ndarray] = None
	velocity_sum: Optional[np.ndarray] = None
	valid_count = 0

	for filepath in files:
		result = read_vtu_data(filepath)
		if result is None:
			print(f"Skipping unreadable file: {filepath}")
			continue

		coords, velocity = result
		coords_2d = coords[:, :2]
		velocity_2d = velocity[:, :2]

		if coords_ref is None:
			coords_ref = coords_2d
			velocity_sum = np.zeros_like(velocity_2d, dtype=float)
		elif coords_ref.shape != coords_2d.shape or not np.allclose(coords_ref, coords_2d):
			raise ValueError(f"Coordinate grid mismatch in {filepath}")

		assert velocity_sum is not None
		velocity_sum += velocity_2d
		valid_count += 1

	if coords_ref is None or velocity_sum is None or valid_count == 0:
		raise ValueError("No valid VTU files could be loaded.")

	return coords_ref, velocity_sum / float(valid_count), valid_count


def plot_velocity_vectors(
	coords: np.ndarray,
	velocity: np.ndarray,
	title: str,
	output_png: str,
	output_pkl: Optional[str] = None,
	arrow_fraction: float = DEFAULT_ARROW_FRACTION,
	occupancy_path: Optional[str] = None,
	x_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
	y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> plt.Figure:
	"""Create and save a quiver plot of the velocity vectors."""
	if not 0.0 < arrow_fraction <= 1.0:
		raise ValueError("--arrow-fraction must be greater than 0 and less than or equal to 1.")

	coords_2d = coords[:, :2]
	velocity_2d = velocity[:, :2]
# Load occupancy grid if provided (or generate fallback) and filter out occupied points
	occ_grid = None
	occ_origin_x = None
	occ_origin_y = None
	occ_resolution = None
	occ_extent = None
	if occupancy_path is not None and occupancy_path != "":
		try:
			data = np.load(occupancy_path)
			if isinstance(data, np.lib.npyio.NpzFile):
				occ_grid = data.get("grid")
				occ_origin_x = float(data.get("origin_x", 0.0))
				occ_origin_y = float(data.get("origin_y", 0.0))
				occ_resolution = float(data.get("resolution", 1.0))
			else:
				occ_grid = data
				# infer origin/resolution from coords when possible
				if occ_grid is not None:
					h, w = occ_grid.shape
					min_x, min_y = float(np.min(coords[:, 0])), float(np.min(coords[:, 1]))
					max_x, max_y = float(np.max(coords[:, 0])), float(np.max(coords[:, 1]))
					occ_origin_x = min_x
					occ_origin_y = min_y
					occ_resolution = max((max_x - min_x) / max(1, w), (max_y - min_y) / max(1, h))

			if occ_grid is not None:
				h, w = occ_grid.shape
				if occ_origin_x is None or occ_origin_y is None or occ_resolution is None:
					occ_extent = [float(np.min(coords[:,0])), float(np.max(coords[:,0])), float(np.min(coords[:,1])), float(np.max(coords[:,1]))]
				else:
					occ_extent = [occ_origin_x, occ_origin_x + w * occ_resolution, occ_origin_y, occ_origin_y + h * occ_resolution]
		except Exception as e:
			print(f"Warning: could not load occupancy grid '{occupancy_path}': {e}")
			# fallback: try to rasterize polygon geo file
			try:
				geo_path = REPO_ROOT / "meshes" / "sydney_regatta" / "water_polygon.geo"
				if geo_path.exists():
					def parse_geo_file(filepath: str) -> List[Tuple[float, float]]:
						pts = {}
						edges = []
						with open(filepath, "r") as f:
							lines = f.readlines()
							for line in lines:
								m = re.match(r'Point\((\d+)\)\s*=\s*\{\s*([\-\d.]+),\s*([\-\d.]+)', line)
								if m:
									pid = int(m.group(1))
									x = float(m.group(2))
									y = float(m.group(3))
									pts[pid] = (x, y)
									m2 = re.match(r'Line\((\d+)\)\s*=\s*\{(\d+),\s*(\d+)\}', line)
									if m2:
										edges.append((int(m2.group(2)), int(m2.group(3))))
							if not edges:
								return [pts[k] for k in sorted(pts.keys())]
							polygon = []
							current = edges[0][0]
							visited = set()
							for _ in range(len(edges)):
								polygon.append(pts[current])
								for idx_e, (a, b) in enumerate(edges):
									if idx_e in visited:
										continue
									if a == current:
										visited.add(idx_e)
										current = b
										break
									if b == current:
										visited.add(idx_e)
										current = a
										break
								return polygon
				polygon_coords = parse_geo_file(str(geo_path))
				poly_array = np.array(polygon_coords)
				padding = 50
				occ_resolution = 5.0
				min_x, min_y = poly_array.min(axis=0) - padding
				max_x, max_y = poly_array.max(axis=0) + padding
				grid_width = int(np.ceil((max_x - min_x) / occ_resolution))
				grid_height = int(np.ceil((max_y - min_y) / occ_resolution))
				xs = min_x + (np.arange(grid_width) + 0.5) * occ_resolution
				ys = min_y + (np.arange(grid_height) + 0.5) * occ_resolution
				xv, yv = np.meshgrid(xs, ys)
				pts = np.vstack((xv.ravel(), yv.ravel())).T
				path = MplPath(np.array(polygon_coords))
				contains = path.contains_points(pts)
				occ_grid = np.where(contains.reshape((grid_height, grid_width)), 0, 100).astype(np.int8)
				occ_origin_x = min_x
				occ_origin_y = min_y
				occ_extent = [min_x, min_x + grid_width * occ_resolution, min_y, min_y + grid_height * occ_resolution]
			except Exception as e2:
				print(f"Fallback occupancy generation failed: {e2}")

	# Build a free_mask (True where points are free / not occupied)
	free_mask = np.ones(len(coords_2d), dtype=bool)
	if occ_grid is not None:
		h, w = occ_grid.shape
		# ensure we have origin and resolution
		if occ_origin_x is None or occ_origin_y is None or occ_resolution is None:
			# if extent is available, infer origin/resolution
			if occ_extent is not None:
				occ_origin_x = occ_extent[0]
				occ_origin_y = occ_extent[2]
				occ_resolution = (occ_extent[1] - occ_extent[0]) / max(1, w)
		# map coords to grid indices
		ix = np.floor((coords_2d[:, 0] - occ_origin_x) / occ_resolution).astype(int)
		iy = np.floor((coords_2d[:, 1] - occ_origin_y) / occ_resolution).astype(int)
		valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
		occupied = np.zeros(len(coords_2d), dtype=bool)
		occupied[valid] = (occ_grid[iy[valid], ix[valid]] != 0)
		free_mask = ~occupied

	# Apply free mask
	coords_2d = coords_2d[free_mask]
	velocity_2d = velocity_2d[free_mask]

	if len(coords_2d) == 0:
		print("No free points remain after removing occupied cells.")
		raise ValueError("No free points remain after removing occupied cells.")

	# Subsample and normalize arrows by one global speed scale.
	step = max(1, int(round(1.0 / arrow_fraction)))
	indices = np.arange(0, len(coords_2d), step)
	print(f"Drawing {len(indices)} of {len(coords_2d)} velocity vectors (arrow_fraction={arrow_fraction}, step={step}).")
	velocity_sampled = velocity_2d[indices]
	speed_sampled = np.linalg.norm(velocity_sampled, axis=1)
	speed_all = np.linalg.norm(velocity_2d, axis=1)
	global_speed = max(float(np.percentile(speed_all, 95)), 1e-12)
	span_x = float(np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0]))
	span_y = float(np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1]))
	span = max(span_x, span_y)
	max_arrow_length = 0.035 * span
	direction = np.zeros_like(velocity_sampled, dtype=float)
	nonzero_speed = speed_sampled > 1e-12
	direction[nonzero_speed] = velocity_sampled[nonzero_speed] / speed_sampled[nonzero_speed, None]
	normalized_length = np.clip(speed_sampled / global_speed, 0.15, 1.0)
	normalized_length[~nonzero_speed] = 0.0
	quiver_u = direction[:, 0] * normalized_length * max_arrow_length
	quiver_v = direction[:, 1] * normalized_length * max_arrow_length

	# Publication-standard figure size and DPI
	FIGURE_SIZE_INCHES = (9, 9)
	OUTPUT_DPI = 300

	fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)

	# If we have an occupancy grid, display it beneath the vectors
	if occ_grid is not None:
		cmap = ListedColormap(["white", "#8B4513"])  # free=white, occupied=saddlebrown
		norm = Normalize(vmin=0, vmax=100)
		if occ_extent is None and occ_origin_x is not None:
			h, w = occ_grid.shape
			occ_extent = [occ_origin_x, occ_origin_x + w * occ_resolution, occ_origin_y, occ_origin_y + h * occ_resolution]
		ax.imshow(occ_grid, cmap=cmap, norm=norm, origin="lower", extent=occ_extent, alpha=0.6, zorder=0)

	quiver = ax.quiver(
		coords_2d[indices, 0],
		coords_2d[indices, 1],
		quiver_u,
		quiver_v,
		color="blue",
		angles="xy",
		scale_units="xy",
		scale=0.5,
		width=0.003,
		headwidth=2.5,
		headlength=3.0,
		headaxislength=2.8,
		pivot="mid",
		alpha=0.9,
	)

	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(False)

	# Ensure the saved figure/pickle keeps requested limits when provided.
	try:
		x_min = float(np.min(coords_2d[:, 0]))
		x_max = float(np.max(coords_2d[:, 0]))
		y_min = float(np.min(coords_2d[:, 1]))
		y_max = float(np.max(coords_2d[:, 1]))
		if x_limits is not None:
			x_min = x_limits[0] if x_limits[0] is not None else x_min
			x_max = x_limits[1] if x_limits[1] is not None else x_max
		if y_limits is not None:
			y_min = y_limits[0] if y_limits[0] is not None else y_min
			y_max = y_limits[1] if y_limits[1] is not None else y_max
		ax.set_xlim(x_min, x_max)
		ax.set_ylim(y_min, y_max)
	except Exception:
		# If something goes wrong, ignore and keep default limits
		pass

	plt.tight_layout()
	plt.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight")
	print(f"Saved velocity vector figure to {output_png}")

	if output_pkl is not None:
		# Ensure the pickled figure has the publication size recorded
		fig.set_size_inches(*FIGURE_SIZE_INCHES)
		with open(output_pkl, "wb") as f:
			pickle.dump(fig, f)
		print(f"Saved figure object to {output_pkl}")

	return fig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot velocity vectors from VTU files.")
	parser.add_argument(
		"search_path",
		nargs="?",
		default=default_search_path(),
		help="Glob pattern for VTU files.",
	)
	parser.add_argument(
		"--arrow-fraction",
		type=float,
		default=DEFAULT_ARROW_FRACTION,
		help="Fraction of arrows to draw from the averaged field.",
	)
	parser.add_argument(
		"--xmin",
		type=float,
		default=None,
		help="Minimum x to include (inclusive).",
	)
	parser.add_argument(
		"--xmax",
		type=float,
		default=None,
		help="Maximum x to include (inclusive).",
	)
	parser.add_argument(
		"--ymin",
		type=float,
		default=None,
		help="Minimum y to include (inclusive).",
	)
	parser.add_argument(
		"--ymax",
		type=float,
		default=None,
		help="Maximum y to include (inclusive).",
	)
	parser.add_argument(
		"--output-png",
		default=str(DEFAULT_OUTPUT_PNG),
		help="Output PNG path.",
	)
	parser.add_argument(
		"--output-pkl",
		default=str(DEFAULT_OUTPUT_PKL),
		help="Output pickle path for the matplotlib Figure object.",
	)
	parser.add_argument(
		"--output-dir",
		default=str(DATA_DIR),
		help="Directory to write outputs into when output paths are bare filenames.",
	)
	parser.add_argument(
		"--occupancy",
		default=str(DEFAULT_OCCUPANCY_PATH),
		help="Path to occupancy grid (.npz or .npy) to overlay beneath the vectors.",
	)
	parser.add_argument(
		"--max-vectors",
		type=int,
		default=3000,
		help="Deprecated: kept for compatibility.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	files = sorted(glob.glob(args.search_path))

	if not files:
		print(f"No VTU files found at {args.search_path}")
		return

	print(f"Averaging {len(files)} VTU files")
	try:
		coords, velocity, valid_count = load_average_velocity_field(files)
	except ValueError as e:
		print(str(e))
		return

	# Apply spatial filter if requested
	mask = np.ones(len(coords), dtype=bool)
	if args.xmin is not None:
		mask &= coords[:, 0] >= args.xmin
	if args.xmax is not None:
		mask &= coords[:, 0] <= args.xmax
	if args.ymin is not None:
		mask &= coords[:, 1] >= args.ymin
	if args.ymax is not None:
		mask &= coords[:, 1] <= args.ymax

	if not mask.any():
		print("No points remain after applying spatial filter.")
		return

	coords_f = coords[mask]
	velocity_f = velocity[mask]

	output_png = ensure_output_path(args.output_png, args.output_dir)
	output_pkl = ensure_output_path(args.output_pkl, args.output_dir)

	plot_velocity_vectors(
		coords=coords_f,
		velocity=velocity_f,
		title=f"Average Velocity Field over {valid_count} VTU Files (filtered)",
		output_png=output_png or args.output_png,
		output_pkl=output_pkl,
		arrow_fraction=args.arrow_fraction,
		occupancy_path=args.occupancy,
		x_limits=(args.xmin, args.xmax),
		y_limits=(args.ymin, args.ymax),
	)


if __name__ == "__main__":
	main()
