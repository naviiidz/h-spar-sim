#!/usr/bin/env python3
"""
Simulate Lagrangian particles from VTU velocity fields.

Pipeline:
    VTU velocity -> RK2 advection + random walk -> CSV trajectories

Particles are seeded along the upstream side of the mesh. Once a particle
leaves the loaded mesh, it is marked as exited and no longer advected.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
VELOCITY_ARRAY_NAMES = ("Velocity2d", "Depth averaged velocity", "velocity")
COORDINATE_ARRAY_NAMES = ("firedrake_default_coordinates", "Coordinates")
VTK_TRIANGLE = 5
VTK_POLYGON = 7
VTK_QUAD = 9
CONFIG_ALIASES = {
	"gif": "generate_gif",
	"plot": "generate_gif",
}


@dataclass(frozen=True)
class MeshGeometry:
	xy: np.ndarray
	z: np.ndarray
	original_to_unique: np.ndarray
	triangulation: mtri.Triangulation
	flow_dir: np.ndarray
	perp_dir: np.ndarray


@dataclass
class VelocityFrame:
	u_interp: mtri.LinearTriInterpolator
	v_interp: mtri.LinearTriInterpolator
	source: Path


def natural_key(path: Path) -> tuple:
	parts = re.split(r"(\d+)", path.name)
	return tuple(int(part) if part.isdigit() else part for part in parts)


def resolve_vtu_files(pattern: str) -> list[Path]:
	files = sorted((Path(path) for path in glob.glob(pattern)), key=natural_key)
	if not files:
		raise FileNotFoundError(f"No VTU files matched: {pattern}")
	return files


def infer_map_name(files: list[Path], pattern: str) -> str:
	if files:
		parts = files[0].parts
		if "raw" in parts:
			raw_index = parts.index("raw")
			if raw_index > 0:
				return parts[raw_index - 1]

	pattern_path = Path(pattern)
	for parent in [pattern_path.parent, *pattern_path.parents]:
		if parent.name and parent.name not in {"*", "Velocity2d", "raw", "."}:
			return parent.name
	return "map"


def default_output_paths(map_name: str) -> tuple[Path, Path]:
	map_output_dir = OUTPUT_DIR / map_name
	return (
		map_output_dir / f"{map_name}_lagrangian_particle_trajectories.csv",
		map_output_dir / f"{map_name}_lagrangian_particle_trajectories.gif",
	)


def normalize_config_key(key: str) -> str:
	return CONFIG_ALIASES.get(key.replace("-", "_"), key.replace("-", "_"))


def load_config_file(path: Path) -> dict:
	with path.open() as handle:
		config = json.load(handle)
	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a JSON object: {path}")
	return {normalize_config_key(str(key)): value for key, value in config.items()}


def extract_xml_header(filepath: Path) -> str:
	with filepath.open("rb") as handle:
		content = handle.read()

	appended_marker = b'<AppendedData encoding="raw">'
	index = content.find(appended_marker)
	if index == -1:
		return content.decode("utf-8", errors="ignore")

	xml_part = content[:index].decode("utf-8", errors="ignore")
	return xml_part + "</VTKFile>"


def split_vtu_content(filepath: Path) -> tuple[str, bytes, int]:
	with filepath.open("rb") as handle:
		content = handle.read()

	appended_marker = b'<AppendedData encoding="raw">'
	header_index = content.find(appended_marker)
	if header_index == -1:
		return content.decode("utf-8", errors="ignore"), content, -1

	xml_part = content[:header_index].decode("utf-8", errors="ignore") + "</VTKFile>"
	raw_marker = b'<AppendedData encoding="raw">\n_'
	raw_index = content.find(raw_marker)
	if raw_index == -1:
		raise ValueError(f"{filepath} does not contain raw appended VTU data.")
	return xml_part, content, raw_index + len(raw_marker)


def vtk_dtype(vtk_type: str) -> np.dtype:
	dtypes = {
		"Float64": np.dtype("<f8"),
		"Float32": np.dtype("<f4"),
		"Int64": np.dtype("<i8"),
		"Int32": np.dtype("<i4"),
		"UInt32": np.dtype("<u4"),
		"UInt8": np.dtype("u1"),
	}
	if vtk_type not in dtypes:
		raise ValueError(f"Unsupported VTU DataArray type: {vtk_type}")
	return dtypes[vtk_type]


def read_appended_array_from_content(
	filepath: Path,
	content: bytes,
	appended_start: int,
	data_array: ET.Element,
	count: Optional[int] = None,
) -> np.ndarray:
	if appended_start < 0:
		raise ValueError(f"{filepath} does not contain raw appended VTU data.")
	offset = int(data_array.get("offset", "0"))
	data_offset = appended_start + offset
	size_bytes = content[data_offset : data_offset + 4]
	if len(size_bytes) < 4:
		raise ValueError(f"Could not read appended block size in {filepath}")
	size = struct.unpack("<I", size_bytes)[0]
	data_start = data_offset + 4
	data_bytes = content[data_start : data_start + size]
	if len(data_bytes) < size:
		raise ValueError(f"Truncated appended block in {filepath}")

	array = np.frombuffer(data_bytes, dtype=vtk_dtype(data_array.get("type", "")))
	if count is not None:
		if len(array) < count:
			raise ValueError(f"VTU block has {len(array)} values; expected {count}.")
		array = array[:count]
	return array


def read_appended_array(filepath: Path, data_array: ET.Element, count: Optional[int] = None) -> np.ndarray:
	_, content, appended_start = split_vtu_content(filepath)
	return read_appended_array_from_content(filepath, content, appended_start, data_array, count)


def find_named_data_array(root: ET.Element, parent_name: str, names: tuple[str, ...]) -> ET.Element:
	parent = root.find(f".//{parent_name}")
	if parent is None:
		raise ValueError(f"VTU is missing {parent_name}.")
	for data_array in parent.findall("DataArray"):
		if data_array.get("Name") in names:
			return data_array
	raise ValueError(f"VTU {parent_name} has no DataArray named one of: {', '.join(names)}")


def read_vtu(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	xml_header, content, appended_start = split_vtu_content(filepath)
	root = ET.fromstring(xml_header)
	piece = root.find(".//Piece")
	if piece is None:
		raise ValueError(f"{filepath} has no VTU Piece element.")

	n_points = int(piece.get("NumberOfPoints", "0"))
	n_cells = int(piece.get("NumberOfCells", "0"))

	coords_data = find_named_data_array(root, "Points", COORDINATE_ARRAY_NAMES)
	coords_components = int(coords_data.get("NumberOfComponents", "3") or "3")
	coords = read_appended_array_from_content(filepath, content, appended_start, coords_data, n_points * coords_components).reshape(-1, coords_components)

	velocity_data = find_named_data_array(root, "PointData", VELOCITY_ARRAY_NAMES)
	velocity_components = int(velocity_data.get("NumberOfComponents", "2") or "2")
	velocity = read_appended_array_from_content(filepath, content, appended_start, velocity_data, n_points * velocity_components).reshape(-1, velocity_components)

	cells_parent = root.find(".//Cells")
	if cells_parent is None:
		raise ValueError(f"{filepath} has no Cells section.")
	cell_arrays = {data_array.get("Name"): data_array for data_array in cells_parent.findall("DataArray")}
	for required in ("connectivity", "offsets", "types"):
		if required not in cell_arrays:
			raise ValueError(f"{filepath} Cells section is missing '{required}'.")

	connectivity = read_appended_array_from_content(filepath, content, appended_start, cell_arrays["connectivity"])
	offsets = read_appended_array_from_content(filepath, content, appended_start, cell_arrays["offsets"], n_cells)
	types = read_appended_array_from_content(filepath, content, appended_start, cell_arrays["types"], n_cells)
	triangles = triangles_from_vtk_cells(connectivity, offsets, types)
	return coords[:, :3], velocity[:, :2], triangles


def read_vtu_velocity(filepath: Path) -> np.ndarray:
	xml_header, content, appended_start = split_vtu_content(filepath)
	root = ET.fromstring(xml_header)
	piece = root.find(".//Piece")
	if piece is None:
		raise ValueError(f"{filepath} has no VTU Piece element.")

	n_points = int(piece.get("NumberOfPoints", "0"))
	velocity_data = find_named_data_array(root, "PointData", VELOCITY_ARRAY_NAMES)
	velocity_components = int(velocity_data.get("NumberOfComponents", "2") or "2")
	velocity = read_appended_array_from_content(
		filepath,
		content,
		appended_start,
		velocity_data,
		n_points * velocity_components,
	).reshape(-1, velocity_components)
	return velocity[:, :2]


def triangles_from_vtk_cells(connectivity: np.ndarray, offsets: np.ndarray, types: np.ndarray) -> np.ndarray:
	triangles: list[list[int]] = []
	start = 0
	for stop, vtk_type in zip(offsets.astype(int), types.astype(int)):
		cell = connectivity[start:stop].astype(int)
		start = stop
		if vtk_type == VTK_TRIANGLE and len(cell) >= 3:
			triangles.append(cell[:3].tolist())
		elif vtk_type == VTK_QUAD and len(cell) >= 4:
			a, b, c, d = cell[:4]
			triangles.append([int(a), int(b), int(c)])
			triangles.append([int(a), int(c), int(d)])
		elif vtk_type == VTK_POLYGON and len(cell) >= 3:
			for i in range(1, len(cell) - 1):
				triangles.append([int(cell[0]), int(cell[i]), int(cell[i + 1])])
	if not triangles:
		raise ValueError("Could not build a 2D triangular surface from VTU cells.")
	return np.asarray(triangles, dtype=int)


def load_geometry(first_vtu: Path) -> tuple[MeshGeometry, np.ndarray]:
	points, first_velocity, triangles = read_vtu(first_vtu)
	if points.ndim != 2 or points.shape[1] < 2:
		raise ValueError(f"{first_vtu} does not contain 2D/3D mesh points.")

	xy_raw = points[:, :2]
	z_raw = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points), dtype=float)
	xy, original_to_unique = np.unique(xy_raw, axis=0, return_inverse=True)
	z = average_point_values(z_raw[:, None], original_to_unique, len(xy))[:, 0]
	unique_triangles = original_to_unique[triangles]
	nondegenerate = np.array([len(set(cell.tolist())) == 3 for cell in unique_triangles], dtype=bool)
	unique_triangles = unique_triangles[nondegenerate]
	if len(unique_triangles) == 0:
		raise ValueError("All VTU cells became degenerate after merging duplicate coordinates.")
	triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], unique_triangles)

	first_velocity_unique = average_point_values(first_velocity, original_to_unique, len(xy))
	mean_velocity = np.nanmean(first_velocity_unique, axis=0)
	if not np.all(np.isfinite(mean_velocity)) or np.linalg.norm(mean_velocity) < 1e-12:
		span = np.ptp(xy, axis=0)
		mean_velocity = np.array([1.0, 0.0]) if span[0] >= span[1] else np.array([0.0, 1.0])

	flow_dir = mean_velocity / np.linalg.norm(mean_velocity)
	perp_dir = np.array([-flow_dir[1], flow_dir[0]], dtype=float)
	return (
		MeshGeometry(
			xy=xy,
			z=z,
			original_to_unique=original_to_unique,
			triangulation=triangulation,
			flow_dir=flow_dir,
			perp_dir=perp_dir,
		),
		first_velocity_unique,
	)


def average_point_values(values: np.ndarray, inverse: np.ndarray, unique_count: int) -> np.ndarray:
	values = np.asarray(values, dtype=float)
	if values.ndim == 1:
		values = values[:, None]
	if values.shape[0] != inverse.shape[0]:
		raise ValueError(f"Expected {inverse.shape[0]} point values, got {values.shape[0]}.")
	out = np.zeros((unique_count, values.shape[1]), dtype=float)
	counts = np.bincount(inverse, minlength=unique_count).astype(float)
	for component in range(values.shape[1]):
		out[:, component] = np.bincount(inverse, weights=values[:, component], minlength=unique_count)
	out /= counts[:, None]
	return out


class VtuVelocitySeries:
	def __init__(self, files: list[Path], geometry: MeshGeometry, first_velocity: np.ndarray, field_dt: float, cache_size: int = 4):
		if field_dt <= 0.0:
			raise ValueError("--field-dt must be positive.")
		self.files = files
		self.geometry = geometry
		self.field_dt = field_dt
		self.cache_size = max(1, cache_size)
		self._cache: OrderedDict[int, VelocityFrame] = OrderedDict()
		self._add_frame_to_cache(0, first_velocity)

	def _add_frame_to_cache(self, index: int, velocity: np.ndarray) -> VelocityFrame:
		if velocity.shape[0] == self.geometry.original_to_unique.shape[0]:
			velocity = average_point_values(velocity, self.geometry.original_to_unique, len(self.geometry.xy))
		if velocity.shape[0] != self.geometry.xy.shape[0]:
			raise ValueError(
				f"Velocity point count mismatch in {self.files[index]}: "
				f"{velocity.shape[0]} values for {self.geometry.xy.shape[0]} mesh points."
			)
		frame = VelocityFrame(
			u_interp=mtri.LinearTriInterpolator(self.geometry.triangulation, velocity[:, 0]),
			v_interp=mtri.LinearTriInterpolator(self.geometry.triangulation, velocity[:, 1]),
			source=self.files[index],
		)
		self._cache[index] = frame
		self._cache.move_to_end(index)
		while len(self._cache) > self.cache_size:
			self._cache.popitem(last=False)
		return frame

	def frame(self, index: int) -> VelocityFrame:
		index = int(np.clip(index, 0, len(self.files) - 1))
		if index in self._cache:
			self._cache.move_to_end(index)
			return self._cache[index]
		velocity = read_vtu_velocity(self.files[index])
		return self._add_frame_to_cache(index, velocity)

	def velocity(self, points: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
		if len(self.files) == 1:
			return interpolate_frame(self.frame(0), points), np.zeros(len(points), dtype=int)

		scaled = max(0.0, time / self.field_dt)
		i0 = min(int(math.floor(scaled)), len(self.files) - 1)
		i1 = min(i0 + 1, len(self.files) - 1)
		alpha = 0.0 if i0 == i1 else scaled - i0

		v0 = interpolate_frame(self.frame(i0), points)
		if alpha <= 1e-12:
			return v0, np.full(len(points), i0, dtype=int)

		v1 = interpolate_frame(self.frame(i1), points)
		velocity = (1.0 - alpha) * v0 + alpha * v1
		return velocity, np.full(len(points), i0, dtype=int)


def interpolate_frame(frame: VelocityFrame, points: np.ndarray) -> np.ndarray:
	u = frame.u_interp(points[:, 0], points[:, 1])
	v = frame.v_interp(points[:, 0], points[:, 1])
	return np.column_stack([np.ma.filled(u, np.nan), np.ma.filled(v, np.nan)]).astype(float)


def inside_mesh(geometry: MeshGeometry, points: np.ndarray) -> np.ndarray:
	finder = geometry.triangulation.get_trifinder()
	return finder(points[:, 0], points[:, 1]) >= 0


def boundary_line_segments(geometry: MeshGeometry) -> np.ndarray:
	triangles = np.asarray(geometry.triangulation.triangles, dtype=int)
	edges = np.vstack(
		[
			triangles[:, [0, 1]],
			triangles[:, [1, 2]],
			triangles[:, [2, 0]],
		]
	)
	edges.sort(axis=1)
	unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
	boundary_edges = unique_edges[counts == 1]
	return geometry.xy[boundary_edges]


def seed_upstream_particles(
	geometry: MeshGeometry,
	count: int,
	band_width: Optional[float],
	jitter: float,
	rng: np.random.Generator,
) -> np.ndarray:
	if count <= 0:
		raise ValueError("--particles must be positive.")

	projection = geometry.xy @ geometry.flow_dir
	perp = geometry.xy @ geometry.perp_dir
	domain_span = float(np.linalg.norm(np.ptp(geometry.xy, axis=0)))
	width = band_width if band_width is not None else max(domain_span * 0.02, 1e-9)

	upstream_min = float(np.min(projection))
	candidate_mask = projection <= upstream_min + width
	candidates = geometry.xy[candidate_mask]
	candidate_perp = perp[candidate_mask]
	if len(candidates) == 0:
		raise ValueError("No mesh points found on the upstream seeding band.")

	if count == 1:
		target_perp = np.array([0.5 * (candidate_perp.min() + candidate_perp.max())], dtype=float)
	else:
		target_perp = np.linspace(float(candidate_perp.min()), float(candidate_perp.max()), count)

	seeds = []
	for target in target_perp:
		index = int(np.argmin(np.abs(candidate_perp - target)))
		seeds.append(candidates[index])
	seeds_array = np.asarray(seeds, dtype=float)

	if jitter > 0.0:
		seeds_array += rng.normal(0.0, jitter, size=seeds_array.shape)
		valid = inside_mesh(geometry, seeds_array)
		if not np.all(valid):
			seeds_array[~valid] = np.asarray(seeds, dtype=float)[~valid]

	return seeds_array


def seed_fixed_particles(
	geometry: MeshGeometry,
	count: int,
	start_x: float,
	start_y: float,
	jitter: float,
	rng: np.random.Generator,
) -> np.ndarray:
	if count <= 0:
		raise ValueError("--particles must be positive.")

	start = np.array([start_x, start_y], dtype=float)
	if not inside_mesh(geometry, start[None, :])[0]:
		raise ValueError(f"Initial particle position ({start_x}, {start_y}) is outside the VTU mesh.")

	seeds = np.repeat(start[None, :], count, axis=0)
	if jitter > 0.0:
		jittered = seeds + rng.normal(0.0, jitter, size=seeds.shape)
		valid = inside_mesh(geometry, jittered)
		seeds[valid] = jittered[valid]

	return seeds


def seed_rectangle_particles(
	geometry: MeshGeometry,
	count: int,
	x_min: float,
	x_max: float,
	y_min: float,
	y_max: float,
	rng: np.random.Generator,
) -> np.ndarray:
	if count <= 0:
		raise ValueError("--particles must be positive.")
	if x_max < x_min:
		raise ValueError("--seed-x-max must be greater than or equal to --seed-x-min.")
	if y_max < y_min:
		raise ValueError("--seed-y-max must be greater than or equal to --seed-y-min.")

	seeds: list[np.ndarray] = []
	batch_size = max(count * 4, 256)
	max_attempts = 100
	for _ in range(max_attempts):
		candidates = np.column_stack(
			[
				rng.uniform(x_min, x_max, size=batch_size),
				rng.uniform(y_min, y_max, size=batch_size),
			]
		)
		inside = candidates[inside_mesh(geometry, candidates)]
		for point in inside:
			seeds.append(point)
			if len(seeds) == count:
				return np.asarray(seeds, dtype=float)

	raise ValueError(
		f"Only found {len(seeds)} valid seed points inside the mesh after {max_attempts} attempts. "
		"Check the rectangle bounds or reduce --particles."
	)


def seed_particles(
	geometry: MeshGeometry,
	args: argparse.Namespace,
	rng: np.random.Generator,
	count: int,
	first_vtu: Path,
) -> tuple[np.ndarray, str]:
	if args.seed_upstream:
		return (
			seed_upstream_particles(geometry, count, args.upstream_band_width, args.seed_jitter, rng),
			f"upstream side of {first_vtu.name}",
		)
	if args.seed_fixed:
		return (
			seed_fixed_particles(geometry, count, args.start_x, args.start_y, args.seed_jitter, rng),
			f"({args.start_x}, {args.start_y})",
		)
	return (
		seed_rectangle_particles(
			geometry,
			count,
			args.seed_x_min,
			args.seed_x_max,
			args.seed_y_min,
			args.seed_y_max,
			rng,
		),
		f"x=[{args.seed_x_min}, {args.seed_x_max}], y=[{args.seed_y_min}, {args.seed_y_max}]",
	)


def rk2_random_walk_step(
	series: VtuVelocitySeries,
	positions: np.ndarray,
	active: np.ndarray,
	time: float,
	dt: float,
	diffusivity: float,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	next_positions = positions.copy()
	velocity_now = np.full_like(positions, np.nan, dtype=float)
	frame_indices = np.full(len(positions), -1, dtype=int)

	active_indices = np.flatnonzero(active)
	if len(active_indices) == 0:
		return next_positions, active.copy(), velocity_now, frame_indices

	p0 = positions[active_indices]
	v0, frame_ids = series.velocity(p0, time)
	valid_v0 = np.all(np.isfinite(v0), axis=1)

	mid = p0 + 0.5 * dt * v0
	v_mid = np.full_like(v0, np.nan)
	if np.any(valid_v0):
		v_mid[valid_v0], _ = series.velocity(mid[valid_v0], time + 0.5 * dt)

	valid_mid = valid_v0 & np.all(np.isfinite(v_mid), axis=1)
	displacement = np.zeros_like(v_mid)
	displacement[valid_mid] = dt * v_mid[valid_mid]
	if diffusivity > 0.0:
		displacement[valid_mid] += math.sqrt(2.0 * diffusivity * dt) * rng.normal(size=displacement[valid_mid].shape)

	p1 = p0 + displacement
	still_inside = np.zeros(len(active_indices), dtype=bool)
	if np.any(valid_mid):
		still_inside[valid_mid] = inside_mesh(series.geometry, p1[valid_mid])

	next_positions[active_indices] = p1
	new_active = active.copy()
	new_active[active_indices] = valid_mid & still_inside
	velocity_now[active_indices] = v0
	frame_indices[active_indices] = frame_ids
	return next_positions, new_active, velocity_now, frame_indices


def write_rows(
	writer: csv.DictWriter,
	step: int,
	time: float,
	positions: np.ndarray,
	active: np.ndarray,
	previous_active: np.ndarray,
	velocity: np.ndarray,
	frame_indices: np.ndarray,
	generations: np.ndarray,
	series: VtuVelocitySeries,
	particle_indices: Optional[np.ndarray] = None,
	status_override: Optional[str] = None,
) -> None:
	if particle_indices is None:
		particle_indices = np.arange(len(positions), dtype=int)
	for particle_id in particle_indices:
		position = positions[particle_id]
		if status_override is None and not previous_active[particle_id] and step > 0:
			continue
		status = status_override if status_override is not None else ("active" if active[particle_id] else "exited")
		frame_index = frame_indices[particle_id]
		source = series.files[frame_index].name if frame_index >= 0 else ""
		writer.writerow(
			{
				"particle_id": particle_id,
				"generation": int(generations[particle_id]),
				"step": step,
				"time": f"{time:.9g}",
				"x": f"{position[0]:.9g}",
				"y": f"{position[1]:.9g}",
				"u": f"{velocity[particle_id, 0]:.9g}" if np.isfinite(velocity[particle_id, 0]) else "",
				"v": f"{velocity[particle_id, 1]:.9g}" if np.isfinite(velocity[particle_id, 1]) else "",
				"status": status,
				"vtu_frame": int(frame_index) if frame_index >= 0 else "",
				"vtu_file": source,
			}
		)


def read_trajectory_csv(csv_path: Path) -> dict[int, dict[str, list]]:
	trajectories: dict[tuple[int, int], dict[str, list]] = {}
	with csv_path.open(newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			x = float(row["x"])
			y = float(row["y"])
			if not np.isfinite(x) or not np.isfinite(y):
				continue
			particle_id = int(row["particle_id"])
			generation = int(row.get("generation", 0) or 0)
			trajectory = trajectories.setdefault((particle_id, generation), {"x": [], "y": [], "time": [], "status": []})
			trajectory["x"].append(x)
			trajectory["y"].append(y)
			trajectory["time"].append(float(row["time"]))
			trajectory["status"].append(row["status"])
	return trajectories


def animate_trajectories(
	geometry: MeshGeometry,
	csv_path: Path,
	output_gif: Path,
	title: str,
	show_mesh: bool,
	max_mesh_edges: int,
	mesh_color: str,
	mesh_linewidth: float,
	mesh_alpha: float,
	border_color: str,
	border_linewidth: float,
	border_alpha: float,
	fps: int,
	max_gif_frames: int,
	gif_stride: int,
	dpi: int,
	trail_duration: float,
) -> None:
	trajectories = read_trajectory_csv(csv_path)
	if not trajectories:
		raise ValueError(f"No trajectory rows found in {csv_path}")

	output_gif.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)

	triangulation = geometry.triangulation
	if show_mesh:
		if max_mesh_edges > 0 and len(triangulation.triangles) > max_mesh_edges:
			step = max(1, len(triangulation.triangles) // max_mesh_edges)
			mesh_triangulation = mtri.Triangulation(
				geometry.xy[:, 0],
				geometry.xy[:, 1],
				triangulation.triangles[::step],
			)
		else:
			mesh_triangulation = triangulation
		ax.triplot(
			mesh_triangulation,
			color=mesh_color,
			linewidth=mesh_linewidth,
			alpha=mesh_alpha,
			zorder=1,
		)
	border_segments = boundary_line_segments(geometry)
	ax.add_collection(
		LineCollection(
			border_segments,
			colors=border_color,
			linewidths=border_linewidth,
			alpha=border_alpha,
			zorder=2,
		)
	)

	cmap = plt.get_cmap("viridis")
	particle_ids = sorted(trajectories)
	particle_ids = [particle_id for particle_id in particle_ids if trajectories[particle_id]["x"]]
	if not particle_ids:
		raise ValueError(f"No finite trajectory coordinates found in {csv_path}")
	denom = max(1, len(particle_ids) - 1)
	all_times_unique = np.unique(
		np.concatenate([np.asarray(trajectories[particle_id]["time"], dtype=float) for particle_id in particle_ids])
	)
	if gif_stride > 1:
		frame_times = all_times_unique[::gif_stride]
	else:
		frame_count = len(all_times_unique) if max_gif_frames <= 0 else min(len(all_times_unique), max_gif_frames)
		frame_times = np.linspace(float(all_times_unique[0]), float(all_times_unique[-1]), frame_count)
	if len(frame_times) == 0:
		raise ValueError(f"No animation frames found in {csv_path}")

	lines = []
	for index, particle_id in enumerate(particle_ids):
		color = cmap(index / denom)
		(line,) = ax.plot([], [], color=color, linewidth=1.4, alpha=0.85, zorder=3)
		lines.append((particle_id, line))

	active_points = ax.scatter([], [], s=24, color="#2563eb", edgecolor="white", linewidth=0.35, zorder=5)
	exited_points = ax.scatter([], [], s=34, marker="x", color="#dc2626", linewidth=1.2, zorder=6)
	time_label = ax.text(
		0.02,
		0.98,
		"",
		transform=ax.transAxes,
		ha="left",
		va="top",
		color="#111827",
		bbox={"facecolor": "white", "edgecolor": "#d0d5dd", "alpha": 0.85, "pad": 4},
		zorder=7,
	)

	all_x = geometry.xy[:, 0]
	all_y = geometry.xy[:, 1]
	finite_bounds = np.isfinite(all_x) & np.isfinite(all_y)
	all_x = all_x[finite_bounds]
	all_y = all_y[finite_bounds]
	if len(all_x) == 0:
		raise ValueError("No finite mesh coordinates found for plot bounds.")
	x_pad = max(float(np.ptp(all_x)) * 0.02, 10.0)
	y_pad = max(float(np.ptp(all_y)) * 0.02, 10.0)
	ax.set_xlim(float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad))
	ax.set_ylim(float(np.min(all_y) - y_pad), float(np.max(all_y) + y_pad))

	def update(frame_time: float):
		active_offsets = []
		exited_offsets = []
		for particle_id, line in lines:
			trajectory = trajectories[particle_id]
			times = np.asarray(trajectory["time"], dtype=float)
			x_all = np.asarray(trajectory["x"], dtype=float)
			y_all = np.asarray(trajectory["y"], dtype=float)
			visible = times <= frame_time + 1e-12
			if trail_duration > 0.0:
				visible &= times >= frame_time - trail_duration - 1e-12
			x = x_all[visible]
			y = y_all[visible]
			line.set_data(x, y)
			if len(x) == 0:
				continue
			point = [float(x[-1]), float(y[-1])]
			last_time = float(times[-1])
			if frame_time >= last_time and trajectory["status"][-1] == "exited":
				exited_offsets.append(point)
			else:
				active_offsets.append(point)

		active_points.set_offsets(np.asarray(active_offsets, dtype=float) if active_offsets else np.empty((0, 2)))
		exited_points.set_offsets(np.asarray(exited_offsets, dtype=float) if exited_offsets else np.empty((0, 2)))
		if trail_duration > 0.0:
			time_label.set_text(f"t = {frame_time:.3g} | trail = {trail_duration:.3g}s")
		else:
			time_label.set_text(f"t = {frame_time:.3g} | full trails")
		return [line for _, line in lines] + [active_points, exited_points, time_label]

	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.2)
	ani = animation.FuncAnimation(fig, update, frames=frame_times, interval=1000 / max(1, fps), blit=False)
	ani.save(output_gif, writer=animation.PillowWriter(fps=max(1, fps)), dpi=max(40, dpi))
	plt.close(fig)


def simulate(args: argparse.Namespace) -> None:
	rng = np.random.default_rng(args.seed)
	files = resolve_vtu_files(args.vtu_glob)
	map_name = args.map_name or infer_map_name(files, args.vtu_glob)
	default_csv, default_gif = default_output_paths(map_name)
	output_csv = args.output_csv or default_csv
	output_gif = args.output_gif or default_gif

	geometry, first_velocity = load_geometry(files[0])
	series = VtuVelocitySeries(files, geometry, first_velocity, args.field_dt, args.cache_size)

	positions, seed_description = seed_particles(geometry, args, rng, args.particles, files[0])
	active = inside_mesh(geometry, positions)
	if not np.any(active):
		raise ValueError("No seeded particles are inside the mesh.")
	generations = np.zeros(args.particles, dtype=int)
	release_count = args.release_particles if args.release_particles is not None else args.particles
	next_release_time = args.release_interval if args.release_interval > 0.0 else math.inf
	release_generation = 0
	total_released = len(positions)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="") as handle:
		fieldnames = ["particle_id", "generation", "step", "time", "x", "y", "u", "v", "status", "vtu_frame", "vtu_file"]
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()

		initial_velocity, initial_frames = series.velocity(positions, 0.0)
		write_rows(writer, 0, 0.0, positions, active, active, initial_velocity, initial_frames, generations, series)

		for step in range(1, args.steps + 1):
			previous_active = active.copy()
			positions, active, velocity, frame_indices = rk2_random_walk_step(
				series=series,
				positions=positions,
				active=active,
				time=(step - 1) * args.dt,
				dt=args.dt,
				diffusivity=args.diffusivity,
				rng=rng,
			)
			current_time = step * args.dt
			should_write_step = args.output_stride <= 1 or step % args.output_stride == 0 or step == args.steps
			if should_write_step:
				write_rows(writer, step, current_time, positions, active, previous_active, velocity, frame_indices, generations, series)

			while current_time + 1e-12 >= next_release_time:
				release_generation += 1
				new_positions, _ = seed_particles(geometry, args, rng, release_count, files[0])
				new_active = inside_mesh(geometry, new_positions)
				new_velocity, new_frame_indices = series.velocity(new_positions, current_time)
				start_index = len(positions)
				new_indices = np.arange(start_index, start_index + len(new_positions), dtype=int)

				positions = np.vstack([positions, new_positions])
				active = np.concatenate([active, new_active])
				generations = np.concatenate([generations, np.full(len(new_positions), release_generation, dtype=int)])

				velocity_for_release = np.full_like(positions, np.nan, dtype=float)
				frame_indices_for_release = np.full(len(positions), -1, dtype=int)
				velocity_for_release[new_indices] = new_velocity
				frame_indices_for_release[new_indices] = new_frame_indices
				write_rows(
					writer,
					step,
					current_time,
					positions,
					active,
					active,
					velocity_for_release,
					frame_indices_for_release,
					generations,
					series,
					particle_indices=new_indices,
					status_override="released",
				)
				total_released += len(new_positions)
				next_release_time += args.release_interval

			if not np.any(active) and args.release_interval <= 0.0:
				break

	print(f"Map name: {map_name}")
	print(f"Wrote {output_csv}")
	print(f"Seeded {args.particles} particles from {seed_description}")
	if args.release_interval > 0.0:
		print(f"Timed releases enabled: added {release_count} particles every {args.release_interval:g} seconds.")
		print(f"Total particles released: {total_released}")
	print(f"Active at finish: {int(np.count_nonzero(active))}/{len(active)}")
	if args.generate_gif:
		animate_trajectories(
			geometry=geometry,
			csv_path=output_csv,
			output_gif=output_gif,
			title=f"Lagrangian particle trajectories from {seed_description}",
			show_mesh=args.gif_show_mesh,
			max_mesh_edges=args.gif_mesh_edges,
			mesh_color=args.gif_mesh_color,
			mesh_linewidth=args.gif_mesh_linewidth,
			mesh_alpha=args.gif_mesh_alpha,
			border_color=args.gif_border_color,
			border_linewidth=args.gif_border_linewidth,
			border_alpha=args.gif_border_alpha,
			fps=args.gif_fps,
			max_gif_frames=args.gif_max_frames,
			gif_stride=args.gif_stride,
			dpi=args.gif_dpi,
			trail_duration=args.gif_trail_duration,
		)
		print(f"Wrote {output_gif}")


def build_parser(config_defaults: Optional[dict] = None) -> argparse.ArgumentParser:
	config_defaults = config_defaults or {}
	has_config_value = set(config_defaults)

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=None, help="JSON config file. Command-line options override config values.")
	parser.add_argument("--vtu-glob", required="vtu_glob" not in has_config_value, help="Glob for VTU velocity files.")
	parser.add_argument("--map-name", default=None, help="Map name used for default output folder and filenames. Inferred from the VTU path when omitted.")
	parser.add_argument("--output-csv", type=Path, default=None, help="Output trajectory CSV path. Defaults to lagrangian_sim/outputs/<map_name>/<map_name>_lagrangian_particle_trajectories.csv.")
	parser.add_argument("--output-stride", type=int, default=1, help="Write trajectory rows every N simulation steps. Releases are always written.")
	parser.add_argument("--particles", type=int, default=100, help="Number of particles to seed.")
	parser.add_argument("--steps", type=int, default=300, help="Maximum number of advection steps.")
	parser.add_argument("--dt", type=float, default=1.0, help="Particle integration timestep in seconds.")
	parser.add_argument("--field-dt", type=float, default=1.0, help="Time spacing between consecutive VTU files in seconds.")
	parser.add_argument("--diffusivity", type=float, default=0.0, help="Random-walk diffusivity D in m^2/s.")
	parser.add_argument("--start-x", type=float, default=300.0, help="Initial particle x coordinate.")
	parser.add_argument("--start-y", type=float, default=280.0, help="Initial particle y coordinate.")
	parser.add_argument("--seed-fixed", action="store_true", help="Seed all particles at --start-x/--start-y instead of the random rectangle.")
	parser.add_argument("--seed-upstream", action="store_true", help="Seed particles along the inferred upstream mesh edge instead of the random rectangle.")
	parser.add_argument("--seed-x-min", type=float, default=25.0, help="Minimum x for random rectangle seeding.")
	parser.add_argument("--seed-x-max", type=float, default=500.0, help="Maximum x for random rectangle seeding.")
	parser.add_argument("--seed-y-min", type=float, default=200.0, help="Minimum y for random rectangle seeding.")
	parser.add_argument("--seed-y-max", type=float, default=300.0, help="Maximum y for random rectangle seeding.")
	parser.add_argument("--release-interval", type=float, default=0.0, help="Add a new particle batch every N simulation seconds. Use 0 to disable timed releases.")
	parser.add_argument("--release-particles", type=int, default=None, help="Particles per timed release. Defaults to --particles.")
	parser.add_argument("--upstream-band-width", type=float, default=None, help="Width of the upstream seeding band in mesh units.")
	parser.add_argument("--seed-jitter", type=float, default=0.0, help="Gaussian jitter applied to seed points in mesh units.")
	parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible random walk.")
	parser.add_argument("--cache-size", type=int, default=4, help="Number of VTU frames to keep in memory.")
	parser.add_argument("--gif", action="store_true", dest="generate_gif", help="Write an animated GIF of the simulated trajectories.")
	parser.add_argument("--plot", action="store_true", dest="generate_gif", help=argparse.SUPPRESS)
	parser.add_argument("--output-gif", type=Path, default=None, help="Output path for --gif. Defaults to lagrangian_sim/outputs/<map_name>/<map_name>_lagrangian_particle_trajectories.gif.")
	parser.add_argument("--gif-fps", type=int, default=10, help="Frames per second for --gif.")
	parser.add_argument("--gif-max-frames", type=int, default=80, help="Maximum animation frames to render. Use 0 to render every simulation step.")
	parser.add_argument("--gif-stride", type=int, default=1, help="Render every Nth simulation step. Values greater than 1 override --gif-max-frames.")
	parser.add_argument("--gif-trail-duration", type=float, default=60.0, help="Seconds of trajectory history to show in the GIF. Use 0 to keep full trails.")
	parser.add_argument("--gif-dpi", type=int, default=90, help="GIF render DPI.")
	parser.add_argument("--gif-show-mesh", action="store_true", help="Draw interior mesh triangle lines. By default only map borders are drawn.")
	parser.add_argument("--gif-mesh-edges", type=int, default=1500, help="Approximate maximum number of mesh triangles to draw in the GIF. Use 0 for all.")
	parser.add_argument("--gif-mesh-color", default="#667085", help="Mesh line color for the GIF.")
	parser.add_argument("--gif-mesh-linewidth", type=float, default=0.45, help="Mesh line width for the GIF.")
	parser.add_argument("--gif-mesh-alpha", type=float, default=0.75, help="Mesh line opacity for the GIF.")
	parser.add_argument("--gif-border-color", default="#101828", help="Map border line color for the GIF.")
	parser.add_argument("--gif-border-linewidth", type=float, default=1.4, help="Map border line width for the GIF.")
	parser.add_argument("--gif-border-alpha", type=float, default=0.95, help="Map border line opacity for the GIF.")
	parser.add_argument("--plot-mesh-edges", type=int, dest="gif_mesh_edges", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
	parser.set_defaults(**config_defaults)
	return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
	argv_list = list(argv) if argv is not None else None
	config_parser = argparse.ArgumentParser(add_help=False)
	config_parser.add_argument("--config", type=Path, default=None)
	config_args, _ = config_parser.parse_known_args(argv_list)

	config_defaults = load_config_file(config_args.config) if config_args.config is not None else {}
	parser = build_parser(config_defaults)
	valid_dests = {action.dest for action in parser._actions}
	unknown_keys = sorted(set(config_defaults) - valid_dests)
	if unknown_keys:
		raise ValueError(f"Unknown config option(s): {', '.join(unknown_keys)}")

	args = parser.parse_args(argv_list)
	if args.output_csv is not None:
		args.output_csv = Path(args.output_csv)
	if args.output_gif is not None:
		args.output_gif = Path(args.output_gif)
	if args.dt <= 0.0:
		raise ValueError("--dt must be positive.")
	if args.steps < 0:
		raise ValueError("--steps must be non-negative.")
	if args.diffusivity < 0.0:
		raise ValueError("--diffusivity must be non-negative.")
	if args.release_interval < 0.0:
		raise ValueError("--release-interval must be non-negative.")
	if args.release_particles is not None and args.release_particles <= 0:
		raise ValueError("--release-particles must be positive.")
	if args.gif_trail_duration < 0.0:
		raise ValueError("--gif-trail-duration must be non-negative.")
	if args.output_stride <= 0:
		raise ValueError("--output-stride must be positive.")
	simulate(args)


if __name__ == "__main__":
	main()
