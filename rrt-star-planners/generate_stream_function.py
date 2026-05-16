'''
Author: Navid Zarrabi
Date: 2026-04-22
For SVF-RRT* we need stream function values at each occupancy grid cell to compute edge costs.
This script loads the velocity field and occupancy grid, then performs a breadth-first graph integration
Stores the stream function values in a grid as .npy and visualizes them.
'''

from __future__ import annotations


from collections import deque
from typing import Deque, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

map_name = "sydney_regatta"

LOOKUP_FOLDER = f"./lookup_tables/{map_name}"
OUTPUT_DIR = f"./lookup_tables/{map_name}"


def load_vector_field(folder: str = LOOKUP_FOLDER) -> Tuple[Tuple[RegularGridInterpolator, RegularGridInterpolator], Tuple[float, float, float, float]]:
	"""Load (u, v) lookup tables and return interpolators + (xmin, xmax, ymin, ymax)."""
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

	bounds = (
		float(x_coords.min()),
		float(x_coords.max()),
		float(y_coords.min()),
		float(y_coords.max()),
	)
	return (u_interp, v_interp), bounds


def load_occupancy_from_npz(output_dir: str = OUTPUT_DIR) -> Tuple[np.ndarray, Tuple[float, float], float]:
	"""Load occupancy grid, origin and resolution from NPZ."""
	data = np.load(f"{output_dir}/occupancy_grid.npz")
	grid = data["grid"]
	origin = (float(data["origin_x"]), float(data["origin_y"]))
	resolution = float(data["resolution"])
	return grid, origin, resolution


def choose_datum_cell(free_mask: np.ndarray) -> Tuple[int, int]:
	"""Choose a datum in free space (closest free cell to map center)."""
	h, w = free_mask.shape
	cy, cx = 200, -1000 #h // 2, w // 2
	free_indices = np.argwhere(free_mask)
	if free_indices.size == 0:
		raise ValueError("No free cells in occupancy grid.")
	d2 = (free_indices[:, 0] - cy) ** 2 + (free_indices[:, 1] - cx) ** 2
	idx = int(np.argmin(d2))
	gy, gx = free_indices[idx]
	return int(gy), int(gx)


def grid_to_world_center(gx: int, gy: int, origin: Tuple[float, float], resolution: float) -> Tuple[float, float]:
	x = origin[0] + (gx + 0.5) * resolution
	y = origin[1] + (gy + 0.5) * resolution
	return float(x), float(y)


def edge_delta_psi(
	u_interp: RegularGridInterpolator,
	v_interp: RegularGridInterpolator,
	p_from: Tuple[float, float],
	p_to: Tuple[float, float],
) -> Optional[float]:
	"""Compute dpsi along one edge using dpsi = u*dy - v*dx at edge midpoint."""
	mx = 0.5 * (p_from[0] + p_to[0])
	my = 0.5 * (p_from[1] + p_to[1])
	dx = p_to[0] - p_from[0]
	dy = p_to[1] - p_from[1]

	u = float(u_interp([[mx, my]])[0])
	v = float(v_interp([[mx, my]])[0])
	if not np.isfinite(u) or not np.isfinite(v):
		return None

	return float(u * dy - v * dx)


def compute_stream_function(
	occupancy_grid: np.ndarray,
	origin: Tuple[float, float],
	resolution: float,
	u_interp: RegularGridInterpolator,
	v_interp: RegularGridInterpolator,
	datum_cell: Optional[Tuple[int, int]] = None,
	occupied_threshold: float = 50.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
	"""
	Numerically integrate stream function from a datum point within free space.

	Integration is graph-based over 4-connected neighbors and only traverses free cells,
	so all integration paths remain inside the occupancy-grid free area.
	"""
	free_mask = occupancy_grid <= occupied_threshold
	h, w = occupancy_grid.shape

	if datum_cell is None:
		datum_cell = choose_datum_cell(free_mask)

	gy0, gx0 = datum_cell
	if gy0 < 0 or gy0 >= h or gx0 < 0 or gx0 >= w or not free_mask[gy0, gx0]:
		raise ValueError("Chosen datum cell is not in free space.")

	psi = np.full((h, w), np.nan, dtype=float)
	visited = np.zeros((h, w), dtype=bool)

	psi[gy0, gx0] = 0.0
	visited[gy0, gx0] = True
	q: Deque[Tuple[int, int]] = deque([(gy0, gx0)])

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	while q:
		gy, gx = q.popleft()
		p_from = grid_to_world_center(gx, gy, origin, resolution)

		for dgy, dgx in neighbors:
			ngy, ngx = gy + dgy, gx + dgx
			if ngy < 0 or ngy >= h or ngx < 0 or ngx >= w:
				continue
			if not free_mask[ngy, ngx]:
				continue

			p_to = grid_to_world_center(ngx, ngy, origin, resolution)
			dpsi = edge_delta_psi(u_interp, v_interp, p_from, p_to)
			if dpsi is None:
				continue

			if not visited[ngy, ngx]:
				psi[ngy, ngx] = psi[gy, gx] + dpsi
				visited[ngy, ngx] = True
				q.append((ngy, ngx))

	return psi, (gy0, gx0)


def visualize_stream_function(
	psi: np.ndarray,
	occupancy_grid: np.ndarray,
	origin: Tuple[float, float],
	resolution: float,
	datum_cell: Tuple[int, int],
	output_path: str = f"{OUTPUT_DIR}/stream_function.png",
	occupied_threshold: float = 50.0,
) -> None:
	"""Visualize stream function in free space with obstacle overlay and datum marker."""
	h, w = occupancy_grid.shape
	extent = [
		origin[0],
		origin[0] + w * resolution,
		origin[1],
		origin[1] + h * resolution,
	]

	free_mask = occupancy_grid <= occupied_threshold
	psi_masked = np.where(free_mask, psi, np.nan)

	fig, ax = plt.subplots(figsize=(14, 8))
	im = ax.imshow(psi_masked, origin="lower", extent=extent, cmap="turbo")
	plt.colorbar(im, ax=ax, label="Stream function ψ")

	# contour lines over free space
	if np.isfinite(psi_masked).sum() > 10:
		xs = origin[0] + (np.arange(w) + 0.5) * resolution
		ys = origin[1] + (np.arange(h) + 0.5) * resolution
		xx, yy = np.meshgrid(xs, ys)
		try:
			ax.contour(xx, yy, psi_masked, levels=24, colors="k", linewidths=0.4, alpha=0.6)
		except Exception:
			pass

	# obstacle overlay
	occ = np.where(~free_mask, 1.0, np.nan)
	ax.imshow(occ, origin="lower", extent=extent, cmap="gray", alpha=0.9)

	# datum marker
	gy0, gx0 = datum_cell
	x0, y0 = grid_to_world_center(gx0, gy0, origin, resolution)
	ax.scatter(x0, y0, c="red", s=120, marker="x", linewidths=2, label="Datum (ψ=0)")

	ax.set_title("Stream Function from Free-Space-Constrained Integration")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.legend(loc="upper right")
	ax.grid(True, alpha=0.2)

	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Saved stream-function visualization to {output_path}")
	plt.show()


if __name__ == "__main__":
	(u_interp, v_interp), bounds = load_vector_field()
	grid, origin, resolution = load_occupancy_from_npz()
	print(f"Loaded vector field bounds: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}]")
	print(f"Loaded occupancy grid: shape={grid.shape}, origin={origin}, res={resolution}")

	psi, datum_cell = compute_stream_function(
		occupancy_grid=grid,
		origin=origin,
		resolution=resolution,
		u_interp=u_interp,
		v_interp=v_interp,
		datum_cell=None,
	)

	n_valid = int(np.isfinite(psi).sum())
	print(f"Datum cell (gy, gx): {datum_cell}, psi(datum)=0")
	print(f"Integrated stream function on {n_valid} free cells.")

	np.save(f"{OUTPUT_DIR}/stream_function.npy", psi)
	print(f"Saved stream function grid to {OUTPUT_DIR}/stream_function.npy")

	visualize_stream_function(
		psi=psi,
		occupancy_grid=grid,
		origin=origin,
		resolution=resolution,
		datum_cell=datum_cell,
		output_path=f"{OUTPUT_DIR}/stream_function.png",
	)
