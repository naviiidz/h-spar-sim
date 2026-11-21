#!/usr/bin/env python3
"""Build the planner lookup tables from a field written by the flow generator.

Replaces generate_occupancy_npy.py + generate_vel_npy.py for flow-generator fields
(velocity_fields/<field>/ with region.json), then runs the stream-function step:

    lookup_tables/<map>/occupancy_grid.npz/.npy, occupancy_grid_metadata.json
    lookup_tables/<map>/grid_x.npy, grid_y.npy
    lookup_tables/<map>/average_velocity_{u,v,magnitude}_lookup.npy
    lookup_tables/<map>/stream_function.npy

The velocity is the time average of the field. With a periodically varying inflow
the average is taken over a whole number of inflow periods, so it does not depend
on where in the cycle the window ends.

    python3 build_lookup_tables.py --field sydney_regatta_roi_from1326s --map sydney_regatta_roi
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolyPath

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "particle_simulator"))
import engine  # noqa: E402  (flow-generator field reader)


def occupancy(polygon: np.ndarray, resolution: float):
    """100 = land, 0 = water; cell (i, j) has its centre at min + (j + 0.5, i + 0.5) * resolution."""
    lo, hi = polygon.min(0), polygon.max(0)
    w = int(np.ceil((hi[0] - lo[0]) / resolution))
    h = int(np.ceil((hi[1] - lo[1]) / resolution))
    xs = lo[0] + (np.arange(w) + 0.5) * resolution
    ys = lo[1] + (np.arange(h) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    inside = PolyPath(polygon).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(h, w)
    grid = np.where(inside, 0, 100).astype(np.int8)
    return grid, (float(lo[0]), float(lo[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True, help="folder in velocity_fields/")
    ap.add_argument("--map", required=True, help="output folder in lookup_tables/")
    ap.add_argument("--occupancy-resolution", type=float, default=5.0, help="m per occupancy cell")
    ap.add_argument("--grid", type=int, default=300, help="velocity lookup points per axis")
    ap.add_argument("--t0", type=float, default=0.0, help="start of the averaging window (field time, s)")
    ap.add_argument("--t1", type=float, default=None,
                    help="end of the averaging window (default: the longest whole number of inflow periods)")
    args = ap.parse_args()

    field = engine.Field(args.field)
    region = json.loads((engine.FIELDS_DIR / args.field / "region.json").read_text())
    polygon = np.array(region["polygon"], dtype=float)
    out = HERE / "lookup_tables" / args.map
    out.mkdir(parents=True, exist_ok=True)

    # occupancy
    grid, origin = occupancy(polygon, args.occupancy_resolution)
    np.save(out / "occupancy_grid.npy", grid)
    np.savez_compressed(out / "occupancy_grid.npz", grid=grid, origin_x=origin[0], origin_y=origin[1],
                        resolution=args.occupancy_resolution)
    (out / "occupancy_grid_metadata.json").write_text(json.dumps({
        "origin_x": origin[0], "origin_y": origin[1], "resolution": args.occupancy_resolution,
        "width": int(grid.shape[1]), "height": int(grid.shape[0]),
        "threshold_occupied": 50, "threshold_free": 0, "field": args.field}, indent=2))
    print(f"occupancy {grid.shape}, origin {origin}, free cells {int((grid == 0).sum())}")

    # time-averaged velocity at the mesh nodes
    period = field.period_s if field.variation else 0.0
    t1 = args.t1
    if t1 is None:
        span = field.duration - args.t0
        t1 = args.t0 + (np.floor(span / period) * period if period > 0 and span >= period else span)
    k0, k1 = int(round(args.t0 / field.field_dt)), int(round(t1 / field.field_dt))
    series = field.new_series()
    xy = field.geometry.xy
    sum_v, sum_mag, n = np.zeros((len(xy), 2)), np.zeros(len(xy)), 0
    for k in range(k0, k1):  # [t0, t1): one sample per snapshot, a whole number of periods
        v, _ = series.velocity(xy, k * field.field_dt)
        v = np.nan_to_num(v)
        sum_v += v
        sum_mag += np.linalg.norm(v, axis=1)
        n += 1
    mean_v, mean_mag = sum_v / n, sum_mag / n
    print(f"averaged {n} snapshots, t = {args.t0:g} .. {t1:g} s"
          + (f" ({(t1 - args.t0) / period:g} inflow periods)" if period > 0 else ""))

    # regular lookup grid, linear within the triangles; outside the mesh the nearest node
    import matplotlib.tri as mtri
    from scipy.interpolate import griddata
    lo, hi = xy.min(0), xy.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], args.grid), np.linspace(lo[1], hi[1], args.grid), indexing="ij")
    tri = field.geometry.triangulation
    looked = []
    for vals in (mean_v[:, 0], mean_v[:, 1], mean_mag):
        g = np.ma.filled(mtri.LinearTriInterpolator(tri, vals)(gx, gy), np.nan)
        near = griddata(xy, vals, (gx, gy), method="nearest")
        looked.append(np.where(np.isfinite(g), g, near))
    np.save(out / "grid_x.npy", gx)
    np.save(out / "grid_y.npy", gy)
    for name, g in zip(("u", "v", "magnitude"), looked):
        np.save(out / f"average_velocity_{name}_lookup.npy", g)
    print(f"velocity lookup {gx.shape}, mean speed {mean_mag.mean():.3f} m/s")

    # stream function
    subprocess.run([sys.executable, str(HERE / "generate_stream_function.py"), "--map", args.map],
                   check=True, cwd=HERE, env={**os.environ, "MPLBACKEND": "Agg"})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
