#!/usr/bin/env python3
"""Solve 2D shallow-water flow on a region mesh with Thetis (Firedrake).

Run with the Firedrake environment's Python. Writes velocity snapshots in the
layout the rest of H-SPAR expects:

    velocity_fields/<name>/raw/Velocity2d.pvd
    velocity_fields/<name>/raw/Velocity2d/Velocity2d_<n>.vtu

Usage:
    python run_flow.py meshes/<name>/region.json [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region import SPEED_KINDS, Region  # noqa: E402

TOOL_MARKER = "region.json"  # present in fields_dir only when this tool made it


def check_output(region: Region, overwrite: bool):
    """Refuse to touch fields this tool did not make (no side effects)."""
    fields = region.fields_dir
    raw = fields / "raw"
    if raw.exists() and any(raw.iterdir()):
        if not (fields / TOOL_MARKER).is_file():
            sys.exit(f"[flow] {raw} already holds velocity fields that this tool did not create. "
                     "Choose another output name.")
        if not overwrite:
            sys.exit(f"[flow] {raw} already exists. Re-run with --overwrite to replace it.")


def prepare_output(region: Region, overwrite: bool) -> Path:
    check_output(region, overwrite)
    fields = region.fields_dir
    if (fields / "raw").exists():
        shutil.rmtree(fields / "raw")
        shutil.rmtree(fields / "thetis", ignore_errors=True)
    fields.mkdir(parents=True, exist_ok=True)
    region.save(fields / TOOL_MARKER)
    return fields


def collect_velocity(thetis_dir: Path, raw: Path):
    """Move Thetis' Velocity2d output into raw/ (handles both Thetis output layouts)."""
    pvds = sorted(thetis_dir.rglob("Velocity2d.pvd"))
    if not pvds:
        raise FileNotFoundError(f"No Velocity2d.pvd under {thetis_dir}")
    pvd = pvds[0]
    raw.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pvd.parent / "Velocity2d"), str(raw / "Velocity2d"))
    shutil.move(str(pvd), str(raw / "Velocity2d.pvd"))


def run(region: Region, overwrite: bool):
    check_output(region, overwrite)  # fail fast, before the slow Firedrake import
    print("[flow] loading Firedrake/Thetis ...", flush=True)
    from thetis import (Constant, Function, Mesh, get_functionspace, solver2d)

    s = region.sim
    fields = prepare_output(region, overwrite)  # only deletes old output once Firedrake has loaded
    thetis_dir = fields / "thetis"

    mesh2d = Mesh(str(region.mesh_file))
    print(f"[flow] mesh {region.mesh_file}: {mesh2d.num_vertices()} vertices, {mesh2d.num_cells()} cells",
          flush=True)

    bathymetry = Function(get_functionspace(mesh2d, "CG", 1), name="Bathymetry").assign(s.depth_m)
    solver = solver2d.FlowSolver2d(mesh2d, bathymetry)
    o = solver.options
    o.element_family = "dg-dg"
    o.swe_timestepper_type = "CrankNicolson"
    # Direct solve: robust and fast at the 2D mesh sizes used here.
    o.swe_timestepper_options.solver_parameters = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_type": "aij",
    }
    o.timestep = s.timestep_s
    o.simulation_export_time = s.export_interval_s
    o.simulation_end_time = s.duration_h * 3600.0
    o.output_directory = str(thetis_dir)
    o.fields_to_export = ["uv_2d", "elev_2d"]
    o.horizontal_viscosity = Constant(s.viscosity_m2s)
    if s.manning_n > 0:
        o.manning_drag_coefficient = Constant(s.manning_n)
    speeds = [e.value for e in region.edges if e.kind in SPEED_KINDS]
    o.horizontal_velocity_scale = Constant(max(speeds + [0.5]) * (1.0 + s.inflow_variation))

    # Boundary values ramp from 0 to full over ramp_s to avoid a start-up shock.
    ramp = Constant(1.0 if s.ramp_s <= 0 else 0.0)
    # Speeds also follow the slow variation: factor = ramp(t) * (1 + a sin(2 pi t / P)).
    speed = Constant(1.0 if s.ramp_s <= 0 else 0.0)
    bnd = {}
    for i, e in enumerate(region.edges):
        tag = i + 1
        if e.kind == "wall":
            bnd[tag] = {"un": Constant(0.0)}
        elif e.kind == "inflow":    # Thetis 'un' is positive OUT of the domain
            bnd[tag] = {"un": Constant(-e.value) * speed}
        elif e.kind == "inflow_open":
            # Level and velocity both given: the Riemann flux takes the incoming part from
            # these values and lets waves from inside leave (no trapped sloshing).
            bnd[tag] = {"elev": Constant(0.0), "un": Constant(-e.value) * speed}
        elif e.kind == "outflow":
            bnd[tag] = {"un": Constant(e.value) * speed}
        elif e.kind == "open":
            bnd[tag] = {"elev": Constant(e.value) * ramp}
    solver.bnd_functions["shallow_water"] = bnd
    for tag, bc in bnd.items():
        print(f"[flow] boundary {tag}: {region.edges[tag - 1].kind} {region.edges[tag - 1].value:g}", flush=True)

    # Bed friction uses |u| = sqrt(u.u), whose derivative is 0/0 at rest; start from a
    # negligible velocity so the first Newton step has a finite Jacobian.
    u0 = 1e-6 if s.manning_n > 0 else 0.0
    solver.assign_initial_conditions(uv=Constant((u0, u0)), elev=Constant(0.0))

    n_exports = int(math.floor(o.simulation_end_time / o.simulation_export_time + 1e-9)) + 1
    t0 = time.time()
    count = {"n": 0}

    def update_forcings(t):
        r = min(1.0, t / s.ramp_s) if s.ramp_s > 0 else 1.0
        ramp.assign(r)
        speed.assign(r * (1.0 + s.inflow_variation * math.sin(2.0 * math.pi * t / s.variation_period_s)))

    def on_export():
        count["n"] += 1
        print(f"[flow] export {count['n']}/{n_exports} t={solver.simulation_time:.0f}s "
              f"wall={time.time() - t0:.0f}s", flush=True)

    solver.iterate(update_forcings=update_forcings, export_func=on_export)

    collect_velocity(thetis_dir, fields / "raw")
    info = {"exports": count["n"], "wall_time_s": round(time.time() - t0, 1),
            "mesh": str(region.mesh_file), "thetis_output": str(thetis_dir)}
    (fields / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"[flow] done: {fields / 'raw'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("region", type=Path, help="region.json written by the flow generator UI")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace fields previously generated by this tool under the same name")
    args = ap.parse_args()

    region = Region.load(args.region)
    problems = region.problems()
    if problems:
        sys.exit("Region is not valid:\n  - " + "\n  - ".join(problems))
    if not region.mesh_file.is_file():
        sys.exit(f"[flow] mesh not found: {region.mesh_file}. Build it first (build_mesh.py).")
    run(region, args.overwrite)


if __name__ == "__main__":
    main()
