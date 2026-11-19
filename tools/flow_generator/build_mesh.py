#!/usr/bin/env python3
"""Mesh a region with gmsh.

Each polygon edge becomes its own physical curve (ID = edge number), so the
flow solver can give every edge its own boundary condition. The water surface
is physical surface 1000.

Usage:
    python3 build_mesh.py meshes/<name>/region.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region import SURFACE_ID, Region  # noqa: E402


def build_mesh(region: Region) -> Path:
    import gmsh

    lc = float(region.sim.mesh_size_m)
    region.mesh_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(region.output_name)
        geo = gmsh.model.geo
        pts = [geo.addPoint(x, y, 0.0, lc) for x, y in region.polygon]
        n = len(pts)
        lines = [geo.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        surf = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for i, (line, bc) in enumerate(zip(lines, region.edges)):
            gmsh.model.addPhysicalGroup(1, [line], tag=i + 1, name=f"edge{i + 1}_{bc.kind}")
        gmsh.model.addPhysicalGroup(2, [surf], tag=SURFACE_ID, name="water")

        gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 4)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")

        gmsh.write(str(region.mesh_file))
        gmsh.write(str(region.mesh_dir / f"{region.output_name}.geo_unrolled"))
    finally:
        gmsh.finalize()
    return region.mesh_file


def read_msh_triangles(path: Path):
    """(nodes (N, 2), triangles (M, 3) as 0-based indices) from an ASCII gmsh 2.2 file."""
    lines = Path(path).read_text().splitlines()
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    ids, xy = [], []
    for ln in lines[i + 2:i + 2 + n]:
        p = ln.split()
        ids.append(int(p[0]))
        xy.append((float(p[1]), float(p[2])))
    index = {nid: k for k, nid in enumerate(ids)}
    j = lines.index("$Elements")
    m = int(lines[j + 1])
    tris = []
    for ln in lines[j + 2:j + 2 + m]:
        p = list(map(int, ln.split()))
        if p[1] == 2:  # 3-node triangle
            ntags = p[2]
            tris.append([index[v] for v in p[3 + ntags:6 + ntags]])
    return np.array(xy), np.array(tris, dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("region", type=Path, help="region.json written by the flow generator UI")
    args = ap.parse_args()

    region = Region.load(args.region)
    problems = region.problems()
    if problems:
        sys.exit("Region is not valid:\n  - " + "\n  - ".join(problems))
    path = build_mesh(region)
    nodes, tris = read_msh_triangles(path)
    print(f"[mesh] wrote {path}")
    print(f"[mesh] {len(nodes)} nodes, {len(tris)} triangles, {len(region.edges)} boundary edges")


if __name__ == "__main__":
    main()
