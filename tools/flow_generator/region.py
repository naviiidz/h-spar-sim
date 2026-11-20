"""Region definition shared by the UI, the mesher and the flow solver.

A region is a closed polygon in the Gazebo *world* frame. Edge i runs from
polygon[i] to polygon[i + 1] (wrapping around) and becomes gmsh physical
curve / Thetis boundary ID i + 1.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACE_ID = 1000  # gmsh physical surface tag of the water domain

# kind -> (label, explanation, unit of the value or None)
EDGE_KINDS = {
    "wall": ("Shore / wall", "no flow through the edge", None),
    "inflow": ("Inflow", "water enters at this speed", "m/s"),
    "inflow_open": ("Inflow, waves pass out", "inflow, level held at 0 so waves can leave", "m/s"),
    "outflow": ("Outflow", "water leaves at this speed", "m/s"),
    "open": ("Open water level", "water level held at this elevation", "m"),
}
DEFAULT_VALUES = {"wall": 0.0, "inflow": 1.0, "inflow_open": 1.0, "outflow": 1.0, "open": 0.0}
SPEED_KINDS = ("inflow", "inflow_open", "outflow")  # edges whose value is a speed (> 0)
LEVEL_KINDS = ("open", "inflow_open")              # edges that hold the water level
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class EdgeBC:
    kind: str = "wall"
    value: float = 0.0


@dataclass
class SimSettings:
    depth_m: float = 10.0
    mesh_size_m: float = 15.0
    duration_h: float = 1.0
    export_interval_s: float = 10.0
    timestep_s: float = 2.0
    ramp_s: float = 300.0
    viscosity_m2s: float = 1.0
    manning_n: float = 0.0
    inflow_variation: float = 0.0     # inflow/outflow speeds vary by ± this fraction (0 = steady)
    variation_period_s: float = 600.0  # period of that variation

    # (attribute, label, minimum allowed value, must be > minimum)
    FIELDS = (
        ("depth_m", "Water depth (m)", 0.0, True),
        ("mesh_size_m", "Mesh cell size (m)", 0.0, True),
        ("duration_h", "Simulated time (h)", 0.0, True),
        ("export_interval_s", "Save every (s)", 0.0, True),
        ("timestep_s", "Solver time step (s)", 0.0, True),
        ("ramp_s", "Boundary ramp-up (s)", 0.0, False),
        ("viscosity_m2s", "Viscosity (m²/s)", 0.0, False),
        ("manning_n", "Manning bed friction (0 = off)", 0.0, False),
        ("inflow_variation", "Inflow speed variation (± fraction, 0 = steady)", 0.0, False),
        ("variation_period_s", "Variation period (s)", 0.0, True),
    )


@dataclass
class Region:
    map_name: str
    output_name: str
    world_offset: tuple = (0.0, 0.0)  # world xy = DAE xy + offset
    polygon: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    sim: SimSettings = field(default_factory=SimSettings)

    # ---- output locations (h-spar-sim per-map layout) ----
    @property
    def mesh_dir(self) -> Path:
        return REPO_ROOT / "meshes" / self.output_name

    @property
    def fields_dir(self) -> Path:
        return REPO_ROOT / "velocity_fields" / self.output_name

    @property
    def mesh_file(self) -> Path:
        return self.mesh_dir / f"{self.output_name}.msh"

    @property
    def region_file(self) -> Path:
        return self.mesh_dir / "region.json"

    # ---- (de)serialisation ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["world_offset"] = list(self.world_offset)
        d["boundary_ids"] = {str(i + 1): e.kind for i, e in enumerate(self.edges)}
        d["frame"] = "Gazebo world (x, y in metres)"
        return d

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.region_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Region":
        d = json.loads(Path(path).read_text())
        sim_keys = {f[0] for f in SimSettings.FIELDS}
        return cls(
            map_name=d["map_name"],
            output_name=d["output_name"],
            world_offset=tuple(d.get("world_offset", (0.0, 0.0))),
            polygon=[list(map(float, p)) for p in d["polygon"]],
            edges=[EdgeBC(**e) for e in d["edges"]],
            sim=SimSettings(**{k: v for k, v in d.get("sim", {}).items() if k in sim_keys}),
        )

    # ---- checks ----
    def problems(self) -> list[str]:
        out = []
        if not NAME_RE.match(self.output_name or ""):
            out.append("Output name may only contain letters, digits, '_' and '-'.")
        if len(self.polygon) < 3:
            out.append("The region needs at least 3 points.")
        elif self_intersects(self.polygon):
            out.append("The region outline crosses itself.")
        if len(self.edges) != len(self.polygon):
            out.append("Every edge needs a boundary condition.")
        elif all(e.kind == "wall" for e in self.edges):
            out.append("All edges are walls: at least one edge must let water in or out.")
        for e in self.edges:
            if e.kind not in EDGE_KINDS:
                out.append(f"Unknown boundary type '{e.kind}'.")
            elif e.kind in SPEED_KINDS and e.value <= 0:
                out.append("Inflow/outflow speeds must be greater than 0.")
        for attr, label, lo, strict in SimSettings.FIELDS:
            v = getattr(self.sim, attr)
            if (v <= lo) if strict else (v < lo):
                out.append(f"{label} must be {'>' if strict else '>='} {lo:g}.")
        if self.sim.inflow_variation >= 1.0:
            out.append("The inflow speed variation must be below 1 (100 %).")
        if not export_fits_timestep(self.sim):
            out.append("'Save every' must be a whole multiple of the solver time step.")
        return out


def export_fits_timestep(sim: SimSettings) -> bool:
    if sim.timestep_s <= 0 or sim.export_interval_s <= 0:
        return True  # reported separately
    ratio = sim.export_interval_s / sim.timestep_s
    return ratio >= 1 - 1e-9 and abs(ratio - round(ratio)) < 1e-6


# ---------------------------------------------------------------- geometry
def edge_length(poly, i) -> float:
    (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
    return math.hypot(x1 - x0, y1 - y0)


def polygon_area(poly) -> float:
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def self_intersects(poly) -> bool:
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # neighbouring edges share a vertex
            if _segments_cross(a1, a2, poly[j], poly[(j + 1) % n]):
                return True
    return False


# ---------------------------------------------------------------- import
def polygon_from_geo(path: Path):
    """Read points (and INLET/OUTLET/LAND curve labels, if present) from a gmsh .geo file.

    Returns (polygon, edges or None). Only plain `Point(n) = {x, y, ...}` and
    `Line(n) = {a, b}` statements are understood; edges are assumed to follow
    the point order, as in the hand-written H-SPAR .geo files.
    """
    text = Path(path).read_text()
    pts = re.findall(r"Point\((\d+)\)\s*=\s*\{\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)", text)
    poly = [[float(x), float(y)] for _, x, y in sorted(pts, key=lambda p: int(p[0]))]
    if len(poly) > 1 and math.dist(poly[0], poly[-1]) < 1e-9:
        poly.pop()
    edges = None
    groups = re.findall(r'Physical\s+(?:Curve|Line)\s*\(\s*"([^"]+)"\s*,\s*\d+\s*\)\s*=\s*\{([^}]*)\}', text)
    if groups:
        edges = [EdgeBC() for _ in poly]
        for name, ids in groups:
            u = name.upper()
            kind = "inflow" if "IN" in u and "LET" in u else "open" if "OUT" in u else "wall"
            for tok in re.findall(r"\d+", ids):
                k = int(tok) - 1
                if 0 <= k < len(edges):
                    edges[k] = EdgeBC(kind, DEFAULT_VALUES[kind])
    return poly, edges
