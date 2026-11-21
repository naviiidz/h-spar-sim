"""Particle simulation behind the particle simulator UI (no Tk here).

Loads a velocity field from velocity_fields/<name>/raw, splits the mesh boundary
into edges, seeds particles along chosen edges and runs the Lagrangian
simulation of lagrangian_sim (RK2 midpoint + random walk). The CSV it writes
has the same columns as lagrangian_sim, so particle_sampling reads it unchanged.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lagrangian_sim"))
import simulate_lagrangian_particles as lsim  # noqa: E402

FIELDS_DIR = REPO_ROOT / "velocity_fields"
OUTPUT_DIR = REPO_ROOT / "lagrangian_sim" / "outputs"
CORNER_DEG = 35.0  # the boundary is split into edges where it turns more than this
CSV_FIELDS = ["particle_id", "generation", "step", "time", "x", "y", "u", "v", "status", "vtu_frame", "vtu_file"]
MAX_VIEW_FRAMES = 1500  # snapshots kept for playback


def list_fields() -> list[str]:
    """Names under velocity_fields/ that have VTU velocity snapshots."""
    out = []
    for raw in sorted(FIELDS_DIR.glob("*/raw")):
        if (raw / "Velocity2d.pvd").is_file() or any((raw / "Velocity2d").glob("*.vtu")):
            out.append(raw.parent.name)
    return out


def read_pvd(pvd: Path) -> tuple[list[Path], np.ndarray]:
    """Snapshot files and their timestep values, in time order."""
    rows = []
    for ds in ET.parse(pvd).getroot().iter("DataSet"):
        rows.append((float(ds.get("timestep", len(rows))), pvd.parent / ds.get("file")))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], np.array([r[0] for r in rows])


def field_info(name: str) -> dict:
    """What the flow generator recorded for this field (empty for hand-made fields)."""
    path = FIELDS_DIR / name / "region.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("sim", {})
    except (OSError, ValueError):
        return {}


def default_field_dt(name: str, times: np.ndarray) -> float:
    """Seconds between snapshots: from the flow generator, else from the .pvd timesteps."""
    dt = field_info(name).get("export_interval_s")
    if dt:
        return float(dt)
    d = np.diff(times)
    if len(d) and d[0] > 0 and np.allclose(d, d[0]):
        return float(d[0])
    return 1.0


def _cropped_from(name: str) -> dict | None:
    path = FIELDS_DIR / name / "region.json"
    try:
        return json.loads(path.read_text()).get("cropped_from") if path.is_file() else None
    except (OSError, ValueError):
        return None


@dataclass
class Edge:
    points: np.ndarray   # (k + 1, 2) polyline along the boundary, water on the left
    normals: np.ndarray  # (k, 2) outward unit normal of each segment
    kind: str = "shore"  # "inflow", "outflow" or "shore", from the flow through the edge

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def probes(self) -> tuple[np.ndarray, np.ndarray]:
        """Points just inside the water at each segment's middle, and the segment lengths."""
        mid = 0.5 * (self.points[:-1] + self.points[1:])
        seg_len = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return mid - self.normals * np.minimum(0.5, 0.1 * seg_len)[:, None], seg_len


def boundary_edges(tri, corner_deg: float = CORNER_DEG) -> list[Edge]:
    """Boundary loops of the mesh, split into edges at corners."""
    T = tri.triangles
    xy = np.column_stack([tri.x, tri.y])
    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    third = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
    _, inv, cnt = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
    on_boundary = cnt[inv.ravel()] == 1
    be, bc = e[on_boundary], third[on_boundary]
    # orient each boundary segment so the water (its triangle) is on the left
    a, b, c = xy[be[:, 0]], xy[be[:, 1]], xy[bc]
    flip = np.cross(b - a, c - a) < 0
    be[flip] = be[flip][:, ::-1]

    nxt: dict[int, list[int]] = {}
    for s, t in be:
        nxt.setdefault(int(s), []).append(int(t))
    edges: list[Edge] = []
    while nxt:
        start = node = next(iter(nxt))
        loop = []
        while node in nxt:
            loop.append(node)
            outs = nxt[node]
            nxt_node = outs.pop()
            if not outs:
                del nxt[node]
            node = nxt_node
            if node == start:
                break
        if len(loop) >= 3:
            edges.extend(_split_loop(xy[loop], corner_deg))
    return edges


def _split_loop(P: np.ndarray, corner_deg: float) -> list[Edge]:
    n = len(P)
    seg = np.roll(P, -1, axis=0) - P  # segment i: P[i] -> P[i + 1]
    ang = np.degrees(np.abs(np.arctan2(np.cross(np.roll(seg, 1, axis=0), seg),
                                       np.einsum("ij,ij->i", np.roll(seg, 1, axis=0), seg))))
    corners = np.flatnonzero(ang > corner_deg)  # vertex i starts a new edge
    if len(corners) == 0:
        corners = np.array([0])
    out = []
    for k, c0 in enumerate(corners):
        c1 = corners[(k + 1) % len(corners)]
        idx = np.arange(c0, c1 + (n if c1 <= c0 else 0) + 1) % n
        pts = P[idx]
        d = np.diff(pts, axis=0)
        normals = np.column_stack([d[:, 1], -d[:, 0]]) / np.linalg.norm(d, axis=1)[:, None]
        out.append(Edge(points=pts, normals=normals))
    return out


class Field:
    """A velocity field: mesh, snapshots and boundary edges."""

    def __init__(self, name: str, field_dt: float | None = None):
        self.name = name
        raw = FIELDS_DIR / name / "raw"
        pvd = raw / "Velocity2d.pvd"
        if pvd.is_file():
            self.files, times = read_pvd(pvd)
        else:
            self.files = lsim.resolve_vtu_files(str(raw / "Velocity2d" / "*.vtu"))
            times = np.arange(len(self.files), dtype=float)
        self.field_dt = float(field_dt or default_field_dt(name, times))
        self.geometry, self._first = lsim.load_geometry(self.files[0])
        info = field_info(name)
        self.ramp_s = float(info.get("ramp_s", 0.0))
        # inflow varied periodically by the flow generator: spin-up is judged by periodicity
        self.variation = float(info.get("inflow_variation", 0.0))
        self.period_s = float(info.get("variation_period_s", 0.0))
        self.cropped_from = _cropped_from(name)  # set for fields written by crop()
        self.edges = self._split_by_flow(boundary_edges(self.geometry.triangulation))

    @property
    def duration(self) -> float:
        return (len(self.files) - 1) * self.field_dt

    def new_series(self) -> lsim.VtuVelocitySeries:
        """Snapshot reader for one thread (its frame cache is not thread-safe)."""
        return lsim.VtuVelocitySeries(self.files, self.geometry, self._first, self.field_dt)

    def _inward_speed(self, series, edge: Edge, t: float) -> np.ndarray:
        probe, _ = edge.probes()
        v, _ = series.velocity(probe, t)
        return -np.einsum("ij,ij->i", np.nan_to_num(v), edge.normals)

    def _split_by_flow(self, edges: list[Edge]) -> list[Edge]:
        """Split edges where the flow through them changes between inflow, outflow and shore.

        Uses the mean over a few snapshots from the second half of the field,
        after the spin-up. Runs of a single segment are merged into their
        neighbours so corner noise does not create tiny edges.
        """
        series = self.new_series()
        times = np.linspace(0.5 * self.duration, self.duration, 5)
        un = [np.mean([self._inward_speed(series, e, t) for t in times], axis=0) for e in edges]
        allun = np.concatenate(un)
        thr = max(0.05, 0.2 * np.percentile(np.abs(allun), 99))
        out = []
        for e, u in zip(edges, un):
            cls = np.where(u > thr, 1, np.where(u < -thr, -1, 0))
            for i in range(1, len(cls) - 1):  # drop one-segment blips
                if cls[i] != cls[i - 1] and cls[i] != cls[i + 1]:
                    cls[i] = cls[i - 1]
            cuts = [0, *(np.flatnonzero(np.diff(cls)) + 1).tolist(), len(cls)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                kind = {1: "inflow", -1: "outflow", 0: "shore"}[int(cls[a])] if b - a > 1 else "shore"
                out.append(Edge(points=e.points[a:b + 1], normals=e.normals[a:b], kind=kind))
        return out

    def edge_inflow(self, series, t: float) -> np.ndarray:
        """Mean inward normal speed (m/s) through each edge at field time t."""
        out = np.zeros(len(self.edges))
        for i, e in enumerate(self.edges):
            un = self._inward_speed(series, e, t)
            _, seg_len = e.probes()
            out[i] = float((un * seg_len).sum() / max(seg_len.sum(), 1e-12))
        return out

    def spin_up(self, series, max_samples: int = 600, tol: float = 0.02) -> "SpinUp":
        """When the flow has settled after the start from rest.

        The domain-mean speed is sampled over the whole field. Its settled level is
        the mean over the last 30 % of the record. The spin-up ends at the first time
        after which the mean speed stays within `tol` of that level for the rest of
        the record (never before the boundary ramp-up ends). If the settled flow
        itself varies by more than `tol`, the band is widened to that variation.
        """
        n = int(min(max_samples, len(self.files)))
        t = np.linspace(0.0, self.duration, n)
        speed = np.array([np.nanmean(np.linalg.norm(series.velocity(self.geometry.xy, ti)[0], axis=1)) for ti in t])
        tail = speed[int(0.7 * n):]
        level = float(np.mean(tail)) if len(tail) else 0.0
        if level <= 0 or self.cropped_from:  # nothing to detect, or already cropped
            return SpinUp(t, speed, level, tol, 0.0)
        if self.variation > 0 and self.period_s > 0 and self.duration > 2 * self.period_s:
            return self._spin_up_periodic(series, t, speed, level, tol, max_samples)
        band = max(tol, 1.1 * float(np.max(np.abs(tail - level))) / level)
        outside = np.flatnonzero(np.abs(speed - level) > band * level)
        t_spin = float(t[min(outside[-1] + 1, n - 1)]) if len(outside) else 0.0
        t_spin = math.ceil(max(t_spin, self.ramp_s) / self.field_dt - 1e-9) * self.field_dt
        return SpinUp(t, speed, level, band, min(t_spin, self.duration))

    def _spin_up_periodic(self, series, t, speed, level, tol, max_samples) -> "SpinUp":
        """Spin-up for a periodically varying inflow: the flow must repeat one period later.

        mismatch(t) = mean over the nodes of |u(t + P) - u(t)|, relative to the mean speed.
        The spin-up ends at the first time after which the mismatch stays below `tol`.
        """
        P = self.period_s
        n = int(min(max_samples, (self.duration - P) / self.field_dt + 1))
        tm = np.linspace(0.0, self.duration - P, n)
        xy = self.geometry.xy
        mis = np.array([np.nanmean(np.linalg.norm(series.velocity(xy, ti + P)[0] - series.velocity(xy, ti)[0], axis=1))
                        for ti in tm]) / level
        above = np.flatnonzero(mis > tol)
        if len(above) and above[-1] == n - 1:
            t_spin = self.duration  # never repeats within the record
        else:
            t_spin = float(tm[above[-1] + 1]) if len(above) else 0.0
        t_spin = math.ceil(max(t_spin, self.ramp_s) / self.field_dt - 1e-9) * self.field_dt
        return SpinUp(t, speed, level, tol, min(t_spin, self.duration), mismatch_t=tm, mismatch=mis, period_s=P)

    def crop(self, t_start: float, name: str | None = None) -> Path:
        """Write a new field that starts at field time t_start (e.g. after the spin-up).

        Snapshots before t_start are dropped and the rest renumbered from 0, so every
        reader (particle simulator, Gazebo current, vtu_converter) starts after the
        spin-up. The .vtu files are hard links to the originals (no extra disk space;
        copies if links are not possible). Returns the new field directory.
        """
        k0 = int(round(t_start / self.field_dt))
        if not 0 < k0 < len(self.files) - 1:
            raise ValueError(f"crop time must be between {self.field_dt:g} and {self.duration - self.field_dt:g} s")
        name = name or f"{self.name}_from{k0 * self.field_dt:g}s"
        out = FIELDS_DIR / name
        if out.exists():
            raise FileExistsError(f"{out} already exists")
        vtu_dir = out / "raw" / "Velocity2d"
        vtu_dir.mkdir(parents=True)
        lines = ['<?xml version="1.0" ?>',
                 '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">', "<Collection>"]
        for k, src in enumerate(self.files[k0:]):
            dst = vtu_dir / f"Velocity2d_{k}.vtu"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            lines.append(f'<DataSet timestep="{k * self.field_dt:g}" file="Velocity2d/{dst.name}" />')
        lines += ["</Collection>", "</VTKFile>", ""]
        (out / "raw" / "Velocity2d.pvd").write_text("\n".join(lines))
        src_info = FIELDS_DIR / self.name / "region.json"
        info = json.loads(src_info.read_text()) if src_info.is_file() else {"sim": {}}
        info.setdefault("sim", {})
        info["sim"]["export_interval_s"] = self.field_dt
        info["sim"]["ramp_s"] = 0.0
        info["output_name"] = name
        info["cropped_from"] = {"field": self.name, "start_s": k0 * self.field_dt,
                                "note": "snapshot k of this field = snapshot k + "
                                        f"{k0} of {self.name}"}
        (out / "region.json").write_text(json.dumps(info, indent=2))
        return out


@dataclass
class SpinUp:
    t: np.ndarray       # sample times (s)
    speed: np.ndarray   # domain-mean speed (m/s)
    level: float        # settled mean speed (mean of the last 30 %)
    band: float         # relative band the speed stays within after t_spin
    t_spin: float       # end of the spin-up (s, a whole number of snapshots)
    mismatch_t: np.ndarray | None = None  # periodic inflow only: sample times
    mismatch: np.ndarray | None = None    # relative change of the flow over one period
    period_s: float = 0.0

    @property
    def periodic(self) -> bool:
        return self.mismatch is not None


@dataclass
class Settings:
    per_release: int = 100      # particles per release
    interval_s: float = 10.0    # time between releases (0 = only one release at t = 0)
    placement: str = "even"     # "even" or "random" along the chosen edges
    offset_m: float = 1.0       # distance from the edge into the water
    duration_s: float = 600.0   # simulated time
    dt_s: float = 1.0           # particle time step
    diffusivity: float = 0.05   # random-walk diffusivity D (m²/s)
    field_start_s: float = 0.0  # field time at particle time 0 (skips the spin-up)
    seed: int = 1
    output_stride: int = 1      # write CSV rows every N steps

    FIELDS = (  # (attribute, label, minimum, must be > minimum)
        ("per_release", "Particles per release", 0, True),
        ("interval_s", "Release every (s, 0 = once)", 0.0, False),
        ("offset_m", "Start distance from edge (m)", 0.0, False),
        ("duration_s", "Simulated time (s)", 0.0, True),
        ("dt_s", "Time step (s)", 0.0, True),
        ("diffusivity", "Diffusivity D (m²/s, 0 = off)", 0.0, False),
        ("field_start_s", "Field time at start (s)", 0.0, False),
        ("seed", "Random seed", 0, False),
        ("output_stride", "Write CSV every N steps", 0, True),
    )

    @property
    def steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    @property
    def releases(self) -> int:
        return 1 if self.interval_s <= 0 else int(math.floor(self.duration_s / self.interval_s + 1e-9)) + 1


def seed_on_edges(field: Field, edge_ids, n: int, placement: str, offset: float, rng) -> np.ndarray:
    """n points along the chosen edges, `offset` metres into the water."""
    a = np.vstack([field.edges[i].points[:-1] for i in edge_ids])
    b = np.vstack([field.edges[i].points[1:] for i in edge_ids])
    nrm = np.vstack([field.edges[i].normals for i in edge_ids])
    seg = np.linalg.norm(b - a, axis=1)
    cum = np.cumsum(seg)
    total = cum[-1]
    s = (np.arange(n) + 0.5) / n * total if placement == "even" else np.sort(rng.uniform(0.0, total, n))
    k = np.minimum(np.searchsorted(cum, s, side="right"), len(seg) - 1)
    frac = (s - (cum[k] - seg[k])) / np.maximum(seg[k], 1e-12)
    on_edge = a[k] + (b[k] - a[k]) * frac[:, None]
    pts = on_edge - nrm[k] * offset
    inside = lsim.inside_mesh(field.geometry, pts)
    for shrink in (0.5, 0.25, 0.1, 0.01):  # pull back points that landed outside (sharp corners)
        if inside.all():
            break
        retry = on_edge[~inside] - nrm[k][~inside] * max(offset * shrink, 1e-3)
        pts[~inside] = retry
        inside = lsim.inside_mesh(field.geometry, pts)
    return pts[inside]


@dataclass
class Snapshot:
    time: float           # particle time (s); the field time is field_start_s + time
    xy: np.ndarray        # positions of active particles
    generation: np.ndarray
    released: int
    exited: int


class Simulation:
    """Runs in a worker thread; the UI reads `snapshots` and `progress`."""

    def __init__(self, field: Field, edge_ids: list[int], s: Settings, csv_path: Path):
        self.field, self.edge_ids, self.s, self.csv_path = field, list(edge_ids), s, Path(csv_path)
        self.snapshots: list[Snapshot] = []
        self.progress = 0.0
        self.error: str | None = None
        self.stop = threading.Event()
        self.view_stride = max(1, math.ceil(s.steps / MAX_VIEW_FRAMES))

    def settings_path(self) -> Path:
        return self.csv_path.with_suffix(".json")

    def save_settings(self):
        d = {
            "field": self.field.name,
            "field_dt_s": self.field.field_dt,
            "settings": asdict(self.s),
            "edge_ids": self.edge_ids,
            "edges": [self.field.edges[i].points.round(3).tolist() for i in self.edge_ids],
            "csv": self.csv_path.name,
            "integrator": "RK2 midpoint + random walk sqrt(2 D dt) N(0, 1); linear in space (P1) and time",
            "frame": "Gazebo world (x, y in metres); CSV time = particle time, field time = field_start_s + time",
        }
        self.settings_path().write_text(json.dumps(d, indent=2))

    def run(self, on_snapshot: Callable[[], None] | None = None):
        try:
            self._run(on_snapshot)
        except Exception as exc:  # reported in the UI
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self, on_snapshot):
        s, f = self.s, self.field
        rng = np.random.default_rng(s.seed)
        series = f.new_series()
        t0 = s.field_start_s
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_settings()

        pos = np.zeros((0, 2))
        active = np.zeros(0, dtype=bool)
        gen = np.zeros(0, dtype=int)
        exited = 0
        next_release, release_no = 0.0, 0

        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for step in range(0, s.steps + 1):
                if self.stop.is_set():
                    break
                t = step * s.dt_s
                if step > 0:
                    prev = active.copy()
                    pos, active, vel, frames = lsim.rk2_random_walk_step(
                        series, pos, active, t0 + t - s.dt_s, s.dt_s, s.diffusivity, rng)
                    exited += int(np.count_nonzero(prev & ~active))
                    if s.output_stride <= 1 or step % s.output_stride == 0 or step == s.steps:
                        lsim.write_rows(writer, step, t, pos, active, prev, vel, frames, gen, series,
                                        particle_indices=np.flatnonzero(prev))
                while next_release <= t + 1e-9:
                    new = seed_on_edges(f, self.edge_ids, s.per_release, s.placement, s.offset_m, rng)
                    idx = np.arange(len(pos), len(pos) + len(new))
                    pos = np.vstack([pos, new])
                    active = np.concatenate([active, np.ones(len(new), dtype=bool)])
                    gen = np.concatenate([gen, np.full(len(new), release_no)])
                    v_new, fr_new = series.velocity(new, t0 + t) if len(new) else (np.zeros((0, 2)), np.zeros(0, int))
                    vel_all = np.full_like(pos, np.nan)
                    fr_all = np.full(len(pos), -1, dtype=int)
                    vel_all[idx], fr_all[idx] = v_new, fr_new
                    lsim.write_rows(writer, step, t, pos, active, active, vel_all, fr_all, gen, series,
                                    particle_indices=idx, status_override="released")
                    release_no += 1
                    next_release = next_release + s.interval_s if s.interval_s > 0 else math.inf
                if step % self.view_stride == 0 or step == s.steps:
                    self.snapshots.append(Snapshot(t, pos[active].astype(np.float32), gen[active].astype(np.int32),
                                                   len(pos), exited))
                    if on_snapshot:
                        on_snapshot()
                self.progress = step / max(s.steps, 1)
