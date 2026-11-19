"""Gazebo maps the flow generator knows, and how to get a top view of them.

Each map points at a Gazebo Fuel model. The model is looked up in the local
Fuel caches first and is only downloaded when it is not there.
"""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FUEL_SERVER = "https://fuel.gazebosim.org/1.0"
FUEL_CACHES = [
    Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org",
    Path.home() / ".ignition" / "fuel" / "fuel.gazebosim.org",
]
BG_CACHE = Path.home() / ".cache" / "hspar_flow_generator"


@dataclass(frozen=True)
class MapInfo:
    name: str
    owner: str
    model: str
    dae_files: tuple          # paths inside the Fuel model folder
    world_offset: tuple       # world xy = DAE xy + offset (pose of the model in the world .sdf)
    water_level_dae: float    # DAE z of the water surface (for colouring only)
    description: str


MAPS = {
    "sydney_regatta": MapInfo(
        name="sydney_regatta",
        owner="openrobotics",
        model="sydney_regatta",
        dae_files=("meshes/sydney_regatta_shore.dae",),
        world_offset=(0.0, 0.0),
        water_level_dae=-0.2,
        description="VRX Sydney Regatta. World file: vrx_gz/worlds/sydney_regatta.sdf "
                    "(model at 0 0 0.2, so world = DAE).",
    ),
    "baylands": MapInfo(
        name="baylands",
        owner="openrobotics",
        model="baylands",
        dae_files=("media/baylands_01.DAE", "media/baylands_02.DAE", "media/baylands_03.DAE"),
        world_offset=(205.0, 155.0),
        water_level_dae=0.3,
        description="Baylands park. World file: vrx_gz/worlds/baylands.sdf "
                    "(park at 205 155, so world = DAE + (205, 155)). The park itself is land.",
    ),
}


# ------------------------------------------------------------------ lookup
def _dae_paths(info: MapInfo, model_dir: Path):
    return [model_dir / f for f in info.dae_files]


def find_model_dir(info: MapInfo) -> Path | None:
    """Newest cached version of the model that has all DAE files, or None."""
    for root in FUEL_CACHES:
        if not root.is_dir():
            continue
        for owner in root.iterdir():
            if owner.name.lower() != info.owner.lower():
                continue
            models = owner / "models"
            if not models.is_dir():
                continue
            for model in models.iterdir():
                if model.name.lower() != info.model.lower():
                    continue
                versions = sorted((v for v in model.iterdir() if v.is_dir()),
                                  key=lambda v: int(v.name) if v.name.isdigit() else -1, reverse=True)
                for v in versions:
                    if all(p.is_file() for p in _dae_paths(info, v)):
                        return v
    return None


def download_model(info: MapInfo, progress=None) -> Path:
    """Download the model from Gazebo Fuel into the gz Fuel cache layout."""
    base = f"{FUEL_SERVER}/{info.owner}/models/{urllib.parse.quote(info.model)}"
    with urllib.request.urlopen(f"{base}/tip/files", timeout=60) as r:
        version = str(json.load(r).get("version", "1"))
    target = FUEL_CACHES[0] / info.owner / "models" / info.model / version
    with urllib.request.urlopen(f"{base}/tip/{urllib.parse.quote(info.model)}.zip", timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            if progress:
                progress(buf.tell(), total)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(target)
    missing = [p for p in _dae_paths(info, target) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Downloaded model is missing {missing}")
    return target


# ------------------------------------------------------------------ terrain
def load_terrain(info: MapInfo, model_dir: Path):
    """All DAE triangles as (N, 3, 3) array, shifted into the Gazebo world frame."""
    import trimesh

    tris = []
    for p in _dae_paths(info, model_dir):
        loaded = trimesh.load(str(p))
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        tris.append(np.asarray(mesh.triangles))
    t = np.concatenate(tris)
    t[:, :, 0] += info.world_offset[0]
    t[:, :, 1] += info.world_offset[1]
    return t


def background(info: MapInfo, model_dir: Path, max_px: int = 3000):
    """Top-view image of the map (cached on disk) and its extent in world metres."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    BG_CACHE.mkdir(parents=True, exist_ok=True)
    png, meta = BG_CACHE / f"{info.name}.png", BG_CACHE / f"{info.name}.json"
    if png.is_file() and meta.is_file():
        return matplotlib.image.imread(png), tuple(json.loads(meta.read_text())["extent"])

    tris = load_terrain(info, model_dir)
    z = tris[:, :, 2].mean(axis=1)
    lo = tris[:, :, :2].reshape(-1, 2).min(axis=0)
    hi = tris[:, :, :2].reshape(-1, 2).max(axis=0)
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    extent = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

    wl = info.water_level_dae
    land = z > wl + 0.05
    cmap = matplotlib.colormaps["terrain"]
    h = np.clip((z - wl) / 15.0, 0, 1)
    colors = cmap(0.28 + 0.7 * h)
    colors[~land] = matplotlib.colors.to_rgba("#8fbbe0")  # modelled ground below water level

    w, hgt = hi - lo
    scale = max_px / max(w, hgt)
    fig = Figure(figsize=(w * scale / 100, hgt * scale / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("#dfe9f2")  # no geometry (usually open water)
    ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=colors, edgecolors="none",
                                     linewidths=0, antialiased=False))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    fig.savefig(png, dpi=100, facecolor="#dfe9f2")
    meta.write_text(json.dumps({"extent": extent, "faces": int(len(tris))}))
    return matplotlib.image.imread(png), extent
