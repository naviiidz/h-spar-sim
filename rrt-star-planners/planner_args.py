"""Command-line options shared by the planners (rrt_star, vf_rrt_star, svf_rrt_star).

    --map NAME    lookup tables in lookup_tables/NAME (default sydney_regatta)
    --seed N      seed of the random sampling; outputs get the suffix _seedN
    --no-show     save the figures without opening a window
    --max-iter N  planner iterations (default: the value in each script)
    --length-factor K  VF/SVF: weight of path length, in units of the mean current speed
    --clearance M  minimum distance of the path from land (m, default 10)
"""
from __future__ import annotations

import argparse
import os

import numpy as np


# Mission of Exp. 1, shared by the three planners. The goal keeps the 10 m clearance
# from the south-facing shore at y ~ 286.
START = (-400.0, 200.0)
GOAL = (-100.0, 270.0)


def parse():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--map", default="sydney_regatta")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-iter", type=int, default=None)
    args, _ = ap.parse_known_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    if args.seed is not None:
        np.random.seed(args.seed)
    lookup = f"./lookup_tables/{args.map}"
    output = f"{lookup}/output"
    os.makedirs(output, exist_ok=True)
    suffix = "" if args.seed is None else f"_seed{args.seed}"
    return lookup, output, suffix


def show() -> bool:
    """False with --no-show: no windows and no live plot (redrawing the tree every
    iteration takes most of the planning time)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--no-show", action="store_true")
    return not ap.parse_known_args()[0].no_show


def length_factor(default: float) -> float:
    """--length-factor k: VF-RRT*/SVF-RRT* edge cost = upstream cost + k * mean current speed * length."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--length-factor", type=float, default=None)
    value = ap.parse_known_args()[0].length_factor
    return default if value is None else value


def max_iter(default: int) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max-iter", type=int, default=None)
    value = ap.parse_known_args()[0].max_iter
    return default if value is None else value


def clearance(default: float = 10.0) -> float:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--clearance", type=float, default=None)
    value = ap.parse_known_args()[0].clearance
    return default if value is None else value


def inflate(grid: np.ndarray, resolution: float, margin: float) -> np.ndarray:
    """Mark as occupied (100) every cell whose centre is closer than `margin` to land.

    Distances are between cell centres, and land starts half a cell from the centre of
    a land cell, so a cell stays free when (centre distance - resolution / 2) >= margin.
    """
    if margin <= 0:
        return grid
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(grid <= 50) * resolution - resolution / 2.0
    out = grid.copy()
    out[(grid <= 50) & (d < margin)] = 100
    return out
