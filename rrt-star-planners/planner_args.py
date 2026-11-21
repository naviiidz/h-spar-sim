"""Command-line options shared by the planners (rrt_star, vf_rrt_star, svf_rrt_star).

    --map NAME    lookup tables in lookup_tables/NAME (default sydney_regatta)
    --seed N      seed of the random sampling; outputs get the suffix _seedN
    --no-show     save the figures without opening a window
    --max-iter N  planner iterations (default: the value in each script)
"""
from __future__ import annotations

import argparse
import os

import numpy as np


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


def max_iter(default: int) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max-iter", type=int, default=None)
    value = ap.parse_known_args()[0].max_iter
    return default if value is None else value
