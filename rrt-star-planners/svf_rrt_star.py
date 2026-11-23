#!/usr/bin/env python3
"""SVF-RRT* (Zhang et al., RA-L 2023, Algs. 1-4): see vf_rrt_core.py.

    python3 svf_rrt_star.py --map sydney_regatta_roi --seed 1 --no-show [--max-iter 2000] [--length-factor 0]

Writes lookup_tables/<map>/output/svf_waypoints_seed<N>.csv and a figure of the tree.
Needs lookup_tables/<map>/stream_function.npy (generate_stream_function.py).
"""
from vf_rrt_core import main

if __name__ == "__main__":
    main(svf=True)
