#!/usr/bin/env python3
"""VF-RRT* (Zhang et al., RA-L 2023, Sec. II-B): see vf_rrt_core.py.

    python3 vf_rrt_star.py --map sydney_regatta_roi --seed 1 --no-show [--max-iter 2000] [--length-factor 0]

Writes lookup_tables/<map>/output/vf_waypoints_seed<N>.csv and a figure of the tree.
"""
from vf_rrt_core import main

if __name__ == "__main__":
    main(svf=False)
