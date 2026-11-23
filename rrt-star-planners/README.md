# RRT-Star Planners

This folder contains the planner implementations and the small generator scripts used to convert CFD-oriented data into Python-friendly lookup tables.

## What the generator scripts do

The `generate_*.py` scripts are used to convert CFD compatible files into Python compatible files for planning and visualization.

The typical pipeline is:

1. Generate the occupancy grid.
2. Generate the velocity lookup tables.
3. Generate the stream function lookup table.

After these three steps, all lookup tables are available for the RRT* planners.

## Implemented planner methods

Three planner variants are implemented here:

- `rrt_star.py` - baseline RRT* using the occupancy grid.
- `vf_rrt_star.py` - VF-RRT* using the velocity field lookup tables.
- `svf_rrt_star.py` - stream-function-aware VF-RRT* variant.

## Required inputs

To run this folder successfully, provide the following data:

- Mesh-derived occupancy data.
- Velocity field data.

These inputs are expected to be converted into the lookup tables used by the planner scripts.

## Output location

Generated planner outputs are stored under:

`./lookup_tables/{map_name}/output`

This folder is created automatically by the planner scripts if it does not already exist.

## Typical workflow

1. Prepare the mesh and velocity field sources.
2. Run the generators in order to create the occupancy, velocity, and stream lookup tables.
3. Run any of the `rrt*-*` planner methods.
4. Review the generated waypoints, plots, and diagnostics in the output folder.

## Fields from the flow generator

For a field written by the flow generator (`velocity_fields/<field>/` with `region.json`), one script builds all lookup tables, including the stream function:

```bash
python3 build_lookup_tables.py --field sydney_regatta_roi_from1326s --map sydney_regatta_roi
```

The occupancy grid comes from the region polygon, and the velocity is the time average of the field over a whole number of inflow periods.

Then plan with a documented seed, one run per seed:

```bash
for seed in $(seq 1 10); do
  python3 rrt_star.py     --map sydney_regatta_roi --seed $seed --no-show
  python3 vf_rrt_star.py  --map sydney_regatta_roi --seed $seed --no-show
  python3 svf_rrt_star.py --map sydney_regatta_roi --seed $seed --no-show
done
```

Each run writes `lookup_tables/<map>/output/<planner>_waypoints_seed<N>.csv` (columns `x,y`), which the goal sender follows with `ros2 run usv_planners sequential_goal_sender --preset astar_wp_file <csv>`.

All three planners run a fixed budget of 2000 iterations (`--max-iter`) and return the cheapest path to the goal, with a step and goal radius of 30 m. RRT\* minimises path length. VF-RRT\* and SVF-RRT\* (`vf_rrt_core.py`) follow Zhang et al., "SVF-RRT\*: A Stream-Based VF-RRT\* for USVs Path Planning Considering Ocean Currents", IEEE RA-L 8(4), 2023: upstream-criterion cost (Eq. 1), VF-RRT direction sampling with the λ update of Eq. 2, inefficient nodes counted but kept, R_rewire = min(δ, γ (log|V|/|V|)^(1/2)); SVF-RRT\* adds, whenever a cheaper path is found, the upstream coefficient ε (Eq. 7), the stream-function interval (Alg. 2), tree pruning (Alg. 4), biased sampling (Alg. 3) and the λ update of Eqs. 12–13. Parameters are those of the paper's Table I (E_s = 0.3, λ₀ = 3, n = 100, δ = r = 30). `--length-factor k` adds k × (mean current speed) × length to the cost, a variant that is not part of the paper.

All three planners keep a clearance from the shore (`--clearance`, default 10 m): water cells closer to land than the clearance are treated as obstacles. Start (−400, 200) and goal (−100, 270) are shared through `planner_args.py`.

## Notes

- The planner scripts read lookup data from `./lookup_tables/{map_name}`.
- The `output` subfolder is reserved for generated results.
- If your data lives in a different location, update the path constants in the scripts before running them.
