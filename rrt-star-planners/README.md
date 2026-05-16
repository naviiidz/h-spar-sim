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

`./lookup_tables/sydney_regatta/output`

This folder is created automatically by the planner scripts if it does not already exist.

## Typical workflow

1. Prepare the mesh and velocity field sources.
2. Run the generators in order to create the occupancy, velocity, and stream lookup tables.
3. Run any of the `rrt*-*` planner methods.
4. Review the generated waypoints, plots, and diagnostics in the output folder.

## Notes

- The planner scripts read lookup data from `./lookup_tables/sydney_regatta`.
- The `output` subfolder is reserved for generated results.
- If your data lives in a different location, update the path constants in the scripts before running them.
