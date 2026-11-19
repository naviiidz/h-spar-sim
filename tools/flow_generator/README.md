# Flow Generator

A step-by-step UI that turns a region of a Gazebo world into a velocity field:

```text
Gazebo map (DAE) -> draw region -> boundary conditions -> depth -> gmsh mesh -> Thetis/Firedrake
```

The output uses the same per-map layout as the rest of H-SPAR, so `tools/vtu_converter`,
`drag_query` and `lagrangian_sim` work on it without changes.

## Start

```bash
tools/flow_generator/launch.sh
```

`launch.sh` runs the UI with the Firedrake virtual environment and sets `PETSC_DIR` /
`PETSC_ARCH`. Override `FIREDRAKE_VENV`, `PETSC_DIR` or `PETSC_ARCH` by exporting them first.

## Steps

| Step | What you do |
|---|---|
| 1 Map | Pick the Gazebo world and an output name. The map is taken from the local Gazebo Fuel cache (`~/.gz/fuel`); it is downloaded only if it is missing. |
| 2 Draw region | Click the outline of the water region. Click the first point again to close it. Right-click undoes. You can also load an outline from an existing `.geo` or `region.json`. |
| 3 Boundaries | Each edge starts as shore (no flow). Select edges on the map or in the list and set them to inflow (m/s), outflow (m/s) or open water level (m). |
| 4 Depth & settings | Water depth (constant over the region) plus mesh size, simulated time, save interval, time step, ramp-up, viscosity and bed friction. |
| 5 Mesh & run | Build the mesh, check it on the map, then run the flow. Progress and the solver log are shown live. |

## Output

```text
meshes/<name>/region.json                       # everything chosen in the UI
meshes/<name>/<name>.msh                        # gmsh 2.2 mesh; edge i = boundary ID i
meshes/<name>/<name>.geo_unrolled
velocity_fields/<name>/raw/Velocity2d.pvd       # what lagrangian_sim and vtu_converter read
velocity_fields/<name>/raw/Velocity2d/Velocity2d_<n>.vtu
velocity_fields/<name>/thetis/                  # remaining Thetis output (elevation, logs)
velocity_fields/<name>/region.json, run_info.json
```

The tool never replaces velocity fields it did not create. A name whose `raw/` folder exists
without a `region.json` next to it is refused.

## Coordinates

Everything is in the **Gazebo world frame**. Each map stores the offset between its DAE and the
world (from the model's `<pose>` in the world `.sdf`):

| Map | World .sdf | world = DAE + |
|---|---|---|
| `sydney_regatta` | `vrx_gz/worlds/sydney_regatta.sdf` | (0, 0) |
| `baylands` | `vrx_gz/worlds/baylands.sdf` | (205, 155) |

Add a map by adding an entry to `MAPS` in `maps.py`.

## Running steps without the UI

```bash
python3 tools/flow_generator/build_mesh.py meshes/<name>/region.json     # needs gmsh
python  tools/flow_generator/run_flow.py  meshes/<name>/region.json     # Firedrake Python
```

## Solver setup

Thetis 2D shallow water, DG-DG P1, Crank-Nicolson, nonlinear. Boundary IDs map to:

| Edge type | Thetis condition |
|---|---|
| Shore / wall | `un = 0` |
| Inflow, speed v | `un = -v` (Thetis `un` is positive out of the domain) |
| Outflow, speed v | `un = +v` |
| Open water level h | `elev = h` |

Inflow, outflow and open levels ramp linearly from 0 over the ramp-up time.

## Firedrake environment

If the run fails with `cannot open shared object file`, PETSc cannot find a system library it was
built against. Install Firedrake's system packages:

```bash
sudo apt install $(python3 /home/navid/h-spar-test/Hspar-backup/particle_backend/firedrake-configure --show-system-packages)
```
