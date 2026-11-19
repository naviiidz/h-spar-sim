# Particle Simulator

A step-by-step UI for the Lagrangian particle simulation:

```text
velocity field -> spawn edge -> release & settings -> run & play back
```

It uses the integrator of `lagrangian_sim` (second-order midpoint, linear interpolation in
space and time, random walk for diffusion) and writes the same CSV columns, so
`particle_sampling` reads the output unchanged.

## Start

```bash
python3 tools/particle_simulator/app.py
```

Needs only `numpy`, `matplotlib` (with Tk) and the repository's `lagrangian_sim`.

## Steps

| Step | What you do |
|---|---|
| 1 Velocity field | Pick a field under `velocity_fields/<name>/raw` and press Load. The snapshot interval comes from the flow generator's `region.json`, else from `Velocity2d.pvd`; you can override it. The slider shows the flow at any time. The tool also finds the end of the spin-up from rest and starts particles after it. |
| 2 Spawn edge | The mesh boundary is split into edges at corners and where the flow through it changes. Edges with water flowing in are green, out orange, shore grey. Inflow edges are selected to start with; click to choose others (Shift-click for several). |
| 3 Release & settings | Particles per release, release interval, placement along the edge (even or random), start distance from the edge, simulated time, time step, diffusivity, field start time, random seed and CSV stride. The black dots on the map show one release. |
| 4 Run & view | Run writes the CSV while the particles are shown live. Afterwards, play back at 1–60× or drag the slider. Particles are coloured by release. |

## Output

```text
lagrangian_sim/outputs/<field>/<name>.csv    # particle_id, generation, step, time, x, y, u, v, status, vtu_frame, vtu_file
lagrangian_sim/outputs/<field>/<name>.json   # every setting, the chosen edges and their coordinates
```

`time` in the CSV is particle time from 0; the velocity field is read at
`field_start_s + time`. After the last snapshot the field holds its last value.

## Files

| File | Contents |
|---|---|
| `app.py` | Tk UI |
| `engine.py` | Field loading, boundary edges, seeding along edges and the simulation loop (no UI) |
