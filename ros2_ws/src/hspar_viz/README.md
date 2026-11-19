# hspar_viz

An illustration-only ROS 2 package that publishes HDF5-backed velocity vectors as RViz arrows.

The ROS package name is `hspar_viz`, while the installed command is `hspar-viz`.

## Build

```bash
cd /home/navid/h-spar-sim/ros2_ws
colcon build --merge-install --packages-select hspar_viz
```

## Run

```bash
source install/setup.bash
ros2 run hspar_viz hspar-viz
```

By default the node reads:

`/home/navid/h-spar-sim/velocity_fields/sydney_regatta/h5/velocity_lookup.h5`

## RViz

Add a `MarkerArray` display and subscribe to `hspar_viz/vector_field`.

Useful parameters:

- `frame_id`
- `timer_period`
- `h5_path`
- `timestamp_step`
- `sample_stride`
- `vector_length`
- `line_width`
- `colormap_min_speed`
- `colormap_max_speed`