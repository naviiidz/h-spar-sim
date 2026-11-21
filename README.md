# H-SPAR Simulation

![H-SPAR front/back](docs/assets/h-spar-front-back.gif)

A ROS2-based hydrodynamic simulation environment for autonomous surface vehicles (ASVs) using Gazebo and the Virtual RobotX (VRX) framework.

## Overview

This project simulates an autonomous surface vehicle in a water environment with:
- **Gazebo simulation** - Physics-based 3D environment
- **VRX framework** - Marine robotics simulation
- **Water current modeling** - Hydrodynamic damping relative to the local current from velocity field data
- **ROS2 integration** - Full ROS2 middleware support for sensors, actuators, and control




## Prerequisites

- **ROS2 Jazzy** (or compatible version)
- **Gazebo** (installed via VRX)
- **Python 3.10+**
- **colcon** build system
- **h5py** - For HDF5 velocity field database access

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd h-spar-sim
```

### 2. Install VRX Simulator

Follow the official VRX installation guide:
https://github.com/osrf/vrx

H-SPAR needs a patched VRX in which the WAM-V hydrodynamic damping acts on
the velocity relative to the water current. Clone it from the fork into the
workspace:

```bash
git clone -b hspar-current https://github.com/naviiidz/vrx.git ros2_ws/src/vrx
```

### 3. Build the Workspace

```bash
colcon build --merge-install
source install/setup.bash
```

## Architecture

### Main Components

**1. Gazebo/VRX** - Physics simulation environment
- Simulates the ASV in water
- Applies forces and torques to the vehicle
- Publishes sensor data and pose information

**2. drag_query Service** - Water current lookup
- Queries velocity field database (HDF5)
- Returns the current velocity at the vehicle position (world frame)
- Provides the current via ROS2 service interface

**3. Current Bridge** - Current input to Gazebo
- Bridges `/wamv/water_current` from ROS2 to Gazebo
- The patched VRX hydrodynamics plugin applies damping to the velocity relative to the water

**4. USV Bringup** - Vehicle initialization
- Publishes vehicle pose in Gazebo coordinates
- Initializes actuators and sensors

## Usage Workflow

### Step 1: Start the Gazebo Simulation

```bash
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
```

This launches:
- Gazebo with the Sydney Regatta environment
- VRX plugins and simulation world

### Step 2: Bridge Front- and Back-end 

```bash
ros2 launch h_spar_experiments experiment1.launch.py  # optional: field:=<name> field_time_offset_s:=1200.0 current_enabled:=false
```

This launch file starts:
- `ros_gz_bridge` to forward the water current into Gazebo
- `drag_query_server` to provide the current velocity
- `usv_bringup.launch.py` from `usv_control` for vehicle bringup
- `drag_force_node` to query the current and publish it on `/wamv/water_current`

### Step 3: Launch DWA planner

```bash
ros2 launch usv_planners dwa_planner.launch.py
```

This launch file starts the `simple_dwa_planner` node and a `path_visualizer` node, with simulation time enabled by default and ROS topic remaps for `/odom`, `/cmd_vel`, and `/goal_pose`.

### Step 4: Bridge Global and Local Planners

```bash
source ros2_ws/install/setup.bash
ros2 run usv_planners sequential_goal_sender --preset rrt
```

This script sends a preset sequence of navigation goals to the planner. Survey patrol presets are available at 0 degrees (`survey`), 45 degrees (`survey_45`), and 90 degrees (`survey_90`). The RRT-style presets include `rrt`, `vf-rrt`, and `svf-rrt`.


### Step 5: Run Particle Sampling Mechanism

```bash
ros2 run particle_sampling particle_sampling_node --ros-args \
  -p csv_path:=lagrangian_sim/outputs/<run>/<run>_particles.csv \
  -p probabilistic_sampling:=true -p seed:=1 -p particle_time_offset_s:=1200.0

```

Particle sampling parameters:

- `probabilistic_sampling`: `true` captures each particle within `sampling_radius` of the USV with probability 1 − exp(−λΔt) per step, with λ = Q / (π r_s² h); `false` samples every particle within `sampling_radius`.
- `sampling_radius`: radius r_s of the sampling region, in meters (default 5).
- `pump_flow_m3h`, `intake_layer_m`: pump flow rate Q (default 25 m³/h) and intake layer thickness h (default 0.1 m).
- `seed`: seed of the capture draws; record it with the results.
- `particle_time_offset_s`: particle (CSV) time when the Gazebo clock reads 0, so that particles have spread before the mission starts (default 1200 s). Use the same value as `field_time_offset_s` of `drag_query_server` when the particles were generated on that field from its start.

On shutdown the node writes, next to `report_path`, the first-sampled particles (`.csv`), a summary with the encountered, sampled and expected sampled counts (`_summary.json`), and a 1 s time series of the particles in the sampling region and the running counts (`_timeseries.csv`).

The current felt by the USV is set by `drag_query_server` parameters: `field` (a folder in `velocity_fields/`, default `sydney_regatta_roi_from1326s`), `field_time_offset_s` (field time when the Gazebo clock reads 0) and `current_enabled` (`false` for no current, the VRX baseline). The current is interpolated within each triangle and in time, as for the particles.

This starts the particle sampling node, which generates and tracks particle-based sampling behavior for the simulation workflow.

## References

- [VRX GitHub](https://github.com/osrf/vrx)
- [ROS2 Documentation](https://docs.ros.org/)
- [Gazebo Documentation](https://gazebosim.org/)

## License

This project is licensed under the [MIT License](LICENSE).






