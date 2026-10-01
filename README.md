# H-SPAR Simulation

![H-SPAR front/back](docs/assets/h-spar-front-back.gif)

A ROS2-based hydrodynamic simulation environment for autonomous surface vehicles (ASVs) using Gazebo and the Virtual RobotX (VRX) framework.

## Overview

This project simulates an autonomous surface vehicle in a water environment with:
- **Gazebo simulation** - Physics-based 3D environment
- **VRX framework** - Marine robotics simulation
- **Drag force modeling** - Real-time hydrodynamic drag calculations from velocity field data
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

**2. drag_query Service** - Hydrodynamic drag calculation
- Queries velocity field database (HDF5)
- Calculates drag forces in x and y directions
- Provides forces via ROS2 service interface

**3. Wrench Bridge** - Force application interface
- Bridges ROS2 drag force topics to Gazebo
- Applies calculated forces to the simulated vehicle

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
ros2 launch h_spar_experiments experiment1.launch.py
```

This launch file starts:
- `ros_to_gazebo_wrench_bridge` to forward wrench commands into Gazebo
- `drag_query_server` to provide hydrodynamic drag values
- `usv_bringup.launch.py` from `usv_control` for vehicle bringup
- `drag_force_node` to query drag and publish wrench commands

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
ros2 run particle_sampling particle_sampling_node --ros-args   -p probabilistic_sampling:=false   -p sampling_radius:=5.0   -p capture_v0:=0.5   -p capture_sigma:=0.2

```

Particle sampling parameters:

- `probabilistic_sampling`: Enables the probabilistic capture model. Set to `true` to enable it; `false` uses deterministic geometric sampling.
- `sampling_radius`: Candidate sampling radius around the USV, in meters.
- `capture_v0`: Design intake velocity, in m/s. Capture probability is highest when relative particle speed equals this value.
- `capture_sigma`: Sensitivity of capture probability to speed differences. Smaller values make capture more selective.

This starts the particle sampling node, which generates and tracks particle-based sampling behavior for the simulation workflow.

## References

- [VRX GitHub](https://github.com/osrf/vrx)
- [ROS2 Documentation](https://docs.ros.org/)
- [Gazebo Documentation](https://gazebosim.org/)

## License

This project is licensed under the [MIT License](LICENSE).






