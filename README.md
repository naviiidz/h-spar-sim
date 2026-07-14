# H-SPAR Simulation

## TODO: 
Merge instances of VTU folder

A ROS2-based hydrodynamic simulation environment for autonomous surface vehicles (ASVs) using Gazebo and the Virtual RobotX (VRX) framework.

## Overview

This project simulates an autonomous surface vehicle in a water environment with:
- **Gazebo simulation** - Physics-based 3D environment
- **VRX framework** - Marine robotics simulation
- **Drag force modeling** - Real-time hydrodynamic drag calculations from velocity field data
- **ROS2 integration** - Full ROS2 middleware support for sensors, actuators, and control

## Lagrangian Particle Simulation

![Lagrangian particle trajectories](backend/lagrangian_sim/docs/assets/lagrangian_particle_trajectories.gif)
![H-SPAR front/back](docs/assets/h-spar-front-back.gif)

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

### Step 2: Run the Drag Query Service

```bash
ros2 run drag_query drag_query_server.py
```

This starts the service that:
- Loads the velocity field database (`velocity_lookup.h5`)
- Loads robot configuration from `robot_config.yaml`
- Listens for drag force requests on `/get_drag_force`

### Step 3: Bridge Forces to Gazebo

```bash
ros2 run h_spar_force ros_to_gazebo_wrench_bridge
```

This connects:
- Drag force calculations from `drag_query` service
- Force application to the simulated vehicle in Gazebo

### Step 4: TODO
To be completed

## References

- [VRX GitHub](https://github.com/osrf/vrx)
- [ROS2 Documentation](https://docs.ros.org/)
- [Gazebo Documentation](https://gazebosim.org/)

## License

See LICENSE file in repository.






