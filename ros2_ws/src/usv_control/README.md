# USV Control Package

Control and actuation system for autonomous surface vehicles (ASVs) in the Gazebo/VRX simulation environment.

## Overview

The `usv_control` package manages vehicle dynamics, thruster commands, pose estimation, and manual teleoperation for the WAM-V ASV platform. It bridges high-level velocity commands to low-level thruster forces and provides vehicle pose feedback via ROS2 and TF transforms.

## Components

### 1. USV Velocity Controller (`usv_velocity_controller`)

**Type**: C++ ROS2 Node  
**Purpose**: Convert velocity commands to differential thrust commands

**Subscriptions**:
- `/wamv/cmd_vel` (geometry_msgs/Twist) - Linear and angular velocity commands

**Publications**:
- `/wamv/thrusters/left/thrust` (std_msgs/Float64) - Left thruster force (Newtons)
- `/wamv/thrusters/right/thrust` (std_msgs/Float64) - Right thruster force (Newtons)

**Parameters**:
- `max_thrust` (default: 250.0 N) - Maximum thrust per thruster
- `boat_length` (default: 4.9 m) - Distance between thrusters
- `thrust_deadband` (default: 0.1 N) - Minimum force to overcome static friction

---

### 2. GZ Dynamic Pose Listener (`gz_dynamic_pose_listener.py`)

**Type**: Python ROS2 Node  
**Purpose**: Extract vehicle pose from Gazebo and broadcast as TF transforms

**Listens to**: Gazebo dynamic pose topic via `gz topic` CLI

**Publications**:
- TF2 Transforms: `map` → `wamv_simple` (vehicle position and orientation)
- `/wamv/simple_pose` (geometry_msgs/PoseStamped) - Vehicle pose in map frame

**Parameters**:
- `world_name` (default: sydney_regatta) - Gazebo world name
- `robot_name` (default: wamv) - Robot name in Gazebo

**Features**:
- Filters dynamic pose messages to extract only the specified robot
- Publishes TF transforms for navigation and planning
- Handles variable world names dynamically

---

### 3. USV Teleop Keyboard (`usv_teleop_keyboard`)

**Type**: C++ ROS2 Node  
**Purpose**: Manual keyboard-based vehicle control

**Publications**:
- `/wamv/cmd_vel` (geometry_msgs/Twist) - Velocity commands from keyboard input

**Parameters**:
- `linear_speed` (default: 2.0 m/s) - Maximum forward speed
- `angular_speed` (default: 1.0 rad/s) - Maximum turning rate
- `linear_step` (default: 0.1 m/s) - Velocity increment per keystroke
- `angular_step` (default: 0.1 rad/s) - Angular velocity increment

**Keyboard Controls**:
- `w` - Increase forward velocity
- `s` - Decrease forward velocity (reverse)
- `a` - Increase counterclockwise turning
- `d` - Increase clockwise turning
- `SPACE` - Stop all motion
- `q` - Quit

---

### 4. USV Path Tracker (`usv_path_tracker`)

**Type**: C++ ROS2 Node  
**Purpose**: Record robot path history for visualization

**Subscriptions**:
- `/wamv/pose` (nav_msgs/Odometry) - Robot pose updates

**Publications**:
- `usv_path` (nav_msgs/Path) - Complete path history

**Parameters**:
- `map_frame` (default: map) - Reference frame for path
- `min_distance` (default: 0.5 m) - Minimum distance between path points

**Note**: Not launched by default. Enable in launch configuration if needed.

---

## Launch Files

### `usv_bringup.launch.py` (Main)

Launches complete vehicle control stack:
- ROS-Gazebo bridges for thrusters and cmd_vel
- Pose estimation node
- Velocity controller

**Usage**:
```bash
ros2 launch usv_control usv_bringup.launch.py
```

**Arguments**:
- `max_thrust` (default: 250.0 N)
- `world_name` (default: sydney_regatta)

---

### `_usv_velocity_controller.launch.py` (Helper)

Launches only the velocity controller for independent testing.

**Usage**:
```bash
ros2 launch usv_control _usv_velocity_controller.launch.py
```

---

### `multi_usv_bringup.launch.py` (Multi-robot)

Launches multiple ASVs with independent control stacks.

**Usage**:
```bash
ros2 launch usv_control multi_usv_bringup.launch.py robot_names:=wamv1,wamv2
```

## Notes

- Velocity commands are converted to thrust using differential drive kinematics
- The velocity controller includes thrust limiting and deadband to prevent servo jitter
- Pose estimation requires active Gazebo simulation
- All topics use the `/wamv/` namespace - modify launch files for different robot names

