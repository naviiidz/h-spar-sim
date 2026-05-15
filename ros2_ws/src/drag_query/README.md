# drag_query

A ROS2 package for querying water drag forces on marine vehicles using velocity field lookup.

## Overview

The `drag_query` package provides a service to calculate drag forces acting on a marine vehicle given its position, orientation, and a water velocity field. It uses an HDF5 database containing pre-computed velocity fields and applies directional drag coefficients in x and y directions.

## Features

- **Velocity Field Lookup**: Queries velocity data from HDF5 database
- **Directional Drag Calculation**: Applies separate drag coefficients for longitudinal (x) and lateral (y) directions
- **Configurable Parameters**: Load robot properties and drag parameters from YAML config
- **ROS2 Service**: Provides `get_drag_force` service for drag force queries

## Service Definition

**Service**: `get_drag_force`

**Request**:
- `float64 timestamp` - Time index for velocity field query
- `float64 x` - Robot x position (meters)
- `float64 y` - Robot y position (meters)
- `float64 theta` - Robot heading angle (radians)

**Response**:
- `float64 fx` - Drag force in x-direction (Newtons)
- `float64 fy` - Drag force in y-direction (Newtons)
- `float64 torque` - Rotational torque (currently set to 0)

## Configuration

Edit `robot_config.yaml` to customize:

```yaml
robot_width: 2.0        # Robot width (meters)
robot_length: 3.0       # Robot length (meters)
water_density: 1025.0   # Seawater density (kg/m³)
cx: 0.8                 # Longitudinal drag coefficient
cy: 1.2                 # Lateral drag coefficient
Ax: 3.0                 # Reference area x-direction (m²)
Ay: 6.0                 # Reference area y-direction (m²)
```

## Running

```bash
# Start the server
ros2 run drag_query drag_query_server.py

# Call the service (in another terminal)
ros2 service call /get_drag_force drag_query/srv/GetDragForce "{timestamp: 200.0, x: -400.0, y: 300.0, theta: 0.0}"
```

## Drag Force Formula

Drag forces are calculated in the robot body frame:
- **X-direction**: `Fx = -0.5 * cx * ρ * Ax * vx * |vx|`
- **Y-direction**: `Fy = -0.5 * cy * ρ * Ay * vy * |vy|`

Then transformed back to the world frame based on robot orientation.
