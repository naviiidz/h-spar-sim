#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from drag_query.srv import GetDragForce
import os
import sys
import yaml
import math
import numpy as np

# Add velocity converter directory to path for importing velocity_lookup
from drag_query.velocity_lookup import VelocityLookup

class DragQueryServer(Node):
    def __init__(self):
        super().__init__('drag_query_server')
        self.srv = self.create_service(GetDragForce, 'get_drag_force', self.handle_get_drag_force)
        
        # Get the package path
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load robot configuration
        self.load_robot_config()
        
        # Load velocity field lookup database (HDF5)
        self.lookup_db = '/home/navid/h-spar-sim/velocity_fields/sydney_regatta/h5/velocity_lookup.h5'
        #self.lookup_db = '/home/navid/h-spar-sim/velocity_fields/sydney_regatta/h5/average_spatial_velocity.h5'
        try:
            self.velocity_lookup = VelocityLookup(self.lookup_db)
            self.get_logger().info(f'Loaded velocity lookup database: {self.lookup_db}')
        except Exception as e:
            self.get_logger().error(f'Failed to load velocity lookup database: {e}')
            self.velocity_lookup = None
        
        self.get_logger().info('Velocity Field Lookup Service is ready')

    def load_robot_config(self):
        """Load robot configuration from YAML file"""
        # Try multiple locations
        possible_paths = [
            os.path.join(self.package_dir, 'velocity_field_lookup', 'robot_config.yaml'),
            os.path.join(os.path.dirname(__file__), 'robot_config.yaml'),
            '/opt/ros/jazzy/lib/velocity_field_lookup/robot_config.yaml',
        ]
        
        config_file = None
        for path in possible_paths:
            if os.path.exists(path):
                config_file = path
                break
        
        if config_file is None:
            self.get_logger().warn(f'Robot config file not found in any expected location')
            config_file = possible_paths[0]  # Use default path for error message
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.robot_width = config['robot_width']
            self.robot_length = config['robot_length']
            self.water_density = config['water_density']
            # Hull damping coefficients (defaults match VRX WAM-V)
            self.xU = config.get('xU', 100.0)
            self.xUU = config.get('xUU', 150.0)
            self.yV = config.get('yV', 100.0)
            self.yVV = config.get('yVV', 100.0)
            self.get_logger().info(f'Loaded robot config: {config_file}')
        except Exception as e:
            self.get_logger().warn(f'Failed to load robot config ({config_file}): {e}, using defaults')
            # Use defaults
            self.robot_width = 2.0
            self.robot_length = 3.0
            self.water_density = 1025.0
            self.xU = 100.0
            self.xUU = 150.0
            self.yV = 100.0
            self.yVV = 100.0
        self.get_logger().info(
            f'Hull damping: xU={self.xU}, xUU={self.xUU}, yV={self.yV}, yVV={self.yVV}'
        )

    def get_velocity_at_position(self, timestamp, x, y, tolerance=0.5):
        """Query velocity field at a position for a given timestamp
        
        Args:
            timestamp: Time index
            x, y: Position coordinates
            tolerance: Tolerance for nearest neighbor matching
        
        Returns:
            [vx, vy] velocity vector or None
        """
        if self.velocity_lookup is None:
            return None
        
        try:
            # Convert timestamp to int (service receives floats)
            timestamp = int(timestamp)
            
            # Try exact match within tolerance first
            vel = self.velocity_lookup.get_velocity_at_point(timestamp, x, y, tolerance=tolerance)
            if vel is not None:
                return vel
            
            # Fall back to nearest neighbor
            vel, _ = self.velocity_lookup.get_nearest_velocity(timestamp, x, y)
            return vel
        except Exception as e:
            self.get_logger().warn(f'Error querying velocity at ({x}, {y}) t={timestamp}: {e}')
            return None

    def find_closest_velocity(self, timestamp, robot_x, robot_y):
        """Find velocity at robot position using HDF5 lookup
        
        Args:
            timestamp: Time index
            robot_x, robot_y: Robot position
            
        Returns:
            [vx, vy] velocity vector or None
        """
        vel = self.get_velocity_at_position(timestamp, robot_x, robot_y, tolerance=0.5)
        
        if vel is not None:
            self.get_logger().debug(
                f'Velocity at ({robot_x:.2f}, {robot_y:.2f}): {vel}'
            )
        
        return vel

    @staticmethod
    def _damping(u, lin, quad):
        """Damping magnitude D(u) = lin*u + quad*u*|u| (VRX applies -D(u))."""
        return lin * u + quad * u * abs(u)

    def calculate_drag_force(self, velocity, robot_theta, boat_vx, boat_vy):
        """Current-induced force on the hull, world frame.

        VRX already applies still-water damping -D(v_boat). Adding
        D(v_boat) - D(v_boat - u_current) makes the net hull damping
        -D(v_boat - u_current), i.e. damping on the velocity relative to the water.
        """
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)

        # World -> body frame for current and boat velocity
        u_cx = velocity[0] * cos_theta + velocity[1] * sin_theta
        u_cy = -velocity[0] * sin_theta + velocity[1] * cos_theta
        v_bx = boat_vx * cos_theta + boat_vy * sin_theta
        v_by = -boat_vx * sin_theta + boat_vy * cos_theta

        # Velocity relative to the water, body frame
        v_rx = v_bx - u_cx
        v_ry = v_by - u_cy

        drag_x = self._damping(v_bx, self.xU, self.xUU) - self._damping(v_rx, self.xU, self.xUU)
        drag_y = self._damping(v_by, self.yV, self.yVV) - self._damping(v_ry, self.yV, self.yVV)

        # Body -> world frame
        fx = drag_x * cos_theta - drag_y * sin_theta
        fy = drag_x * sin_theta + drag_y * cos_theta
        
        # Torque - ignored for now
        torque = 0.0
        
        return fx, fy, torque

    def handle_get_drag_force(self, request, response):
        """
        Handle GetDragForce service request.
        Look up velocity field from HDF5 database, and calculate drag forces.
        """
        self.get_logger().debug(
            f'Received request: timestamp={request.timestamp}, x={request.x}, y={request.y}, '
            f'theta={request.theta}, vx={request.vx}, vy={request.vy}'
        )
        
        try:
            # Find velocity at robot position from HDF5 lookup
            velocity = self.find_closest_velocity(request.timestamp, request.x, request.y)
            if velocity is None:
                self.get_logger().warn(f'No velocity data found for timestamp {request.timestamp}')
                response.fx = 0.0
                response.fy = 0.0
                response.torque = 0.0
                return response
            
            # Calculate drag forces
            fx, fy, torque = self.calculate_drag_force(
                velocity, request.theta, request.vx, request.vy
            )

            response.fx = fx
            response.fy = fy
            response.torque = torque

            self.get_logger().debug(
                f'Velocity at position: {velocity}, Drag forces: fx={fx:.4f}, fy={fy:.4f}, torque={torque:.4f}'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing request: {e}')
            response.fx = 0.0
            response.fy = 0.0
            response.torque = 0.0
        
        return response


def main():
    rclpy.init()
    server = DragQueryServer()
    rclpy.spin(server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
