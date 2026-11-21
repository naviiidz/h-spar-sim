#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from drag_query.srv import GetDragForce
import os
import sys
import yaml
import math
import numpy as np

from pathlib import Path

class DragQueryServer(Node):
    def __init__(self):
        super().__init__('drag_query_server')
        self.srv = self.create_service(GetDragForce, 'get_drag_force', self.handle_get_drag_force)
        
        # Get the package path
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load robot configuration
        self.load_robot_config()
        
        # Velocity field: a field written by the flow generator (velocity_fields/<field>),
        # interpolated linearly within each triangle and linearly in time, as for the
        # particles (tools/particle_simulator/engine.py).
        # field time = field_time_offset_s + Gazebo time; use the same offset as the
        # particle_sampling node's particle_time_offset_s when the particles were
        # generated on this field from its t = 0.
        repo_root = Path(os.environ.get('HSPAR_ROOT', str(Path.home() / 'h-spar-sim')))
        self.declare_parameter('repo_root', str(repo_root))
        self.declare_parameter('field', 'sydney_regatta_roi_from1326s')
        self.declare_parameter('field_time_offset_s', 1200.0)
        # False: no current (VRX baseline); the service then returns zero velocity.
        self.declare_parameter('current_enabled', True)
        self.field_name = str(self.get_parameter('field').value)
        self.field_time_offset_s = float(self.get_parameter('field_time_offset_s').value)
        self.current_enabled = bool(self.get_parameter('current_enabled').value)
        self.series = None
        self.field_duration = 0.0
        self._warned_end = False
        if self.current_enabled:
            try:
                sys.path.insert(0, str(Path(self.get_parameter('repo_root').value) / 'tools' / 'particle_simulator'))
                import engine  # noqa: E402  (flow generator / particle simulator field reader)
                field = engine.Field(self.field_name)
                self.series = field.new_series()
                self.field_duration = float(field.duration)
                self.get_logger().info(
                    f'Loaded field {self.field_name}: {len(field.files)} snapshots, {self.field_duration:g} s; '
                    f'field time = {self.field_time_offset_s:g} s + Gazebo time')
            except Exception as e:
                self.get_logger().error(f'Failed to load velocity field {self.field_name}: {e}')
        else:
            self.get_logger().info('Current disabled (current_enabled=false): returning zero velocity')
        
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
            self.get_logger().info(f'Loaded robot config: {config_file}')
        except Exception as e:
            self.get_logger().warn(f'Failed to load robot config ({config_file}): {e}, using defaults')
            # Use defaults
            self.robot_width = 2.0
            self.robot_length = 3.0
            self.water_density = 1025.0

    def find_closest_velocity(self, timestamp, robot_x, robot_y):
        """Current [vx, vy] at the robot position and Gazebo time `timestamp`.

        Zero when the current is disabled or the robot is outside the mesh; after the
        end of the field the last snapshot is held (warned once).
        """
        if not self.current_enabled or self.series is None:
            return np.zeros(2)
        t_field = self.field_time_offset_s + float(timestamp)
        if t_field > self.field_duration and not self._warned_end:
            self.get_logger().warn(
                f'Field time {t_field:.0f} s is past the end of {self.field_name} '
                f'({self.field_duration:g} s); holding the last snapshot')
            self._warned_end = True
        vel, _ = self.series.velocity(np.array([[robot_x, robot_y]], dtype=float), t_field)
        return np.nan_to_num(vel[0])

    def handle_get_drag_force(self, request, response):
        """
        Handle GetDragForce service request.
        Look up the water current velocity at the robot position from the
        velocity field. The force follows from VRX SimpleHydrodynamics, which damps
        the velocity relative to this current.
        """
        self.get_logger().debug(
            f'Received request: timestamp={request.timestamp}, x={request.x}, y={request.y}, '
            f'theta={request.theta}'
        )
        
        try:
            # Find velocity at robot position from HDF5 lookup
            velocity = self.find_closest_velocity(request.timestamp, request.x, request.y)

            response.vx = float(velocity[0])
            response.vy = float(velocity[1])

            self.get_logger().debug(f'Velocity at position: {velocity}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing request: {e}')
            response.vx = 0.0
            response.vy = 0.0
        
        return response


def main():
    rclpy.init()
    server = DragQueryServer()
    rclpy.spin(server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
