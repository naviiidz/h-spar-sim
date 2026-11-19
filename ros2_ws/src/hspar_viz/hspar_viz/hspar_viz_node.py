#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from drag_query.velocity_lookup import VelocityLookup


class HsparVizNode(Node):
    def __init__(self) -> None:
        super().__init__('hspar_viz_node')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('timer_period', 0.5)
        self.declare_parameter(
            'h5_path',
            '/home/navid/h-spar-sim/velocity_fields/sydney_regatta/h5/velocity_lookup.h5',
        )
        self.declare_parameter('timestamp_step', 0.2)
        self.declare_parameter('sample_stride', 1)
        self.declare_parameter('vector_length', 10.0)
        self.declare_parameter('line_width', 1)
        self.declare_parameter('min_speed_for_vector', 1e-6)
        self.declare_parameter('colormap_min_speed', 0.0)
        self.declare_parameter('colormap_max_speed', 2.0)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.timer_period = float(self.get_parameter('timer_period').value)
        self.h5_path = str(self.get_parameter('h5_path').value)
        self.timestamp_step = max(1, int(self.get_parameter('timestamp_step').value))
        self.sample_stride = max(1, int(self.get_parameter('sample_stride').value))
        self.vector_length = float(self.get_parameter('vector_length').value)
        self.line_width = float(self.get_parameter('line_width').value)
        self.min_speed_for_vector = float(self.get_parameter('min_speed_for_vector').value)
        self.colormap_min_speed = float(self.get_parameter('colormap_min_speed').value)
        self.colormap_max_speed = float(self.get_parameter('colormap_max_speed').value)

        if self.colormap_max_speed <= self.colormap_min_speed:
            self.colormap_max_speed = self.colormap_min_speed + 1.0

        self.lookup = VelocityLookup(self.h5_path)
        self.timestamps = list(self.lookup.timestamps)
        if not self.timestamps:
            raise ValueError(f'No timestamps found in HDF5 file: {self.h5_path}')
        self.timestamp_index = 0

        self.publisher = self.create_publisher(MarkerArray, 'hspar_viz/vector_field', 10)
        self.timer = self.create_timer(self.timer_period, self.publish_field)

        self.get_logger().info(
            'hspar-viz ready: '
            f'frame_id={self.frame_id}, timer_period={self.timer_period:.2f}s, '
            f'h5_path={self.h5_path}, timestamps={len(self.timestamps)}, '
            f'vector_length={self.vector_length:.2f}, sample_stride={self.sample_stride}'
        )

    def publish_field(self) -> None:
        timestamp = self.timestamps[self.timestamp_index]
        coords, velocities = self.lookup.get_all_velocities(timestamp)
        markers = MarkerArray()

        published_vectors = 0
        for index, (coord, velocity) in enumerate(zip(coords, velocities)):
            if index % self.sample_stride != 0:
                continue

            vx = float(velocity[0])
            vy = float(velocity[1])
            speed = math.sqrt(vx * vx + vy * vy)
            if speed < self.min_speed_for_vector:
                continue

            norm = speed if speed > 1e-9 else 1.0
            ux = vx / norm
            uy = vy / norm
            color = self._speed_to_color(speed)

            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'hspar_viz_field'
            marker.id = published_vectors
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.line_width
            marker.scale.y = self.line_width * 2.0
            marker.scale.z = self.line_width * 2.5
            marker.color.r = color.r
            marker.color.g = color.g
            marker.color.b = color.b
            marker.color.a = color.a
            marker.points = [
                Point(x=float(coord[0]), y=float(coord[1]), z=0.0),
                Point(x=float(coord[0] + ux * self.vector_length), y=float(coord[1] + uy * self.vector_length), z=0.0),
            ]
            markers.markers.append(marker)
            published_vectors += 1

        self.publisher.publish(markers)
        self.get_logger().info(
            f'Published {published_vectors} velocity vectors for timestamp {timestamp}'
        )

        self.timestamp_index = (self.timestamp_index + self.timestamp_step) % len(self.timestamps)

    def _speed_to_color(self, speed: float):
        ratio = (speed - self.colormap_min_speed) / (self.colormap_max_speed - self.colormap_min_speed)
        ratio = max(0.0, min(1.0, ratio))

        # Blue -> cyan -> yellow -> red gradient.
        if ratio < 0.33:
            t = ratio / 0.33
            r, g, b = 0.0, t, 1.0
        elif ratio < 0.66:
            t = (ratio - 0.33) / 0.33
            r, g, b = t, 1.0, 1.0 - t
        else:
            t = (ratio - 0.66) / 0.34
            r, g, b = 1.0, 1.0 - t, 0.0

        color = Marker().color
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = 0.95
        return color


def main() -> None:
    rclpy.init()
    node = HsparVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()