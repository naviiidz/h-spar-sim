#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


@dataclass(frozen=True)
class ParticleSample:
    particle_id: int
    x: float
    y: float


class ParticleSamplingNode(Node):
    def __init__(self) -> None:
        super().__init__('particle_sampling_node')

        self.declare_parameter('pose_topic', '/wamv/simple_pose')
        self.declare_parameter(
            'csv_path',
            str(Path('backend/lagrangian_sim/outputs/sydney_regatta/sydney_regatta_lagrangian_particle_trajectories.csv')),
        )
        self.declare_parameter('sim_time_column', 'sim_tim')
        self.declare_parameter('fallback_time_column', 'time')
        self.declare_parameter('front_length', 20.0)
        self.declare_parameter('front_half_width', 5.0)
        self.declare_parameter('yaw_offset', 0.0)
        self.declare_parameter('position_tolerance', 0.5)
        self.declare_parameter(
            'mesh_resource',
            'package://usv_control/worlds/sydney_regatta_shore.dae',
        )
        self.declare_parameter('mesh_frame_id', 'map')
        self.declare_parameter('mesh_scale', 1.0)

        pose_topic = self.get_parameter('pose_topic').value
        csv_path = Path(self.get_parameter('csv_path').value).expanduser()
        self.sim_time_column = str(self.get_parameter('sim_time_column').value)
        self.fallback_time_column = str(self.get_parameter('fallback_time_column').value)
        self.front_length = float(self.get_parameter('front_length').value)
        self.front_half_width = float(self.get_parameter('front_half_width').value)
        self.yaw_offset = float(self.get_parameter('yaw_offset').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.mesh_resource = str(self.get_parameter('mesh_resource').value)
        self.mesh_frame_id = str(self.get_parameter('mesh_frame_id').value)
        self.mesh_scale = float(self.get_parameter('mesh_scale').value)

        self.csv_path = self._resolve_csv_path(csv_path)
        self.samples_by_time = self._load_samples(self.csv_path)
        self.csv_max_time = max(self.samples_by_time) if self.samples_by_time else 0.0
        self.last_reported_key: tuple[float, float, float, float] | None = None

        self.pose_subscription = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'particle_markers', 10)

        self.get_logger().info(
            f'Particle sampling node ready. topic={pose_topic}, csv={self.csv_path}, '
            f'front_length={self.front_length:.2f}, front_half_width={self.front_half_width:.2f}'
        )

    def _resolve_csv_path(self, csv_path: Path) -> Path:
        if csv_path.exists():
            return csv_path

        repo_candidate = Path.cwd().parent / csv_path
        if repo_candidate.exists():
            return repo_candidate

        workspace_candidate = Path.cwd() / csv_path
        if workspace_candidate.exists():
            return workspace_candidate

        raise FileNotFoundError(f'Particle CSV not found: {csv_path}')

    def _load_samples(self, csv_path: Path) -> DefaultDict[float, list[ParticleSample]]:
        samples_by_time: DefaultDict[float, list[ParticleSample]] = defaultdict(list)
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f'Particle CSV has no header: {csv_path}')

            time_column = self.sim_time_column if self.sim_time_column in reader.fieldnames else self.fallback_time_column
            if time_column not in reader.fieldnames:
                raise ValueError(
                    f'Particle CSV must contain either {self.sim_time_column!r} or {self.fallback_time_column!r}; '
                    f'found {reader.fieldnames}'
                )
            for row in reader:
                try:
                    sim_time = float(row[time_column])
                    particle_id = int(float(row['particle_id']))
                    x = float(row['x'])
                    y = float(row['y'])
                except (KeyError, TypeError, ValueError):
                    continue

                samples_by_time[sim_time].append(ParticleSample(particle_id=particle_id, x=x, y=y))

        self.get_logger().info(
            f'Loaded {sum(len(items) for items in samples_by_time.values())} particle samples '
            f'from {len(samples_by_time)} unique simulation times'
        )
        return samples_by_time

    def pose_callback(self, msg: PoseStamped) -> None:
        pose = msg.pose.position
        orientation = msg.pose.orientation
        yaw = self._quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w) + self.yaw_offset

        sim_time = self._extract_sim_time(msg)
        if sim_time is None:
            self.get_logger().warn('Pose message has no usable timestamp; skipping particle lookup')
            return

        lookup_time = self._normalize_sim_time(sim_time)
        samples = self._samples_for_time(lookup_time)
        if not samples:
            self.get_logger().debug(f'No particle samples found for sim time {sim_time:.6f} (lookup={lookup_time:.6f})')
            return

        key = (round(lookup_time, 3), round(pose.x, 3), round(pose.y, 3), round(yaw, 3))
        if self.last_reported_key == key:
            return
        self.last_reported_key = key

        ids_in_front = []
        for sample in samples:
            rel_x, rel_y = self._world_to_robot_frame(sample.x, sample.y, pose.x, pose.y, yaw)
            if 0.0 <= rel_x <= self.front_length and abs(rel_y) <= self.front_half_width:
                ids_in_front.append(sample.particle_id)

        if ids_in_front:
            ids_text = ', '.join(str(particle_id) for particle_id in sorted(set(ids_in_front)))
            self.get_logger().info(
                f'sim_time={sim_time:.6f} lookup_time={lookup_time:.6f} robot=({pose.x:.2f}, {pose.y:.2f}) yaw={yaw:.2f} '
                f'particles_in_front=[{ids_text}]'
            )
        else:
            self.get_logger().info(
                f'sim_time={sim_time:.6f} lookup_time={lookup_time:.6f} robot=({pose.x:.2f}, {pose.y:.2f}) yaw={yaw:.2f} '
                'particles_in_front=[]'
            )

        # Publish all samples at this lookup time as visualization markers
        try:
            marker_array = MarkerArray()

            mesh_marker = Marker()
            mesh_marker.header.frame_id = self.mesh_frame_id
            mesh_marker.header.stamp = self.get_clock().now().to_msg()
            mesh_marker.ns = 'scene_mesh'
            mesh_marker.id = 0
            mesh_marker.type = Marker.MESH_RESOURCE
            mesh_marker.action = Marker.ADD
            mesh_marker.pose.orientation.w = 1.0
            mesh_marker.scale.x = self.mesh_scale
            mesh_marker.scale.y = self.mesh_scale
            mesh_marker.scale.z = self.mesh_scale
            mesh_marker.color.r = 0.59
            mesh_marker.color.g = 0.29
            mesh_marker.color.b = 0.0
            mesh_marker.color.a = 1.0
            mesh_marker.mesh_resource = self.mesh_resource
            mesh_marker.mesh_use_embedded_materials = True
            mesh_marker.lifetime = Duration(sec=1, nanosec=0)
            marker_array.markers.append(mesh_marker)

            for sample in samples:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'particles'
                # Marker id must be non-negative int32
                m.id = int(sample.particle_id)
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(sample.x)
                m.pose.position.y = float(sample.y)
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = 1.0
                m.scale.y = 1.0
                m.scale.z = 1.0
                m.color.r = 0.0
                m.color.g = 0.5
                m.color.b = 1.0
                m.color.a = 0.8
                m.lifetime = Duration(sec=1, nanosec=0)
                marker_array.markers.append(m)
            self.marker_publisher.publish(marker_array)
        except Exception:
            # Don't let visualization errors disrupt node operation
            pass

    def _normalize_sim_time(self, sim_time: float) -> float:
        if self.csv_max_time <= 0.0:
            return sim_time
        return sim_time % self.csv_max_time

    def _samples_for_time(self, sim_time: float) -> list[ParticleSample]:
        if sim_time in self.samples_by_time:
            return self.samples_by_time[sim_time]

        rounded_time = round(sim_time, 3)
        if rounded_time in self.samples_by_time:
            return self.samples_by_time[rounded_time]

        nearest_time = min(self.samples_by_time, key=lambda value: abs(value - sim_time))
        if abs(nearest_time - sim_time) <= self.position_tolerance:
            return self.samples_by_time[nearest_time]
        return []

    def _extract_sim_time(self, msg: PoseStamped) -> float | None:
        stamp = msg.header.stamp
        if stamp is None:
            return None
        return float(stamp.sec) + float(stamp.nanosec) / 1e9

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _world_to_robot_frame(px: float, py: float, rx: float, ry: float, yaw: float) -> tuple[float, float]:
        dx = px - rx
        dy = py - ry
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw
        return forward, lateral


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParticleSamplingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
