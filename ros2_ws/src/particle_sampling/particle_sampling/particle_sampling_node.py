#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Optional, Dict, List

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


@dataclass(frozen=True)
class ParticleDetectionRecord:
    particle_id: int
    robot_x: float
    robot_y: float
    sim_time: float
    lookup_time: float


class ParticleSamplingNode(Node):
    def __init__(self) -> None:
        super().__init__('particle_sampling_node')

        self.declare_parameter('pose_topic', '/wamv/simple_pose')
        self.declare_parameter(
            'csv_path',
            str(Path('/home/navid/h-spar-sim/lagrangian_sim/outputs/sydney_regatta/sydney_regatta_lagrangian_particle_trajectories.csv')),
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

        # New parameter: particle marker diameter in RViz/Gazebo visualization, in meters.
        self.declare_parameter('particle_marker_size', 2.0)

        self.declare_parameter('report_path', '')

        # Probabilistic sampling parameters. Disabled by default to preserve
        # original deterministic behavior; enable with the `probabilistic_sampling`
        # parameter.
        self.declare_parameter('probabilistic_sampling', False)
        self.declare_parameter('sampling_radius', 5.0)
        self.declare_parameter('capture_v0', 0.5)
        self.declare_parameter('capture_sigma', 0.2)

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

        # Read marker-size parameter.
        self.particle_marker_size = float(self.get_parameter('particle_marker_size').value)

        # Probabilistic sampling runtime options
        self.probabilistic_sampling = bool(self.get_parameter('probabilistic_sampling').value)
        self.sampling_radius = float(self.get_parameter('sampling_radius').value)
        self.capture_v0 = float(self.get_parameter('capture_v0').value)
        self.capture_sigma = float(self.get_parameter('capture_sigma').value)

        report_path_value = str(self.get_parameter('report_path').value).strip()

        self.csv_path = self._resolve_csv_path(csv_path)
        self.report_path = self._resolve_report_path(report_path_value)
        self.samples_by_time = self._load_samples(self.csv_path)
        # Build an index of particle trajectories for velocity estimation.
        # Populated by _load_samples as `self.particle_trajectories`.
        self.particle_trajectories: Dict[int, List[tuple[float, float, float]]] = getattr(self, 'particle_trajectories', {})
        self.csv_max_time = max(self.samples_by_time) if self.samples_by_time else 0.0
        self.last_reported_key: tuple[float, float, float, float] | None = None
        self.detection_records: dict[int, ParticleDetectionRecord] = {}
        self.published_particle_ids: set[int] = set()

        # For estimating robot velocity between successive poses
        self.last_pose_x: Optional[float] = None
        self.last_pose_y: Optional[float] = None
        # Real-time stamp of last received pose (seconds since epoch)
        self.last_real_time: Optional[float] = None

        # Internal simulation clock (starts at 0 every run) used to lookup
        # particle samples from the CSV. This is independent of absolute
        # ROS time in incoming poses.
        self.sim_clock: float = 500.0

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            'particle_markers',
            10,
        )

        self.get_logger().info(
            f'Particle sampling node ready. topic={pose_topic}, csv={self.csv_path}, '
            f'front_length={self.front_length:.2f}, front_half_width={self.front_half_width:.2f}, '
            f'particle_marker_size={self.particle_marker_size:.2f}, '
            f'probabilistic_sampling={self.probabilistic_sampling}, sampling_radius={self.sampling_radius:.2f}'
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

    def _resolve_report_path(self, report_path: str) -> Path:
        if report_path:
            return Path(report_path).expanduser()
        return self.csv_path.parent / 'outputs' / 'particle_detections.csv'

    def _load_samples(self, csv_path: Path) -> DefaultDict[float, list[ParticleSample]]:
        samples_by_time: DefaultDict[float, list[ParticleSample]] = defaultdict(list)

        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f'Particle CSV has no header: {csv_path}')

            time_column = (
                self.sim_time_column
                if self.sim_time_column in reader.fieldnames
                else self.fallback_time_column
            )

            if time_column not in reader.fieldnames:
                raise ValueError(
                    f'Particle CSV must contain either {self.sim_time_column!r} '
                    f'or {self.fallback_time_column!r}; found {reader.fieldnames}'
                )

            # Build particle trajectories on-the-fly assuming CSV rows are
            # ordered by simulation time. This avoids a separate per-particle
            # sort after loading which is expensive for large files.
            particle_trajectories: DefaultDict[int, list[tuple[float, float, float]]] = defaultdict(list)

            for row in reader:
                try:
                    sim_time = float(row[time_column])
                    particle_id = int(float(row['particle_id']))
                    x = float(row['x'])
                    y = float(row['y'])
                except (KeyError, TypeError, ValueError):
                    continue

                samples_by_time[sim_time].append(
                    ParticleSample(
                        particle_id=particle_id,
                        x=x,
                        y=y,
                    )
                )

                # Append to this particle's trajectory in time order
                particle_trajectories[particle_id].append((sim_time, x, y))

        # Attach particle trajectories (unsorted). We'll sort per-particle
        # lazily on first velocity request to avoid large startup cost.
        self.particle_trajectories = {pid: v for pid, v in particle_trajectories.items()}
        self._particle_sorted: Dict[int, bool] = {}

        self.get_logger().info(
            f'Loaded {sum(len(items) for items in samples_by_time.values())} particle samples '
            f'from {len(samples_by_time)} unique simulation times; '
            f'particle trajectories={len(self.particle_trajectories)}'
        )

        return samples_by_time

    def pose_callback(self, msg: PoseStamped) -> None:
        pose = msg.pose.position
        orientation = msg.pose.orientation

        yaw = (
            self._quaternion_to_yaw(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
            + self.yaw_offset
        )

        # Advance internal sim clock based on elapsed real time between poses.
        # Use the pose header stamp when available, otherwise use node clock.
        stamp = msg.header.stamp
        if stamp is not None:
            current_real_time = float(stamp.sec) + float(stamp.nanosec) / 1e9
        else:
            # Fallback: use node wall clock
            current_real_time = float(self.get_clock().now().sec) + float(self.get_clock().now().nanosec) / 1e9

        if self.last_real_time is None:
            dt_real = 0.0
        else:
            dt_real = max(0.0, current_real_time - float(self.last_real_time))

        # Advance sim clock (starts at 0 on node startup)
        self.sim_clock += dt_real

        # Lookup particle samples according to the internal sim clock
        lookup_time = self._normalize_sim_time(self.sim_clock)
        samples = self._samples_for_time(lookup_time)

        if not samples:
            self.get_logger().debug(
                f'No particle samples found for sim time {self.sim_clock:.6f} '
                f'(lookup={lookup_time:.6f})'
            )
            return

        key = (
            round(lookup_time, 3),
            round(pose.x, 3),
            round(pose.y, 3),
            round(yaw, 3),
        )

        if self.last_reported_key == key:
            return

        self.last_reported_key = key

        ids_in_front = []

        # Estimate robot velocity from last pose using real timestamps
        v_r_x, v_r_y = 0.0, 0.0
        if self.last_real_time is not None:
            dt_robot = dt_real
            if dt_robot > 0.0 and self.last_pose_x is not None and self.last_pose_y is not None:
                v_r_x = (pose.x - float(self.last_pose_x)) / dt_robot
                v_r_y = (pose.y - float(self.last_pose_y)) / dt_robot

        for sample in samples:
            rel_x, rel_y = self._world_to_robot_frame(
                sample.x,
                sample.y,
                pose.x,
                pose.y,
                yaw,
            )

            if self.probabilistic_sampling:
                # Circular sampling region
                distance = math.hypot(rel_x, rel_y)
                if distance <= self.sampling_radius:
                    # Estimate particle velocity and compute relative speed
                    v_p_x, v_p_y = self._estimate_particle_velocity(sample.particle_id, lookup_time)
                    rel_v_x = v_p_x - v_r_x
                    rel_v_y = v_p_y - v_r_y
                    rel_speed = math.hypot(rel_v_x, rel_v_y)

                    # Capture probability (Gaussian about design intake velocity v0)
                    # Guard against sigma == 0
                    sigma = max(self.capture_sigma, 1e-6)
                    p_c = math.exp(-((rel_speed - self.capture_v0) ** 2) / (2.0 * sigma * sigma))

                    if random.random() < p_c:
                        ids_in_front.append(sample.particle_id)
            else:
                # Original deterministic rectangular front region
                if 0.0 <= rel_x <= self.front_length and abs(rel_y) <= self.front_half_width:
                    ids_in_front.append(sample.particle_id)

        # Update stored robot pose/time for next velocity estimate
        self.last_pose_x = float(pose.x)
        self.last_pose_y = float(pose.y)
        self.last_real_time = float(current_real_time)

        if ids_in_front:
            ids_text = ', '.join(str(particle_id) for particle_id in sorted(set(ids_in_front)))

            self.get_logger().info(
                f'sim_time={self.sim_clock:.6f} lookup_time={lookup_time:.6f} '
                f'robot=({pose.x:.2f}, {pose.y:.2f}) yaw={yaw:.2f} '
                f'particles_in_front=[{ids_text}]'
            )

            for particle_id in ids_in_front:
                if particle_id not in self.detection_records:
                    self.detection_records[particle_id] = ParticleDetectionRecord(
                        particle_id=particle_id,
                        robot_x=float(pose.x),
                        robot_y=float(pose.y),
                        sim_time=float(self.sim_clock),
                        lookup_time=float(lookup_time),
                    )
        else:
            self.get_logger().info(
                f'sim_time={self.sim_clock:.6f} lookup_time={lookup_time:.6f} '
                f'robot=({pose.x:.2f}, {pose.y:.2f}) yaw={yaw:.2f} '
                'particles_in_front=[]'
            )

        # Publish all samples at this lookup time as visualization markers.
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
            # Zero lifetime keeps the latest scene marker visible until it is
            # explicitly replaced or deleted.
            mesh_marker.lifetime = Duration(sec=0, nanosec=0)
            marker_array.markers.append(mesh_marker)

            for sample in samples:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'particles'

                # Marker id must be non-negative int32.
                m.id = int(sample.particle_id)

                m.type = Marker.SPHERE
                m.action = Marker.ADD

                m.pose.position.x = float(sample.x)
                m.pose.position.y = float(sample.y)
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0

                # Particle marker diameter in meters.
                # Increase particle_marker_size to make particles appear larger.
                m.scale.x = self.particle_marker_size
                m.scale.y = self.particle_marker_size
                m.scale.z = self.particle_marker_size

                m.color.r = 0.0
                m.color.g = 0.5
                m.color.b = 1.0
                m.color.a = 0.8

                # Keep particle markers visible when pose messages pause.
                m.lifetime = Duration(sec=0, nanosec=0)
                marker_array.markers.append(m)

            current_particle_ids = {int(sample.particle_id) for sample in samples}
            for particle_id in self.published_particle_ids - current_particle_ids:
                delete_marker = Marker()
                delete_marker.header.frame_id = 'map'
                delete_marker.ns = 'particles'
                delete_marker.id = particle_id
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)

            self.published_particle_ids = current_particle_ids

            self.marker_publisher.publish(marker_array)

        except Exception as exc:
            # Do not let visualization errors disrupt node operation.
            self.get_logger().warn(f'Particle marker publishing failed: {exc}')

    def write_detection_report(self) -> None:
        if not self.detection_records:
            self.get_logger().info(
                'No new particle detections recorded; skipping CSV report'
            )
            return

        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        with self.report_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    'particle_id',
                    'robot_x',
                    'robot_y',
                    'sim_time',
                    'lookup_time',
                ]
            )

            for particle_id in sorted(self.detection_records):
                record = self.detection_records[particle_id]
                writer.writerow(
                    [
                        record.particle_id,
                        f'{record.robot_x:.6f}',
                        f'{record.robot_y:.6f}',
                        f'{record.sim_time:.6f}',
                        f'{record.lookup_time:.6f}',
                    ]
                )

        self.get_logger().info(
            f'Wrote {len(self.detection_records)} first-seen particle detections '
            f'to {self.report_path}'
        )

    def _normalize_sim_time(self, sim_time: float) -> float:
        # The particle simulation uses an internal sim clock that starts at 0
        # each run and advances independently of ROS time. Do not wrap the
        # simulation time; return it directly so lookup is performed against
        # the CSV times as-is.
        return sim_time

    def _samples_for_time(self, sim_time: float) -> list[ParticleSample]:
        if sim_time in self.samples_by_time:
            return self.samples_by_time[sim_time]

        rounded_time = round(sim_time, 3)

        if rounded_time in self.samples_by_time:
            return self.samples_by_time[rounded_time]

        nearest_time = min(
            self.samples_by_time,
            key=lambda value: abs(value - sim_time),
        )

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
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def _world_to_robot_frame(
        px: float,
        py: float,
        rx: float,
        ry: float,
        yaw: float,
    ) -> tuple[float, float]:
        dx = px - rx
        dy = py - ry

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        return forward, lateral

    def _estimate_particle_velocity(self, particle_id: int, sim_time: float) -> tuple[float, float]:
        """Estimate particle velocity by finite differences from loaded samples.

        Performs lazy per-particle sorting: the trajectory for `particle_id`
        is sorted in-place on first use to avoid a heavy startup cost.
        Returns (v_x, v_y) in the same frame as particle positions. If
        insufficient samples exist, returns (0.0, 0.0).
        """
        traj = self.particle_trajectories.get(particle_id)
        if not traj:
            return 0.0, 0.0

        # Sort trajectory on first access if needed
        if not self._particle_sorted.get(particle_id, False):
            try:
                traj.sort(key=lambda e: e[0])
            except Exception:
                pass
            self._particle_sorted[particle_id] = True

        # traj is now sorted list of (time, x, y).
        idx = 0
        while idx < len(traj) and traj[idx][0] < sim_time:
            idx += 1

        # If sim_time is before the first sample, use forward difference
        if idx == 0:
            if len(traj) >= 2:
                t1, x1, y1 = traj[0]
                t2, x2, y2 = traj[1]
                dt = t2 - t1
                if dt > 0:
                    return ((x2 - x1) / dt, (y2 - y1) / dt)
            return 0.0, 0.0

        # If sim_time is after the last sample, use backward difference
        if idx >= len(traj):
            if len(traj) >= 2:
                t1, x1, y1 = traj[-2]
                t2, x2, y2 = traj[-1]
                dt = t2 - t1
                if dt > 0:
                    return ((x2 - x1) / dt, (y2 - y1) / dt)
            return 0.0, 0.0

        # Otherwise use centered difference between surrounding samples
        t1, x1, y1 = traj[idx - 1]
        t2, x2, y2 = traj[idx]
        dt = t2 - t1
        if dt <= 0.0:
            return 0.0, 0.0
        return ((x2 - x1) / dt, (y2 - y1) / dt)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParticleSamplingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_detection_report()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
