#!/usr/bin/env python3

from __future__ import annotations

import bisect
import csv
import math
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Optional

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
        repo_root = Path(os.environ.get('HSPAR_ROOT', str(Path.home() / 'h-spar-sim')))
        self.declare_parameter(
            'csv_path',
            str(repo_root / 'lagrangian_sim/outputs/sydney_regatta/sydney_regatta_lagrangian_particle_trajectories.csv'),
        )
        self.declare_parameter('sim_time_column', 'sim_tim')
        self.declare_parameter('fallback_time_column', 'time')
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

        # Probabilistic sampling (paper Sec. 2.6): a particle within sampling_radius of
        # the USV is captured as a Poisson process with rate lambda_p = Q / V_s,
        # V_s = pi r_s^2 h, i.e. with probability 1 - exp(-lambda_p dt) per step.
        # Disabled by default: deterministic sampling, every particle within
        # sampling_radius is sampled.
        self.declare_parameter('probabilistic_sampling', False)
        self.declare_parameter('sampling_radius', 5.0)
        self.declare_parameter('pump_flow_m3h', 25.0)
        self.declare_parameter('intake_layer_m', 0.1)
        # Seed of the capture draws; record it with the results so a run can be repeated.
        self.declare_parameter('seed', 0)
        # Particle (CSV) time when the Gazebo clock reads 0. The particle time is
        # particle_time_offset_s + Gazebo time, so particles and the current
        # (drag_query_server field_time_offset_s) stay on the same clock.
        self.declare_parameter('particle_time_offset_s', 1200.0)

        pose_topic = self.get_parameter('pose_topic').value
        csv_path = Path(self.get_parameter('csv_path').value).expanduser()
        self.sim_time_column = str(self.get_parameter('sim_time_column').value)
        self.fallback_time_column = str(self.get_parameter('fallback_time_column').value)
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
        self.pump_flow_m3s = float(self.get_parameter('pump_flow_m3h').value) / 3600.0
        self.intake_layer_m = float(self.get_parameter('intake_layer_m').value)
        self.seed = int(self.get_parameter('seed').value)
        self.rng = random.Random(self.seed)
        self.particle_time_offset_s = float(self.get_parameter('particle_time_offset_s').value)
        v_s = math.pi * self.sampling_radius ** 2 * self.intake_layer_m
        self.capture_rate = self.pump_flow_m3s / v_s if v_s > 0.0 else 0.0  # lambda_p (1/s)

        # Running totals for the report: particles that entered the sampling region,
        # expected number sampled (lambda_p * integral of N_c dt), and a 1 s time series.
        self.encountered_ids: set[int] = set()
        self.expected_sampled = 0.0
        self.timeseries: list[tuple[float, float, int, float, int]] = []
        self.first_stamp: Optional[float] = None
        self.last_stamp: Optional[float] = None
        self._last_series_t: Optional[float] = None

        report_path_value = str(self.get_parameter('report_path').value).strip()

        self.csv_path = self._resolve_csv_path(csv_path)
        self.report_path = self._resolve_report_path(report_path_value)
        self.samples_by_time = self._load_samples(self.csv_path)
        self.csv_max_time = max(self.samples_by_time) if self.samples_by_time else 0.0
        self.sorted_times = sorted(self.samples_by_time)
        self.last_reported_key: tuple[float, float, float, float] | None = None
        self.detection_records: dict[int, ParticleDetectionRecord] = {}
        self.published_particle_ids: set[int] = set()
        self._last_published_samples: Optional[list] = None

        # For estimating robot velocity between successive poses
        self.last_pose_x: Optional[float] = None
        self.last_pose_y: Optional[float] = None
        # Real-time stamp of last received pose (seconds since epoch)
        self.last_real_time: Optional[float] = None

        self.sim_clock: float = self.particle_time_offset_s

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
            f'particle_marker_size={self.particle_marker_size:.2f}, '
            f'probabilistic_sampling={self.probabilistic_sampling}, sampling_radius={self.sampling_radius:.2f}, '
            f'lambda_p={self.capture_rate:.3e} 1/s, seed={self.seed}, '
            f'particle_time_offset_s={self.particle_time_offset_s:g}'
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

        self.get_logger().info(
            f'Loaded {sum(len(items) for items in samples_by_time.values())} particle samples '
            f'from {len(samples_by_time)} unique simulation times'
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

        # Particle time = offset + Gazebo time from the pose stamp (simulation time).
        stamp = msg.header.stamp
        current_real_time = float(stamp.sec) + float(stamp.nanosec) / 1e9
        if self.last_real_time is None:
            dt_real = 0.0
        else:
            dt_real = max(0.0, current_real_time - float(self.last_real_time))
        if self.first_stamp is None:
            self.first_stamp = current_real_time
        self.last_stamp = current_real_time
        self.sim_clock = self.particle_time_offset_s + current_real_time

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
        n_candidates = 0
        # Per-step capture probability; composes to the same cumulative probability
        # for any pose rate.
        p_c = 1.0 - math.exp(-self.capture_rate * dt_real)

        for sample in samples:
            rel_x, rel_y = self._world_to_robot_frame(
                sample.x,
                sample.y,
                pose.x,
                pose.y,
                yaw,
            )

            # Encounter: the particle is within sampling_radius of the USV.
            if math.hypot(rel_x, rel_y) > self.sampling_radius:
                continue
            n_candidates += 1
            self.encountered_ids.add(sample.particle_id)
            if sample.particle_id in self.detection_records:
                continue
            # Deterministic: every encountered particle is sampled.
            # Probabilistic: sampled with probability p_c in this step.
            if not self.probabilistic_sampling or self.rng.random() < p_c:
                ids_in_front.append(sample.particle_id)

        if self.probabilistic_sampling:
            self.expected_sampled += self.capture_rate * n_candidates * dt_real
        self._record_series(current_real_time, n_candidates)

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

        # Publish the particles as visualization markers when the snapshot changes
        # (particle positions are stored once per CSV time step).
        if samples is self._last_published_samples:
            return
        self._last_published_samples = samples
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

    def _record_series(self, t: float, n_candidates: int) -> None:
        """One row per second of Gazebo time: time, particle time, N_c, E[N_s] so far, sampled so far."""
        if self._last_series_t is not None and t - self._last_series_t < 1.0:
            return
        self._last_series_t = t
        self.timeseries.append((t, self.sim_clock, n_candidates, self.expected_sampled, len(self.detection_records)))

    def write_summary(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'csv_path': str(self.csv_path),
            'probabilistic_sampling': self.probabilistic_sampling,
            'seed': self.seed,
            'sampling_radius_m': self.sampling_radius,
            'pump_flow_m3h': self.pump_flow_m3s * 3600.0,
            'intake_layer_m': self.intake_layer_m,
            'capture_rate_per_s': self.capture_rate,
            'particle_time_offset_s': self.particle_time_offset_s,
            'gazebo_time_first_pose_s': self.first_stamp,
            'gazebo_time_last_pose_s': self.last_stamp,
            'encountered': len(self.encountered_ids),
            'sampled': len(self.detection_records),
            'expected_sampled': self.expected_sampled if self.probabilistic_sampling else None,
        }
        stem = self.report_path.with_suffix('')
        Path(f'{stem}_summary.json').write_text(json.dumps(summary, indent=2))
        with Path(f'{stem}_timeseries.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['gazebo_time', 'particle_time', 'n_candidates', 'expected_sampled', 'sampled'])
            for row in self.timeseries:
                writer.writerow([f'{row[0]:.3f}', f'{row[1]:.3f}', row[2], f'{row[3]:.6f}', row[4]])
        self.get_logger().info(
            f'Encountered {summary["encountered"]}, sampled {summary["sampled"]}, '
            f'expected {summary["expected_sampled"]}; wrote {stem}_summary.json and _timeseries.csv'
        )

    def write_detection_report(self) -> None:
        self.write_summary()
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

        if not self.sorted_times:
            return []
        i = bisect.bisect_left(self.sorted_times, sim_time)
        nearest_time = min(self.sorted_times[max(i - 1, 0):i + 1], key=lambda value: abs(value - sim_time))

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
