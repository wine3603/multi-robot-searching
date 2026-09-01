#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import math
import os
import signal

class TiltDetector(Node):
    def __init__(self, pid):
        super().__init__('tilt_detector')
        self.pid = pid  # PID of livox_monitor to signal when restart needed
        self.min_points = 100
        self.max_topic_timeout = 10.0        # 10s timeout for cloud_registered
        self.get_logger().info(f'tilt_detector started, will send SIGUSR1 to livox_monitor PID {pid} on illegal angle or timeout')
        self.get_logger().info('legal angle range: [0-15]° or [75-105]°')
        self.get_logger().info(f'cloud_registered timeout: {self.max_topic_timeout}s')

        # Subscribe to cloud_registered
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_callback,
            10
        )

        self.latest_cloud = None
        self.last_cloud_time = None
        self.check_timer = self.create_timer(1.0, self.check_tilt)

    def cloud_callback(self, msg):
        self.latest_cloud = msg
        self.last_cloud_time = self.get_clock().now().nanoseconds / 1e9

    def pointcloud2_to_xyz(self, cloud_msg):
        points = []
        if cloud_msg.width == 0:
            return np.array(points)

        x_offset = None
        y_offset = None
        z_offset = None
        for field in cloud_msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        if x_offset is None or y_offset is None or z_offset is None:
            return np.array(points)

        point_step = cloud_msg.point_step
        data = cloud_msg.data
        for i in range(cloud_msg.width * cloud_msg.height):
            x = np.frombuffer(data[(i*point_step + x_offset):(i*point_step + x_offset + 4)], dtype=np.float32)[0]
            y = np.frombuffer(data[(i*point_step + y_offset):(i*point_step + y_offset + 4)], dtype=np.float32)[0]
            z = np.frombuffer(data[(i*point_step + z_offset):(i*point_step + z_offset + 4)], dtype=np.float32)[0]
            if not np.isnan(x) and z >= 0.5:
                points.append((x, y, z))
        return np.array(points, dtype=np.float32)

    def calculate_tilt_angle(self, points):
        """Find the largest plane (ground/ceiling) using RANSAC, calculate tilt angle from z-axis"""
        if len(points) < 100:
            return None

        num_iterations = 200
        distance_threshold = 0.1
        best_inliers = []
        best_normal = None

        for _ in range(num_iterations):
            sample_idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm
            d = -normal.dot(p1)
            distances = np.abs(points.dot(normal) + d)
            inliers = points[distances < distance_threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_normal = normal

        if len(best_inliers) < 200:
            return None

        # Re-fit plane to best inliers using PCA
        mean = np.mean(best_inliers, axis=0)
        centered = best_inliers - mean
        cov = np.cov(centered.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
        except np.linalg.LinAlgError:
            return None

        min_idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, min_idx]

        if normal[2] < 0:
            normal = -normal

        dot = abs(normal[2])
        dot = np.clip(dot, 0.0, 1.0)
        angle_deg = math.acos(dot) * 180.0 / np.pi
        return angle_deg

    def check_tilt(self):
        # Check cloud_registered timeout first
        if self.last_cloud_time is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.last_cloud_time
            if elapsed > self.max_topic_timeout:
                # Timeout: send SIGUSR1 to trigger restart
                self.get_logger().warn(f"cloud_registered TIMEOUT {elapsed:.1f}s > {self.max_topic_timeout:.1f}s → sending SIGUSR1 to livox_monitor PID {self.pid}")
                os.kill(self.pid, signal.SIGUSR1)
                return

        if self.latest_cloud is None:
            return

        points = self.pointcloud2_to_xyz(self.latest_cloud)
        if len(points) < self.min_points:
            return

        angle_deg = self.calculate_tilt_angle(points)
        if angle_deg is None:
            return

        legal = (angle_deg >= 0 and angle_deg <= 15) or (angle_deg >= 75 and angle_deg <= 105)
        self.get_logger().info(f"Current angle: {angle_deg:.1f}° | Legal range: [0-15]° or [75-105]° | Status: {'OK' if legal else 'ILLEGAL'}")

        if not legal:
            # Send SIGUSR1 to livox_monitor to trigger restart
            self.get_logger().warn(f"ILLEGAL angle {angle_deg:.1f}° → sending SIGUSR1 to livox_monitor PID {self.pid}")
            os.kill(self.pid, signal.SIGUSR1)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: tilt_detector.py <livox_monitor_pid>")
        sys.exit(1)

    try:
        pid = int(sys.argv[1])
    except ValueError:
        print("Error: PID must be integer")
        sys.exit(1)

    rclpy.init(args=None)
    node = TiltDetector(pid)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
