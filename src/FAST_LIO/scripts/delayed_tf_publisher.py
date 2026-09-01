#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster
from sensor_msgs_py import point_cloud2 as pc2
from collections import deque
from std_msgs.msg import Header
import numpy as np
import struct
import sys
import signal
import os
import json

FIELD_TYPES = {
    PointField.INT8: 'b', PointField.UINT8: 'B', PointField.INT16: 'h',
    PointField.UINT16: 'H', PointField.INT32: 'i', PointField.UINT32: 'I',
    PointField.FLOAT32: 'f', PointField.FLOAT64: 'd'
}

def transform_to_matrix(t: TransformStamped) -> np.ndarray:
    """Convert transform to 4x4 homogeneous transformation matrix"""
    from scipy.spatial.transform import Rotation as R
    q = t.transform.rotation
    t_vec = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
    rot = R.from_quat([q.x, q.y, q.z, q.w])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = t_vec
    return T

def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to Nx3 points"""
    n = points.shape[0]
    points_h = np.hstack([points, np.ones((n, 1))])  # Nx4
    points_transformed_h = (T @ points_h.T).T  # Nx4
    return points_transformed_h[:, :3]  # Nx3

class DelayedPublisher(Node):
    def __init__(self, n):
        super().__init__('delayed_publisher')

        # Parameters from input
        self.n = n
        self.delay_frames = 30
        self.cumulative_mode = True  # all n are cumulative (need to accumulate and save final transform)
        self.voxel_size = 0.2  # voxel downsample resolution
        self.update_frequency = 1.0  # publish cumulative cloud at 1Hz
        self.cumulative_points = np.empty((0, 3), dtype=np.float32)
        if n == 0:
            self.output_cloud_topic = 'cloud_safe'          # special topic name for first segment
        else:
            self.output_cloud_topic = f'cloud_{n}'          # output point cloud topic: cloud_n

        self.output_cloud_frame = 'camera_init'        # output point cloud frame_id: camera_init
        self.output_tf_parent = 'camera_init'          # output TF parent: camera_init
        self.output_tf_child = 'body_safe'             # output TF child: body_safe
        self.query_parent = 'camera_init'
        self.query_child = 'body'                       # query: camera_init -> body

        # Auto-cleanup when starting n=0: remove all previous /tmp/delayed_tf_*.json
        if n == 0:
            import glob
            old_files = glob.glob('/tmp/delayed_tf_*.json')
            for f in old_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
            if len(old_files) > 0:
                self.get_logger().info(f"Auto-cleaned {len(old_files)} old transform files from /tmp")

        # Coordinate alignment: for n > 0, align new points to previous segment's final pose
        # Accumulate translation and apply rotation from previous segment end
        # New segment starts at previous segment's ending pose with z=0
        self.alignment_T = None  # 4x4 full transformation matrix
        self.first_body_transform = None
        self.prev_final_transform = None  # previous segment's final transform (camera_init -> body_prev)
        self.cumulative_transform = np.eye(4)  # cumulative alignment transform from all previous segments

        # Load and compose all previous transforms from 0 to n-1
        # Align current segment to the final pose of previous segment
        if n > 0:
            loaded_count = 0
            for i in range(n):
                prev_file = f'/tmp/delayed_tf_{i}_final.json'
                if os.path.exists(prev_file):
                    try:
                        with open(prev_file, 'r') as f:
                            data = json.load(f)
                            # Load full transform
                            qx = data['qx']
                            qy = data['qy']
                            qz = data['qz']
                            qw = data['qw']
                            tx = data['tx']
                            ty = data['ty']
                            tz = data['tz']
                            # Convert to matrix
                            from scipy.spatial.transform import Rotation as R
                            T_prev = np.eye(4)
                            rot = R.from_quat([qx, qy, qz, qw])
                            # Extract only yaw rotation (around z-axis) for horizontal orientation
                            # Keep original yaw, discard roll and pitch
                            yaw = rot.as_euler('xyz')[2]
                            rot_yaw_only = R.from_euler('z', yaw)
                            T_prev[:3, :3] = rot_yaw_only.as_matrix()
                            # Set translation: use original xy, set z translation to 0
                            # New segment starts at previous segment end with z=0
                            T_prev[0, 3] = tx
                            T_prev[1, 3] = ty
                            T_prev[2, 3] = 0.0
                            # Compose into cumulative transform
                            self.cumulative_transform = T_prev @ self.cumulative_transform
                            # Keep the last one as prev_final_transform
                            self.prev_final_transform = TransformStamped()
                            self.prev_final_transform.header.stamp.sec = data['stamp_sec']
                            self.prev_final_transform.header.stamp.nanosec = data['stamp_nanosec']
                            self.prev_final_transform.header.frame_id = data['frame_id']
                            self.prev_final_transform.child_frame_id = data['child_frame_id']
                            self.prev_final_transform.transform.translation.x = tx
                            self.prev_final_transform.transform.translation.y = ty
                            self.prev_final_transform.transform.translation.z = data['tz']
                            self.prev_final_transform.transform.rotation.x = qx
                            self.prev_final_transform.transform.rotation.y = qy
                            self.prev_final_transform.transform.rotation.z = qz
                            self.prev_final_transform.transform.rotation.w = qw
                        loaded_count += 1
                    except Exception as e:
                        self.get_logger().warn(f'Failed to load transform {prev_file}: {e}, skipping')
            if loaded_count > 0:
                last_T = self.cumulative_transform
                yaw_deg = np.arctan2(last_T[1, 0], last_T[0, 0]) * 180 / np.pi
                self.get_logger().info(f'Loaded and composed {loaded_count} previous transforms. '
                                      f'Final alignment: tx={last_T[0,3]:.3f}, ty={last_T[1,3]:.3f}, yaw={yaw_deg:.1f}°, z=0')
            else:
                self.get_logger().warn(f'No previous transforms found (0/{n} loaded), alignment disabled')
                self.prev_final_transform = None
                self.cumulative_transform = np.eye(4)

        # State: queue to store (cloud_msg, transform) for delayed output
        self.buffer = deque(maxlen=self.delay_frames + 1)

        # Directly subscribe to /tf to get camera_init -> body
        # Avoid TF buffer lookup failures
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import qos_profile_system_default
        self.latest_body_transform = None
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            qos_profile_system_default
        )

        # Publisher
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, qos)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriber
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_callback,
            10
        )

        # Flag: whether accepting new points (set to False on SIGUSR1 from livox_monitor)
        self.accepting_new_points = True

        # Handle SIGUSR1: stop accepting new points but keep publishing existing, save final transform
        signal.signal(signal.SIGUSR1, self.handle_sigusr1)

        # Timer for cumulative mode
        if self.cumulative_mode:
            self.create_timer(1.0 / self.update_frequency, self.publish_cumulative_cloud)

        self.get_logger().info(f'delayed_publisher initialized:')
        if self.cumulative_mode:
            self.get_logger().info(f'  Mode: CUMULATIVE (voxel downsample {self.voxel_size}m)')
            self.get_logger().info(f'  Delay: {self.delay_frames} frames (add 50-frame-old cloud to cumulative map)')
            if self.prev_final_transform is not None:
                self.get_logger().info(f'  Coordinate alignment: ENABLED (using cloud_{n-1} final transform)')
            else:
                self.get_logger().info(f'  Coordinate alignment: DISABLED (no previous transform file)')
            self.get_logger().info(f'  SIGUSR1 handler installed: will stop accepting new points on signal')
        else:
            self.get_logger().info(f'  Mode: DELAYED (publish 50-frame-old single cloud)')
        self.get_logger().info(f'  Input cloud: /cloud_registered')
        self.get_logger().info(f'  Output cloud: {self.output_cloud_topic} (frame_id: {self.output_cloud_frame})')
        self.get_logger().info(f'  Output TF: {self.output_tf_parent} -> {self.output_tf_child}')

    def tf_callback(self, msg):
        """Get camera_init -> body transform directly from /tf topic"""
        for transform in msg.transforms:
            if transform.header.frame_id == self.query_parent and transform.child_frame_id == self.query_child:
                self.latest_body_transform = transform

    def handle_sigusr1(self, signum, frame):
        """Handle SIGUSR1: stop accepting new points, keep publishing existing, save final transform"""
        if self.accepting_new_points:
            self.accepting_new_points = False

            # If cumulative mode, save current latest body transform to file for next restart
            if self.cumulative_mode:
                try:
                    # Get latest body transform from our subscribed list
                    latest_transform = self.latest_body_transform
                    if latest_transform is None:
                        raise RuntimeError("No body_transform received yet")
                    # Only save when cumulative points count meets minimum requirement
                    if len(self.cumulative_points) < 10000:
                        raise RuntimeError(f"Insufficient points ({len(self.cumulative_points)} < 10000), skipping save")
                    # Save to file for next cloud_{n+1} to use
                    save_file = f'/tmp/delayed_tf_{self.n}_final.json'
                    data = {
                        'stamp_sec': latest_transform.header.stamp.sec,
                        'stamp_nanosec': latest_transform.header.stamp.nanosec,
                        'frame_id': latest_transform.header.frame_id,
                        'child_frame_id': latest_transform.child_frame_id,
                        'tx': latest_transform.transform.translation.x,
                        'ty': latest_transform.transform.translation.y,
                        'tz': latest_transform.transform.translation.z,
                        'qx': latest_transform.transform.rotation.x,
                        'qy': latest_transform.transform.rotation.y,
                        'qz': latest_transform.transform.rotation.z,
                        'qw': latest_transform.transform.rotation.w
                    }
                    with open(save_file, 'w') as f:
                        json.dump(data, f)
                    self.get_logger().info(f'Received SIGUSR1 -> STOPPED accepting new points, '
                                           f'saved final transform to {save_file}, '
                                           f'will keep publishing {len(self.cumulative_points)} existing points')
                except Exception as e:
                    self.get_logger().error(f'Failed to save final transform: {e}')
                    self.get_logger().info(f'Received SIGUSR1 -> STOPPED accepting new points')
            else:
                self.get_logger().info(f'Received SIGUSR1 -> STOPPED accepting new points')

    def pc2numpy(self, msg):
        """Convert PointCloud2 to numpy array (x,y,z)"""
        try:
            fs = {f.name:(f.offset, FIELD_TYPES[f.datatype]) for f in msg.fields}
            out = []
            for i in range(0, len(msg.data), msg.point_step):
                d = msg.data[i:i+msg.point_step]
                x = struct.unpack(fs['x'][1], d[fs['x'][0]:fs['x'][0]+4])[0]
                y = struct.unpack(fs['y'][1], d[fs['y'][0]:fs['y'][0]+4])[0]
                z = struct.unpack(fs['z'][1], d[fs['z'][0]:fs['z'][0]+4])[0]
                if not np.isnan(x):
                    out.append((x,y,z))
            return np.array(out, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f'Failed to parse point cloud: {e}', throttle_duration_sec=5.0)
            return None

    def remove_duplicate_points(self, points, voxel_size=0.2):
        """Voxel downsampling: keep one point per voxel"""
        if len(points) < 2:
            return points
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        voxel_dict = {}
        for idx, coords in enumerate(voxel_coords):
            voxel_key = (coords[0], coords[1], coords[2])
            voxel_dict[voxel_key] = idx  # newer points overwrite older ones
        kept_indices = list(voxel_dict.values())
        return points[kept_indices]

    def publish_cumulative_cloud(self):
        """Publish accumulated point cloud (cumulative mode only)"""
        # Only publish when we have at least 10000 points
        if len(self.cumulative_points) < 10000:
            return
        h = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.output_cloud_frame)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg = pc2.create_cloud(h, fields, self.cumulative_points.tolist())
        self.cloud_pub.publish(cloud_msg)
        #self.get_logger().info(f'[cumulative] Published {len(self.cumulative_points)} points (voxel size {self.voxel_size}m)', throttle_duration_sec=5.0)

    def cloud_callback(self, msg):
        # If stopped accepting new points, do nothing
        # We still keep the timer running for cumulative mode to keep publishing
        if not self.accepting_new_points:
            return

        # Get current transform directly from subscribed /tf topic
        if self.latest_body_transform is None:
            self.get_logger().warn('Waiting for camera_init -> body transform...', throttle_duration_sec=5.0)
            return
        transform = self.latest_body_transform

        # Add current cloud and transform to buffer
        self.buffer.append( (msg, transform) )

        # If we have enough frames, process the oldest one (50 frames ago)
        if len(self.buffer) >= self.delay_frames:
            # Get 50-frame-old data from front of queue
            delayed_cloud, delayed_transform = self.buffer.popleft()

            if self.cumulative_mode and self.accepting_new_points:
                # Cumulative mode: add points to cumulative buffer with voxel downsampling
                # Only add if we are still accepting new points
                new_points = self.pc2numpy(delayed_cloud)
                if new_points is not None and len(new_points) > 0:

                    # Coordinate alignment if we have previous transform
                    if self.prev_final_transform is not None:
                        # First frame: get first body transform and compute full alignment
                        # Align current segment to previous segment's ending pose
                        if self.first_body_transform is None:
                            self.first_body_transform = delayed_transform
                            # Directly use the already-composed cumulative transform from all previous segments
                            # All points in current segment are already in camera_init frame,
                            # just apply the cumulative alignment: move to previous end, rotate to previous orientation, set z=0
                            self.alignment_T = self.cumulative_transform

                            # Log final alignment
                            T_align = self.cumulative_transform
                            yaw = np.arctan2(T_align[1, 0], T_align[0, 0])
                            yaw_deg = yaw * 180 / np.pi
                            self.get_logger().info(f'Computed full alignment for cloud_{self.n} '
                                                  f'(tx={T_align[0,3]:.3f}, ty={T_align[1,3]:.3f}, yaw={yaw_deg:.1f}°, z=0)')

                        # Apply alignment to all new points
                        if self.alignment_T is not None:
                            new_points = apply_transform(new_points, self.alignment_T)

                    if len(self.cumulative_points) > 0:
                        combined = np.vstack([self.cumulative_points, new_points])
                    else:
                        combined = new_points
                    # Voxel downsample to control density
                    if len(combined) > 100:
                        self.cumulative_points = self.remove_duplicate_points(combined, voxel_size=self.voxel_size)
                    else:
                        self.cumulative_points = combined
            elif not self.cumulative_mode:
                # Delayed mode: publish single cloud
                # Publish delayed point cloud with updated frame_id and current timestamp
                output_cloud = PointCloud2()
                output_cloud.header.stamp = self.get_clock().now().to_msg()
                output_cloud.header.frame_id = self.output_cloud_frame
                output_cloud.height = delayed_cloud.height
                output_cloud.width = delayed_cloud.width
                output_cloud.fields = delayed_cloud.fields
                output_cloud.is_bigendian = delayed_cloud.is_bigendian
                output_cloud.point_step = delayed_cloud.point_step
                output_cloud.row_step = delayed_cloud.row_step
                output_cloud.data = delayed_cloud.data
                output_cloud.is_dense = delayed_cloud.is_dense
                self.cloud_pub.publish(output_cloud)

            # Always publish delayed transform as camera_init -> body_safe
            # Use same current timestamp
            out_tf = TransformStamped()
            out_tf.header.stamp = self.get_clock().now().to_msg()
            out_tf.header.frame_id = self.output_tf_parent
            out_tf.child_frame_id = self.output_tf_child
            out_tf.transform = delayed_transform.transform
            self.tf_broadcaster.sendTransform(out_tf)

def main():
    if len(sys.argv) < 2:
        print("Usage: delayed_publisher.py <n>")
        print("  n - The number for output cloud topic 'cloud_n'")
        print("      if n=0, output topic is 'cloud_safe' (special case)")
        print("  Example: delayed_publisher.py 100")
        print("  Output:")
        print("   - Topic: /cloud_100 (frame_id=camera_init)")
        print("   - TF: camera_init -> body_safe (50 frames delayed transform)")
        print("  Example: delayed_publisher.py 0")
        print("  Output:")
        print("   - Topic: /cloud_safe (frame_id=camera_init)")
        print("   - TF: camera_init -> body_safe (50 frames delayed transform)")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: n must be an integer")
        sys.exit(1)

    rclpy.init()
    node = DelayedPublisher(n)
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, rclpy._rclpy_pybind11.RCLError):
        # Expected when killed by parent process, exit cleanly
        pass
    except Exception:
        # Any other exception during spin, still exit cleanly
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except (rclpy._rclpy_pybind11.RCLError, Exception):
            # Context already shutdown or other errors, ignore
            pass

if __name__ == '__main__':
    main()
