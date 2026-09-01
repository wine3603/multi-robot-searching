#!/usr/bin/env python3
"""
自动校准 LiDAR 到 IMU 的时间偏移 (time_offset_lidar_to_imu)
原理：通过网格搜索不同偏移值，选择去畸变后点云平面平整度最好（方差最小）的偏移量

用法:
  python calib_time_offset.py --bag your_bag.db3 --lidar_topic /livox/lidar --imu_topic /livox/imu --frame_id camera_init

输出:
  会画出误差曲线，输出最优时间偏移值
"""

import argparse
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from livox_interfaces.msg import CustomMsg as LivoxCustomMsg
from sensor_msgs.msg import Imu
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Auto calibrate time_offset_lidar_to_imu')
    parser.add_argument('--bag', required=True, help='ROS2 bag file path')
    parser.add_argument('--lidar_topic', default='/livox/lidar', help='LiDAR topic name')
    parser.add_argument('--imu_topic', default='/livox/imu', help='IMU topic name')
    parser.add_argument('--min_offset', default=-0.2, type=float, help='Minimum time offset to search (seconds)')
    parser.add_argument('--max_offset', default=0.2, type=float, help='Maximum time offset to search (seconds)')
    parser.add_argument('--step', default=0.01, type=float, help='Search step (seconds)')
    parser.add_argument('--num_scans', default=20, type=int, help='Number of scans to use for evaluation')
    parser.add_argument('--min_points', default=100, type=int, help='Minimum points per scan')
    parser.add_argument('--planarity_threshold', default=0.1, type=float,
                        help='Maximum allowed variance for planar fit. Lower = stricter')
    args = parser.parse_args()
    return args


def read_ros2_bag(bag_path, lidar_topic, imu_topic):
    """Read lidar and imu messages from ROS2 bag"""
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    print(f"Topics in bag:")
    for topic in topic_types:
        print(f"  - {topic.name}: {topic.type}")

    if lidar_topic not in type_map:
        raise ValueError(f"Lidar topic {lidar_topic} not found in bag")
    if imu_topic not in type_map:
        raise ValueError(f"IMU topic {imu_topic} not found in bag")

    lidar_msgs = []
    imu_msgs = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == lidar_topic:
            if 'CustomMsg' in type_map[topic]:
                # Livox custom message format
                msg = LivoxCustomMsg()
                msg.deserialize(data)
                points = []
                for p in msg.points:
                    points.append([p.x, p.y, p.z, p.timestamp / 1e9])  # timestamp in seconds
                if len(points) > 0:
                    lidar_msgs.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                      np.array(points, dtype=np.float32)))
            else:
                # Standard PointCloud2
                msg = PointCloud2()
                msg.deserialize(data)
                points_list = []
                for p in read_points(msg, field_names=('x', 'y', 'z', 'timestamp'), skip_nans=True):
                    if len(p) == 4:
                        points_list.append([p[0], p[1], p[2], p[3]])
                    else:
                        points_list.append([p[0], p[1], p[2], 0.0])
                if len(points_list) > 0:
                    lidar_msgs.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                      np.array(points_list, dtype=np.float32)))
        elif topic == imu_topic:
            msg = Imu()
            msg.deserialize(data)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            imu_msgs.append((t, gyro, acc))

    print(f"\nLoaded:")
    print(f"  - {len(lidar_msgs)} lidar scans")
    print(f"  - {len(imu_msgs)} IMU messages")

    return lidar_msgs, imu_msgs


def integrate_gyro(gyro_sequence, time_sequence, start_time, end_time):
    """Integrate gyro from start_time to end_time to get incremental rotation"""
    # Find all gyro measurements between start and end
    mask = (time_sequence >= start_time) & (time_sequence <= end_time)
    times = time_sequence[mask]
    gyros = gyro_sequence[mask]

    if len(times) < 2:
        return np.eye(3)

    dR = np.eye(3)
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        w = (gyros[i] + gyros[i-1]) / 2
        angle = np.linalg.norm(w) * dt
        if angle < 1e-10:
            continue
        axis = w / np.linalg.norm(w)
        # Rodrigues rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        dR = R @ dR

    return dR


def undistort_point_cloud(points, scan_start_time, time_offset, gyro_data, imu_times):
    """
    Undistort point cloud using gyro data
    points: [N, 4] - x, y, z, point_timestamp_rel
    """
    undistorted = []
    avg_point_time = scan_start_time + np.mean(points[:, 3]) if points.shape[0] > 0 else scan_start_time

    for pt in points:
        x, y, z, pt_rel_t = pt
        pt_time = scan_start_time + pt_rel_t + time_offset
        # Get rotation from point_time to avg_point_time
        dR = integrate_gyro(gyro_data, imu_times, pt_time, avg_point_time)
        # Rotate point to undistorted frame
        pt_original = np.array([x, y, z])
        pt_undistorted = dR @ pt_original
        undistorted.append(pt_undistorted)

    return np.array(undistorted)


def compute_planarity_score(points):
    """
    Compute planarity score for point cloud.
    Lower score = more planar (better)
    Returns None if too few points
    """
    if len(points) < 10:
        return None

    # Use Open3D to compute covariance and eigenvalues
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals and get covariance
    if len(pcd.points) < 5:
        return None

    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        # Get eigenvalues from covariance analysis
        cov = np.cov(points.T)
        eigenvalues, _ = np.linalg.eig(cov)
        eigenvalues.sort()
        # Smallest eigenvalue / (sum of eigenvalues) = planarity
        # Lower = more planar
        planarity = eigenvalues[0] / np.sum(eigenvalues)
        return planarity
    except:
        return None


def evaluate_time_offset(lidar_msgs, imu_msgs, time_offset, num_scans_eval, min_points):
    """Evaluate a given time offset, return average planarity score (lower = better)"""
    # Extract IMU data
    imu_times = np.array([t for t, _, _ in imu_msgs])
    gyro_data = np.array([gyro for _, gyro, _ in imu_msgs])

    # Select evenly space scans for evaluation
    step = max(1, len(lidar_msgs) // num_scans_eval)
    selected_scans = lidar_msgs[::step][:num_scans_eval]

    scores = []
    for scan_time, points in selected_scans:
        if len(points) < min_points:
            continue
        # Need point timestamps (relative to scan start)
        if np.allclose(points[:, 3], 0):
            # If no per-point timestamp, approximate by assuming linear sweep
            # This is common for some lidars
            points[:, 3] = np.linspace(0, 1.0 / 10, len(points))  # assume 10Hz scan

        undistorted = undistort_point_cloud(points, scan_time, time_offset, gyro_data, imu_times)
        score = compute_planarity_score(undistorted)
        if score is not None:
            scores.append(score)

    if len(scores) == 0:
        return 1e9  # bad score

    return np.mean(scores)


def plot_results(offset_values, scores, best_offset):
    """Plot search results"""
    plt.figure(figsize=(10, 6))
    plt.plot(offset_values, scores, 'b.-', linewidth=1, markersize=8)
    plt.scatter([best_offset], [scores[np.argmin(scores)]],
                color='red', s=100, zorder=5,
                label=f'Best offset: {best_offset:.4f} s')
    plt.xlabel('Time Offset (seconds)')
    plt.ylabel('Planarity Score (lower = better)')
    plt.title('Time Offset Calibration Search')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_offset_calib_result.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: time_offset_calib_result.png")
    plt.show()


def main():
    args = parse_args()

    print("=" * 60)
    print("Auto LiDAR-IMU Time Offset Calibration")
    print("=" * 60)

    # Read bag
    print(f"\nReading bag: {args.bag}")
    lidar_msgs, imu_msgs = read_ros2_bag(args.bag, args.lidar_topic, args.imu_topic)

    if len(lidar_msgs) < 5:
        print("Error: Not enough lidar scans in bag")
        return
    if len(imu_msgs) < 100:
        print("Error: Not enough IMU messages in bag")
        return

    # Create search grid
    offset_values = np.arange(args.min_offset, args.max_offset + args.step, args.step)
    print(f"\nSearching {len(offset_values)} offset values from "
          f"{args.min_offset:.3f}s to {args.max_offset:.3f}s (step = {args.step:.3f}s)")

    # Evaluate each offset
    scores = []
    for offset in tqdm(offset_values):
        score = evaluate_time_offset(lidar_msgs, imu_msgs, offset,
                                    args.num_scans, args.min_points)
        scores.append(score)

    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_offset = offset_values[best_idx]
    best_score = scores[best_idx]

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Best time_offset_lidar_to_imu = {best_offset:.4f} seconds")
    print(f"Score = {best_score:.6f} (lower = better)")
    print("\nUpdate your mid360.yaml:")
    print(f"    time_offset_lidar_to_imu: {best_offset:.4f}")
    print("=" * 60)

    # Plot results
    try:
        plot_results(offset_values, scores, best_offset)
    except Exception as e:
        print(f"\nCould not plot: {e}")
        print("Raw results:")
        for o, s in zip(offset_values, scores):
            print(f"  {o:.4f}: {s:.6f}")


if __name__ == '__main__':
    main()
