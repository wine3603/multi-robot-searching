#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_system_default
import subprocess
import os
import signal
import time
import numpy as np

# 仓库根: <root>/src/FAST_LIO/scripts/ ，用于定位同目录脚本与 workspace install
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class LivoxMonitor(Node):
    def __init__(self, ros_setup, workspace_setup):
        super().__init__('livox_monitor')
        self.ros_setup = ros_setup
        self.workspace_setup = workspace_setup

        # Subscribe to /Odometry to get latest pose
        # Trigger detection every time we get a new odometry message
        self.latest_odom = None
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            qos_profile_system_default
        )

        # 订阅livox自定义点云话题 - 用于检查点数量
        self.lidar_sub = self.create_subscription(
            CustomMsg,
            '/livox/lidar',
            self.lidar_callback,
            10
        )

        # Handle SIGUSR1 from tilt_detector - illegal angle or timeout triggers restart
        signal.signal(signal.SIGUSR1, self.handle_sigusr1)

        # 统计信息
        self.frame_count = 0
        self.restart_count = 0
        self.last_restart_time = time.time()
        self.good_frames_count = 0  # 连续正常帧数计数，达到safety_buffer_frames才安全
        # 连续超时重启计数 - 连续两次超时就不再重启
        self.consecutive_timeout_restarts = 0
        self.max_consecutive_timeouts = 2  # 连续多少次超时就停止重启

        # 参数设置
        self.cooldown_sec = 3.0              # 重启后冷却时间，至少X秒才开始检查速度位置
        self.safety_buffer_frames = 10       # 连续 N 帧正常才认为安全
        self.max_topic_timeout = 10.0        # 最大超时，超过10s重启
        self.speed_threshold = 2.0           # 最大总速度，m/s
        self.point_threshold = 15000         # 最少点云数量
        self.target_frame = 'camera_init'
        self.source_frame = 'body'

        # Only keep previous frame data for velocity calculation (no buffer)
        self.last_time = None
        self.last_x = None
        self.last_y = None
        self.last_z = None

        # Buffer recent Z positions for moving average detection
        # Trigger restart if current Z deviates too much from moving average
        self.z_buffer = []
        self.max_z_buffer = 10
        self.z_deviation_threshold = 0.5  # max allowed deviation from moving average

        # Store previous orientation for angular velocity check
        self.last_orientation_time = None
        self.last_roll = None
        self.last_pitch = None
        self.last_yaw = None
        self.max_angular_speed = 80.0  # degrees per second, any component > this triggers restart

        # Track consecutive huge jumps - only restart after N consecutive jumps
        self.consecutive_huge_jumps = 0
        self.max_consecutive_huge_jumps = 3  # restart after N consecutive huge jumps

        # Speed check: require N consecutive frames speed over threshold to trigger restart
        self.consecutive_speed_over = 0
        self.required_consecutive_over = 2  # require this many consecutive over speed to restart

        # Parameter to control speed check printing
        self.enable_speed_print = True  # set to True to enable printing every 0.1s

        # 进程管理
        self.launch_process = None
        self.rviz_process = None
        self.gmg_process = None
        # self.tf_echo_process = None  # removed - no longer needed
        self.tilt_detector_process = None
        # Keep all delayed_tf_publisher processes:
        # - Each restart creates a NEW process for new cumulative cloud cloud_n
        # - Old processes keep running and continue publishing cloud_0 ... cloud_{n-1}
        # - Old processes are signaled to STOP ACCEPTING new points, only publish existing
        self.delayed_tf_processes = []

        self.get_logger().info('Livox monitor initialized')
        self.get_logger().info('Auto-restart conditions (all trigger if any):')
        self.get_logger().info('  1. XY speed > %.1f m/s (check after 5s cooldown)' % self.speed_threshold)
        self.get_logger().info('  2. Z speed abs > 0.5 m/s (check after 5s cooldown)')
        self.get_logger().info('  3. Z position out of [-1.0, 1.5] (check after 5s cooldown)')
        self.get_logger().info('  4. Point cloud too small (< %d points) (check anytime)' % self.point_threshold)
        self.get_logger().info('  5. Tilt angle not in [0-15] or [75-105] deg (checked by separate tilt_detector process)' )
        self.get_logger().info('  6. No cloud_registered > 10s (checked by separate tilt_detector process)')
        self.get_logger().info('Safe buffer: %d consecutive good frames' % self.safety_buffer_frames)
        self.get_logger().info('Watching TF: %s -> %s' % (self.source_frame, self.target_frame))
        self.get_logger().info('Tilt detection + timeout checking in separate process: tilt_detector.py')

        # 启动各模块
        self.start_fastlio()
        self.start_rviz()
        self.start_grid_map_generator()
        self.start_delayed_tf_publisher(0)
        # self.start_tf_echo()  # no longer needed since we get pose from /Odometry
        self.start_tilt_detector()

    def start_tf_echo(self):
        self.get_logger().info('Starting ros2 topic echo /tf (output suppressed)...')
        cmd = 'ros2 topic echo /tf'
        self.tf_echo_process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def start_tilt_detector(self):
        main_pid = os.getpid()
        cmd = f'python3 {SCRIPT_DIR}/tilt_detector.py {main_pid}'
        self.tilt_detector_process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def is_safe(self):
        # 检查连续正常帧数是否达到安全缓冲
        return self.good_frames_count >= self.safety_buffer_frames

    def start_fastlio(self):
        if self.launch_process is not None and self.launch_process.poll() is None:
            self.get_logger().info('Killing old fast_lio process...')
            os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
            time.sleep(1)

        cmd = f'bash -c "source {self.ros_setup} && source {self.workspace_setup} && ros2 launch fast_lio mapping.launch.py rviz:=false"'
        self.get_logger().info('Starting fast_lio mapping.launch.py (silent mode)...')
        self.launch_process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)
        self.last_restart_time = time.time()
        self.get_logger().info('fast_lio started, restart count: %d' % self.restart_count)

    def start_rviz(self):
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            return  # already running
        self.get_logger().info('Starting rviz2...')
        self.rviz_process = subprocess.Popen(
            'rviz2',
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def start_grid_map_generator(self):
        self.get_logger().info('Starting grid_map_generator.py...')
        cmd = f'python3 {SCRIPT_DIR}/grid_map_generator.py'
        self.gmg_process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def start_delayed_tf_publisher(self, restart_n: int):
        # If there is a previous current delayed_tf_publisher,
        # send SIGUSR1 to it to stop accepting new points
        # DO NOT KILL it - it will keep publishing existing cumulative cloud forever
        if len(self.delayed_tf_processes) > 0:
            last_proc = self.delayed_tf_processes[-1]
            if last_proc is not None and last_proc.poll() is None:
                self.get_logger().info(f'Sending SIGUSR1 to previous delayed_tf_publisher (PID {last_proc.pid}) to stop accepting new points...')
                # Send signal to entire process group
                os.killpg(os.getpgid(last_proc.pid), signal.SIGUSR1)

        self.get_logger().info(f'Starting NEW delayed_tf_publisher.py for cloud_{restart_n}...')
        cmd = f'python3 {SCRIPT_DIR}/delayed_tf_publisher.py {restart_n}'
        new_proc = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            #stdout=subprocess.DEVNULL,
            #stderr=subprocess.DEVNULL
        )
        self.delayed_tf_processes.append(new_proc)

    def restart_fastlio(self, reason):
        self.restart_count += 1
        self.get_logger().warn('=' * 60)
        self.get_logger().warn('AUTO RESTART TRIGGERED: %s' % reason)
        self.get_logger().warn('Total restarts: %d' % self.restart_count)
        self.get_logger().warn('=' * 60)

        # Kill old tilt_detector before restart
        if self.tilt_detector_process is not None and self.tilt_detector_process.poll() is None:
            os.killpg(os.getpgid(self.tilt_detector_process.pid), signal.SIGTERM)

        # Reset all counters immediately BEFORE restart
        self.good_frames_count = 0
        self.last_time = None
        self.last_x = None
        self.last_y = None
        self.last_z = None
        # Reset consecutive timeout count - this is a new restart attempt
        self.consecutive_timeout_restarts = 0
        # Reset consecutive huge jumps counter - this is a new restart attempt
        self.consecutive_huge_jumps = 0
        # Reset consecutive speed over counter - this is a new restart attempt
        self.consecutive_speed_over = 0
        # Reset Z buffer for moving average - start fresh after restart
        self.z_buffer.clear()
        # Reset orientation history - start fresh after restart
        self.last_orientation_time = None
        self.last_roll = None
        self.last_pitch = None
        self.last_yaw = None

        self.last_restart_time = time.time()
        self.start_fastlio()
        # Restart tilt_detector for new session
        self.start_tilt_detector()
        # DO NOT kill old delayed_tf_publisher:
        # - Previous cloud_{n-1} is signaled to stop accepting new points
        # - Old process keeps running and continues publishing cloud_{n-1}
        # - Start NEW delayed_tf_publisher for new cloud_n to accumulate new points
        self.start_delayed_tf_publisher(self.restart_count)

    def lidar_callback(self, msg: CustomMsg):
        # CustomMsg中points数量就是点云数量
        num_points = len(msg.points)
        timestamp = msg.header.stamp
        self.frame_count += 1

        # 检查点云数量，少于阈值触发重启
        if num_points < self.point_threshold:
            print(f"[LIDAR RESTART] Frame: {self.frame_count:<4} | Points: {num_points:>6} | Stamp: {timestamp.sec:>10}.{timestamp.nanosec:09d}", flush=True)
            self.restart_fastlio(f"Point cloud too small: {num_points} < {self.point_threshold}")
            return
        else:
            # This frame is good, increment counter
            self.good_frames_count += 1

    def odom_callback(self, msg):
        # Update latest odometry and trigger speed/pose check
        self.latest_odom = msg
        self.check_speed()

    def check_speed(self):
        # Check cooldown: restart after at least X seconds
        elapsed = time.time() - self.last_restart_time
        if elapsed < self.cooldown_sec:
            # No printing during cooldown
            return

        # Get latest pose from /Odometry
        if self.latest_odom is None:
            # No odometry received yet
            return

        current_time = self.latest_odom.header.stamp.sec + self.latest_odom.header.stamp.nanosec * 1e-9
        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        z = self.latest_odom.pose.pose.position.z
        q = self.latest_odom.pose.pose.orientation

        # Check for huge jump (>2m) in any coordinate (SLAM drift jump)
        huge_jump = False
        if self.last_time is not None and self.last_x is not None:
            dx = abs(x - self.last_x)
            dy = abs(y - self.last_y)
            dz = abs(z - self.last_z)
            if dx > 2.0 or dy > 2.0 or dz > 2.0:
                huge_jump = True

        # Always add current Z to moving average buffer
        self.z_buffer.append(z)
        if len(self.z_buffer) > self.max_z_buffer:
            self.z_buffer.pop(0)

        # Track consecutive huge jumps
        vx = vy = vz = v_mag = 0.0
        restart = False
        reason = ""
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q.x, q.y, q.z, q.w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        if huge_jump:
            self.consecutive_huge_jumps += 1
            if self.consecutive_huge_jumps >= self.max_consecutive_huge_jumps:
                restart = True
                reason = f"Huge position jump {self.consecutive_huge_jumps} consecutive times (dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f})"
        else:
            # Reset counter when no huge jump
            self.consecutive_huge_jumps = 0

        # Only do detection if no huge jump (or already restarting due to consecutive jumps)
        if not huge_jump or restart:
            # Calculate speed
            if self.last_time is not None and self.last_x is not None:
                dt = current_time - self.last_time
                if dt > 0:
                    dx = x - self.last_x
                    dy = y - self.last_y
                    dz = z - self.last_z
                    vx = dx / dt
                    vy = dy / dt
                    vz = dz / dt
                    v_mag = ((dx**2 + dy**2)**0.5) / dt
                else:
                    v_mag = 0.0

            # Check position Z deviation from moving average
            if len(self.z_buffer) >= self.max_z_buffer:
                z_avg = np.mean(self.z_buffer)
                z_deviation = abs(z - z_avg)
                if z_deviation > self.z_deviation_threshold:
                    restart = True
                    reason = f"Z deviation {z_deviation:.3f} > {self.z_deviation_threshold:.1f} (current={z:.3f}, avg={z_avg:.3f})"
            # Check speed - require consecutive frames over threshold to restart
            if v_mag > self.speed_threshold:
                self.consecutive_speed_over += 1
                if self.consecutive_speed_over >= self.required_consecutive_over:
                    restart = True
                    reason += f" | {self.consecutive_speed_over} consecutive frames: XY speed {v_mag:.3f} > {self.speed_threshold} m/s"
            else:
                self.consecutive_speed_over = 0

            # Check angular velocity - any component > 120 deg/s triggers restart
            def angle_diff(a, b):
                """Calculate minimal absolute angle difference in degrees, handle wrapping at 360"""
                diff = abs(a - b)
                return min(diff, 360 - diff)

            if self.last_orientation_time is not None:
                dt = current_time - self.last_orientation_time
                if dt > 0:
                    droll = angle_diff(roll, self.last_roll) / dt
                    dpitch = angle_diff(pitch, self.last_pitch) / dt
                    dyaw = angle_diff(yaw, self.last_yaw) / dt
                    if droll > self.max_angular_speed or dpitch > self.max_angular_speed or dyaw > self.max_angular_speed:
                        restart = True
                        reason += f" | Angular speed too high (roll={droll:.1f}°, pitch={dpitch:.1f}°, yaw={dyaw:.1f}°)/s > {self.max_angular_speed}°/s"

        # Always update stored state - this frame becomes previous frame for next check
        # Even if huge jump, we still cache it so next check compares with it
        self.last_time = current_time
        self.last_x = x
        self.last_y = y
        self.last_z = z
        self.last_orientation_time = current_time
        self.last_roll = roll
        self.last_pitch = pitch
        self.last_yaw = yaw

        # Z speed check removed - no longer restart due to fast Z speed
        # if abs(vz) > 0.5:
        #     restart = True
        #     reason += f" | Z speed {abs(vz):.3f} > 0.5 m/s"

        # Print current state only if enabled
        if self.enable_speed_print:
            t_sec = self.latest_odom.header.stamp.sec
            t_nano = self.latest_odom.header.stamp.nanosec
            z_avg = np.mean(self.z_buffer) if len(self.z_buffer) > 0 else z
            self.get_logger().info(f"[Speed check] stamp={t_sec}.{t_nano:09d} | x={x:.3f} y={y:.3f} z={z:.3f} | speed={v_mag:.3f} m/s | z_avg={z_avg:.3f}")

        # Only print when restart is triggered
        if restart:
            t_sec = self.latest_odom.header.stamp.sec
            t_nano = self.latest_odom.header.stamp.nanosec
            print(f"[RESTART] Stamp: {t_sec:>10}.{t_nano:09d} | {reason}", flush=True)
            self.restart_fastlio(reason)

    def handle_sigusr1(self, signum, frame):
        """Handle SIGUSR1 from tilt_detector - illegal angle or timeout triggers restart"""
        # Ignore all signals during cooldown period
        elapsed = time.time() - self.last_restart_time
        if elapsed < self.cooldown_sec:
            self.get_logger().debug(f"Ignoring SIGUSR1 during cooldown ({elapsed:.1f}s < {self.cooldown_sec}s)")
            return

        self.get_logger().warn('Received SIGUSR1 from tilt_detector → triggering restart')

        # Check if this is a timeout - tilt_detector sends SIGUSR1 for both illegal angle and timeout
        # We can't distinguish, but consecutive timeouts will be counted correctly if it's actually timeout
        self.consecutive_timeout_restarts += 1

        if self.consecutive_timeout_restarts >= self.max_consecutive_timeouts:
            # 连续达到最大超时次数，不再重启，只kill fastlio
            self.get_logger().warn('=' * 60)
            self.get_logger().warn(f"STOP RESTARTING: {self.consecutive_timeout_restarts} consecutive timeouts/errors")
            self.get_logger().warn(f"Killing fastlio process, will NOT restart again")
            self.get_logger().warn('=' * 60)
            # Kill fastlio only, keep node running
            if self.launch_process is not None and self.launch_process.poll() is None:
                os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
            self.launch_process = None
        else:
            # Less than max, restart as usual
            self.restart_fastlio('Illegal tilt angle or cloud_registered timeout detected by tilt_detector')

    def shutdown(self):
        if self.launch_process is not None and self.launch_process.poll() is None:
            os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            os.killpg(os.getpgid(self.rviz_process.pid), signal.SIGTERM)
        if self.gmg_process is not None and self.gmg_process.poll() is None:
            os.killpg(os.getpgid(self.gmg_process.pid), signal.SIGTERM)
        # self.tf_echo_process removed - no longer used
        if self.tilt_detector_process is not None and self.tilt_detector_process.poll() is None:
            os.killpg(os.getpgid(self.tilt_detector_process.pid), signal.SIGTERM)
        # Kill all delayed_tf_publisher processes on shutdown
        for proc in self.delayed_tf_processes:
            if proc is not None and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Force kill all remaining python and ros2 child processes
        try:
            subprocess.run(['pkill', '-f', 'python'], check=False)
            subprocess.run(['pkill', '-f', 'ros2.*launch'], check=False)
        except:
            pass

def main(args=None):
    workspace_setup = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'install', 'setup.bash'))
    ros_setup = '/opt/ros/humble/setup.bash'

    # Restart ros2 daemon once at startup
    print('[livox_monitor] Restarting ros2 daemon at startup...')
    try:
        subprocess.run(['ros2', 'daemon', 'stop'], check=True, capture_output=True)
        time.sleep(2)
        subprocess.run(['ros2', 'daemon', 'start'], check=True, capture_output=True)
        time.sleep(2)
        print('[livox_monitor] ros2 daemon restart done')
    except Exception as e:
        print(f'[livox_monitor] Failed to restart: {e}')

    rclpy.init(args=args)
    monitor = LivoxMonitor(ros_setup, workspace_setup)

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user. Total restarts: %d" % monitor.restart_count)
    except Exception:
        pass
    finally:
        monitor.shutdown()
        monitor.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            # Ignore if already shutdown
            pass

if __name__ == '__main__':
    main()
