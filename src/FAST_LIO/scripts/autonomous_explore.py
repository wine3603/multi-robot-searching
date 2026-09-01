#!/usr/bin/env python3
import os
import platform
import sys
import rclpy
import time
import math
import json
import numpy as np
import subprocess
import threading
import termios
import tty
from datetime import datetime
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

# 不直接使用SDK，改为通过ROS2话题发布命令给listen_command.py处理

EXIT_FLAG = False
EMERGENCY_STOP = False  # 空格键按下后紧急停止标志
USER_CONFIRM = False   # q键按下确认移动
OBSTACLE_THRESHOLD = 50
UNKNOWN_THRESHOLD = -1
FREE_THRESHOLD = 50
OBSTACLE_SAFE_RADIUS = 4  # 障碍物安全距离，单位：格

# 结果保存目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploration_result")


class AutonomousExplorer(Node):
    def __init__(self, robot_ip="192.168.234.18", robot_port=43988):
        super().__init__('autonomous_explorer')

        # 探索参数配置
        self.goal_tolerance = 0.2       # 到达目标容差（米）
        self.min_frontier_size = 1
        self.cluster_distance = 0.6     # 前沿点聚类距离（米）
        self.effective_min_dist = 0.2   # 过滤离已访问目标太近的前沿（米）
        self.robot_safe_radius_grid = 4  # 机器人离障碍物安全距离（格）
        self.unreachable_distance_threshold = 0.4  # 不可达目标过滤距离（米）
        self.max_move_duration = 30.0   # 单次移动最大持续时间（秒），防止卡住
        self.stop_no_frontier_timeout = 30.0  # 无前沿多久后停止探索

        # 当前SLAM地图
        self.current_map = None
        self.map_received = False
        self.visited_grid = None  # 对应已探索区域（map中 != -1）
        self.last_map_update_time = 0

        # 机器人当前位姿（从里程计获取）
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False

        # 探索状态
        self.frontiers = []
        self.exploration_goals = []
        self.current_goal = None
        self.current_planned_path = []
        self.exploration_active = False

        # 不可达目标缓存，避免重复选择
        self.unreachable_targets = set()
        # 记录已访问位置，避免重复回到同一个地方
        self.visited_positions = []

        # 安全缓存：每个栅格只计算一次安全性
        self.safety_cache = None
        self.safety_cache_computed = None

        # 保存所有A*规划路径的历史，用于可视化
        self.all_planned_paths = []

        # 图片保存标志：探索线程设置，主线程定时器执行matplotlib保存
        self.need_save_map = False

        # 运动状态
        self.nav_state = 'idle'  # idle, navigating
        self.move_start_time = 0.0
        self.last_frontier_time = 0.0

        # 机器人运动速度参数（根据你的机器人调整）
        self.linear_vel = 0.4    # 线速度 m/s
        self.angular_vel = 0.5   # 角速度 rad/s
        self.yaw_tolerance = math.radians(2.0)  # 角度容差 1.5度

        # ==================== 数据记录与保存 ====================
        self.start_time = time.time()
        self.total_steps = 0
        self.exploration_trajectory = []  # 机器人实际轨迹 [(x, y, timestamp), ...]
        self.frontier_detection_times = []  # 前沿检测耗时历史
        self.coverage_history = []  # 覆盖率历史
        self.time_history = []  # 时间历史

        # 创建结果保存目录
        os.makedirs(RESULT_DIR, exist_ok=True)
        self.result_json_path = os.path.join(RESULT_DIR, "result.json")
        self.get_logger().info(f"💾 探索数据将实时保存到: {RESULT_DIR}")

        # 机器人控制 - 通过ROS2话题发布命令给listen_command.py
        # 需要用户手动启动listen_command.py
        self.cmd_pub = self.create_publisher(String, '/agibot_cmd', 10)

        # 发送起立命令
        self.get_logger().info("🤖 发送机器人起立命令...")
        self.publish_command('1')
        time.sleep(5)
        self.get_logger().info("🤖 机器人已准备探索")

        # 订阅SLAM建图得到的地图
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # 订阅机器人里程计获取实时位姿
        self.odom_sub = self.create_subscription(
            Odometry, '/Odometry', self.odom_callback, 10)

        # 可视化发布器
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.goal_path_pub = self.create_publisher(Path, '/exploration_goals', 10)
        self.planned_path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.robot_marker_pub = self.create_publisher(Marker, '/robot_position', 10)

        # 后台探索线程
        self.exploration_thread = threading.Thread(target=self.exploration_loop, daemon=True)
        self.exploration_thread.start()

        # 键盘监听线程 - 空格键按下触发紧急停止
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        # 零速度发送线程 - 紧急停止时持续发送0速度
        self.stop_thread = None

        # 定时记录机器人位置（0.1s一次）
        self.traj_timer = self.create_timer(0.1, self.record_trajectory)

        # 定时更新可视化
        self.vis_timer = self.create_timer(1.0, self.publish_visualization)

        self.get_logger().info("🚀 自主探索节点已启动")
        self.get_logger().info("⌨️  按下空格键：清除不可达/已到达判断 + 紧急停止")
        self.get_logger().info(f"📐 参数: 目标容差={self.goal_tolerance}m, 聚类距离={self.cluster_distance}m")

    def publish_command(self, command):
        """发布命令到/agibot_cmd话题"""
        msg = String()
        msg.data = str(command)
        self.cmd_pub.publish(msg)
        self.get_logger().debug(f"📤 Published command: {command}")

    def map_callback(self, msg):
        """接收SLAM建图得到的最新地图"""
        self.current_map = msg
        self.map_received = True
        width = self.current_map.info.width
        height = self.current_map.info.height

        # 初始化已探索标记：map中 != -1 的区域就是已探索
        if self.visited_grid is None or self.visited_grid.shape != (height, width):
            self.visited_grid = np.zeros((height, width), dtype=bool)
            self.safety_cache = np.zeros((height, width), dtype=bool)
            self.safety_cache_computed = np.zeros((height, width), dtype=bool)
            self.get_logger().info(
                f"🗺️  收到SLAM地图: {width}x{height}, "
                f"分辨率={msg.info.resolution}m, 原点=({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})"
            )

        # 更新已探索区域标记
        map_data = np.array(msg.data).reshape(height, width)
        self.visited_grid = (map_data != UNKNOWN_THRESHOLD)
        # 地图数据更新后，安全缓存失效，下次访问时重新计算
        self.safety_cache_computed[:] = False
        self.last_map_update_time = self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg):
        """接收里程计获取机器人实时位姿
        IMU在机器人头部，需要沿朝向向后偏移0.2米得到机器人中心坐标
        """
        imu_x = msg.pose.pose.position.x
        imu_y = msg.pose.pose.position.y
        # 从四元数提取yaw角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        # IMU在头部，沿着yaw方向后退0.2米得到机器人中心坐标
        offset_dist = 0.2
        self.robot_x = imu_x - offset_dist * math.cos(self.robot_yaw)
        self.robot_y = imu_y - offset_dist * math.sin(self.robot_yaw)
        self.odom_received = True

    def record_trajectory(self):
        """定时记录机器人位置（0.1s一次）"""
        if self.odom_received and self.exploration_active:
            self.exploration_trajectory.append((self.robot_x, self.robot_y, time.time()))

    def world_to_grid(self, wx, wy):
        """世界坐标转栅格坐标"""
        if self.current_map is None:
            return None
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution
        gx = int(round((wx - origin_x) / res))
        gy = int(round((wy - origin_y) / res))
        if gx < 0 or gx >= self.current_map.info.width or gy < 0 or gy >= self.current_map.info.height:
            return None
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        """栅格坐标转世界坐标"""
        if self.current_map is None:
            return None
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution
        wx = origin_x + gx * res
        wy = origin_y + gy * res
        return (wx, wy)

    def get_cell_value(self, gx, gy):
        """获取当前地图中栅格的值"""
        if self.current_map is None:
            return UNKNOWN_THRESHOLD
        width = self.current_map.info.width
        idx = gy * width + gx
        if idx < 0 or idx >= len(self.current_map.data):
            return UNKNOWN_THRESHOLD
        return self.current_map.data[idx]

    def detect_frontiers(self):
        """检测前沿（未知区域相邻的已知自由区域边界）
        使用分层安全距离：优先8格，逐步递减到4格
        """
        if self.current_map is None or self.visited_grid is None:
            return []

        import time
        t_start = time.time()

        # 分层尝试不同安全距离
        for safe_radius in [8, 6, 4]:
            frontiers = self._detect_frontiers_with_radius(safe_radius)
            if len(frontiers) > 0:
                if safe_radius != 4:
                    self.get_logger().info(f"🛡️ 使用 {safe_radius} 格安全距离找到 {len(frontiers)} 个前沿")
                break

        # 对找到的前沿进行聚类（使用较大距离减少密度）
        self.frontiers = self.cluster_frontiers(frontiers, self.cluster_distance)
        original_count = len(self.frontiers)
        # 过滤掉已经标记为不可达的目标
        self.frontiers = [f for f in self.frontiers if not any(
            math.hypot(f[0] - ur[0], f[1] - ur[1]) < self.unreachable_distance_threshold
            for ur in self.unreachable_targets
        )]

        # 如果没有找到前沿，生成已探索区域内的安全点作为替代
        if len(self.frontiers) == 0:
            self.get_logger().info("🔍 未检测到有效前沿，尝试生成已探索区域内的安全点...")
            self.frontiers = self.generate_safe_points_as_frontiers()
            # 如果生成了安全点但聚类后为空，说明无法继续探索，直接结束
            if len(self.frontiers) == 0:
                self.get_logger().warning("🏁 已探索区域无可用安全点，探索完成")
                self.exploration_active = False

        t_elapsed = time.time() - t_start
        if original_count != len(self.frontiers):
            self.get_logger().info(f"🔍 前沿检测完成: {original_count} → {len(self.frontiers)} 个有效前沿，耗时 {t_elapsed*1000:.1f}ms")
        else:
            self.get_logger().info(f"🔍 前沿检测完成: {len(self.frontiers)} 个有效前沿，耗时 {t_elapsed*1000:.1f}ms")

        if len(self.frontiers) > 0:
            self.last_frontier_time = time.time()

        return self.frontiers

    def _detect_frontiers_with_radius(self, safe_radius):
        """使用指定安全半径检测前沿（内部函数）
        Args:
            safe_radius: 安全半径（格）
        Returns:
            前沿点列表 [(wx, wy), ...]
        """
        frontiers = []
        width = self.current_map.info.width
        height = self.current_map.info.height

        # 8邻域
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0), (1, 1)]

        # 先创建一个二值图：1 = 已探索，0 = 未探索
        explored_binary = self.visited_grid.astype(np.float32)

        # 使用Sobel边缘检测找出所有边缘点（已探索和未探索的边界）
        from scipy.ndimage import sobel
        edge_mag = np.abs(sobel(explored_binary))
        # 边缘点就是梯度大于0.1的地方（有变化就是边界）
        edge_coords = np.argwhere(edge_mag > 0.1)

        # 只遍历边缘候选点，大大减少循环次数
        for (gy, gx) in edge_coords:
            val = self.get_cell_value(gx, gy)
            # 如果当前格子是已知自由
            if val >= 0 and val < FREE_THRESHOLD:
                    # 确认相邻确实有未知格子
                    has_unknown_neighbor = False
                    for dx, dy in neighbors:
                        ngx = gx + dx
                        ngy = gy + dy
                        if 0 <= ngx < width and 0 <= ngy < height:
                            nval = self.get_cell_value(ngx, ngy)
                            if nval == UNKNOWN_THRESHOLD:
                                has_unknown_neighbor = True
                                break
                    if has_unknown_neighbor:
                        # 检查周围safe_radius格内是否有障碍物
                        has_obstacle_nearby = False
                        check_start = -safe_radius
                        check_end = safe_radius
                        for dx in range(check_start, check_end + 1):
                            for dy in range(check_start, check_end + 1):
                                ngx = gx + dx
                                ngy = gy + dy
                                if 0 <= ngx < width and 0 <= ngy < height:
                                    nval = self.get_cell_value(ngx, ngy)
                                    if nval > OBSTACLE_THRESHOLD:
                                        has_obstacle_nearby = True
                                        break
                            if has_obstacle_nearby:
                                break
                        if not has_obstacle_nearby:
                            wx, wy = self.grid_to_world(gx, gy)
                            # 检查是否离已选目标太近，避免重复选择同一区域
                            too_close = False
                            for (tx, ty) in self.exploration_goals:
                                if math.hypot(tx - wx, ty - wy) < self.effective_min_dist:
                                    too_close = True
                                    break
                            if not too_close:
                                # 检查是否离当前机器人太近
                                if len(self.exploration_goals) > 0 or math.hypot(wx - self.robot_x, wy - self.robot_y) > 1.0:
                                    frontiers.append((wx, wy))

        return frontiers

    def generate_safe_points_as_frontiers(self):
        """在已探索区域的边界上生成安全点作为备选前沿
        策略：找到已探索与未探索的边界点，筛选其中安全的点
        """
        if self.current_map is None:
            return []

        width = self.current_map.info.width
        height = self.current_map.info.height
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution

        # 用Sobel找到已探索区域的边界
        from scipy.ndimage import sobel
        explored_binary = self.visited_grid.astype(np.float32)
        edge_mag = np.abs(sobel(explored_binary))
        edge_coords = np.argwhere(edge_mag > 0.1)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        safe_points = []
        for (gy, gx) in edge_coords:
            val = self.get_cell_value(gx, gy)
            # 必须是已探索的自由格子
            if val < 0 or val >= FREE_THRESHOLD:
                continue
            # 必须是边界：至少一个邻居是未探索的
            is_boundary = False
            for dx, dy in neighbors:
                ngx, ngy = gx + dx, gy + dy
                if 0 <= ngx < width and 0 <= ngy < height:
                    if not self.visited_grid[ngy, ngx]:
                        is_boundary = True
                        break
            if not is_boundary:
                continue

            wx = gx * res + origin_x
            wy = gy * res + origin_y
            # 检查安全性
            if not self.is_position_safe(wx, wy):
                continue
            # 过滤离已选目标太近的
            too_close = False
            for (tx, ty) in self.exploration_goals:
                if math.hypot(tx - wx, ty - wy) < self.effective_min_dist:
                    too_close = True
                    break
            if too_close:
                continue
            # 过滤离当前机器人太近的
            if math.hypot(wx - self.robot_x, wy - self.robot_y) < 1.5:
                continue
            # 过滤不可达目标
            if any(math.hypot(wx - ur[0], wy - ur[1]) < self.unreachable_distance_threshold
                   for ur in self.unreachable_targets):
                continue

            safe_points.append((wx, wy))

        # 直接返回安全点（不聚类，只进行可达性过滤）
        if len(safe_points) > 0:
            self.get_logger().info(f"🔍 在已探索边界生成了 {len(safe_points)} 个安全点（不聚类）")
            return safe_points
        else:
            self.get_logger().warning("⚠️ 无法在边界生成安全点，探索可能卡住")
            return []

    def cluster_frontiers(self, frontiers, cluster_distance=1.0):
        """对前沿点进行聚类，返回聚类中心
        聚类后再次过滤，移除离障碍物太近的中心
        """
        if not frontiers:
            return []

        clusters = []
        for point in frontiers:
            wx, wy = point
            found = False
            for i, cluster in enumerate(clusters):
                cx, cy = np.mean(cluster, axis=0)
                dist = math.sqrt((wx - cx)**2 + (wy - cy)**2)
                if dist < cluster_distance:
                    cluster.append(point)
                    found = True
                    break
            if not found:
                clusters.append([point])

        # 过滤掉太小的簇，返回中心，并且过滤掉离障碍物太近的中心
        centers = []
        for cluster in clusters:
            if len(cluster) >= self.min_frontier_size:
                cx = np.mean([p[0] for p in cluster])
                cy = np.mean([p[1] for p in cluster])
                # 再次检查中心是否安全
                if self.is_position_safe(cx, cy):
                    # 检查是否在不可达目标附近，避免重复选择同一位置
                    if not any(math.hypot(cx - ur[0], cy - ur[1]) < self.unreachable_distance_threshold
                               for ur in self.unreachable_targets):
                        centers.append((cx, cy))

        return centers

    def find_farthest_frontier(self):
        """找到离机器人当前位置最远的前沿"""
        if not self.frontiers:
            return None
        max_dist = -1
        farthest = None
        for fx, fy in self.frontiers:
            dist = math.sqrt((fx - self.robot_x)**2 + (fy - self.robot_y)**2)
            if dist > max_dist:
                max_dist = dist
                farthest = (fx, fy)
        return farthest

    def is_position_safe(self, wx, wy, safe_radius=None):
        """检查位置是否安全：距离障碍物至少指定格数
        使用缓存：每个栅格只计算一次，结果永久复用，避免重复计算
        Args:
            wx, wy: 世界坐标
            safe_radius: 安全半径（格），默认使用robot_safe_radius_grid。
                        使用非默认值时跳过缓存。
        """
        if safe_radius is None:
            safe_radius = self.robot_safe_radius_grid

        gp = self.world_to_grid(wx, wy)
        if gp is None:
            return False
        gx, gy = gp
        width = self.current_map.info.width
        height = self.current_map.info.height

        # 使用默认半径时，检查缓存
        use_cache = (safe_radius == self.robot_safe_radius_grid)
        if use_cache and self.safety_cache_computed[gy, gx]:
            return self.safety_cache[gy, gx]

        # 第一次计算，检查周围
        check_start = -safe_radius
        check_end = safe_radius
        safe = True
        for dx in range(check_start, check_end + 1):
            for dy in range(check_start, check_end + 1):
                ngx = gx + dx
                ngy = gy + dy
                if 0 <= ngx < width and 0 <= ngy < height:
                    # 只检查已经探索过的区域，未知区域不算障碍物
                    if self.visited_grid[ngy, ngx]:
                        nval = self.get_cell_value(ngx, ngy)
                        if nval > OBSTACLE_THRESHOLD:
                            # 找到了障碍物，位置不安全
                            safe = False
                            break
                if not safe:
                    break
            if not safe:
                break

        # 缓存结果（仅限默认半径）
        if use_cache:
            self.safety_cache[gy, gx] = safe
            self.safety_cache_computed[gy, gx] = True
        return safe

    def calculate_coverage(self):
        """计算当前探索覆盖率"""
        if self.current_map is None:
            return 0.0
        width = self.current_map.info.width
        height = self.current_map.info.height
        total_cells = width * height
        explored_cells = np.sum(self.visited_grid)
        return explored_cells / total_cells

    def save_exploration_result(self):
        """实时保存探索结果（覆盖式保存）"""
        if self.current_map is None:
            return

        total_elapsed = time.time() - self.start_time
        coverage = self.calculate_coverage()

        # 计算总移动距离
        total_distance = 0.0
        for i in range(1, len(self.exploration_trajectory)):
            prev = self.exploration_trajectory[i-1]
            curr = self.exploration_trajectory[i]
            total_distance += math.hypot(curr[0] - prev[0], curr[1] - prev[1])

        # 前沿检测性能统计
        avg_frontier_time = np.mean(self.frontier_detection_times) if self.frontier_detection_times else 0.0

        result = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": float(total_elapsed),
            "total_steps": int(self.total_steps),
            "current_coverage": float(coverage),
            "total_distance_meters": float(total_distance),
            "performance": {
                "avg_frontier_detection_ms": float(avg_frontier_time * 1000),
                "frontier_detection_times_ms": [t * 1000 for t in self.frontier_detection_times]
            },
            "parameters": {
                "cluster_distance_meters": float(self.cluster_distance),
                "effective_min_dist_meters": float(self.effective_min_dist),
                "obstacle_safe_radius_grid": int(self.robot_safe_radius_grid),
                "min_frontier_size": int(self.min_frontier_size),
                "stop_no_frontier_timeout_seconds": float(self.stop_no_frontier_timeout)
            },
            "history": {
                "time_seconds": self.time_history,
                "coverage": self.coverage_history
            },
            "trajectory": [(float(x), float(y)) for (x, y, _) in self.exploration_trajectory],
            "exploration_goals": [(float(x), float(y)) for (x, y) in self.exploration_goals],
            "current_robot_pose": {
                "x": float(self.robot_x),
                "y": float(self.robot_y),
                "yaw": float(self.robot_yaw)
            }
        }

        # 覆盖式保存
        with open(self.result_json_path, 'w') as f:
            json.dump(result, f, indent=2)

        self.get_logger().debug(f"💾 探索数据已保存: 覆盖率 {coverage:.2%}, 步数 {self.total_steps}")

    def save_map_visualization(self):
        """保存地图可视化（PNG）"""
        if self.current_map is None:
            return

        import matplotlib.pyplot as plt

        width = self.current_map.info.width
        height = self.current_map.info.height
        map_data = np.array(self.current_map.data).reshape(height, width)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(map_data, cmap='gray_r', vmin=-1, vmax=100, origin='lower')

        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution

        # 绘制实际轨迹
        if self.exploration_trajectory:
            traj_x = [(p[0] - origin_x) / res for p in self.exploration_trajectory]
            traj_y = [(p[1] - origin_y) / res for p in self.exploration_trajectory]
            ax.plot(traj_x, traj_y, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')

        # 绘制目标点
        if self.exploration_goals:
            goals_x = [(p[0] - origin_x) / res for p in self.exploration_goals]
            goals_y = [(p[1] - origin_y) / res for p in self.exploration_goals]
            ax.scatter(goals_x, goals_y, c='red', s=30, marker='*', label='Goals')

        # 绘制当前机器人位置
        rx = (self.robot_x - origin_x) / res
        ry = (self.robot_y - origin_y) / res
        ax.scatter(rx, ry, c='green', s=100, marker='o', label='Robot')

        ax.set_title(f"Exploration Map - Coverage: {self.calculate_coverage():.2%}")
        ax.legend()
        ax.grid(True, alpha=0.2)

        map_png_path = os.path.join(RESULT_DIR, "exploration_map.png")
        fig.savefig(map_png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.get_logger().info(f"🗺️  地图可视化已保存: {map_png_path}")

    def can_connect_directly(self, start_wx, start_wy, end_wx, end_wy):
        """检查两点之间是否可以直接连通（沿直线没有障碍物）"""
        # Bresenham直线算法检查
        start_gp = self.world_to_grid(start_wx, start_wy)
        end_gp = self.world_to_grid(end_wx, end_wy)
        if start_gp is None or end_gp is None:
            return False
        x0, y0 = start_gp
        x1, y1 = end_gp

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x = x0
        y = y0

        while True:
            # 如果是未知栅格，不允许直连（未知区域可能有障碍物，不能走）
            # 只有整条路径都在已探索区域内才允许直连
            if not self.visited_grid[y, x]:
                return False
            # 检查当前点是否安全（距离障碍物足够远）
            if not self.is_position_safe(
                self.grid_to_world(x, y)[0],
                self.grid_to_world(x, y)[1]
            ):
                return False
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True

    def simplify_path(self, path):
        """路径简化：递归去掉中间不必要的路点，只保留拐点
        两点可以直接连通就去掉中间的所有路点
        A*搜索时已经保证所有路点都是安全的，这里不需要重复检查
        """
        if len(path) <= 2:
            return path

        # 递归简化
        start = path[0]
        end = path[-1]

        if self.can_connect_directly(start[0], start[1], end[0], end[1]):
            # 可以直接连接，只保留起点和终点
            return [start, end]

        # 否则找最远的能连通点，分两段递归
        max_dist = -1
        best_idx = 1
        for i in range(1, len(path)-1):
            if self.can_connect_directly(start[0], start[1], path[i][0], path[i][1]):
                dist = (path[i][0] - start[0])**2 + (path[i][1] - start[1])**2
                if dist > max_dist:
                    max_dist = dist
                    best_idx = i

        left = self.simplify_path(path[:best_idx+1])
        right = self.simplify_path(path[best_idx:])

        simplified = left[:-1] + right
        return simplified

    def a_star_planning(self, start_wx, start_wy, goal_wx, goal_wy, safe_radius=None):
        """简单A*路径规划，使用heapq优先队列加速
        Args:
            start_wx, start_wy: 起点世界坐标
            goal_wx, goal_wy: 终点世界坐标
            safe_radius: 安全半径（格），默认使用robot_safe_radius_grid
        """
        import heapq
        start_gp = self.world_to_grid(start_wx, start_wy)
        goal_gp = self.world_to_grid(goal_wx, goal_wy)
        if start_gp is None or goal_gp is None:
            return None
        sgx, sgy = start_gp
        ggx, ggy = goal_gp

        width = self.current_map.info.width
        height = self.current_map.info.height

        # 8方向移动
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        open_heap = []
        closed_set = set()
        came_from = {}
        g_score = {}
        f_score = {}

        h = math.sqrt((ggx - sgx)**2 + (ggy - sgy)**2)
        g_score[(sgx, sgy)] = 0
        f_score[(sgx, sgy)] = h
        heapq.heappush(open_heap, (f_score[(sgx, sgy)], (sgx, sgy)))

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            closed_set.add(current)
            cx, cy = current

            if current == (ggx, ggy):
                # 重构路径
                path = []
                while current in came_from:
                    wx, wy = self.grid_to_world(current[0], current[1])
                    path.append((wx, wy))
                    current = came_from[current]
                path.append((start_wx, start_wy))
                path.reverse()

                # 路径简化：去掉中间不必要路点，过滤掉障碍物附近的路点
                simplified = self.simplify_path(path)
                if len(simplified) >= 2:
                    self.get_logger().info(f"🚀 A*规划完成: 原路径 {len(path)} 点 → 简化后 {len(simplified)} 点")
                    return simplified
                else:
                    self.get_logger().info(f"🚀 A*规划完成")
                    return path

            for dx, dy in moves:
                ngx = cx + dx
                ngy = cy + dy
                neighbor = (ngx, ngy)
                if neighbor in closed_set:
                    continue
                if ngx < 0 or ngx >= width or ngy < 0 or ngy >= height:
                    continue
                # 检查是否是障碍物
                val = self.get_cell_value(ngx, ngy)
                if val > OBSTACLE_THRESHOLD:
                    continue  # 障碍物绝对不能走
                # 终点允许是未知区域（目标本来就在前沿未知边缘）
                # 路径中间只允许已探索自由，终点可以是未知
                if (ngx, ngy) != (ggx, ggy) and val == UNKNOWN_THRESHOLD:
                    continue
                # 检查位置是否安全：周围不能有已知障碍物
                wx_world, wy_world = self.grid_to_world(ngx, ngy)
                if not self.is_position_safe(wx_world, wy_world, safe_radius=safe_radius):
                    continue  # 不安全，不允许走
                # 允许走
                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + math.sqrt((ggx - ngx)**2 + (ggy - ngy)**2)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, neighbor))

        # 找不到路径
        return None

    def a_star_planning_tiered(self, start_wx, start_wy, goal_wx, goal_wy):
        """分层安全距离的A*规划：优先6格，逐步递减到4格
        返回: (path, radius_used) 或 (None, None)
        """
        # 从6到4，依次尝试
        for radius in [6, 5, 4]:
            path = self.a_star_planning(start_wx, start_wy, goal_wx, goal_wy, safe_radius=radius)
            if path is not None:
                if radius != self.robot_safe_radius_grid:
                    self.get_logger().info(f"🛡️ 使用 {radius} 格安全距离找到路径")
                return path, radius
        self.get_logger().warning(f"❌ 所有安全距离均无法找到路径")
        return None, None

    def move_robot_to_waypoint(self, target_x, target_y):
        """控制机器人移动到目标路点，通过发布ROS2话题命令"""
        if self.cmd_pub is None:
            # 仿真模式，直接更新位置（用于测试）
            self.get_logger().info(f"🧪 [仿真] 移动到 ({target_x:.2f}, {target_y:.2f})")
            # 模拟移动时间
            dist = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
            time.sleep(dist / self.linear_vel)
            self.robot_x = target_x
            self.robot_y = target_y
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            self.robot_yaw = math.atan2(dy, dx)
            return True

        # 计算目标方向
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        target_yaw = math.atan2(dy, dx)

        # 在执行运动前向用户描述动作，等待q确认
        dyaw = target_yaw - self.robot_yaw
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        dyaw_abs = abs(dyaw)
        dist = math.hypot(dx, dy)

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🧭 计划移动:")
        self.get_logger().info(f"   当前位置: ({self.robot_x:.2f}, {self.robot_y:.2f})")
        self.get_logger().info(f"   目标位置: ({target_x:.2f}, {target_y:.2f})")
        self.get_logger().info(f"   直线距离: {dist:.2f} m")
        if dyaw_abs > 0.1:
            turn_dir_text = "左转" if dyaw > 0 else "右转"
            self.get_logger().info(f"   需要先{turn_dir_text} {math.degrees(dyaw_abs):.1f}°")
        self.get_logger().info(f"   预计总时长: {(dyaw_abs / self.angular_vel if dyaw_abs > 0.1 else 0) + (dist / self.linear_vel):.1f} 秒")
        self.get_logger().info("=" * 60)

        if EXIT_FLAG:
            self.get_logger().info("退出信号，停止移动")
            return False

        if EMERGENCY_STOP:
            return False

        # 先转向目标方向
        dyaw = target_yaw - self.robot_yaw
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        dyaw_abs = abs(dyaw)

        if dyaw_abs > self.yaw_tolerance:
            # 迭代式转向：停转后等待里程计稳定，再验证是否到位，补偿残余角度
            max_turn_iterations = 5
            for turn_iter in range(max_turn_iterations):
                # 重新计算当前朝向误差
                dyaw = target_yaw - self.robot_yaw
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                dyaw_abs = abs(dyaw)

                if dyaw_abs < self.yaw_tolerance:
                    self.get_logger().info(f"✅ 转向完成，剩余误差 {math.degrees(dyaw_abs):.1f}°")
                    break

                turn_dir = 1 if dyaw > 0 else -1
                angular_z = turn_dir * self.angular_vel
                turn_time = max(3.0, dyaw_abs / self.angular_vel * 6.0)
                self.get_logger().info(
                    f"🔄 转向 (第{turn_iter+1}次) {math.degrees(dyaw_abs):.1f}°, "
                    f"超时 {turn_time:.1f}s"
                )
                self.publish_command(f"0 0 {angular_z}")
                start_turn_time = time.time()
                while time.time() - start_turn_time < turn_time and not EMERGENCY_STOP:
                    time.sleep(0.01)
                    remaining_dyaw = target_yaw - self.robot_yaw
                    while remaining_dyaw > math.pi:
                        remaining_dyaw -= 2 * math.pi
                    while remaining_dyaw < -math.pi:
                        remaining_dyaw += 2 * math.pi
                    if abs(remaining_dyaw) < self.yaw_tolerance:
                        self.get_logger().info(
                            f"✅ 转向完成，剩余误差 {math.degrees(abs(remaining_dyaw)):.1f}°"
                        )
                        break
                # 停止转动
                self.publish_command("0 0 0")
                if EMERGENCY_STOP:
                    return False
                # 等待里程计追赶实际朝向（补偿检测延迟）
                time.sleep(0.5)

        # 直行前确认角度是否对准
        final_dyaw = target_yaw - self.robot_yaw
        while final_dyaw > math.pi:
            final_dyaw -= 2 * math.pi
        while final_dyaw < -math.pi:
            final_dyaw += 2 * math.pi
        if abs(final_dyaw) > self.yaw_tolerance:
            self.get_logger().warn(
                f"⚠️ 转向未到位，剩余角度误差 {math.degrees(abs(final_dyaw)):.1f}°，放弃本次直行"
            )
            self.publish_command("0 0 0")
            return False

        # 直线前进到目标
        dist = math.hypot(dx, dy)
        move_time = dist / self.linear_vel
        if move_time > self.max_move_duration:
            move_time = self.max_move_duration
            self.get_logger().info(f"🚶 走向目标，距离 {dist:.2f}m，限时 {move_time:.1f}s (受限于最大时长)")
        else:
            self.get_logger().info(f"🚶 走向目标，距离 {dist:.2f}m，预计 {move_time:.1f}s")

        # 发布前进命令
        self.publish_command(f"{self.linear_vel} 0 0")
        start_time = time.time()
        initial_dist = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
        max_dist = initial_dist
        moving_away_start_time = None  # 开始持续远离的时间

        while time.time() - start_time < move_time and not EMERGENCY_STOP:
            # 主线程的rclpy.spin已经在处理ROS消息更新位姿了
            time.sleep(0.01)
            # 实时检查是否已经到达
            current_dist = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
            if current_dist < self.goal_tolerance:
                break

            # 检查是否越来越远
            if current_dist > max_dist:
                max_dist = current_dist
                if moving_away_start_time is None:
                    moving_away_start_time = time.time()
                else:
                    # 如果已经持续远离超过1秒，自动停止
                    if (time.time() - moving_away_start_time) > 1.0:
                        self.get_logger().warn(f"⚠️  机器人持续远离目标，当前距离 {current_dist:.2f}m > 初始 {initial_dist:.2f}m，持续1秒，自动停止")
                        self.publish_command("0 0 0")
                        return False
            else:
                # 距离没有增加，重置计时器
                moving_away_start_time = None

        # 停止
        self.publish_command("0 0 0")
        if EMERGENCY_STOP:
            return False
        time.sleep(0.2)

        # 最后再检查一次到达（位姿已经由主线程spin更新）
        time.sleep(0.1)
        final_dist = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
        if final_dist < self.goal_tolerance:
            self.get_logger().info(f"✅ 到达路点 ({target_x:.2f}, {target_y:.2f})")
            # 回到站立姿态稳定
            self.publish_command("1")  # 起立命令
            time.sleep(0.5)
            return True
        else:
            self.get_logger().warn(f"⚠️  未完全到达，剩余距离 {final_dist:.2f}m，继续下一个路点")
            return False

    def add_unreachable(self, goal):
        """标记目标为不可达"""
        if goal is not None:
            fx, fy = goal
            self.unreachable_targets.add((round(fx, 1), round(fy, 1)))
            self.get_logger().warn(f"⚠️  标记目标 ({fx:.1f}, {fy:.1f}) 为不可达")

    def exploration_loop(self):
        """后台自主探索主循环"""
        # 等待初始数据
        while not EXIT_FLAG and (not self.map_received or not self.odom_received):
            time.sleep(0.1)

        if EXIT_FLAG:
            return

        self.get_logger().info("🎯 开始自主探索...")
        self.exploration_active = True
        self.last_frontier_time = time.time()

        while not EXIT_FLAG and self.exploration_active:
            t_frontier_start = time.time()

            # 检测前沿
            self.detect_frontiers()

            # 记录前沿检测时间
            frontier_elapsed = time.time() - t_frontier_start
            self.frontier_detection_times.append(frontier_elapsed)

            # 检查停止条件：长时间没有前沿，说明探索完成
            if len(self.frontiers) == 0:
                elapsed_no_frontier = time.time() - self.last_frontier_time
                if elapsed_no_frontier > self.stop_no_frontier_timeout:
                    self.get_logger().info("🏁 超过 %.1f 秒没有找到前沿，探索完成" % self.stop_no_frontier_timeout)
                    break
                else:
                    self.get_logger().info("⏳ 暂时没有前沿，等待地图更新... (%.1f/%.1f秒)" % (elapsed_no_frontier, self.stop_no_frontier_timeout))
                    time.sleep(2)
                    continue

            # 找最远的前沿
            self.current_goal = self.find_farthest_frontier()
            if self.current_goal is None:
                self.get_logger().info("⚠️ 没有可用前沿，等待地图更新...")
                time.sleep(2)
                continue

            fx, fy = self.current_goal
            self.get_logger().info(f"🎯 步骤 {self.total_steps + 1}: 选择目标 ({fx:.2f}, {fy:.2f})")

            self.exploration_goals.append(self.current_goal)
            self.nav_state = 'navigating'

            # 沿着路径依次访问路点，每到一个路点重新规划
            navigation_success = True
            while self.nav_state == 'navigating' and not EXIT_FLAG and not EMERGENCY_STOP:
                # 每次从当前位置重新规划到最终目标（使用分层安全距离）
                path, radius_used = self.a_star_planning_tiered(self.robot_x, self.robot_y, fx, fy)
                if path is None or len(path) < 2:
                    self.get_logger().warning(f"❌ 重新规划失败，放弃当前目标")
                    self.add_unreachable(self.current_goal)
                    self.nav_state = 'idle'
                    navigation_success = False
                    break

                self.current_planned_path = path
                self.all_planned_paths.append(path)  # 保存路径历史用于可视化

                # 检查是否已经非常接近目标
                if math.hypot(fx - self.robot_x, fy - self.robot_y) < self.goal_tolerance:
                    self.get_logger().info(f"✅ 到达目标点 ({fx:.2f}, {fy:.2f})")
                    self.nav_state = 'idle'
                    break

                # 移动到下一个路点（path[0]是当前位置，path[1]是下一个路点）
                target = path[1]  # path至少有2个点（起点+终点）
                success = self.move_robot_to_waypoint(target[0], target[1])
                if not success:
                    # 移动失败，标记不可达，跳出循环重新选择目标
                    self.add_unreachable(self.current_goal)
                    self.nav_state = 'idle'
                    navigation_success = False
                    break

            # 更新统计数据
            self.total_steps += 1
            self.time_history.append(time.time() - self.start_time)
            self.coverage_history.append(self.calculate_coverage())

            # 只有成功行动后（机器人实际移动并获得新地图信息）才清理不可达目标缓存
            if navigation_success and len(self.unreachable_targets) > 0:
                self.get_logger().info(f"🧹 行动完成，清理不可达目标缓存: {len(self.unreachable_targets)} 个")
                self.unreachable_targets.clear()

            # 实时保存结果（覆盖式）
            self.save_exploration_result()

            # 请求主线程保存地图图片（matplotlib不是线程安全的）
            self.need_save_map = True

            # 发布更新可视化
            self.publish_visualization()
            time.sleep(1)

        # 探索结束 - 最终保存
        self.get_logger().info("🏁 探索完成，保存最终结果...")
        self.save_exploration_result()
        self.need_save_map = True
        # 等待主线程保存完成
        time.sleep(2)

        # 发送站立命令
        self.publish_command("1")

        self.exploration_active = False
        self.get_logger().info(f"📊 探索统计: {self.total_steps} 步, 覆盖率 {self.calculate_coverage():.2%}")
        self.get_logger().info(f"📊 总共完成 {len(self.exploration_goals)} 个探索目标")

    def publish_visualization(self):
        """发布所有可视化话题（在主线程中执行，matplotlib安全）"""
        if self.current_map is None:
            return

        # 在主线程中执行matplotlib保存
        if self.need_save_map:
            self.need_save_map = False
            try:
                self.save_map_visualization()
            except Exception as e:
                self.get_logger().warning(f"⚠️ 保存地图图片失败: {e}")

        frame_id = self.current_map.header.frame_id

        # 发布目标点历史
        history_msg = Path()
        history_msg.header.stamp = self.get_clock().now().to_msg()
        history_msg.header.frame_id = frame_id
        for (wx, wy) in self.exploration_goals:
            ps = PoseStamped()
            ps.header = history_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            history_msg.poses.append(ps)
        self.goal_path_pub.publish(history_msg)

        # 发布当前规划路径
        if len(self.current_planned_path) > 0:
            planned_msg = Path()
            planned_msg.header.stamp = self.get_clock().now().to_msg()
            planned_msg.header.frame_id = frame_id
            for (wx, wy) in self.current_planned_path:
                ps = PoseStamped()
                ps.header = planned_msg.header
                ps.pose.position.x = wx
                ps.pose.position.y = wy
                ps.pose.orientation.w = 1.0
                planned_msg.poses.append(ps)
            self.planned_path_pub.publish(planned_msg)

        # 发布前沿点标记
        marker_array = MarkerArray()
        for i, (fx, fy) in enumerate(self.frontiers):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = fx
            marker.pose.position.y = fy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)

        # 删除旧标记（超过当前前沿数量的）
        for i in range(len(self.frontiers), len(self.frontiers) + 100):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.frontier_pub.publish(marker_array)

        # 发布机器人位置立方体标记
        if self.odom_received:
            robot_marker = Marker()
            robot_marker.header.frame_id = frame_id
            robot_marker.header.stamp = self.get_clock().now().to_msg()
            robot_marker.id = 1000
            robot_marker.type = Marker.CUBE
            robot_marker.action = Marker.ADD
            robot_marker.pose.position.x = self.robot_x
            robot_marker.pose.position.y = self.robot_y
            robot_marker.pose.position.z = 0.2  # 抬高一点方便看
            robot_marker.pose.orientation.z = math.sin(self.robot_yaw / 2)
            robot_marker.pose.orientation.w = math.cos(self.robot_yaw / 2)
            robot_marker.scale.x = 0.8  # 长
            robot_marker.scale.y = 0.4  # 宽
            robot_marker.scale.z = 0.4  # 高
            robot_marker.color.r = 0.0
            robot_marker.color.g = 1.0
            robot_marker.color.b = 0.0
            robot_marker.color.a = 0.8
            self.robot_marker_pub.publish(robot_marker)

    def keyboard_listener(self):
        """监听键盘输入，空格键清除判断+紧急停止，q键确认移动"""
        global EMERGENCY_STOP, USER_CONFIRM
        self.get_logger().info("⌨️ 键盘监听已启动")
        self.get_logger().info("  空格键: 清除不可达/已到达判断 + 紧急停止/恢复")
        self.get_logger().info("  q键: 确认移动")

        # 设置终端为raw模式读取单个字符
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not EXIT_FLAG:
                ch = sys.stdin.read(1)
                if ch == ' ':  # 空格键按下
                    # 清除不可达判断和已到达判断
                    cleared_unreachable_count = len(self.unreachable_targets)
                    cleared_goals_count = len(self.exploration_goals)
                    self.unreachable_targets.clear()
                    self.exploration_goals.clear()

                    if cleared_unreachable_count > 0 or cleared_goals_count > 0:
                        self.get_logger().info(f"🧹 已清除不可达目标({cleared_unreachable_count}个)和已到达目标({cleared_goals_count}个)的判断")

                    if not EMERGENCY_STOP:
                        self.get_logger().warn("🛑 空格键按下！触发紧急停止，持续发送零速度")
                        EMERGENCY_STOP = True
                        # 启动停止线程持续发送0 0 0
                        self.stop_thread = threading.Thread(target=self.stop_loop, daemon=True)
                        self.stop_thread.start()
                        # 暂停探索
                        self.exploration_active = False
                    else:
                        self.get_logger().info("▶️  空格键再次按下，恢复探索")
                        EMERGENCY_STOP = False
                        # 恢复探索
                        if not self.exploration_active:
                            self.exploration_active = True
                            self.exploration_thread = threading.Thread(target=self.exploration_loop, daemon=True)
                            self.exploration_thread.start()
                elif ch == 'q' or ch == 'Q':  # q键按下确认移动
                    USER_CONFIRM = True
                    self.get_logger().info("✅ q键按下，确认移动")
                elif ch == '\x03':  # Ctrl+C
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop_loop(self):
        """紧急停止循环，持续发送零速度命令"""
        while EMERGENCY_STOP and not EXIT_FLAG:
            self.publish_command("0 0 0")
            time.sleep(0.1)  # 10Hz发送，保证持续停止

    def safe_exit(self):
        """安全退出"""
        global EXIT_FLAG, EMERGENCY_STOP
        EXIT_FLAG = True
        EMERGENCY_STOP = True
        # 发送坐下命令
        try:
            self.publish_command('2')
            time.sleep(3)
        except:
            pass

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    # 可通过参数修改机器人IP
    robot_ip = "192.168.234.18"
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    node = AutonomousExplorer(robot_ip=robot_ip)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 收到中断信号，保存结果后退出")
        node.save_exploration_result()
        node.save_map_visualization()
        node.safe_exit()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
