#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXIT_FLAG = False
OBSTACLE_THRESHOLD = 50
UNKNOWN_THRESHOLD = -1
FREE_THRESHOLD = 50
OBSTACLE_SAFE_RADIUS = 4  # 障碍物安全距离，单位：格


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # 探索参数配置
        self.goal_tolerance = 0.3
        self.min_frontier_size = 1
        self.cluster_distance = 1.5  # 前沿点聚类距离（米）
        self.effective_min_dist = 2.5  # 过滤离已访问目标太近的前沿（米）
        self.robot_safe_radius_grid = 4  # 机器人离障碍物安全距离（格）

        # 当前SLAM地图
        self.current_map = None
        self.map_received = False
        self.visited_grid = None  # 对应已探索区域（map中 != -1）
        self.last_map_update_time = 0

        # 探索状态
        self.frontiers = []
        self.exploration_goals = []
        self.current_goal = None

        # 不可达目标缓存，避免重复选择
        self.unreachable_targets = set()

        # 安全缓存：每个栅格只计算一次安全性
        self.safety_cache = None
        self.safety_cache_computed = None

        # 机器人当前位姿（从里程计获取）
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False

        # 订阅SLAM建图得到的地图
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # 订阅机器人里程计获取实时位姿
        self.odom_sub = self.create_subscription(
            Odometry, '/Odometry', self.odom_callback, 10)

        # 可视化发布器
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.goal_path_pub = self.create_publisher(Path, '/exploration_goals', 10)
        self.robot_marker_pub = self.create_publisher(Marker, '/robot_position', 10)

        # 定时检测前沿并更新可视化
        self.detection_timer = self.create_timer(1.0, self.detection_callback)

        self.get_logger().info("🔍 前沿检测探索节点已启动")
        self.get_logger().info(f"📐 参数: 最小前沿大小={self.min_frontier_size}, 聚类距离={self.cluster_distance}m")

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
        self.last_map_update_time = self.get_clock().now().nanoseconds / 1e9

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
        """检测前沿（未知区域相邻的已知自由区域边界）"""
        if self.current_map is None or self.visited_grid is None:
            return []

        t_start = time.time()
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
        # 这样只需要处理边缘点，不需要遍历所有已访问栅格
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
                        # 检查周围OBSTACLE_SAFE_RADIUS格内是否有障碍物
                        has_obstacle_nearby = False
                        check_start = -OBSTACLE_SAFE_RADIUS
                        check_end = OBSTACLE_SAFE_RADIUS
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
                                frontiers.append((wx, wy))

        # 如果全部过滤掉了，放宽距离限制重新查找
        if len(frontiers) == 0:
            self.get_logger().debug("🔍 初始过滤后无前沿，放宽距离限制重新查找...")
            frontiers = []
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
                            # 检查周围障碍物
                            has_obstacle_nearby = False
                            check_start = -OBSTACLE_SAFE_RADIUS
                            check_end = OBSTACLE_SAFE_RADIUS
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
                                # 放宽最小距离
                                if len(self.exploration_goals) == 0:
                                    frontiers.append((wx, wy))
                                else:
                                    too_close = False
                                    for (tx, ty) in self.exploration_goals:
                                        if math.hypot(tx - wx, ty - wy) < 1.0:
                                            too_close = True
                                            break
                                    if not too_close:
                                        frontiers.append((wx, wy))

        self.frontiers = self.cluster_frontiers(frontiers, self.cluster_distance)
        # 过滤掉已经标记为不可达的目标
        original_count = len(self.frontiers)
        self.frontiers = [f for f in self.frontiers if not any(
            math.hypot(f[0] - ur[0], f[1] - ur[1]) < 1.0 for ur in self.unreachable_targets
        )]
        t_elapsed = time.time() - t_start
        if original_count != len(self.frontiers):
            self.get_logger().info(f"🔍 前沿检测完成: {original_count} → {len(self.frontiers)} 个有效前沿，耗时 {t_elapsed*1000:.1f}ms")
        else:
            self.get_logger().info(f"🔍 前沿检测完成: {len(self.frontiers)} 个有效前沿，耗时 {t_elapsed*1000:.1f}ms")
        return self.frontiers

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
                    centers.append((cx, cy))

        return centers

    def find_nearest_frontier(self, robot_x, robot_y):
        """找到离机器人当前位置最近的前沿"""
        if not self.frontiers:
            return None
        min_dist = float('inf')
        nearest = None
        for fx, fy in self.frontiers:
            dist = math.sqrt((fx - robot_x)**2 + (fy - robot_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (fx, fy)
        return nearest

    def odom_callback(self, msg):
        """接收里程计获取机器人实时位姿"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # 从四元数提取yaw角
        q = msg.pose.pose.orientation
        # yaw = atan2(2*(qw*qz + qx*qy), 1-2*(qy^2 + qz^2))
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def detection_callback(self):
        """定时检测前沿并更新可视化"""
        if not self.map_received:
            return

        self.detect_frontiers()
        self.publish_visualization()

    def a_star_planning(self, start_wx, start_wy, goal_wx, goal_wy):
        """简单A*路径规划，使用heapq优先队列加速"""
        import heapq
        t_start = time.time()
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
                t_elapsed = time.time() - t_start
                if len(simplified) >= 2:
                    self.get_logger().info(f"🚀 A*规划完成: 原路径 {len(path)} 点 → 简化后 {len(simplified)} 点，耗时 {t_elapsed*1000:.1f}ms")
                    return simplified
                else:
                    self.get_logger().info(f"🚀 A*规划完成，耗时 {t_elapsed*1000:.1f}ms")
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
                if not self.is_position_safe(wx_world, wy_world):
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
        t_elapsed = time.time() - t_start
        self.get_logger().info(f"❌ A*规划失败，找不到路径，耗时 {t_elapsed*1000:.1f}ms")
        return None

    def is_position_safe(self, wx, wy):
        """检查位置是否安全：距离障碍物至少OBSTACLE_SAFE_RADIUS格
        使用缓存：每个栅格只计算一次，结果永久复用，避免重复计算
        """
        gp = self.world_to_grid(wx, wy)
        if gp is None:
            return False
        gx, gy = gp
        width = self.current_map.info.width
        height = self.current_map.info.height

        # 已经计算过，直接返回缓存结果
        if self.safety_cache_computed[gy, gx]:
            return self.safety_cache[gy, gx]

        # 第一次计算，检查周围
        check_radius = OBSTACLE_SAFE_RADIUS
        check_start = -check_radius
        check_end = check_radius
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

        # 缓存结果，以后直接用，不再重复计算
        self.safety_cache[gy, gx] = safe
        self.safety_cache_computed[gy, gx] = True
        return safe

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

    def publish_visualization(self):
        """发布所有可视化话题"""
        if self.current_map is None:
            return

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

        # 发布机器人位置立方体标记（如果有里程计数据）
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

    def safe_exit(self):
        """安全退出"""
        global EXIT_FLAG
        EXIT_FLAG = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 收到中断信号，退出")
        node.safe_exit()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
