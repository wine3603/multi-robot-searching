#!/usr/bin/env python3
"""
自主探索对比实验框架
支持多种前沿检测策略和路径规划算法
结果自动保存到 experiments/ 文件夹

用法:
  ./simulation_explore_framework.py --strategy frontier_greedy
  ./simulation_explore_framework.py --strategy frontier_infogain
  ./simulation_explore_framework.py --strategy rrt
  ./simulation_explore_framework.py --strategy rrt_star
  ./simulation_explore_framework.py --strategy rrt_connect
  ./simulation_explore_framework.py --strategy extended_rrt
  ./simulation_explore_framework.py --strategy dynamic_rrt
  ./simulation_explore_framework.py --strategy rrt_sharp
  ./simulation_explore_framework.py --strategy rrt_star_smart
  ./simulation_explore_framework.py --strategy informed_rrt_star
  ./simulation_explore_framework.py --strategy bit_star
  ./simulation_explore_framework.py --strategy abit_star
  ./simulation_explore_framework.py --strategy abit_star_advanced
  ./simulation_explore_framework.py --strategy fmt_star
  ./simulation_explore_framework.py --strategy astar_info_gain
  ./simulation_explore_framework.py --strategy dijkstra_info_gain
  ./simulation_explore_framework.py --strategy bfs_info_gain
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import threading
import sys
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import json
from datetime import datetime
from scipy.ndimage import sobel

# ========== 导入PathPlanning算法（相对仓库根定位，任意部署位置可运行） ==========
# 仓库布局: <root>/src/FAST_LIO/scripts/ 与 <root>/src/PathPlanning/
# rrt_2D 内部采用顶层导入（from Sampling_based_Planning.rrt_2D import ...），
# 因此需要同时把 <root>/src 与 <root>/src/PathPlanning 加入 sys.path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_SRC_DIR, "PathPlanning"))
sys.path.insert(0, _SRC_DIR)

# 常量定义
EXIT_FLAG = False
OBSTACLE_THRESHOLD = 50
UNKNOWN_THRESHOLD = -1
FREE_THRESHOLD = 50
OBSTACLE_SAFE_RADIUS = 4  # 障碍物安全距离，单位：格

# FOV和传感器参数
FOV_ANGLE = math.radians(360)
MAX_SENSOR_RANGE = 3.0  


# ========== 策略基类 ==========
class ExplorationStrategy:
    """自主探索策略基类"""
    name = "base"
    description = "Base class for exploration strategies"

    def __init__(self, explorer):
        self.explorer = explorer

    def detect_frontiers(self):
        """检测前沿，返回前沿点列表 [(wx, wy), ...]"""
        raise NotImplementedError

    def select_goal(self, frontiers):
        """从前沿中选择目标，返回 (wx, wy) 或 None"""
        raise NotImplementedError


# ========== 基于前沿检测的方法 ==========

# 策略1: 贪心最近前沿 (经典基线)
class FrontierGreedyStrategy(ExplorationStrategy):
    """贪心最近前沿 - 选择离机器人最近的前沿"""
    name = "frontier_greedy"
    description = "Greedy closest frontier (classical baseline)"

    def detect_frontiers(self):
        """使用Sobel边缘检测前沿"""
        return self.explorer.detect_frontiers_sobel()

    def select_goal(self, frontiers):
        """选择最近的前沿"""
        if len(frontiers) == 0:
            return None
        cx, cy = self.explorer.sim_x, self.explorer.sim_y
        frontiers.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        return frontiers[0]


# 策略2: 信息增益最大
class FrontierInfoGainStrategy(ExplorationStrategy):
    """信息增益最大 - 选择预期能探索最多未知区域的前沿"""
    name = "frontier_infogain"
    description = "Maximum information gain frontier selection"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.sensor_range = MAX_SENSOR_RANGE

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_sobel()

    def compute_information_gain(self, wx, wy):
        return self.explorer.compute_expected_gain(wx, wy)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        cx, cy = self.explorer.sim_x, self.explorer.sim_y

        scored = []
        for (wx, wy) in frontiers:
            gain = self.compute_information_gain(wx, wy)
            dist = math.hypot(wx - cx, wy - cy)
            utility = gain / (dist + 0.1)
            scored.append((utility, gain, dist, wx, wy))

        scored.sort(reverse=True, key=lambda x: x[0])
        return (scored[0][3], scored[0][4])


# ========== 基于RRT采样的方法 ==========

# 策略3: RRT 基础
class RRTStrategy(ExplorationStrategy):
    """基础RRT探索 - RRT在已探索区域生长找到前沿边界"""
    name = "rrt"
    description = "Basic RRT exploration"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_rrt(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略4: RRT*
class RRTStarStrategy(ExplorationStrategy):
    """RRT* 渐近最优探索"""
    name = "rrt_star"
    description = "RRT* asymptotically optimal"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 500

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_rrt_star(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略5: RRT-Connect
class RRTConnectStrategy(ExplorationStrategy):
    """RRT-Connect 双向RRT探索"""
    name = "rrt_connect"
    description = "RRT-Connect bidirectional RRT"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_rrt_connect(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略6: Extended RRT
class ExtendedRRTSStrategy(ExplorationStrategy):
    """Extended RRT 扩展RRT"""
    name = "extended_rrt"
    description = "Extended RRT"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_extended_rrt(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略7: Dynamic RRT
class DynamicRRTSStrategy(ExplorationStrategy):
    """Dynamic RRT 动态RRT"""
    name = "dynamic_rrt"
    description = "Dynamic RRT"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_dynamic_rrt(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略8: RRT# (RRT Sharp)
class RRTSharpStrategy(ExplorationStrategy):
    """RRT# (RRT Sharp)"""
    name = "rrt_sharp"
    description = "RRT# (RRT Sharp)"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_rrt_sharp(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略9: RRT*-Smart
class RRTStarSmartStrategy(ExplorationStrategy):
    """RRT*-Smart 智能采样加速收敛"""
    name = "rrt_star_smart"
    description = "RRT*-Smart with intelligent sampling"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_rrt_star_smart(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略10: Informed RRT*
class InformedRRTStarStrategy(ExplorationStrategy):
    """Informed RRT* - 启发式椭圆采样"""
    name = "informed_rrt_star"
    description = "Informed RRT* with ellipsoidal sampling"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_informed_rrt_star(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略11: BIT* (Batch Informed Trees)
class BITStarStrategy(ExplorationStrategy):
    """BIT* - Batch Informed Trees"""
    name = "bit_star"
    description = "Batch Informed Trees (BIT*)"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_bit_star(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略12: ABIT* (Adaptively Informed Trees)
class ABITStarStrategy(ExplorationStrategy):
    """ABIT* - Adaptively Informed Trees"""
    name = "abit_star"
    description = "Adaptively Informed Trees (ABIT*)"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_abit_star(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略13: Advanced ABIT*
class ABITStarAdvancedStrategy(ExplorationStrategy):
    """Advanced ABIT* - Advanced Batch Informed Trees"""
    name = "abit_star_advanced"
    description = "Advanced Adaptively Informed Trees"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_abit_star_advanced(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# 策略14: FMT* (Fast Marching Trees)
class FMTStarStrategy(ExplorationStrategy):
    """FMT* - Fast Marching Trees"""
    name = "fmt_star"
    description = "Fast Marching Trees (FMT*)"

    def __init__(self, explorer):
        super().__init__(explorer)
        self.max_nodes = 400

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_fmt_star(max_nodes=self.max_nodes)

    def select_goal(self, frontiers):
        if len(frontiers) == 0:
            return None
        return self.explorer.select_closest_frontier(frontiers)


# ========== 基于搜索的方法 + 信息增益 ==========

# 策略15: A* + 信息增益
class AStarInfoGainStrategy(ExplorationStrategy):
    """A* 搜索结合信息增益"""
    name = "astar_info_gain"
    description = "A* search with information gain"

    def __init__(self, explorer):
        super().__init__(explorer)

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_sobel()

    def select_goal(self, frontiers):
        return self.explorer.select_by_search_info_gain(frontiers, 'astar')


# 策略16: Dijkstra + 信息增益
class DijkstraInfoGainStrategy(ExplorationStrategy):
    """Dijkstra 搜索结合信息增益"""
    name = "dijkstra_info_gain"
    description = "Dijkstra search with information gain"

    def __init__(self, explorer):
        super().__init__(explorer)

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_sobel()

    def select_goal(self, frontiers):
        return self.explorer.select_by_search_info_gain(frontiers, 'dijkstra')


# 策略17: BFS + 信息增益
class BFSInfoGainStrategy(ExplorationStrategy):
    """BFS 搜索结合信息增益"""
    name = "bfs_info_gain"
    description = "BFS search with information gain"

    def __init__(self, explorer):
        super().__init__(explorer)

    def detect_frontiers(self):
        return self.explorer.detect_frontiers_sobel()

    def select_goal(self, frontiers):
        return self.explorer.select_by_search_info_gain(frontiers, 'bfs')


# ========== 策略工厂 ==========
class StrategyFactory:
    """策略工厂，根据名称创建策略"""

    _strategies = {
        "frontier_greedy": FrontierGreedyStrategy,
        "frontier_infogain": FrontierInfoGainStrategy,
        "rrt": RRTStrategy,
        "rrt_star": RRTStarStrategy,
        "rrt_connect": RRTConnectStrategy,
        "extended_rrt": ExtendedRRTSStrategy,
        "dynamic_rrt": DynamicRRTSStrategy,
        "rrt_star_smart": RRTStarSmartStrategy,
        "informed_rrt_star": InformedRRTStarStrategy,
        "bit_star": BITStarStrategy,
        "abit_star": ABITStarStrategy,
        "abit_star_advanced": ABITStarAdvancedStrategy,
        "fmt_star": FMTStarStrategy,
        "astar_info_gain": AStarInfoGainStrategy,
        "dijkstra_info_gain": DijkstraInfoGainStrategy,
        "bfs_info_gain": BFSInfoGainStrategy,
    }

    @classmethod
    def get_available_strategies(cls):
        return list(cls._strategies.keys())

    @classmethod
    def create(cls, name, explorer):
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}, available: {list(cls._strategies.keys())}")
        return cls._strategies[name](explorer)


# ========== 主探索类 ==========
class SimulationExperimentExplorer(Node):
    def __init__(self, strategy_name, experiment_dir, publish_enabled):
        super().__init__('simulation_explore_experiment')

        self.strategy_name = strategy_name
        self.experiment_dir = experiment_dir
        self.publish_enabled = publish_enabled

        # 模拟机器人位姿
        self.sim_x = 0.0
        self.sim_y = 0.0
        self.sim_yaw = 0.0

        # 模拟移动步长参数
        self.step_size = 1.0  # 每步移动1米

        # 完整真实地图
        self.full_map = None
        self.map_received = False

        # 模拟探测得到的探索地图
        self.explored_map = None
        self.visited_grid = None

        # 自动探索参数
        self.auto_explore_running = True
        self.exploration_thread = None

        # 探索状态
        self.current_goal = None
        self.exploration_path = []
        self.actual_trajectory = []
        self.current_planned_path = []
        self.frontiers = []
        self.nav_state = 'idle'
        self.nav_current_waypoint_idx = 0
        self.nav_path = None

        # RRT树可视化：累积保存所有步的RRT探测结果
        self.all_rrt_nodes = []
        self.all_rrt_edges = []
        self.all_rrt_frontiers = []

        # 性能统计
        self.start_time = None
        self.total_steps = 0
        self.total_motion_time = 0.0
        self.total_turns = 0
        self.turn_angle_total = 0.0
        self.explored_cells_history = []
        self.time_history = []
        self.cpu_time_history = []
        self.planning_time_total = 0.0

        # 各模块累计耗时
        self.total_time_sensor = 0.0
        self.total_time_frontier = 0.0
        self.total_time_planning = 0.0
        self.total_time_astar = 0.0
        self.total_time_safety = 0.0
        self._cached_total_free = None

        # 停止条件
        self.stop_coverage_threshold = 0.95
        self.stop_max_steps = 500
        self.stop_timeout_no_gain = 120  # 如果超过120秒覆盖率没有增长，提前停止
        # 运动速度参数
        self.linear_speed = 1.0
        self.angular_speed = 1.0
        # 不可达缓存
        self.unreachable_targets = set()
        self.no_value_grid = set()
        self.min_new_grid_threshold = 100
        # 超时检测
        self._last_coverage = 0.0
        self._last_coverage_time = 0.0

        # 安全缓存
        self.safety_cache = None
        self.safety_cache_computed = None

        # 创建策略
        self.strategy = StrategyFactory.create(strategy_name, self)

        # ROS publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.full_map_callback, 10)

        self.explored_map_pub = self.create_publisher(
            OccupancyGrid, '/explored_map', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.history_path_pub = self.create_publisher(Path, '/exploration_goals', 10)
        self.actual_trajectory_pub = self.create_publisher(Path, '/sim_trajectory', 10)
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.robot_marker_pub = self.create_publisher(Marker, '/robot_position', 10)

        # 可视化定时发布
        self.enable_visualization_publishing = True
        if self.enable_visualization_publishing:
            self.vis_timer = self.create_timer(1.0, self.publish_visualization)

        # 启动探索线程
        self.exploration_thread = threading.Thread(target=self.wait_map_and_start, daemon=True)
        self.exploration_thread.start()

        self.get_logger().info(f"🚀 实验启动: 策略 = {self.strategy.name} - {self.strategy.description}")
        self.get_logger().info(f"📂 结果将保存到: {self.experiment_dir}")

    def full_map_callback(self, msg):
        self.full_map = msg
        self.map_received = True

        if self.explored_map is None:
            self.init_explored_map()
            self.get_logger().info(
                f"🗺️ 收到完整地图: {msg.info.width}x{msg.info.height}, "
                f"分辨率={msg.info.resolution}m"
            )

    def init_explored_map(self):
        if self.full_map is None:
            return

        self.explored_map = OccupancyGrid()
        self.explored_map.header = self.full_map.header
        self.explored_map.info = self.full_map.info
        width = self.full_map.info.width
        height = self.full_map.info.height
        self.explored_map.data = [-1] * (width * height)

        self.visited_grid = np.zeros((height, width), dtype=bool)
        self.safety_cache = np.zeros((height, width), dtype=bool)
        self.safety_cache_computed = np.zeros((height, width), dtype=bool)
        self.get_logger().info(f"🗺️ 探索地图初始化完成: {width}x{height} = {width*height} 栅格")

    def world_to_grid(self, wx, wy):
        if self.full_map is None:
            return None
        origin_x = self.full_map.info.origin.position.x
        origin_y = self.full_map.info.origin.position.y
        res = self.full_map.info.resolution
        gx = int(round((wx - origin_x) / res))
        gy = int(round((wy - origin_y) / res))
        if gx < 0 or gx >= self.full_map.info.width or gy < 0 or gy >= self.full_map.info.height:
            return None
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        if self.full_map is None:
            return None
        origin_x = self.full_map.info.origin.position.x
        origin_y = self.full_map.info.origin.position.y
        res = self.full_map.info.resolution
        wx = origin_x + gx * res
        wy = origin_y + gy * res
        return (wx, wy)

    def get_explored_cell_value(self, gx, gy):
        if gx < 0 or gx >= self.full_map.info.width or gy < 0 or gy >= self.full_map.info.height:
            return 100
        idx = gy * self.full_map.info.width + gx
        return self.explored_map.data[idx]

    def set_explored_cell_value(self, gx, gy, val):
        if gx < 0 or gx >= self.full_map.info.width or gy < 0 or gy >= self.full_map.info.height:
            return
        idx = gy * self.full_map.info.width + gx
        self.explored_map.data[idx] = val

    def is_in_fov(self, wx, wy):
        dx = wx - self.sim_x
        dy = wy - self.sim_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > MAX_SENSOR_RANGE:
            return False

        if FOV_ANGLE < 2 * math.pi:
            angle = math.atan2(dy, dx)
            diff = abs(angle - self.sim_yaw)
            while diff > math.pi:
                diff = 2 * math.pi - diff
            if diff > FOV_ANGLE / 2:
                return False
        return True

    def check_safety_grid(self, gx, gy):
        if self.safety_cache_computed[gy, gx]:
            return self.safety_cache[gy, gx]

        width = self.full_map.info.width
        height = self.full_map.info.height
        check_start = -OBSTACLE_SAFE_RADIUS
        check_end = OBSTACLE_SAFE_RADIUS

        for dx in range(check_start, check_end + 1):
            for dy in range(check_start, check_end + 1):
                ngx = gx + dx
                ngy = gy + dy
                if 0 <= ngx < width and 0 <= ngy < height:
                    val = self.get_explored_cell_value(ngx, ngy)
                    if val > OBSTACLE_THRESHOLD:
                        self.safety_cache[gy, gx] = False
                        self.safety_cache_computed[gy, gx] = True
                        return False

        self.safety_cache[gy, gx] = True
        self.safety_cache_computed[gy, gx] = True
        return True

    def sensor_simulate(self):
        t_start = time.time()
        width = self.full_map.info.width
        height = self.full_map.info.height
        res = self.full_map.info.resolution

        radius_grid = int(math.ceil(MAX_SENSOR_RANGE / res))
        cx_g, cy_g = self.world_to_grid(self.sim_x, self.sim_y)
        if cx_g is None:
            return 0

        new_cells = 0
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                gx = cx_g + dx
                gy = cy_g + dy
                if 0 <= gx < width and 0 <= gy < height:
                    wx, wy = self.grid_to_world(gx, gy)
                    if self.is_in_fov(wx, wy):
                        if not self.visited_grid[gy, gx]:
                            full_val = self.full_map.data[gy * width + gx]
                            self.set_explored_cell_value(gx, gy, full_val)
                            self.visited_grid[gy, gx] = True
                            new_cells += 1

        t_elapsed = time.time() - t_start
        self.total_time_sensor += t_elapsed
        return new_cells

    def detect_frontiers_sobel(self):
        t_start = time.time()
        frontiers = []
        width = self.explored_map.info.width
        height = self.explored_map.info.height

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0), (1, 1)]

        explored_binary = self.visited_grid.astype(np.float32)
        edge_mag = np.abs(sobel(explored_binary))
        edge_coords = np.argwhere(edge_mag > 0.1)

        for (gy, gx) in edge_coords:
            val = self.get_explored_cell_value(gx, gy)
            if val >= 0 and val < FREE_THRESHOLD:
                has_unknown_neighbor = False
                for dx, dy in neighbors:
                    ngx = gx + dx
                    ngy = gy + dy
                    if 0 <= ngx < width and 0 <= ngy < height:
                        nval = self.get_explored_cell_value(ngx, ngy)
                        if nval == UNKNOWN_THRESHOLD:
                            has_unknown_neighbor = True
                            break
                if has_unknown_neighbor and self.check_safety_grid(gx, gy):
                    if len(self.actual_trajectory) <= 1:
                        effective_min_dist = 1.0
                    else:
                        effective_min_dist = 2.5
                    wx, wy = self.grid_to_world(gx, gy)
                    too_close = False
                    for (tx, ty, _) in self.actual_trajectory:
                        if math.hypot(tx - wx, ty - wy) < effective_min_dist:
                            too_close = True
                            break
                    if not too_close:
                        if (gx, gy) not in self.no_value_grid and not self.is_in_unreachable((wx, wy)):
                            frontiers.append((wx, wy))

        if len(frontiers) == 0:
            for (gy, gx) in edge_coords:
                val = self.get_explored_cell_value(gx, gy)
                if val >= 0 and val < FREE_THRESHOLD:
                    has_unknown_neighbor = False
                    for dx, dy in neighbors:
                        ngx = gx + dx
                        ngy = gy + dy
                        if 0 <= ngx < width and 0 <= ngy < height:
                            nval = self.get_explored_cell_value(ngx, ngy)
                            if nval == UNKNOWN_THRESHOLD:
                                has_unknown_neighbor = True
                                break
                        if has_unknown_neighbor and self.check_safety_grid(gx, gy):
                            wx, wy = self.grid_to_world(gx, gy)
                            if not self.is_in_unreachable((wx, wy)):
                                frontiers.append((wx, wy))

        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = frontiers
        return frontiers

    # ========== 各种RRT变体的前沿检测 ==========

    def check_collision_path_point(self, wx, wy):
        """检查单个点是否碰撞（在已探索自由空间）"""
        g = self.world_to_grid(wx, wy)
        if g is None:
            return True
        gx, gy = g
        val = self.get_explored_cell_value(gx, gy)
        if val < 0 or val >= FREE_THRESHOLD:
            return True
        return not self.check_safety_grid(gx, gy)

    def check_collision_path(self, node1, node2):
        """Bresenham线检查两点路径是否碰撞"""
        x0, y0 = node1.x, node1.y
        x1, y1 = node2.x, node2.y
        g0 = self.world_to_grid(x0, y0)
        g1 = self.world_to_grid(x1, y1)
        if g0 is None or g1 is None:
            return True

        x0, y0 = g0
        x1, y1 = g1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            val = self.get_explored_cell_value(x0, y0)
            if val >= OBSTACLE_THRESHOLD or val < 0:
                return True
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return False

    def is_near_frontier(self, wx, wy):
        """检查该点是否靠近前沿（相邻有未知栅格）"""
        g = self.world_to_grid(wx, wy)
        if g is None:
            return False
        gx, gy = g
        width = self.full_map.info.width
        height = self.full_map.info.height
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0), (1, 1)]
        for dx, dy in neighbors:
            ngx = gx + dx
            ngy = gy + dy
            if 0 <= ngx < width and 0 <= ngy < height:
                if self.get_explored_cell_value(ngx, ngy) == UNKNOWN_THRESHOLD:
                    return True
        return False

    def is_safe_position(self, wx, wy):
        """检查位置是否安全"""
        g = self.world_to_grid(wx, wy)
        if g is None:
            return False
        gx, gy = g
        val = self.get_explored_cell_value(gx, gy)
        if val >= FREE_THRESHOLD or val < 0:
            return False
        return self.check_safety_grid(gx, gy)

    def is_in_unreachable(self, f):
        fx, fy = f
        for (ux, uy) in self.unreachable_targets:
            if math.hypot(ux - fx, uy - fy) < 1.0:
                return True
        return False

    def filter_frontiers(self, frontiers):
        filtered = []
        min_dist = 2.0
        for f in frontiers:
            if self.is_in_unreachable(f):
                continue
            g = self.world_to_grid(f[0], f[1])
            if g in self.no_value_grid:
                continue
            too_close = False
            for p in filtered:
                if math.hypot(f[0] - p[0], f[1] - p[1]) < min_dist:
                    too_close = True
                    break
            for (tx, ty, _) in self.actual_trajectory:
                if math.hypot(tx - f[0], ty - f[1]) < 2.5:
                    too_close = True
                    break
            if not too_close:
                filtered.append(f)
        return filtered

    def generic_rrt_frontier_detection(self, RRTClass, max_nodes,
                                       step_size=None, search_radius=None,
                                       has_extra_param=False):
        """Generic RRT-based frontier detection with improved sampling strategy
        All RRT variants use this same improved logic:
        - 30% biased sampling towards unknown area
        - Add seed nodes (current position + existing frontiers + boundary points)
        - Check nearest node for frontier even if extension fails
        - 3x3 neighborhood counting for frontier verification
        """
        from PathPlanning.Sampling_based_Planning.rrt_2D.rrt import Node
        import math
        import numpy as np

        t_start = time.time()
        frontiers = []
        width = self.full_map.info.width
        height = self.full_map.info.height
        res = self.full_map.info.resolution
        origin_x = self.full_map.info.origin.position.x
        origin_y = self.full_map.info.origin.position.y
        x_min = origin_x
        x_max = origin_x + width * res
        y_min = origin_y
        y_max = origin_y + height * res

        if step_size is None:
            step_size = self.step_size

        # Build initial vertex list with seeds
        rrt_vertex = []
        # Always add robot current position
        if self.is_safe_position(self.sim_x, self.sim_y):
            rrt_vertex.append(Node([self.sim_x, self.sim_y]))

        # Add existing frontiers to tree
        if len(self.frontiers) > 0:
            for fx, fy in self.frontiers:
                if self.is_safe_position(fx, fy):
                    exists = False
                    for node in rrt_vertex:
                        if math.hypot(node.x - fx, node.y - fy) < 0.5:
                            exists = True
                            break
                    if not exists:
                        rrt_vertex.append(Node([fx, fy]))

        # Early exploration: add some boundary points as seeds
        total_visited = np.sum(self.visited_grid)
        if len(rrt_vertex) <= 1 and total_visited > 100:
            visited_y, visited_x = np.where(self.visited_grid > 0)
            if len(visited_x) > 0:
                sample_count = min(20, len(visited_x))
                indices = np.random.choice(len(visited_x), sample_count, replace=False)
                for idx in indices:
                    gx, gy = visited_x[idx], visited_y[idx]
                    wx, wy = self.grid_to_world(gx, gy)
                    if self.is_safe_position(wx, wy):
                        rrt_vertex.append(Node([wx, wy]))

        if len(rrt_vertex) == 0:
            filtered = self.filter_frontiers([])
            t_elapsed = time.time() - t_start
            self.total_time_frontier += t_elapsed
            self.frontiers = filtered
            return filtered

        # Get visited points for sampling
        visited_y, visited_x = np.where(self.visited_grid > 0)

        # Create RRT instance
        start = [self.sim_x, self.sim_y]
        goal = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]

        if search_radius is not None and has_extra_param:
            # RRT*, RRT*-Smart, Informed RRT*: (start, goal, step, goal_sample, search_radius, iter_max)
            rrt = RRTClass(start, goal, step_size, 0.05, search_radius, max_nodes)
        elif has_extra_param:
            # Extended RRT, Dynamic RRT: (start, goal, step, goal_sample, waypoint_sample, iter_max)
            rrt = RRTClass(start, goal, step_size, 0.05, 0.10, max_nodes)
        else:
            # Basic RRT: (start, goal, step, goal_sample, iter_max)
            rrt = RRTClass(start, goal, step_size, 0.05, max_nodes)

        # Replace initial vertex with our seeded one
        if hasattr(rrt, 'vertex'):
            rrt.vertex = rrt_vertex
        if hasattr(rrt, 'V1'):
            rrt.V1 = rrt_vertex

        # Main RRT growing loop
        for i in range(rrt.iter_max):
            # 30% biased sampling towards unknown area
            if np.random.rand() < 0.3 and total_visited < self.visited_grid.size:
                found = False
                rand_gx, rand_gy = None, None
                for _ in range(100):
                    rg = (np.random.randint(0, width), np.random.randint(0, height))
                    if self.get_explored_cell_value(rg[0], rg[1]) == -1:
                        rand_gx, rand_gy = rg
                        found = True
                        break
                if found:
                    rand_x, rand_y = self.grid_to_world(rand_gx, rand_gy)
                else:
                    if len(visited_x) > 0:
                        idx = np.random.randint(0, len(visited_x))
                        rand_gx, rand_gy = visited_x[idx], visited_y[idx]
                        rand_x, rand_y = self.grid_to_world(rand_gx, rand_gy)
                    else:
                        rand_x = np.random.uniform(x_min, x_max)
                        rand_y = np.random.uniform(y_min, y_max)
            else:
                if len(visited_x) > 0:
                    idx = np.random.randint(0, len(visited_x))
                    rand_gx, rand_gy = visited_x[idx], visited_y[idx]
                    rand_x, rand_y = self.grid_to_world(rand_gx, rand_gy)
                else:
                    rand_x = np.random.uniform(x_min, x_max)
                    rand_y = np.random.uniform(y_min, y_max)

            node_rand = Node([rand_x, rand_y])
            # nearest_neighbor 不同实现有不同命名和存储位置
            if hasattr(rrt, 'vertex') and hasattr(rrt, 'nearest_neighbor'):
                node_near = rrt.nearest_neighbor(rrt.vertex, node_rand)
            elif hasattr(rrt, 'V1') and hasattr(rrt, 'nearest_neighbor'):
                node_near = rrt.nearest_neighbor(rrt.V1, node_rand)
            elif hasattr(rrt, 'node_list') and hasattr(rrt, 'nearest_neighbor'):
                node_near = rrt.nearest_neighbor(rrt.node_list, node_rand)
            elif hasattr(rrt, 'V') and hasattr(rrt, 'Nearest'):
                # RrtStarSmart, InformedRRTStar: 顶点在 V, 最近邻方法叫 Nearest (capital N)
                node_near = rrt.Nearest(rrt.V, node_rand)
            elif hasattr(rrt, 'V') and hasattr(rrt, 'nearest'):
                node_near = rrt.nearest(rrt.V, node_rand)
            elif hasattr(rrt, 'vertex') and hasattr(rrt, 'nearestNeighbor'):
                node_near = rrt.nearestNeighbor(rrt.vertex, node_rand)
            else:
                # Try all combinations:
                found = False
                for node_attr in ['vertex', 'V', 'V1', 'node_list']:
                    if hasattr(rrt, node_attr):
                        nodes = getattr(rrt, node_attr)
                        for method_name in ['nearest_neighbor', 'Nearest', 'nearest', 'nearestNeighbor']:
                            if hasattr(rrt, method_name):
                                method = getattr(rrt, method_name)
                                node_near = method(nodes, node_rand)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    # Final fallback - static method on class
                    node_near = RRTClass.nearest_neighbor(rrt.vertex, node_rand)

            # Replicate new_state - use instance method if available
            if hasattr(rrt, 'get_distance_and_angle'):
                dist, theta = rrt.get_distance_and_angle(node_near, node_rand)
            else:
                dist, theta = RRTClass.get_distance_and_angle(node_near, node_rand)
            dist = min(step_size, dist)
            node_new = Node((node_near.x + dist * math.cos(theta),
                             node_near.y + dist * math.sin(theta)))
            node_new.parent = node_near

            extended = False
            if node_new and not self.check_collision_path(node_near, node_new):
                # Find the correct vertex list attribute
                vertex_list = None
                if hasattr(rrt, 'vertex'):
                    vertex_list = 'vertex'
                elif hasattr(rrt, 'V'):
                    vertex_list = 'V'
                elif hasattr(rrt, 'V1'):
                    vertex_list = 'V1'
                elif hasattr(rrt, 'node_list'):
                    vertex_list = 'node_list'

                if vertex_list and vertex_list in ['vertex', 'V', 'node_list']:
                    # Single-tree RRT variants
                    vertices = getattr(rrt, vertex_list)
                    if search_radius is not None and RRTClass.__name__ in ['RrtStar', 'RrtStarSmart', 'IRrtStar', 'RrtStarSmart']:
                        # RRT* needs special handling for near nodes and rewire
                        near_indices = []
                        for idx, node in enumerate(vertices):
                            d = math.hypot(node.x - node_new.x, node.y - node_new.y)
                            if d < search_radius:
                                near_indices.append(idx)
                        if hasattr(rrt, 'choose_parent'):
                            rrt.choose_parent(node_new, near_indices)
                        if node_new.parent is not None:
                            vertices.append(node_new)
                            if hasattr(rrt, 'rewire'):
                                rrt.rewire(node_new, near_indices)
                            extended = True
                    else:
                        # Basic RRT, Extended, Dynamic
                        vertices.append(node_new)
                        extended = True
                elif vertex_list == 'V1':
                    # For RRT-Connect
                    rrt.V1.append(node_new)
                    extended = True
                else:
                    # Fallback
                    rrt.vertex.append(node_new)
                    extended = True

            # Check frontier regardless of extension success
            check_node = node_new if (node_new and extended) else node_near
            if check_node and self.is_safe_position(check_node.x, check_node.y):
                g = self.world_to_grid(check_node.x, check_node.y)
                if g:
                    gx, gy = g
                    unknown_count = 0
                    check_range = 3
                    for dx in range(-check_range, check_range + 1):
                        for dy in range(-check_range, check_range + 1):
                            ngx = gx + dx
                            ngy = gy + dy
                            if 0 <= ngx < width and 0 <= ngy < height:
                                if self.get_explored_cell_value(ngx, ngy) == -1:
                                    unknown_count += 1
                    if unknown_count >= 1:
                        frontiers.append((check_node.x, check_node.y))

        # 累积保存所有步的RRT树节点和边用于最终可视化
        # 去重：只添加不在树中的新节点，避免重复
        existing_set = set((round(x, 3), round(y, 3)) for (x, y) in self.all_rrt_nodes)

        # 找到正确的顶点列表
        vertex_list = None
        if hasattr(rrt, 'vertex'):
            vertex_list = rrt.vertex
        elif hasattr(rrt, 'V'):
            vertex_list = rrt.V
        elif hasattr(rrt, 'V1'):
            vertex_list = rrt.V1
        elif hasattr(rrt, 'node_list'):
            vertex_list = rrt.node_list

        if vertex_list is not None:
            # 提取所有节点坐标
            for node in vertex_list:
                key = (round(node.x, 3), round(node.y, 3))
                if key not in existing_set:
                    self.all_rrt_nodes.append((node.x, node.y))
                    existing_set.add(key)
                # 如果有父节点，记录这条边（避免重复记录）
                if hasattr(node, 'parent') and node.parent is not None:
                    p_key = (round(node.parent.x, 3), round(node.parent.y, 3))
                    edge = ((node.parent.x, node.parent.y), (node.x, node.y))
                    # 不检查重复直接添加，边重复不影响可视化
                    self.all_rrt_edges.append(edge)

        # 保存所有前沿候选点
        for frontier in frontiers:
            self.all_rrt_frontiers.append(frontier)

        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_rrt(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.rrt import Rrt
        return self.generic_rrt_frontier_detection(Rrt, max_nodes)

    def detect_frontiers_rrt_star(self, max_nodes=500):
        from PathPlanning.Sampling_based_Planning.rrt_2D.rrt_star import RrtStar
        return self.generic_rrt_frontier_detection(RrtStar, max_nodes, search_radius=3.0, has_extra_param=True)

    def detect_frontiers_rrt_connect(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.rrt_connect import RrtConnect
        return self.generic_rrt_frontier_detection(RrtConnect, max_nodes)

    def detect_frontiers_extended_rrt(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.extended_rrt import ExtendedRrt
        return self.generic_rrt_frontier_detection(ExtendedRrt, max_nodes, has_extra_param=True)

    def detect_frontiers_dynamic_rrt(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.dynamic_rrt import DynamicRrt
        return self.generic_rrt_frontier_detection(DynamicRrt, max_nodes, has_extra_param=True)

    def detect_frontiers_rrt_sharp(self, max_nodes=400):
        # from PathPlanning.Sampling_based_Planning.rrt_2D.rrt_sharp import RRTSharp
        # File has no toplevel class definition found, skipping for now
        t_start = time.time()
        frontiers = []
        width = self.full_map.info.width
        height = self.full_map.info.height
        res = self.full_map.info.resolution
        origin_x = self.full_map.info.origin.position.x
        origin_y = self.full_map.info.origin.position.y
        x_min = origin_x
        x_max = origin_x + width * res
        y_min = origin_y
        y_max = origin_y + height * res

        start = [self.sim_x, self.sim_y]
        goal = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]

        rrt_s = RRTSharp(start, goal, self.step_size, 0.05, max_nodes, 3.0)

        for i in range(rrt_s.max_iter):
            node_rand = rrt_s.generate_random_node(rrt_s.goal_sample_rate)
            node_near = rrt_s.nearest_neighbor(rrt_s.vertex, node_rand)
            node_new = rrt_s.new_state(node_near, node_rand)

            if node_new and not self.check_collision_path(node_near, node_new):
                near_nodes = rrt_s.find_near_nodes(rrt_s.vertex, node_new)
                rrt_s.choose_parent(node_new, near_nodes)
                if node_new.parent is not None:
                    rrt_s.vertex.append(node_new)
                    rrt_s.rewire(node_new, near_nodes)
                    if self.is_near_frontier(node_new.x, node_new.y) and self.is_safe_position(node_new.x, node_new.y):
                        frontiers.append((node_new.x, node_new.y))

        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_rrt_star_smart(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.rrt_star_smart import RrtStarSmart
        return self.generic_rrt_frontier_detection(RrtStarSmart, max_nodes, search_radius=3.0, has_extra_param=True)

    def detect_frontiers_informed_rrt_star(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.informed_rrt_star import IRrtStar
        return self.generic_rrt_frontier_detection(IRrtStar, max_nodes, search_radius=3.0, has_extra_param=True)

    def detect_frontiers_bit_star(self, max_nodes=400):
        # BIT* has different algorithm structure not suitable for this frontier detection framework
        t_start = time.time()
        frontiers = []
        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_abit_star(self, max_nodes=400):
        # File is empty (0 bytes), no implementation available
        t_start = time.time()
        frontiers = []
        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_abit_star_advanced(self, max_nodes=400):
        # File is empty (0 bytes), no implementation available
        t_start = time.time()
        frontiers = []
        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_fmt_star(self, max_nodes=400):
        # FMT* has different algorithm structure not suitable for this frontier detection framework
        t_start = time.time()
        frontiers = []
        filtered = self.filter_frontiers(frontiers)
        t_elapsed = time.time() - t_start
        self.total_time_frontier += t_elapsed
        self.frontiers = filtered
        return filtered

    def detect_frontiers_informed_rrt_star(self, max_nodes=400):
        from PathPlanning.Sampling_based_Planning.rrt_2D.informed_rrt_star import IRrtStar
        return self.generic_rrt_frontier_detection(IRrtStar, max_nodes, search_radius=3.0, has_extra_param=True)

    # ========== 基于搜索的信息增益选择 ==========

    def compute_expected_gain(self, wx, wy):
        g = self.world_to_grid(wx, wy)
        if g is None:
            return 0
        cx, cy = g
        width = self.full_map.info.width
        height = self.full_map.info.height
        res = self.full_map.info.resolution
        radius_grid = int(math.ceil(MAX_SENSOR_RANGE / res))
        gain = 0
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                gx = cx + dx
                gy = cy + dy
                if 0 <= gx < width and 0 <= gy < height:
                    if not self.visited_grid[gy, gx]:
                        wwx, wwy = self.grid_to_world(gx, gy)
                        if math.hypot(wwx - wx, wwy - wy) <= MAX_SENSOR_RANGE:
                            gain += 1
        return gain

    def select_closest_frontier(self, frontiers):
        if len(frontiers) == 0:
            return None
        cx, cy = self.sim_x, self.sim_y
        frontiers.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        return frontiers[0]

    # ========== A* 路径规划 (完整避障逻辑，来自原框架) ==========
    def a_star_planning(self, start_wx, start_wy, goal_wx, goal_wy):
        """A*路径规划，带避障检查"""
        import heapq
        t_start = time.time()
        start_gp = self.world_to_grid(start_wx, start_wy)
        goal_gp = self.world_to_grid(goal_wx, goal_wy)
        if start_gp is None or goal_gp is None:
            return None
        sgx, sgy = start_gp
        ggx, ggy = goal_gp

        width = self.explored_map.info.width
        height = self.explored_map.info.height

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
                path = []
                while current in came_from:
                    wx, wy = self.grid_to_world(current[0], current[1])
                    path.append((wx, wy))
                    current = came_from[current]
                path.append((start_wx, start_wy))
                path.reverse()

                simplified = self.simplify_path(path)
                t_elapsed = time.time() - t_start
                self.total_time_astar += t_elapsed
                if len(simplified) >= 2:
                    return simplified
                else:
                    return path

            for dx, dy in moves:
                ngx = cx + dx
                ngy = cy + dy
                neighbor = (ngx, ngy)
                if neighbor in closed_set:
                    continue
                if ngx < 0 or ngx >= width or ngy < 0 or ngy >= height:
                    continue
                val = self.get_explored_cell_value(ngx, ngy)
                if val > OBSTACLE_THRESHOLD:
                    continue
                if (ngx, ngy) != (ggx, ggy) and val == UNKNOWN_THRESHOLD:
                    continue
                if not self.is_position_safe(self.grid_to_world(ngx, ngy)[0], self.grid_to_world(ngx, ngy)[1]):
                    continue

                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + math.sqrt((ggx - ngx)**2 + (ggy - ngy)**2)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, neighbor))

        t_elapsed = time.time() - t_start
        self.total_time_astar += t_elapsed
        return None

    def is_position_safe(self, wx, wy):
        """检查位置是否安全：距离障碍物至少OBSTACLE_SAFE_RADIUS格"""
        t_start = time.time()
        gp = self.world_to_grid(wx, wy)
        if gp is None:
            t_elapsed = time.time() - t_start
            self.total_time_safety += t_elapsed
            return False
        gx, gy = gp
        width = self.explored_map.info.width
        height = self.explored_map.info.height

        if self.safety_cache_computed[gy, gx]:
            t_elapsed = time.time() - t_start
            self.total_time_safety += t_elapsed
            return self.safety_cache[gy, gx]

        check_radius = OBSTACLE_SAFE_RADIUS
        check_start = -check_radius
        check_end = check_radius
        safe = True
        for dx in range(check_start, check_end + 1):
            for dy in range(check_start, check_end + 1):
                ngx = gx + dx
                ngy = gy + dy
                if 0 <= ngx < width and 0 <= ngy < height:
                    if self.visited_grid[ngy, ngx]:
                        nval = self.get_explored_cell_value(ngx, ngy)
                        if nval > OBSTACLE_THRESHOLD:
                            safe = False
                            break
                if not safe:
                    break
            if not safe:
                break

        self.safety_cache[gy, gx] = safe
        self.safety_cache_computed[gy, gx] = True
        t_elapsed = time.time() - t_start
        self.total_time_safety += t_elapsed
        return safe

    def can_connect_directly(self, start_wx, start_wy, end_wx, end_wy):
        """检查两点之间是否可以直接连通（Bresenham直线检查）"""
        t_start = time.time()
        start_gp = self.world_to_grid(start_wx, start_wy)
        end_gp = self.world_to_grid(end_wx, end_wy)
        if start_gp is None or end_gp is None:
            t_elapsed = time.time() - t_start
            self.total_time_safety += t_elapsed
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
            if not self.is_position_safe(self.grid_to_world(x, y)[0], self.grid_to_world(x, y)[1]):
                t_elapsed = time.time() - t_start
                self.total_time_safety += t_elapsed
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

        t_elapsed = time.time() - t_start
        self.total_time_safety += t_elapsed
        return True

    def simplify_path(self, path):
        """路径简化：递归去掉中间不必要的路点"""
        if len(path) <= 2:
            return path

        start = path[0]
        end = path[-1]

        if self.can_connect_directly(start[0], start[1], end[0], end[1]):
            return [start, end]

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
        return left[:-1] + right

    def select_by_search_info_gain(self, frontiers, algorithm='astar'):
        if len(frontiers) == 0:
            return None

        cx, cy = self.sim_x, self.sim_y
        best_ratio = 0
        best = None

        for f in frontiers:
            fx, fy = f
            gain = self.compute_expected_gain(fx, fy)
            dist = math.hypot(fx - cx, fy - cy)
            ratio = gain / (dist + 0.1)
            if ratio > best_ratio:
                best_ratio = ratio
                best = f

        return best

    # ========== 辅助方法 ==========

    def get_coverage(self):
        width = self.full_map.info.width
        height = self.full_map.info.height
        if self._cached_total_free is None:
            self.get_logger().info("📊 计算总自由栅格数...")
            # Convert to numpy array for vectorized computation (much faster)
            data_np = np.array(self.full_map.data, dtype=np.float32).reshape(height, width)
            total_free = np.sum((data_np >= 0) & (data_np < FREE_THRESHOLD))
            self._cached_total_free = int(total_free)
            self.get_logger().info(f"📊 总自由栅格数: {total_free}")
        # 只统计已经探索的自由栅格 - vectorized
        data_np = np.array(self.full_map.data, dtype=np.float32).reshape(height, width)
        mask = self.visited_grid & (data_np >= 0) & (data_np < FREE_THRESHOLD)
        explored_free = int(np.sum(mask))
        return explored_free / self._cached_total_free

    def move_to_goal(self, goal):
        gx, gy = goal
        dx = gx - self.sim_x
        dy = gy - self.sim_y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)

        yaw_diff = abs(target_yaw - self.sim_yaw)
        while yaw_diff > math.pi:
            yaw_diff = 2 * math.pi - yaw_diff
        turn_time = yaw_diff / self.angular_speed
        move_time = dist / self.linear_speed
        total_time = turn_time + move_time

        self.sim_yaw = target_yaw
        self.sim_x = gx
        self.sim_y = gy

        self.total_motion_time += total_time
        self.total_turns += 1
        self.turn_angle_total += yaw_diff

        return total_time

    def save_experiment_result(self):
        coverage = self.get_coverage()
        total_dist = 0.0
        for i in range(1, len(self.actual_trajectory)):
            prev = self.actual_trajectory[i-1]
            curr = self.actual_trajectory[i]
            total_dist += math.hypot(curr[0] - prev[0], curr[1] - prev[1])

        result = {
            "strategy": self.strategy_name,
            "strategy_description": self.strategy.description,
            "timestamp": datetime.now().isoformat(),
            "coverage_final": float(coverage),
            "total_steps": int(self.total_steps),
            "total_motion_time": float(self.total_motion_time),
            "total_turns": int(self.total_turns),
            "total_turn_angle": float(self.turn_angle_total),
            "total_distance": float(total_dist),
            "cpu_time": {
                "sensor": float(self.total_time_sensor),
                "frontier": float(self.total_time_frontier),
                "astar": float(self.total_time_astar),
                "planning": float(self.total_time_planning),
                "safety": float(self.total_time_safety),
                "total": float(self.total_time_sensor + self.total_time_frontier +
                              self.total_time_astar + self.total_time_planning + self.total_time_safety)
            },
            "history": {
                "time": self.time_history,
                "explored_cells": self.explored_cells_history,
                "cpu_time_per_step": self.cpu_time_history
            },
            "trajectory": [(float(x), float(y)) for (x, y, _) in self.actual_trajectory],
            "parameters": {
                "max_sensor_range": MAX_SENSOR_RANGE,
                "fov_angle": math.degrees(FOV_ANGLE),
                "stop_coverage_threshold": self.stop_coverage_threshold,
                "stop_timeout_no_gain": self.stop_timeout_no_gain,
                "linear_speed": self.linear_speed,
                "angular_speed": self.angular_speed,
                "step_size": self.step_size
            },
            "rrt_statistics": {
                "total_nodes": len(self.all_rrt_nodes),
                "total_edges": len(self.all_rrt_edges),
                "total_candidate_frontiers": len(self.all_rrt_frontiers)
            }
        }

        json_path = os.path.join(self.experiment_dir, "result.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        plot_path = os.path.join(self.experiment_dir, "coverage_curve.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(self.time_history) > 0:
            coverage_history = [c / self._cached_total_free for c in self.explored_cells_history]
            ax.plot(self.time_history, coverage_history, 'b-', linewidth=2)
            ax.set_xlabel('Motion Time (s)')
            ax.set_ylabel('Coverage')
            ax.set_title(f'{self.strategy.description} (final coverage: {coverage:.2%})')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.08])
            ax.set_xlim([0, max(self.time_history) * 1.02])
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        map_path = os.path.join(self.experiment_dir, "explored_map.png")
        self.save_map_visualization(map_path)

        # 如果是RRT类算法，额外保存RRT树图
        if len(self.all_rrt_nodes) > 0:
            rrt_path = os.path.join(self.experiment_dir, "rrt_tree.png")
            self.save_rrt_tree_visualization(rrt_path)
            self.get_logger().info(f"🌳 RRT树可视化已保存: {rrt_path}, 共 {len(self.all_rrt_nodes)} 节点")

        self.get_logger().info(f"💾 实验结果已保存: {json_path}")
        return result

    def save_map_visualization(self, filepath):
        width = self.full_map.info.width
        height = self.full_map.info.height
        full_map_data = np.array(self.full_map.data).reshape(height, width)
        explored_map_data = np.array(self.explored_map.data).reshape(height, width)

        fig, ax = plt.subplots(figsize=(10, 10))
        # 第一层：背景显示原始全量地图
        ax.imshow(full_map_data, cmap='gray_r', vmin=-1, vmax=100, origin='lower')

        # 第二层：探索地图覆盖在背景上，未知区域保持透明
        # 创建RGBA图像，只有已探索区域显示，未知区域透明
        explored_rgba = np.zeros((height, width, 4))
        # 已探索区域：使用探索地图颜色，全不透明
        # 未知区域：保持透明alpha=0
        norm = plt.Normalize(vmin=-1, vmax=100)
        cmap = plt.cm.get_cmap('gray_r')
        for gy in range(height):
            for gx in range(width):
                val = explored_map_data[gy, gx]
                if val != UNKNOWN_THRESHOLD:
                    rgb = cmap(norm(val))[:3]
                    explored_rgba[gy, gx] = [*rgb, 1.0]
                else:
                    explored_rgba[gy, gx] = [0, 0, 0, 0]
        ax.imshow(explored_rgba, origin='lower')

        if len(self.actual_trajectory) > 0:
            traj = np.array(self.actual_trajectory)
            origin_x = self.full_map.info.origin.position.x
            origin_y = self.full_map.info.origin.position.y
            res = self.full_map.info.resolution
            traj_gx = (traj[:, 0] - origin_x) / res
            traj_gy = (traj[:, 1] - origin_y) / res
            traj_yaw = traj[:, 2]
            ax.plot(traj_gx, traj_gy, 'r-', linewidth=2, label='Trajectory')
            ax.plot(traj_gx[-1], traj_gy[-1], 'ro', markersize=8)

            # 绘制每个轨迹点的FOV范围（半透明扇形/圆形）
            import matplotlib.patches as patches
            range_grid = MAX_SENSOR_RANGE / res

            # 使用蒙版方式一次性绘制，确保叠加区域也保持恒定0.3透明度
            # 创建一个与地图相同大小的蒙版
            fov_mask = np.zeros((height, width), dtype=bool)

            # 遍历每个轨迹点标记FOV区域
            for wx, wy, yaw in zip(traj[:, 0], traj[:, 1], traj_yaw):
                # 世界坐标转栅格坐标
                cx = int(round((wx - origin_x) / res))
                cy = int(round((wy - origin_y) / res))
                radius_pix = int(math.ceil(range_grid))

                # 计算检查范围（限制在地图内）
                x_min = max(0, cx - radius_pix)
                x_max = min(width - 1, cx + radius_pix)
                y_min = max(0, cy - radius_pix)
                y_max = min(height - 1, cy + radius_pix)

                # 遍历邻域标记
                for gx in range(x_min, x_max + 1):
                    for gy in range(y_min, y_max + 1):
                        dx = (gx - cx) * res
                        dy = (gy - cy) * res
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > MAX_SENSOR_RANGE * MAX_SENSOR_RANGE:
                            continue
                        # 在半径内，检查是否在FOV角度内
                        if FOV_ANGLE < 2 * math.pi:
                            angle = math.atan2(dy, dx)
                            diff = abs(angle - yaw)
                            while diff > math.pi:
                                diff = 2 * math.pi - diff
                            if diff > FOV_ANGLE / 2:
                                continue
                        fov_mask[gy, gx] = True

            # 使用RGBA叠加一次性绘制所有FOV，确保叠加后仍然只有0.3透明度
            # 创建一个全透明RGBA图像
            fov_rgba = np.zeros((height, width, 4))
            # Cyan color: RGB (0, 1, 1)
            fov_rgba[fov_mask] = [0, 1, 1, 0.3]
            ax.imshow(fov_rgba, extent=[-0.5, width-0.5, -0.5, height-0.5], origin='lower')

        if len(self.frontiers) > 0:
            frontiers = np.array(self.frontiers)
            origin_x = self.full_map.info.origin.position.x
            origin_y = self.full_map.info.origin.position.y
            res = self.full_map.info.resolution
            fr_gx = (frontiers[:, 0] - origin_x) / res
            fr_gy = (frontiers[:, 1] - origin_y) / res
            ax.scatter(fr_gx, fr_gy, c='blue', marker='x', s=30, alpha=0.6, label='Frontiers')

        ax.set_title(f"{self.strategy.description}\nSteps: {self.total_steps}, Coverage: {self.get_coverage():.2%}", fontsize=18)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_rrt_tree_visualization(self, filepath):
        """保存RRT树的可视化图，显示所有节点、边和剪枝后的前沿"""
        if len(self.all_rrt_nodes) == 0:
            return

        width = self.full_map.info.width
        height = self.full_map.info.height
        origin_x = self.full_map.info.origin.position.x
        origin_y = self.full_map.info.origin.position.y
        res = self.full_map.info.resolution

        fig, ax = plt.subplots(figsize=(10, 10))

        # 第一层：背景显示原始全量地图
        full_map_data = np.array(self.full_map.data).reshape(height, width)
        ax.imshow(full_map_data, cmap='gray_r', vmin=-1, vmax=100, origin='lower')

        # 第二层：探索地图覆盖在背景上，未知区域透明
        explored_map_data = np.array(self.explored_map.data).reshape(height, width)
        norm = plt.Normalize(vmin=-1, vmax=100)
        cmap = plt.cm.get_cmap('gray_r')
        explored_rgba = np.zeros((height, width, 4))
        for gy in range(height):
            for gx in range(width):
                val = explored_map_data[gy, gx]
                if val != UNKNOWN_THRESHOLD:
                    rgb = cmap(norm(val))[:3]
                    explored_rgba[gy, gx] = [*rgb, 1.0]
                else:
                    explored_rgba[gy, gx] = [0, 0, 0, 0]
        ax.imshow(explored_rgba, origin='lower')

        # 绘制每个轨迹点的FOV范围（半透明扇形/圆形）
        if len(self.actual_trajectory) > 0:
            traj = np.array(self.actual_trajectory)
            traj_yaw = traj[:, 2]

            # 使用蒙版方式一次性绘制，确保叠加区域也保持恒定0.3透明度
            fov_mask = np.zeros((height, width), dtype=bool)

            # 遍历每个轨迹点标记FOV区域
            for wx, wy, yaw in zip(traj[:, 0], traj[:, 1], traj_yaw):
                # 世界坐标转栅格坐标
                cx = int(round((wx - origin_x) / res))
                cy = int(round((wy - origin_y) / res))
                radius_pix = int(math.ceil(MAX_SENSOR_RANGE / res))

                # 计算检查范围（限制在地图内）
                x_min = max(0, cx - radius_pix)
                x_max = min(width - 1, cx + radius_pix)
                y_min = max(0, cy - radius_pix)
                y_max = min(height - 1, cy + radius_pix)

                # 遍历邻域标记
                for gx in range(x_min, x_max + 1):
                    for gy in range(y_min, y_max + 1):
                        dx = (gx - cx) * res
                        dy = (gy - cy) * res
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > MAX_SENSOR_RANGE * MAX_SENSOR_RANGE:
                            continue
                        # 在半径内，检查是否在FOV角度内
                        if FOV_ANGLE < 2 * math.pi:
                            angle = math.atan2(dy, dx)
                            diff = abs(angle - yaw)
                            while diff > math.pi:
                                diff = 2 * math.pi - diff
                            if diff > FOV_ANGLE / 2:
                                continue
                        fov_mask[gy, gx] = True

            # 使用RGBA叠加一次性绘制所有FOV，确保叠加后仍然只有0.3透明度
            fov_rgba = np.zeros((height, width, 4))
            # Cyan color: RGB (0, 1, 1)
            fov_rgba[fov_mask] = [0, 1, 1, 0.3]
            ax.imshow(fov_rgba, extent=[-0.5, width-0.5, -0.5, height-0.5], origin='lower')

        # 绘制RRT所有边（蓝色细线），累积所有步
        for (n1, n2) in self.all_rrt_edges:
            x1, y1 = n1
            x2, y2 = n2
            gx1 = (x1 - origin_x) / res
            gy1 = (y1 - origin_y) / res
            gx2 = (x2 - origin_x) / res
            gy2 = (y2 - origin_y) / res
            ax.plot([gx1, gx2], [gy1, gy2], 'b-', linewidth=0.5, alpha=0.4)

        # 绘制RRT所有节点（蓝色小圆点），累积所有步，去重保存
        if len(self.all_rrt_nodes) > 0:
            nodes_np = np.array(self.all_rrt_nodes)
            nodes_gx = (nodes_np[:, 0] - origin_x) / res
            nodes_gy = (nodes_np[:, 1] - origin_y) / res
            ax.scatter(nodes_gx, nodes_gy, c='blue', s=6, alpha=0.5, label='RRT nodes')

        # 绘制所有候选前沿点（绿色圆形），累积所有步
        if len(self.all_rrt_frontiers) > 0:
            # 去重
            seen = set()
            unique_frontiers = []
            for fx, fy in self.all_rrt_frontiers:
                key = (round(fx, 2), round(fy, 2))
                if key not in seen:
                    seen.add(key)
                    unique_frontiers.append((fx, fy))
            frontiers_np = np.array(unique_frontiers)
            fr_gx = (frontiers_np[:, 0] - origin_x) / res
            fr_gy = (frontiers_np[:, 1] - origin_y) / res
            ax.scatter(fr_gx, fr_gy, c='lime', marker='o', s=20, alpha=0.7, label='Candidate frontiers')

        # 绘制最终筛选后的前沿（橙色X标记）
        if len(self.frontiers) > 0:
            final_np = np.array(self.frontiers)
            f_gx = (final_np[:, 0] - origin_x) / res
            f_gy = (final_np[:, 1] - origin_y) / res
            ax.scatter(f_gx, f_gy, c='orange', marker='X', s=50, alpha=0.9, label='Clustered frontiers')

        # 绘制机器人轨迹（红色）
        if len(self.actual_trajectory) > 0:
            traj = np.array(self.actual_trajectory)
            traj_gx = (traj[:, 0] - origin_x) / res
            traj_gy = (traj[:, 1] - origin_y) / res
            ax.plot(traj_gx, traj_gy, 'r-', linewidth=2, label='Trajectory')
            ax.plot(traj_gx[-1], traj_gy[-1], 'ro', markersize=8)

        ax.set_title(f"{self.strategy.description}\nRRT Tree: {len(self.all_rrt_nodes)} nodes, {len(self.all_rrt_edges)} edges\nSteps: {self.total_steps}, Coverage: {self.get_coverage():.2%}", fontsize=18)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def wait_map_and_start(self):
        self.get_logger().info("⌛ 等待地图话题...")
        while rclpy.ok() and not self.map_received:
            time.sleep(0.1)
        self.get_logger().info("🗺️  地图收到，开始探索...")

        # 随机膨胀传感器最大探测范围: 基础 3.0m → 随机到最大 4.5m
        global MAX_SENSOR_RANGE
        MAX_SENSOR_RANGE = 3.0 + np.random.rand() * 1.5
        self.get_logger().info(f"📏 随机传感器范围: {MAX_SENSOR_RANGE:.2f}m (base 3.0m, max 4.5m)")

        self.get_logger().info(f"📊 地图大小: {self.full_map.info.width}x{self.full_map.info.height} = {self.full_map.info.width * self.full_map.info.height} 栅格")
        # 保险：确保探索地图已经初始化
        if self.visited_grid is None:
            self.init_explored_map()
        self.exploration_loop()

    def exploration_loop(self):
        self.start_time = time.time()
        self.actual_trajectory.append((self.sim_x, self.sim_y, self.sim_yaw))
        self.get_logger().info("🔍 初始传感器扫描...")
        self.sensor_simulate()
        self.get_logger().info("🚀 开始探索循环...")

        while rclpy.ok() and self.auto_explore_running:
            coverage = self.get_coverage()

            # 记录当前已探索自由栅格数量，用于绘图 - numpy vectorized (fast)
            width = self.full_map.info.width
            height = self.full_map.info.height
            data_np = np.array(self.full_map.data, dtype=np.float32).reshape(height, width)
            mask = self.visited_grid & (data_np >= 0) & (data_np < FREE_THRESHOLD)
            explored_free_now = int(np.sum(mask))
            self.explored_cells_history.append(explored_free_now)
            self.time_history.append(self.total_motion_time)

            if coverage >= self.stop_coverage_threshold:
                self.get_logger().info(f"🏁 达到目标覆盖率 {coverage:.2%}, 探索完成!")
                break

            if self.total_steps >= self.stop_max_steps:
                self.get_logger().info(f"⚠️ 达到最大步数 {self.stop_max_steps}, 停止探索")
                break

            # 超时检测：覆盖率超过120秒没有增长，提前停止
            if coverage <= self._last_coverage:
                if self.total_motion_time - self._last_coverage_time > self.stop_timeout_no_gain:
                    self.get_logger().info(f"⌛ 覆盖率 {coverage:.2%} {self.stop_timeout_no_gain}s 没有增长，提前停止探索")
                    break
            else:
                self._last_coverage = coverage
                self._last_coverage_time = self.total_motion_time

            step_start_time = time.time()

            frontiers = self.strategy.detect_frontiers()
            goal = self.strategy.select_goal(frontiers)

            step_cpu_time = time.time() - step_start_time
            self.cpu_time_history.append(step_cpu_time)
            self.planning_time_total += step_cpu_time

            if goal is None:
                self.get_logger().warning("⚠️ 没有找到前沿，重试...")
                if not hasattr(self, '_frontier_retries'):
                    self._frontier_retries = 0
                self._frontier_retries += 1
                if self._frontier_retries >= 5:
                    self.get_logger().warning("❌ 连续5次没找到前沿，探索结束")
                    break
                else:
                    continue
            self._frontier_retries = 0

            self.current_goal = goal
            self.exploration_path.append(goal)
            self.get_logger().info(f"📍 步骤 {self.total_steps + 1}: 选择目标 ({goal[0]:.2f}, {goal[1]:.2f}), "
                                  f"覆盖率 {coverage:.2%}")

            # A* 路径规划，避开障碍物
            t_start = time.time()
            path = self.a_star_planning(self.sim_x, self.sim_y, goal[0], goal[1])
            t_elapsed = time.time() - t_start
            self.total_time_astar += t_elapsed

            if path is None:
                self.get_logger().warning(f"❌ 无法规划路径到目标，标记为不可达，重新检测")
                g = self.world_to_grid(goal[0], goal[1])
                if g is not None:
                    self.unreachable_targets.add((goal[0], goal[1]))
                continue

            self.get_logger().info(f"🗺️  A*规划完成: {len(path)} 个路点，耗时 {t_elapsed*1000:.1f}ms")

            # 沿着规划路径逐点移动
            for i in range(1, len(path)):
                waypoint = path[i]
                self.move_to_goal(waypoint)  # move_to_goal内部已经累加total_motion_time
                self.actual_trajectory.append((self.sim_x, self.sim_y, self.sim_yaw))

                # 每走一段就重新扫描
                new_cells_step = self.sensor_simulate()

            new_cells = new_cells_step if 'new_cells_step' in locals() else self.sensor_simulate()
            if new_cells < self.min_new_grid_threshold:
                g = self.world_to_grid(self.sim_x, self.sim_y)
                if g is not None:
                    self.no_value_grid.add(g)
                self.get_logger().info(f"ℹ️ 区域新增栅格太少({new_cells})，标记为无价值")

            self.total_steps += 1
            time.sleep(0.01)

        self.save_experiment_result()
        self.get_logger().info(f"✅ 实验完成: {self.total_steps} 步, 覆盖率 {self.get_coverage():.2%}, "
                              f"总时间 {self.total_motion_time:.2f}s")

        # 探索完成，自动退出，让下一个实验可以开始
        self.get_logger().info("🏁 实验结束，退出程序...")
        rclpy.shutdown()
        os._exit(0)

    def publish_visualization(self):
        if not self.publish_enabled:
            return
        if self.explored_map is not None:
            self.explored_map_pub.publish(self.explored_map)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for (x, y, _) in self.actual_trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)
        if len(self.actual_trajectory) > 0:
            self.actual_trajectory_pub.publish(path_msg)

        marker_array = MarkerArray()
        for i, (x, y) in enumerate(self.frontiers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.6
            marker_array.markers.append(marker)
        self.frontier_pub.publish(marker_array)

        robot_marker = Marker()
        robot_marker.header.frame_id = "map"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = self.sim_x
        robot_marker.pose.position.y = self.sim_y
        robot_marker.pose.position.z = 0.0
        robot_marker.scale.x = 0.5
        robot_marker.scale.y = 0.5
        robot_marker.scale.z = 0.1
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 0.0
        robot_marker.color.a = 0.8
        self.robot_marker_pub.publish(robot_marker)


def main():
    parser = argparse.ArgumentParser(
        description='自主探索对比实验框架 - 支持多种策略'
    )
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        required=True,
        help=f'探索策略名称: {StrategyFactory.get_available_strategies()}'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./experiments',
        help='结果输出目录'
    )
    parser.add_argument(
        '--coverage-threshold', '-c',
        type=float,
        default=0.95,
        help='停止覆盖率阈值'
    )
    parser.add_argument(
        '--max-steps', '-m',
        type=int,
        default=500,
        help='最大步数'
    )
    parser.add_argument(
        '--publish', '-p',
        action='store_true',
        default=False,
        help='启用话题发布（默认关闭，避免干扰建图）'
    )
    parser.add_argument(
        '--linear-speed', '-l',
        type=float,
        default=1.0,
        help='机器人直线速度 (m/s)'
    )
    parser.add_argument(
        '--angular-speed', '-a',
        type=float,
        default=1.0,
        help='机器人旋转速度 (rad/s)'
    )
    parser.add_argument(
        '--timeout-no-gain', '-t',
        type=int,
        default=120,
        help='如果超过N秒覆盖率没有增长，提前停止'
    )
    args = parser.parse_args()

    if args.strategy not in StrategyFactory.get_available_strategies():
        print(f"错误: 未知策略 {args.strategy}")
        print(f"可用策略: {StrategyFactory.get_available_strategies()}")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(args.output_dir, f"{args.strategy}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    rclpy.init()
    node = SimulationExperimentExplorer(args.strategy, experiment_dir, args.publish)
    node.stop_coverage_threshold = args.coverage_threshold
    node.stop_max_steps = args.max_steps
    node.linear_speed = args.linear_speed
    node.angular_speed = args.angular_speed

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，保存当前结果...")
        node.save_experiment_result()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
