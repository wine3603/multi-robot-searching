#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, Imu
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import cv2
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, Float32MultiArray

# ==================== 摄像头地面分割参数 (深度学习推理) ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPIC_NAME = "/rtsp_image"
GROUND_MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints_ground", "ground_best.pth")
GROUND_INFER_FREQ = 5.0  # 推理频率 (Hz)
GROUND_THRESHOLD = 0.5   # 分割阈值
INFERENCE_STEP = 2       # 投影时步长，降采样加速

CAM_HEIGHT = 0.35
CAM_PITCH = np.deg2rad(15)
CAM_FOV_H = 82.0
CAM_FOV_V = 62.0
MAX_GROUND_DISTANCE = 10.0  # 最大地面标记距离，超过此距离不标记

# ==================== 代价分级配置 ====================
COST_CAMERA_GROUND = 10    
COST_EMPTY_SPACE = 20      
COST_OBSTACLE = 100        
COST_GROUND_BASE = 40      
COST_BOTTLE = 35          
COST_BOTTLE_SURROUND = 15

# ==================== 累积点云密度控制 ====================
# 全图体素降采样，每个 voxel_size 格子只保留一个点
# 限制点云密度，不会无限增长
VOXEL_SIZE = 0.1
UPDATE_FREQUENCY = 1.0

# ==================== 其他配置 ====================
LOCAL_CLOUD_TOPIC = "/local_cloud_aligned"  # 可选：实时点云做障碍检测

# ==================== 机器人历史轨迹配置 ====================
TRAJECTORY_MIN_DIST = 0.1    # 轨迹点之间的最小距离(米)，避免过于密集
OBSTACLE_DILATE_KERNEL_SIZE = 3
OBSTACLE_DILATE_ITERATIONS = 1
BOTTLE_SURROUND_RADIUS = 3
BOTTLE_TOPIC = "/bottle_grid_coords"
BOTTLE_GRID_BLOCK_SIZE = 50
BOTTLE_MAX_DISTANCE = 50.0

# ==================== 射线扫描优化参数 ====================
MAX_RAY_DISTANCE = 10.0  # 只处理这个距离内的障碍物点，太远的不处理，节省计算
ANGLE_BIN_SIZE = 1.0     # 角度分bin大小(度)，每个bin只保留最近的障碍物点
# → 360度总共只需要360根线，大大减少计算量

# ==================== 空闲区域标记模式选择 ====================
# FREE_SPACE_MODE: 'ray' - 射线扫描模式(从机器人出发，碰到障碍物停止)
#                 'circle' - 圆形区域模式: 机器人位置为中心，半径范围内无障碍物即为空闲
FREE_SPACE_MODE = 'circle'
CIRCLE_FREE_RADIUS = 2.0  # 圆形空闲区域半径(米)

# ============================================================================
# UNet 模型定义 (地面分割)
# ============================================================================
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c_in, c_out))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, 2)
        self.conv = DoubleConv(c_in//2 + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.shape[2]-x.shape[2], skip.shape[3]-x.shape[3]
        x = F.pad(x, (0,dw,0,dh))
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self, c_in=3, num_classes=1, base=32):
        super().__init__()
        b = base
        self.e1 = DoubleConv(c_in, b)
        self.e2 = Down(b, b*2)
        self.e3 = Down(b*2, b*4)
        self.e4 = Down(b*4, b*8)
        self.bot = DoubleConv(b*8, b*8)
        self.d3 = Up(b*8, b*4, b*4)
        self.d2 = Up(b*4, b*2, b*2)
        self.d1 = Up(b*2, b, b)
        self.head = nn.Conv2d(b, num_classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        x = self.bot(e4)
        x = self.d3(x, e3)
        x = self.d2(x, e2)
        x = self.d1(x, e1)
        return self.head(x)

# Image normalization constants
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FIELD_TYPES = {
    PointField.INT8: 'b', PointField.UINT8: 'B', PointField.INT16: 'h',
    PointField.UINT16: 'H', PointField.INT32: 'i', PointField.UINT32: 'I',
    PointField.FLOAT32: 'f', PointField.FLOAT64: 'd'
}

class GridMapGenerator(Node):
    def __init__(self):
        super().__init__('grid_map_generator')

        self.declare_parameters('', [
            ('map_resolution', 0.05),
            #('map_min_x', -11.5),   # 地图X范围最小值(米)
            #('map_max_x', 5.0),    # 地图X范围最大值(米)
            #('map_min_y', -15.0),   # 地图Y范围最小值(米)
            #('map_max_y', 10.0),    # 地图Y范围最大值(米)
            #('map_rotation_deg', -10.0),  # 地图边界旋转角度(度)，正数逆时针旋转
            ('map_min_x', -1.0),   # 地图X范围最小值(米)
            ('map_max_x', 5.5),    # 地图X范围最大值(米)
            ('map_min_y', -6.0),   # 地图Y范围最小值(米)
            ('map_max_y', 6.5),    # 地图Y范围最大值(米)
            ('map_rotation_deg', 0.0),
            ('border_value', 200),  # 边框4格栅格填充值
            ('max_obstacle_height', 1.5),
            ('min_obstacle_height', -0.1),
            ('point_cloud_topic', '/cloud_registered'),
            ('map_frame_id', 'camera_init'),
            ('robot_frame', 'body'),
            ('merge_local_cloud', False),
        ])

        self.res = self.get_parameter('map_resolution').value
        self.map_min_x = self.get_parameter('map_min_x').value
        self.map_max_x = self.get_parameter('map_max_x').value
        self.map_min_y = self.get_parameter('map_min_y').value
        self.map_max_y = self.get_parameter('map_max_y').value
        self.map_rotation_deg = self.get_parameter('map_rotation_deg').value
        self.map_theta = np.deg2rad(self.map_rotation_deg)
        self.border_value = self.get_parameter('border_value').value

        # 原矩形尺寸
        orig_w = self.map_max_x - self.map_min_x
        orig_h = self.map_max_y - self.map_min_y

        # 计算旋转后矩形的轴对齐包围盒(AABB)尺寸
        cos_theta = abs(math.cos(self.map_theta))
        sin_theta = abs(math.sin(self.map_theta))
        aabb_w = orig_w * cos_theta + orig_h * sin_theta
        aabb_h = orig_w * sin_theta + orig_h * cos_theta

        # 计算栅格地图尺寸
        self.w = int(round(aabb_w / self.res))
        self.h = int(round(aabb_h / self.res))

        # 自动额外拓展100格（每边各50格）
        MAP_MARGIN_GRIDS = 0
        self.w += MAP_MARGIN_GRIDS
        self.h += MAP_MARGIN_GRIDS

        # 计算中心点
        self.map_cx = (self.map_min_x + self.map_max_x) / 2.0
        self.map_cy = (self.map_min_y + self.map_max_y) / 2.0

        # 拓展后的实际尺寸
        final_w = self.w * self.res
        final_h = self.h * self.res

        # 原点 = 中心点 - 拓展后尺寸的一半
        self.ox = self.map_cx - final_w / 2.0
        self.oy = self.map_cy - final_h / 2.0
        self.max_h = self.get_parameter('max_obstacle_height').value
        self.min_h = self.get_parameter('min_obstacle_height').value
        self.cloud_topic = self.get_parameter('point_cloud_topic').value
        self.map_frame = self.get_parameter('map_frame_id').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.merge_local_cloud = self.get_parameter('merge_local_cloud').value

        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OBSTACLE_DILATE_KERNEL_SIZE, OBSTACLE_DILATE_KERNEL_SIZE))
        self.bottle_surround_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*BOTTLE_SURROUND_RADIUS+1, 2*BOTTLE_SURROUND_RADIUS+1))
        
        # 地图数据
        self.grid_map = OccupancyGrid()
        self.grid_map.header.frame_id = self.map_frame
        self.grid_map.info.resolution = self.res
        self.grid_map.info.width = self.w
        self.grid_map.info.height = self.h
        self.grid_map.info.origin.position.x = self.ox
        self.grid_map.info.origin.position.y = self.oy
        self.grid_map.info.origin.orientation.w = 1.0
        # 初始全部标记为未知(-1)，只有被射线扫描过才知道是不是空闲
        self.grid_map.data = [-1]*(self.w*self.h)
        # 保存一个已扫描标记：哪些格子已经被射线扫描过
        self.scanned_grid = np.zeros((self.h, self.w), dtype=bool)

        self.ground_grid = np.zeros((self.h, self.w), dtype=np.int8)
        self.local_obs_grid = np.zeros((self.h, self.w), dtype=bool)  # local_cloud_aligned 检测的障碍
        self.cumulative_obs_grid = np.zeros((self.h, self.w), dtype=bool)  # 累积点云检测的障碍
        self.cumulative_points = np.empty((0, 3), dtype=np.float32)
        
        # 瓶子管理
        self.permanent_bottles = []
        self.bottle_blocks = set()
        self.bottle_core_grid = np.zeros((self.h, self.w), dtype=bool)
        self.bottle_surround_grid = np.zeros((self.h, self.w), dtype=bool)

        # Directly subscribe to /tf topic instead of using TF buffer lookup
        # Avoids "frame does not exist" lookup failures
        # Use default QoS which matches TransformBroadcaster (RELIABLE)
        from rclpy.qos import qos_profile_system_default
        self.latest_transform = None
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.has_robot_pose = False
        self.trajectory_history = []  # 存储历史轨迹点
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            qos_profile_system_default
        )
        self.bridge = CvBridge()
        self.img = None
        self.cam_mat = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        # IMU 静止检测
        self.latest_imu = None
        self.ANGULAR_VEL_THRESHOLD = 0.1  # 角速度阈值，小于此值才认为静止

        # 加载地面分割模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"加载地面分割模型: {GROUND_MODEL_PATH} 设备: {self.device}")
        ckpt = torch.load(GROUND_MODEL_PATH, map_location=self.device, weights_only=True)
        self.ground_model = UNet(c_in=3, num_classes=1, base=32).to(self.device)
        self.ground_model.load_state_dict(ckpt)
        self.ground_model.eval()
        self.get_logger().info("✅ 地面分割模型加载完成")

        # QoS
        qos_cumulative = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_cloud = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        # 发布与订阅
        self.cumulative_cloud_pub = self.create_publisher(PointCloud2, '/cumulative_cloud_registered', qos_cumulative)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.ground_mask_pub = self.create_publisher(Image, '/ground_mask_img', 10)    # 发布分割mask
        self.ground_vis_pub = self.create_publisher(Image, '/ground_vis_img', 10)      # 发布可视化结果
        self.trajectory_pub = self.create_publisher(Path, '/robot_trajectory', 10)      # 发布机器人历史轨迹

        self.cloud_sub = self.create_subscription(PointCloud2, self.cloud_topic, self.cloud_cb, qos_cloud)
        self.img_sub = self.create_subscription(Image, TOPIC_NAME, self.img_cb, qos_cloud)
        self.local_cloud_sub = self.create_subscription(PointCloud2, LOCAL_CLOUD_TOPIC, self.local_cloud_cb, qos_cloud)
        self.bottle_sub = self.create_subscription(Float32MultiArray, BOTTLE_TOPIC, self.bottle_cb, qos_cumulative)
        self.imu_sub = self.create_subscription(Imu, '/livox/imu', self.imu_cb, qos_cloud)

        # 定时器
        self.create_timer(1.0/UPDATE_FREQUENCY, self.publish_cumulative_cloud)
        self.create_timer(0.1, self.update_and_publish_map)
        self.create_timer(0.1, self.update_robot_tf)
        self.create_timer(1.0/GROUND_INFER_FREQ, self.process_ground)

        self.get_logger().info("✅ 节点启动：累积点云已完全移除数量限制，内存将随运行时间持续增长。")
        self.get_logger().info("📷 使用深度学习模型进行路面检测，输出: /ground_mask_img, /ground_vis_img")
        if self.merge_local_cloud:
            self.get_logger().info("🔄 merge_local_cloud: ENABLED - local_cloud_aligned will be merged into cumulative point cloud")
        else:
            self.get_logger().info("🔄 merge_local_cloud: DISABLED - local_cloud_aligned only for real-time obstacle detection, not merged")

    def cloud_cb(self, msg):
        if msg.header.frame_id != self.map_frame: return
        new_pts = self.pc2numpy(msg)
        if new_pts is not None:
            self.update_cumulative_cloud(new_pts)

    def local_cloud_cb(self, msg):
        """local_cloud_aligned 做实时障碍检测，可选合并到全局累积点云
        FREE_SPACE_MODE = 'ray': 射线扫描，从机器人出发，碰到障碍物就停止，障碍物后方不标记空闲
        FREE_SPACE_MODE = 'circle': 圆形区域，机器人位置为中心，半径范围内无障碍物即为空闲
        """
        pts = self.pc2numpy(msg)
        if pts is not None and self.has_robot_pose:
            # 累积到全局点云（仅当参数开启时）
            if self.merge_local_cloud:
                self.update_cumulative_cloud(pts)
            # 做实时障碍检测
            self.local_obs_grid.fill(False)
            valid = pts[(pts[:,2]>=self.min_h) & (pts[:,2]<=self.max_h)]

            # 获取机器人当前位置
            rx = self.robot_x
            ry = self.robot_y
            rx_gx = int((rx - self.ox)/self.res)
            ry_gy = int((ry - self.oy)/self.res)

            # ===== 第一步：所有符合高度条件的点直接标记为障碍物 =====
            # 使用配置的最大距离过滤
            if FREE_SPACE_MODE == 'ray':
                max_dist = MAX_RAY_DISTANCE*10
            else: # circle
                max_dist = CIRCLE_FREE_RADIUS*10

            for (x,y,_) in valid:
                dist = math.hypot(x - rx, y - ry)
                if dist > max_dist or dist < 0.1:
                    continue
                gx = int((x - self.ox)/self.res)
                gy = int((y - self.oy)/self.res)
                if 0<=gx<self.w and 0<=gy<self.h:
                    self.local_obs_grid[gy, gx] = True

            # 合并障碍物：任何地方只要是障碍（local 或 cumulative）
            any_obs_grid = np.logical_or(self.cumulative_obs_grid, self.local_obs_grid)
            processed_count = 0

            if FREE_SPACE_MODE == 'ray':
                # ===== 模式1: 射线扫描 =====
                # 按角度分bin，只保留每个角度最远的点
                angle_bins = {}  # angle_bin -> (max_dist, x, y, gx, gy)
                for (x,y,_) in valid:
                    dist = math.hypot(x - rx, y - ry)
                    if dist > MAX_RAY_DISTANCE or dist < 0.1:
                        continue  # 太远或者太近跳过

                    # 计算角度，离散化
                    angle = math.atan2(y - ry, x - rx)
                    # 只保留前方 240 度视角 (robot_yaw 左右各 120 度)
                    angle_relative = math.degrees(angle - self.robot_yaw)
                    while angle_relative > 180:
                        angle_relative -= 360
                    while angle_relative < -180:
                        angle_relative += 360
                    if abs(angle_relative) > 120:
                        continue  # 超出240度前方视角，跳过

                    angle_deg = math.degrees(angle)
                    if angle_deg < 0:
                        angle_deg += 360
                    bin_idx = int(angle_deg / ANGLE_BIN_SIZE)

                    # 每个角度只需要扫描到最远点，这样会自然被途中障碍物挡住
                    if bin_idx not in angle_bins or dist > angle_bins[bin_idx][0]:
                        gx = int((x - self.ox)/self.res)
                        gy = int((y - self.oy)/self.res)
                        angle_bins[bin_idx] = (dist, x, y, gx, gy)

                # 射线扫描标记空闲区域，碰到障碍物就停止
                for bin_idx in angle_bins:
                    (dist, x, y, gx, gy) = angle_bins[bin_idx]
                    if not (0<=gx<self.w and 0<=gy<self.h):
                        continue

                    # 一步步走，碰到障碍物就停止
                    line_points = self.bresenham_line(rx_gx, ry_gy, gx, gy)
                    for (lgx, lgy) in line_points:
                        # 如果这格已经是障碍物，停止
                        if any_obs_grid[lgy, lgx]:
                            break
                        # 根据距离膨胀，标记空闲
                        expand_radius = max(1, int(dist / 5.0))
                        expand_range = list(range(-expand_radius, expand_radius+1))
                        for dx in expand_range:
                            for dy in expand_range:
                                nlx = lgx + dx
                                nly = lgy + dy
                                if 0<=nlx<self.w and 0<=nly<self.h:
                                    if not any_obs_grid[nly, nlx] and not self.scanned_grid[nly, nlx]:
                                        self.scanned_grid[nly, nlx] = True
                                        processed_count += 1

            elif FREE_SPACE_MODE == 'circle':
                # ===== 模式2: 前方150度用完整半径，后方210度只用1/3半径 =====
                # 获取整个圆形栅格范围（按最大半径）
                radius_gx = int(CIRCLE_FREE_RADIUS / self.res)
                radius_gy = int(CIRCLE_FREE_RADIUS / self.res)

                gx_min = max(0, rx_gx - radius_gx)
                gx_max = min(self.w - 1, rx_gx + radius_gx)
                gy_min = max(0, ry_gy - radius_gy)
                gy_max = min(self.h - 1, ry_gy + radius_gy)

                full_radius_sq = CIRCLE_FREE_RADIUS * CIRCLE_FREE_RADIUS
                half_radius_sq = (CIRCLE_FREE_RADIUS / 3) * (CIRCLE_FREE_RADIUS / 3)

                # 遍历圆形范围内所有栅格
                for gx in range(gx_min, gx_max + 1):
                    for gy in range(gy_min, gy_max + 1):
                        # 计算距离平方
                        dx_g = gx - rx_gx
                        dy_g = gy - ry_gy
                        dist_g = (dx_g * dx_g + dy_g * dy_g) * (self.res * self.res)  # 实际距离平方

                        # 根据角度判断用哪个半径阈值
                        wx = self.ox + gx * self.res
                        wy = self.oy + gy * self.res
                        angle = math.atan2(wy - ry, wx - rx)
                        angle_relative = math.degrees(angle - self.robot_yaw)
                        while angle_relative > 180:
                            angle_relative -= 360
                        while angle_relative < -180:
                            angle_relative += 360

                        if abs(angle_relative) <= 120:
                            # 前方150度，用完整半径
                            if dist_g > full_radius_sq:
                                continue
                        else:
                            # 后方210度，只用1/3半径
                            if dist_g > half_radius_sq:
                                continue

                        # 如果不是障碍物，标记为空闲
                        if not any_obs_grid[gy, gx] and not self.scanned_grid[gy, gx]:
                            self.scanned_grid[gy, gx] = True
                            processed_count += 1

            # 起点也标记
            if 0<=rx_gx<self.w and 0<=ry_gy<self.h:
                if not self.scanned_grid[ry_gy, rx_gx]:
                    self.scanned_grid[ry_gy, rx_gx] = True
                    processed_count += 1

            if processed_count > 0:
                self.get_logger().info(f"[local_cloud] mode={FREE_SPACE_MODE}, 障碍物: {np.sum(self.local_obs_grid)}, 标记空闲: {processed_count} 格, 累积 {len(pts)} 点")

    def update_cumulative_cloud(self, new_points):
        if not self.has_robot_pose:
            self.get_logger().info(f"[cloud_safe] 跳过：has_robot_pose = False (TF查找失败)")
            return
        if len(new_points) == 0:
            self.get_logger().info(f"[cloud_safe] 跳过：点云为空")
            return

        self.get_logger().info(f"[cloud_safe] 成功加入 {len(new_points)} 个点")

        # 全图统一控制密度：合并新点后对全图做去重
        # 保证整个地图任意位置都满足密度限制（任意两点距离 >= CLEAN_DISTANCE_THRESHOLD）
        if len(self.cumulative_points) > 0:
            combined = np.vstack([self.cumulative_points, new_points])
        else:
            combined = new_points

        # 对全图做体素降采样，控制整个地图的点云密度
        if len(combined) > 100:
            self.cumulative_points = self.remove_duplicate_points(combined, voxel_size=VOXEL_SIZE)
        else:
            self.cumulative_points = combined

    def remove_duplicate_points(self, points, voxel_size=0.1):
        """Voxel downsampling for uniform density control.
        Each voxel keeps only one point, guarantees uniform density across whole map.
        """
        if len(points) < 2:
            return points

        # Calculate voxel coordinates
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)

        # Use dictionary to keep one point per voxel
        # Since we process old -> new, later (newer) points overwrite older ones
        voxel_dict = {}
        for idx, coords in enumerate(voxel_coords):
            voxel_key = (coords[0], coords[1], coords[2])
            voxel_dict[voxel_key] = idx  # newer overwrites older

        kept_indices = list(voxel_dict.values())
        return points[kept_indices]

    # --- 以下为辅助与业务逻辑函数 (保持原样) ---
    def pc2numpy(self, msg):
        try:
            fs = {f.name:(f.offset, FIELD_TYPES[f.datatype]) for f in msg.fields}
            out = []
            for i in range(0, len(msg.data), msg.point_step):
                d = msg.data[i:i+msg.point_step]
                x = struct.unpack(fs['x'][1], d[fs['x'][0]:fs['x'][0]+4])[0]
                y = struct.unpack(fs['y'][1], d[fs['y'][0]:fs['y'][0]+4])[0]
                z = struct.unpack(fs['z'][1], d[fs['z'][0]:fs['z'][0]+4])[0]
                if not np.isnan(x): out.append((x,y,z))
            return np.array(out, dtype=np.float32)
        except: return None

    def publish_cumulative_cloud(self):
        if len(self.cumulative_points) == 0: return
        h = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame)
        fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i,n in enumerate('xyz')]
        self.cumulative_cloud_pub.publish(pc2.create_cloud(h, fields, self.cumulative_points.tolist()))

    def tf_callback(self, msg):
        # /tf contains multiple transforms in msg.transforms list
        for transform in msg.transforms:
            # Debug: print what we receive (only first 10 times)
            if not hasattr(self, '_debug_print_count'):
                self._debug_print_count = 0
            if self._debug_print_count < 10:
                self.get_logger().info(f"[DEBUG TF] Received: {transform.header.frame_id} -> {transform.child_frame_id}")
                self._debug_print_count += 1
            # We only care about camera_init -> body_safe
            if transform.header.frame_id == self.map_frame and transform.child_frame_id == self.robot_frame:
                self.latest_transform = transform
                # Update pose immediately
                self.robot_x = transform.transform.translation.x
                self.robot_y = transform.transform.translation.y
                q = transform.transform.rotation
                self.robot_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                if not self.has_robot_pose:
                    self.get_logger().info(f"✅ Received first {self.map_frame} -> {self.robot_frame}, has_robot_pose = True")
                self.has_robot_pose = True

    def update_robot_tf(self):
        # Already updated by tf_callback, nothing to do
        # Keep this timer for compatibility

        # 累计历史轨迹并发布Path消息
        if self.has_robot_pose:
            x, y = self.robot_x, self.robot_y
            if len(self.trajectory_history) == 0:
                self.trajectory_history.append((x, y))
            else:
                last_x, last_y = self.trajectory_history[-1]
                dist = math.hypot(x - last_x, y - last_y)
                if dist > TRAJECTORY_MIN_DIST:
                    self.trajectory_history.append((x, y))

            # 构建并发布Path消息
            path_msg = Path()
            path_msg.header.frame_id = self.map_frame
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for (tx, ty) in self.trajectory_history:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = tx
                pose.pose.position.y = ty
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                path_msg.poses.append(pose)

            self.trajectory_pub.publish(path_msg)

    def dilate_obstacle_grid(self, grid):
        u8 = (grid * 255).astype(np.uint8)
        return cv2.dilate(u8, self.dilate_kernel, iterations=OBSTACLE_DILATE_ITERATIONS) > 0

    def update_cumulative_free_space(self):
        """用累积点云标记空闲区域
        FREE_SPACE_MODE = 'ray': 射线扫描，从机器人出发，碰到障碍物就停止，障碍物后方不标记空闲
        FREE_SPACE_MODE = 'circle': 圆形区域，机器人位置为中心，半径范围内无障碍物即为空闲
        """
        if not self.has_robot_pose or len(self.cumulative_points) == 0:
            return

        # 每次清空，重新标记障碍物，因为累积点云更新了
        self.cumulative_obs_grid.fill(False)

        valid = self.cumulative_points[(self.cumulative_points[:,2]>=self.min_h) & (self.cumulative_points[:,2]<=self.max_h)]

        # 获取机器人当前位置
        rx = self.robot_x
        ry = self.robot_y
        rx_gx = int((rx - self.ox)/self.res)
        ry_gy = int((ry - self.oy)/self.res)

        # ===== 第一步：所有符合高度条件的点直接标记为障碍物 =====
        # 使用配置的最大距离过滤
        if FREE_SPACE_MODE == 'ray':
            max_dist = MAX_RAY_DISTANCE*10
        else: # circle
            max_dist = CIRCLE_FREE_RADIUS*10

        for (x,y,_) in valid:
            dist = math.hypot(x - rx, y - ry)
            if dist > max_dist or dist < 0.1:
                continue
            gx = int((x - self.ox)/self.res)
            gy = int((y - self.oy)/self.res)
            if 0<=gx<self.w and 0<=gy<self.h:
                self.cumulative_obs_grid[gy, gx] = True

        # 合并障碍物：任何地方只要是障碍（local 或 cumulative）
        any_obs_grid = np.logical_or(self.cumulative_obs_grid, self.local_obs_grid)
        processed_count = 0

        if FREE_SPACE_MODE == 'ray':
            # ===== 模式1: 射线扫描 =====
            # 按角度分bin，只保留每个角度最远的点
            angle_bins = {}  # angle_bin -> (max_dist, x, y, gx, gy)
            for (x,y,_) in valid:
                dist = math.hypot(x - rx, y - ry)
                if dist > MAX_RAY_DISTANCE or dist < 0.1:
                    continue

                # 计算角度，离散化
                angle = math.atan2(y - ry, x - rx)
                # 只保留前方 240 度视角 (robot_yaw 左右各 120 度)
                angle_relative = math.degrees(angle - self.robot_yaw)
                while angle_relative > 180:
                    angle_relative -= 360
                while angle_relative < -180:
                    angle_relative += 360
                if abs(angle_relative) > 120:
                    continue  # 超出240度前方视角，跳过

                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360
                bin_idx = int(angle_deg / ANGLE_BIN_SIZE)

                # 每个角度只需要扫描到最远点，这样会自然被途中障碍物挡住
                if bin_idx not in angle_bins or dist > angle_bins[bin_idx][0]:
                    gx = int((x - self.ox)/self.res)
                    gy = int((y - self.oy)/self.res)
                    angle_bins[bin_idx] = (dist, x, y, gx, gy)

            # 射线扫描标记空闲区域，碰到障碍物就停止
            for bin_idx in angle_bins:
                (dist, x, y, gx, gy) = angle_bins[bin_idx]
                if not (0<=gx<self.w and 0<=gy<self.h):
                    continue

                # 一步步走，碰到障碍物就停止
                line_points = self.bresenham_line(rx_gx, ry_gy, gx, gy)
                for (lgx, lgy) in line_points:
                    # 如果这格已经是障碍物，停止
                    if any_obs_grid[lgy, lgx]:
                        break
                    # 根据距离膨胀，标记空闲
                    expand_radius = max(1, int(dist / 5.0))
                    expand_range = list(range(-expand_radius, expand_radius+1))
                    for dx in expand_range:
                        for dy in expand_range:
                            nlx = lgx + dx
                            nly = lgy + dy
                            if 0<=nlx<self.w and 0<=nly<self.h:
                                if not any_obs_grid[nly, nlx] and not self.scanned_grid[nly, nlx]:
                                    self.scanned_grid[nly, nlx] = True
                                    processed_count += 1

        elif FREE_SPACE_MODE == 'circle':
            # ===== 模式2: 前方150度用完整半径，后方210度只用1/3半径 =====
            # 获取整个圆形栅格范围（按最大半径）
            radius_gx = int(CIRCLE_FREE_RADIUS / self.res)
            radius_gy = int(CIRCLE_FREE_RADIUS / self.res)

            gx_min = max(0, rx_gx - radius_gx)
            gx_max = min(self.w - 1, rx_gx + radius_gx)
            gy_min = max(0, ry_gy - radius_gy)
            gy_max = min(self.h - 1, ry_gy + radius_gy)

            full_radius_sq = CIRCLE_FREE_RADIUS * CIRCLE_FREE_RADIUS
            half_radius_sq = (CIRCLE_FREE_RADIUS / 3) * (CIRCLE_FREE_RADIUS / 3)

            # 遍历圆形范围内所有栅格
            for gx in range(gx_min, gx_max + 1):
                for gy in range(gy_min, gy_max + 1):
                    # 计算距离平方
                    dx_g = gx - rx_gx
                    dy_g = gy - ry_gy
                    dist_g = (dx_g * dx_g + dy_g * dy_g) * (self.res * self.res)  # 实际距离平方

                    # 根据角度判断用哪个半径阈值
                    wx = self.ox + gx * self.res
                    wy = self.oy + gy * self.res
                    angle = math.atan2(wy - ry, wx - rx)
                    angle_relative = math.degrees(angle - self.robot_yaw)
                    while angle_relative > 180:
                        angle_relative -= 360
                    while angle_relative < -180:
                        angle_relative += 360

                    if abs(angle_relative) <= 120:
                        # 前方150度，用完整半径
                        if dist_g > full_radius_sq:
                            continue
                    else:
                        # 后方210度，只用1/3半径
                        if dist_g > half_radius_sq:
                            continue

                    # 如果不是障碍物，标记为空闲
                    if not any_obs_grid[gy, gx] and not self.scanned_grid[gy, gx]:
                        self.scanned_grid[gy, gx] = True
                        processed_count += 1

        # 起点也标记
        if 0<=rx_gx<self.w and 0<=ry_gy<self.h:
            if not self.scanned_grid[ry_gy, rx_gx]:
                self.scanned_grid[ry_gy, rx_gx] = True
                processed_count += 1

        if processed_count > 0:
            self.get_logger().info(f"[cumulative] mode={FREE_SPACE_MODE}, 障碍物: {np.sum(self.cumulative_obs_grid)}, 标记空闲: {processed_count} 格")

    def update_and_publish_map(self):
        # 先用累积点云做射线扫描，标记空闲区域和障碍物
        self.update_cumulative_free_space()

        # 合并障碍物：累积点云 + local_cloud_aligned (如果有更新)
        obs_grid = np.logical_or(self.cumulative_obs_grid, self.local_obs_grid)

        dil_obs = self.dilate_obstacle_grid(obs_grid)

        # 初始全部是未知(-1)，只有被扫描过才知道是不是空闲
        res_grid = np.full((self.h, self.w), -1, dtype=np.int8)
        # 被射线扫描过的区域标记为空闲
        res_grid[self.scanned_grid] = COST_EMPTY_SPACE
        # 地面识别结果覆盖
        res_grid[self.ground_grid == COST_GROUND_BASE] = COST_CAMERA_GROUND
        # 障碍物覆盖
        res_grid[dil_obs] = COST_OBSTACLE
        res_grid[self.bottle_surround_grid] = COST_BOTTLE_SURROUND
        res_grid[self.bottle_core_grid] = COST_BOTTLE

        # 将倾斜矩形外的所有区域设为边界值（边界与最大范围完全重合）
        BORDER_WIDTH = 4
        if self.w > 2 * BORDER_WIDTH and self.h > 2 * BORDER_WIDTH:
            if abs(self.map_theta) < 1e-6:
                # 角度为0，使用轴对齐矩形边界
                res_grid[:, :BORDER_WIDTH] = self.border_value
                res_grid[:, -BORDER_WIDTH:] = self.border_value
                res_grid[:BORDER_WIDTH, :] = self.border_value
                res_grid[-BORDER_WIDTH:, :] = self.border_value
            else:
                # 计算矩形中心点（世界坐标系）
                cx = (self.map_min_x + self.map_max_x) / 2.0
                cy = (self.map_min_y + self.map_max_y) / 2.0

                # 计算半宽半高
                half_w = (self.map_max_x - self.map_min_x) / 2.0
                half_h = (self.map_max_y - self.map_min_y) / 2.0

                # 预先计算反向旋转参数
                inv_cos = math.cos(-self.map_theta)
                inv_sin = math.sin(-self.map_theta)

                # 使用numpy向量化计算，加速判断
                gx_coords = np.arange(self.w)
                gy_coords = np.arange(self.h)
                gx_grid, gy_grid = np.meshgrid(gx_coords, gy_coords)

                # 栅格中心的世界坐标
                wx = self.ox + (gx_grid + 0.5) * self.res
                wy = self.oy + (gy_grid + 0.5) * self.res

                # 反向旋转：将点旋转回轴对齐坐标系
                dx = wx - cx
                dy = wy - cy
                rx = dx * inv_cos - dy * inv_sin
                ry = dx * inv_sin + dy * inv_cos

                # 判断是否在轴对齐矩形外部
                outside_mask = (rx < -half_w) | (rx > half_w) | (ry < -half_h) | (ry > half_h)
                res_grid[outside_mask] = self.border_value

        self.grid_map.data = res_grid.flatten().tolist()
        self.grid_map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.grid_map)

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham直线算法，生成直线上所有栅格点"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def bottle_cb(self, msg):
        coords = msg.data
        for i in range(0, len(coords), 2):
            gx, gy = int(coords[i]), int(coords[i+1])
            if 0<=gx<self.w and 0<=gy<self.h and self.calculate_grid_distance(gx,gy) <= BOTTLE_MAX_DISTANCE:
                bk = (gx//BOTTLE_GRID_BLOCK_SIZE, gy//BOTTLE_GRID_BLOCK_SIZE)
                if bk not in self.bottle_blocks:
                    self.permanent_bottles.append((gx,gy))
                    self.bottle_blocks.add(bk)
        self._update_bottle_grids()

    def _update_bottle_grids(self):
        self.bottle_core_grid.fill(False)
        for (gx, gy) in self.permanent_bottles: self.bottle_core_grid[gy, gx] = True
        if np.sum(self.bottle_core_grid) > 0:
            sur = cv2.dilate((self.bottle_core_grid*255).astype(np.uint8), self.bottle_surround_kernel, iterations=1) > 0
            self.bottle_surround_grid = np.logical_and(sur, ~self.bottle_core_grid)

    def calculate_grid_distance(self, gx, gy):
        return math.hypot(self.ox+gx*self.res - self.robot_x, self.oy+gy*self.res - self.robot_y)

    def img_cb(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 计算内参只算一次
            if self.cx is None and self.img is not None:
                h, w = self.img.shape[:2]
                self.fx = (w/2) / math.tan(np.deg2rad(CAM_FOV_H)/2)
                self.fy = (h/2) / math.tan(np.deg2rad(CAM_FOV_V)/2)
                self.cx = w / 2
                self.cy = h / 2
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def imu_cb(self, msg: Imu):
        """保存最新IMU数据"""
        self.latest_imu = msg

    def normalize_img(self, img: np.ndarray):
        """BGR uint8 -> normalized tensor 1x3xHxW"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - IMG_MEAN) / IMG_STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def process_ground(self):
        if self.img is None or not self.has_robot_pose or self.cx is None or self.latest_imu is None:
            return

        # 只在机器人几乎静止时处理地面识别（通过IMU角速度判断）
        wx = self.latest_imu.angular_velocity.x
        wy = self.latest_imu.angular_velocity.y
        wz = self.latest_imu.angular_velocity.z
        angular_vel_norm = math.sqrt(wx*wx + wy*wy + wz*wz)
        if angular_vel_norm > self.ANGULAR_VEL_THRESHOLD:
            # 机器人在转动，跳过本次处理
            return

        t0 = time.perf_counter()
        h, w = self.img.shape[:2]

        # 预处理
        img_tensor = self.normalize_img(self.img).to(self.device)
        img_tensor, (ph, pw) = self.pad_to_multiple(img_tensor, 16)

        # 推理
        with torch.no_grad():
            logits = self.ground_model(img_tensor)
            prob = torch.sigmoid(logits)
            prob = prob[0, 0, :h-ph, :w-pw].cpu().numpy()

        mask = (prob > GROUND_THRESHOLD).astype(np.uint8) * 255

        # 发布检测结果图像
        self.publish_mask(mask)
        self.publish_vis(self.img, prob, mask)

        # 投影到世界坐标网格
        self.project_ground_to_world(mask)

        elapsed = (time.perf_counter() - t0) * 1000
        self.get_logger().debug(f"地面推理完成 {w}x{h} in {elapsed:.1f} ms")

    def pad_to_multiple(self, img, multiple=16):
        h, w = img.shape[2:]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        return F.pad(img, (0, pw, 0, ph)), (ph, pw)

    def publish_mask(self, mask: np.ndarray):
        """发布分割mask图像 (mono8)"""
        msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.ground_mask_pub.publish(msg)

    def publish_vis(self, img: np.ndarray, prob: np.ndarray, mask: np.ndarray):
        """发布可视化结果图像 (原图叠加绿色半透明mask)"""
        vis = img.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 127] = [0, 200, 0]  # BGR 绿色，检测到的路面标记为绿色
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.ground_vis_pub.publish(msg)

    def project_ground_to_world(self, mask: np.ndarray):
        """扇形射线扫描填充法：
        1. 将所有地面像素投影到世界坐标，按角度分组
        2. 从机器人原点沿每个角度发射射线，碰到障碍物就停止
        3. 射线从原点到障碍物之间所有栅格都标记为地面
        4. 最终整个可见扇形区域全部填充为地面，自然连通，自动截断障碍物
        5. 只处理图像下方 1/3 区域（近处地面）
        """
        if self.cx is None:
            return

        mh, mw = mask.shape
        cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)

        # 获取机器人当前位置
        rx = self.robot_x
        ry = self.robot_y

        # 只处理图像下方 1/3 区域（近处地面），上方远景/天空不处理
        start_row = mh * 2 // 3

        # 第一步：投影所有地面像素，按角度分组，每个角度记录最大距离
        angle_bins = {}  # key: 离散角度, value: 最大距离(米)
        ground_points = np.where(mask > 127)

        for v, u in zip(*ground_points):
            if v < start_row:  # 跳过上方区域
                continue
            xc = (u - self.cx) / self.fx
            yc = (v - self.cy) / self.fy
            xr, yr = xc, yc*cp + sp
            zr = -yc*sp + cp
            if abs(zr) < 1e-6:
                continue
            t = -CAM_HEIGHT / zr  # 正确投影公式
            xb, yb = t*xr, t*yr  # 机器人坐标系坐标
            if xb <= 0:  # 过滤相机后方
                continue

            # 计算世界坐标和角度
            c, s = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
            wx = rx + xb*c - yb*s
            wy = ry + xb*s + yb*c
            dist = math.hypot(wx - rx, wy - ry)
            angle = math.atan2(wy - ry, wx - rx)

            # 只保留前方 240 度视角 (robot_yaw 左右各 120 度)
            angle_relative = math.degrees(angle - self.robot_yaw)
            while angle_relative > 180:
                angle_relative -= 360
            while angle_relative < -180:
                angle_relative += 360
            if abs(angle_relative) > 120:
                continue  # 超出240度前方视角，跳过

            # 离散角度（精度0.5度，足够密保证全覆盖）
            angle_bin = int(round(math.degrees(angle) * 2))  # *2 → 每 0.5 度一个bin
            if angle_bin not in angle_bins or dist > angle_bins[angle_bin]:
                angle_bins[angle_bin] = dist

        # 第二步：沿每个角度发射射线，从近到远走，碰到障碍物停止，之前全部标记为地面
        for angle_bin, max_dist in angle_bins.items():
            angle_rad = math.radians(angle_bin / 2.0)
            dx_step = math.cos(angle_rad) * self.res  # 每一步x增量（米）
            dy_step = math.sin(angle_rad) * self.res  # 每一步y增量（米）

            current_dist = 0.0
            step = 0
            while current_dist < max_dist:
                wx = rx + dx_step * step
                wy = ry + dy_step * step
                gx = int((wx - self.ox)/self.res)
                gy = int((wy - self.oy)/self.res)

                if not (0 <= gx < self.w and 0 <= gy < self.h):
                    break  # 超出地图，停止

                # 碰到障碍物，停止
                if self.local_obs_grid[gy, gx] or self.cumulative_obs_grid[gy, gx]:
                    break

                # 标记为已扫描，并且标记为地面
                self.scanned_grid[gy, gx] = True
                self.ground_grid[gy, gx] = COST_GROUND_BASE

                step += 1
                current_dist += self.res

def main():
    rclpy.init()
    node = GridMapGenerator()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
