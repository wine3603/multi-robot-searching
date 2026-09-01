#!/usr/bin/env python3
"""
14_infer_ground.py — 地面分割推理 ROS2 节点
功能:
  - 订阅 /rtsp_image 摄像头图像
  - 用训练好的UNet推理出地面分割掩码
  - 将地面投影到世界坐标网格（和grid_map_generator一样）
  - 发布:
    /ground_mask_img   (Image)     - 可视化分割结果
    /ground_projection - 可以集成到grid_map_generator

用法:
  python3 14_infer_ground.py --model checkpoints/ground_best.pth

配置参数和投影逻辑和grid_map_generator.py保持一致
"""

import argparse
import time
import math
import os
import glob
import threading
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from visualization_msgs.msg import Marker


# ============================================================================
# 模型定义 (和训练一致)
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


# ============================================================================
# 配置参数 (和grid_map_generator.py一致)
# ============================================================================

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOPIC_NAME = "/rtsp_image"
CAM_HEIGHT = 0.35  # 相机离地面高度（米）
CAM_PITCH = np.deg2rad(-20)  # 俯仰角：负=向上仰，正=向下看
# 智元 D1 Ultra 前向广角 FPV 相机
CAM_FOV_H = 111.0  # 水平 FOV: 111°
CAM_FOV_V = 70.0   # 垂直 FOV: 70°
COST_GROUND_BASE = 50  # 和grid_map_generator一致
MAX_GROUND_DISTANCE = 5.0  # 只识别5米半径内的地面


# ============================================================================
# 推理节点
# ============================================================================

class GroundInferNode(Node):
    def __init__(self, model_path: str, base_ch: int, device_str: str,
                 throttle_hz: float=5.0, threshold: float=0.5,
                 map_res: float=0.05, map_w: int=2000, map_h: int=2000,
                 map_ox: float=-50.0, map_oy: float=-50.0, map_frame: str="camera_init",
                 robot_frame: str="body", debug_no_infer: bool=False):
        super().__init__("ground_infer_node")
        self.debug_no_infer = debug_no_infer
        self.device = torch.device(device_str)
        self.threshold = threshold
        self.map_res = map_res
        self.map_w = map_w
        self.map_h = map_h
        self.map_ox = map_ox
        self.map_oy = map_oy
        self.pose_delay = 1.5  # 图像和位姿的延迟匹配：图片对应0.5s后的位姿

        # 创建缓存文件夹（启动时清空）
        self.cache_dir = "ground_infer_cache"
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 创建原始数据保存目录
        self.img_raw_dir = os.path.join(self.cache_dir, "images_raw")
        self.img_done_dir = os.path.join(self.cache_dir, "images_done")
        self.odom_raw_dir = os.path.join(self.cache_dir, "odom_raw")
        os.makedirs(self.img_raw_dir, exist_ok=True)
        os.makedirs(self.img_done_dir, exist_ok=True)
        os.makedirs(self.odom_raw_dir, exist_ok=True)

        # Odometry日志文件（追加写入）
        self.odom_log_file = os.path.join(self.cache_dir, "odom_log.txt")
        self.get_logger().info(f"📁 待处理图像目录: {self.img_raw_dir}")
        self.get_logger().info(f"📁 已处理图像目录: {self.img_done_dir}")
        self.get_logger().info(f"📁 Odometry日志: {self.odom_log_file}")

        # 加载模型（调试模式不加载）
        if not self.debug_no_infer:
            self.get_logger().info(f"加载模型: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model = UNet(c_in=3, num_classes=1, base=base_ch).to(self.device)
            self.model.load_state_dict(ckpt)
            self.model.eval()
            self.get_logger().info(f"✅ 模型加载完成 base_ch={base_ch}")
        else:
            self.get_logger().info("🐛 调试模式：不加载模型，直接投影1/3区域")

        # Odometry 位姿队列 (timestamp, x, y, yaw)
        self.odom_queue = deque()
        self.has_robot_pose = False

        # 图像保存队列（后台线程异步写盘，不阻塞ROS回调）
        self.save_queue = deque()
        self.save_thread = threading.Thread(target=self.save_worker, daemon=True)
        self.save_thread.start()

        # Odometry保存队列（同样异步写盘）
        self.odom_save_queue = deque()
        self.odom_save_thread = threading.Thread(target=self.odom_save_worker, daemon=True)
        self.odom_save_thread.start()

        # 相机内参
        self.cam_mat = None
        self.cx = self.cy = self.fx = self.fy = None

        # 地面网格 (和grid_map_generator相同尺寸，单独累积)
        self.ground_grid = np.zeros((self.map_h, self.map_w), dtype=np.int8)

        # RGB颜色累积网格
        self.rgb_grid = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        self.rgb_count = np.zeros((self.map_h, self.map_w), dtype=np.uint16)

        # ROS
        self.bridge = CvBridge()
        self.current_img = None

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)

        self.img_sub = self.create_subscription(Image, TOPIC_NAME, self.img_cb, qos)
        self.odom_sub = self.create_subscription(Odometry, "/Odometry", self.odom_cb, qos)
        self.mask_pub = self.create_publisher(Image, "/ground_mask_img", 10)
        self.vis_pub = self.create_publisher(Image, "/ground_vis_img", 10)
        self.marker_pub = self.create_publisher(Marker, "/ground_rgb_marker", 10)

        self.frame_count = 0
        self.last_img_time = time.time()  # 最后一张图像到达时间
        self.collection_done = False      # 采集是否完成

        # ROS运行期间：只采集，不推理
        # 5秒没新图像自动停止采集并开始推理
        self.running = True

        # 定时检查是否还有新图像（1Hz检查）
        self.create_timer(1.0, self.check_collection_done)

        # 定时处理延迟匹配（100Hz检查）
        self.create_timer(0.01, self.match_pose_delay)

        self._last_ts = 0.0
        self.get_logger().info(
            f"🎯 地面推理节点启动 [调试模式]\n"
            f"   订阅图像: {TOPIC_NAME}\n"
            f"   订阅位姿: /Odometry\n"
            f"   发布: /ground_mask_img, /ground_vis_img\n"
            f"   发布: /ground_rgb_marker (带颜色的地面Marker)\n"
            f"   模式: 完整图像推理，只投影中间1/3宽度+下面1/3高度\n"
            f"   推理速度: 全速处理，无帧率限制\n"
            f"   位姿延迟: {self.pose_delay}s\n"
            f"   FOV: 水平{CAM_FOV_H}°, 垂直{CAM_FOV_V}°\n"
            f"   机制: 图像/位姿分开保存，动态延迟匹配，处理完不删除"
        )

    def odom_cb(self, msg):
        """接收Odometry位姿，放入保存队列，不阻塞ROS回调"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

        # 存入内存队列，用于快速匹配
        self.odom_queue.append((timestamp, x, y, yaw))
        self.has_robot_pose = True
        # 只保留最近5分钟的位姿在内存
        while len(self.odom_queue) > 0 and timestamp - self.odom_queue[0][0] > 300:
            self.odom_queue.popleft()

        # 放入保存队列，后台线程异步写盘
        self.odom_save_queue.append((timestamp, x, y, yaw))

    def odom_save_worker(self):
        """后台线程：异步批量保存Odometry到磁盘，减少IO开销"""
        buffer = []
        last_flush = time.time()
        while True:
            # 收集数据到缓存
            while len(self.odom_save_queue) > 0:
                buffer.append(self.odom_save_queue.popleft())

            # 条件：缓存满100条或超过0.5秒，一次性写入
            now = time.time()
            if len(buffer) >= 100 or (len(buffer) > 0 and now - last_flush > 0.5):
                try:
                    with open(self.odom_log_file, 'a') as f:
                        lines = []
                        for timestamp, x, y, yaw in buffer:
                            lines.append(f"{timestamp:.6f} {x:.6f} {y:.6f} {yaw:.6f}\n")
                        f.writelines(lines)
                    buffer.clear()
                    last_flush = now
                except Exception as e:
                    self.get_logger().error(f"Odom保存失败: {e}")

            time.sleep(0.001)

    def img_cb(self, msg):
        """收到图像放入保存队列，不阻塞ROS回调"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 计算内参只算一次
            if self.cam_mat is None and img is not None:
                h, w = img.shape[:2]
                self.fx = (w/2) / math.tan(np.deg2rad(CAM_FOV_H)/2)
                self.fy = (h/2) / math.tan(np.deg2rad(CAM_FOV_V)/2)
                self.cx = w / 2
                self.cy = h / 2
                self.get_logger().info(f"✅ 收到图像 {w}x{h}，相机内参已计算")

            # 放入保存队列，后台线程异步写盘
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.save_queue.append((timestamp, img.copy()))  # copy避免ROS复用内存
            self.last_img_time = time.time()  # 更新最后图像时间
        except Exception as e:
            self.get_logger().error(f"图像入队失败: {e}")

    def check_collection_done(self):
        """检查是否5秒没有新图像，自动开始推理"""
        if self.cx is not None and not self.collection_done:
            elapsed = time.time() - self.last_img_time
            if elapsed > 5.0:
                self.get_logger().info(f"\n⏹  5秒没有新图像，采集结束，等待IO队列清空...")
                # 等待所有保存队列清空
                flush_start = time.time()
                while len(self.save_queue) > 0 or len(self.odom_save_queue) > 0:
                    time.sleep(0.01)
                    if time.time() - flush_start > 5.0:
                        self.get_logger().warn(f"⏰ IO超时，强制开始推理 (剩余: img={len(self.save_queue)}, odom={len(self.odom_save_queue)})")
                        break
                self.get_logger().info(f"✅ IO完成，开始批量推理...")
                self.collection_done = True
            elif elapsed > 3.0:
                self.get_logger().info(f"⏳ 无新图像 {elapsed:.1f}s，检测到静止...", throttle_duration_sec=0.5)

    def save_worker(self):
        """后台线程：异步保存图像到磁盘，不阻塞ROS回调"""
        while True:
            if len(self.save_queue) > 0:
                timestamp, img = self.save_queue.popleft()
                try:
                    img_file = os.path.join(self.img_raw_dir, f"{timestamp:.6f}.jpg")
                    cv2.imwrite(img_file, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.frame_count += 1
                    self.get_logger().debug(f"📸 保存图像 {self.frame_count}, 队列积压: {len(self.save_queue)}",
                                           throttle_duration_sec=1.0)
                except Exception as e:
                    self.get_logger().error(f"图像保存失败: {e}")
            else:
                time.sleep(0.001)  # 队列为空时短暂休眠

    def find_pose_for_image(self, img_timestamp: float):
        """根据图像时间戳，找到延迟pose_delay后的位姿"""
        target_time = img_timestamp + self.pose_delay

        # 先从内存队列找最快
        min_diff = float('inf')
        best_pose = None
        for odom_time, x, y, yaw in self.odom_queue:
            diff = abs(odom_time - target_time)
            if diff < min_diff:
                min_diff = diff
                best_pose = (x, y, yaw)

        # 如果内存里找到了且时间差小于1秒
        if best_pose is not None and min_diff < 2.0:
            return best_pose

        # 调试：打印时间差
        self.get_logger().warn(
            f"⏰ 时间差太大: img={img_timestamp:.3f}, target={target_time:.3f}, "
            f"min_diff={min_diff:.3f}s, queue_size={len(self.odom_queue)}",
            throttle_duration_sec=1.0
        )

        # 内存没找到，从日志文件找（处理积压很久的情况）
        try:
            with open(self.odom_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        odom_time = float(parts[0])
                        diff = abs(odom_time - target_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_pose = (float(parts[1]), float(parts[2]), float(parts[3]))
            if best_pose is not None and min_diff < 1.0:
                return best_pose
        except:
            pass

        return None

    def match_pose_delay(self):
        """这个函数现在不需要了，保留空实现兼容定时器"""
        pass

    def normalize_img(self, img: np.ndarray):
        """BGR uint8 -> normalized tensor 1x3xHxW"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - IMG_MEAN) / IMG_STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def pad_to_multiple(self, img, multiple=16):
        _, _, h, w = img.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        return F.pad(img, (0, pw, 0, ph)), (ph, pw)

    def infer_loop(self):
        """后台推理循环 - 先等待pose_delay秒确保位姿就绪，再开始推理"""
        self.get_logger().info(f"⏳ 数据采集中... {self.pose_delay}秒后开始推理")
        start_time = time.time()

        while self.running:
            # 先等待pose_delay时间，确保数据开始进入后再推理
            if time.time() - start_time < self.pose_delay:
                time.sleep(0.1)
                continue

            if self.cx is None:
                time.sleep(0.001)
                continue

            # 获取所有缓存文件，按时间排序
            img_files = sorted(glob.glob(os.path.join(self.img_raw_dir, "*.jpg")))
            if len(img_files) == 0:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()

            try:
                # 读取最早的图像
                img_file = img_files[0]
                filename = os.path.basename(img_file)
                img_timestamp = float(os.path.splitext(filename)[0])

                # 检查是否等够了延迟时间
                now = time.time()
                if now - img_timestamp < self.pose_delay:
                    time.sleep(0.001)
                    continue

                # 匹配位姿
                pose = self.find_pose_for_image(img_timestamp)
                if pose is None:
                    time.sleep(0.001)
                    continue

                robot_x, robot_y, robot_yaw = pose

                # 读取图像
                img = cv2.imread(img_file)
                if img is None:
                    self.get_logger().warn(f"⚠️ 图像损坏: {filename}", throttle_duration_sec=1.0)
                    os.rename(img_file, os.path.join(self.img_done_dir, filename))
                    continue

                h, w = img.shape[:2]

                # 完整图像推理 - 和训练时一致，加入地面先验偏置
                img_tensor = self.normalize_img(img).to(self.device)
                with torch.no_grad():
                    logits = self.model(img_tensor)
                    # 添加地面先验 - 和训练时保持一致：只在底部1/3区域加偏置
                    B, C, H, W = logits.shape
                    pb = torch.zeros_like(logits)
                    pb[:, :, int(H*2/3):, :] += 2.0  # prior_bias：仅底部1/3区域
                    logits = logits + pb
                    prob_full = torch.sigmoid(logits)[0, 0].cpu().numpy()

                # 只取中间1/3宽度 + 下面1/3高度的区域用于投影
                v_start = int(h * 2 / 3)  # 下面1/3开始
                u_start = int(w * 1 / 3)  # 中间1/3开始
                u_end = int(w * 2 / 3)    # 中间1/3结束

                # 创建mask：只保留感兴趣区域推理成功的部分
                mask = np.zeros((h, w), dtype=np.uint8)
                prob_masked = np.zeros((h, w), dtype=np.float32)
                mask[v_start:, u_start:u_end] = ((prob_full[v_start:, u_start:u_end] > 0.5) * 255).astype(np.uint8)
                prob_masked[v_start:, u_start:u_end] = prob_full[v_start:, u_start:u_end]

                # 发布分割结果图像（只显示感兴趣区域）
                self.publish_mask(mask)
                vis_img = self.publish_vis(img, prob_masked, mask)

                # 投影到世界坐标网格
                self.project_to_world_with_pose(mask, img, robot_x, robot_y, robot_yaw)

                # 保存分割结果图到images_done
                result_file = os.path.join(self.img_done_dir, filename)
                cv2.imwrite(result_file, vis_img)
                # 删除原始jpg
                os.remove(img_file)

                elapsed = (time.perf_counter() - t0) * 1000
                pending = len(img_files) - 1
                self.get_logger().info(f"✅ 推理完成 {w}x{h} in {elapsed:.0f} ms, 待处理: {pending}",
                                       throttle_duration_sec=1.0)
            except Exception as e:
                self.get_logger().error(f"推理失败: {e}")
                try:
                    os.remove(img_file)
                except:
                    pass

    def publish_mask(self, mask: np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.mask_pub.publish(msg)

    def publish_vis(self, img: np.ndarray, prob: np.ndarray, mask: np.ndarray):
        """叠加绿色半透明到原图"""
        vis = img.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 127] = [0, 200, 0]  # BGR 绿色
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # 概率图
        prob_vis = (prob * 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        # 放在底部
        h1, w1 = vis.shape[:2]
        h2 = int(h1 * 0.2)
        prob_color = cv2.resize(prob_color, (w1, h2))
        vis[-h2:, :] = cv2.addWeighted(vis[-h2:, :], 0.5, prob_color, 0.5, 0)

        msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.vis_pub.publish(msg)
        return vis

    def project_to_world_with_pose(self, mask: np.ndarray, img: np.ndarray,
                                    robot_x: float, robot_y: float, robot_yaw: float):
        """将图像中的地面像素投影到世界坐标，累积到ground_grid
        使用numpy矢量化加速，比for循环快100倍以上
        """
        if self.cx is None:
            return

        cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)

        # 只提取地面像素，矢量化计算
        ground_pixels = np.where(mask > 127)
        v = ground_pixels[0].astype(np.float32)
        u = ground_pixels[1].astype(np.float32)

        # 相机投影（批量计算）
        xc = (u - self.cx) / self.fx
        yc = (v - self.cy) / self.fy

        # 俯仰角旋转（批量）
        y_rot = yc * cp + sp
        z_rot = -yc * sp + cp

        # 计算深度
        t = CAM_HEIGHT / y_rot

        # 一次性过滤所有无效点
        valid = (z_rot > 0) & (y_rot > 1e-6) & (t > 0) & (t <= MAX_GROUND_DISTANCE)

        # 获取有效点
        t_valid = t[valid]
        xc_valid = xc[valid]
        z_rot_valid = z_rot[valid]
        v_valid = v[valid].astype(np.int32)
        u_valid = u[valid].astype(np.int32)

        # 相机坐标系到机器人坐标系
        xb = t_valid * z_rot_valid  # 向前
        yb = -t_valid * xc_valid    # 向左

        # 世界坐标旋转
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        world_x = robot_x + xb*c - yb*s
        world_y = robot_y + xb*s + yb*c

        # 网格坐标
        gx = ((world_x - self.map_ox) / self.map_res).astype(np.int32)
        gy = ((world_y - self.map_oy) / self.map_res).astype(np.int32)

        # 过滤在地图范围内的点
        in_bounds = (gx >= 0) & (gx < self.map_w) & (gy >= 0) & (gy < self.map_h)
        gx = gx[in_bounds]
        gy = gy[in_bounds]
        v_valid = v_valid[in_bounds]
        u_valid = u_valid[in_bounds]

        count = len(gx)

        # 批量更新网格
        self.ground_grid[gy, gx] = COST_GROUND_BASE

        # 批量更新颜色（BGR转RGB）
        bgr = img[v_valid, u_valid]
        self.rgb_grid[gy, gx, 0] = bgr[:, 2]  # R
        self.rgb_grid[gy, gx, 1] = bgr[:, 1]  # G
        self.rgb_grid[gy, gx, 2] = bgr[:, 0]  # B
        self.rgb_count[gy, gx] = 1

        if count > 0:
            # 验证最前方的点
            forward_x = robot_x + MAX_GROUND_DISTANCE * math.cos(robot_yaw)
            forward_y = robot_y + MAX_GROUND_DISTANCE * math.sin(robot_yaw)
            self.get_logger().info(
                f"✅ 累积 {count} 个地面点 | 机器人朝向(度): {math.degrees(robot_yaw):.1f} | "
                f"前方5米: ({forward_x:.2f}, {forward_y:.2f})",
                throttle_duration_sec=2.0
            )

    def publish_accumulated_marker(self):
        """发布累积的所有地面Marker"""
        if self.rgb_count.sum() == 0:
            return

        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "camera_init"
        marker.ns = "ground_rgb"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.7

        cell_size = self.map_res * 4  # 增倍面片大小，确保连续
        half_size = cell_size / 2

        # 找出所有有地面的格子
        ground_indices = np.where(self.rgb_count > 0)

        for gy, gx in zip(ground_indices[0], ground_indices[1]):
            world_x = self.map_ox + gx * self.map_res
            world_y = self.map_oy + gy * self.map_res
            # 修复：uint8 转换为 float，确保颜色正确
            r = float(self.rgb_grid[gy, gx, 0]) / 255.0
            g = float(self.rgb_grid[gy, gx, 1]) / 255.0
            b = float(self.rgb_grid[gy, gx, 2]) / 255.0

            # 创建一个小方形面片（两个三角形组成）
            p1 = Point(); p1.x = world_x - half_size; p1.y = world_y - half_size; p1.z = 0.0
            p2 = Point(); p2.x = world_x + half_size; p2.y = world_y - half_size; p2.z = 0.0
            p3 = Point(); p3.x = world_x + half_size; p3.y = world_y + half_size; p3.z = 0.0
            p4 = Point(); p4.x = world_x - half_size; p4.y = world_y - half_size; p4.z = 0.0
            p5 = Point(); p5.x = world_x + half_size; p5.y = world_y + half_size; p5.z = 0.0
            p6 = Point(); p6.x = world_x - half_size; p6.y = world_y + half_size; p6.z = 0.0

            marker.points.extend([p1, p2, p3, p4, p5, p6])

            # 每个顶点设置相同颜色
            for _ in range(6):
                c = ColorRGBA()
                c.r = r; c.g = g; c.b = b; c.a = 0.7
                marker.colors.append(c)

        self.marker_pub.publish(marker)
        total_cells = len(ground_indices[0])
        self.get_logger().info(f"🗺️ 发布累积地面Marker: {total_cells} 个格子", throttle_duration_sec=2.0)

    def get_ground_grid(self):
        """可供grid_map_generator读取"""
        return self.ground_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_ground/ground_best.pth", help="模型路径")
    ap.add_argument("--base_ch", type=int, default=32, help="UNet基础通道数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--throttle_hz", type=float, default=5.0, help="推理频率上限")
    ap.add_argument("--threshold", type=float, default=0.5, help="分割阈值")
    # 地图参数和grid_map_generator保持一致
    ap.add_argument("--map_resolution", type=float, default=0.05)
    ap.add_argument("--map_width", type=int, default=2000)
    ap.add_argument("--map_height", type=int, default=2000)
    ap.add_argument("--map_origin_x", type=float, default=-50.0)
    ap.add_argument("--map_origin_y", type=float, default=-50.0)
    ap.add_argument("--map_frame", default="camera_init")
    ap.add_argument("--robot_frame", default="body")
    ap.add_argument("--debug_no_infer", action="store_true", help="调试模式：不推理，直接把1/3区域当作地面投影")

    args, _ = ap.parse_known_args()

    rclpy.init()
    node = GroundInferNode(
        model_path=args.model,
        base_ch=args.base_ch,
        device_str=args.device,
        throttle_hz=args.throttle_hz,
        threshold=args.threshold,
        map_res=args.map_resolution,
        map_w=args.map_width,
        map_h=args.map_height,
        map_ox=args.map_origin_x,
        map_oy=args.map_origin_y,
        map_frame=args.map_frame,
        robot_frame=args.robot_frame,
        debug_no_infer=args.debug_no_infer
    )

    try:
        node.get_logger().info("📸 数据采集中... 5秒无新图像自动开始推理，或按Ctrl+C手动停止")
        while rclpy.ok() and not node.collection_done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("\n⏹  手动停止，等待IO队列清空...")
        # 等待所有保存队列清空
        flush_start = time.time()
        while len(node.save_queue) > 0 or len(node.odom_save_queue) > 0:
            time.sleep(0.01)
            if time.time() - flush_start > 5.0:
                node.get_logger().warn(f"⏰ IO超时，强制开始推理")
                break
        node.get_logger().info(f"✅ IO完成，开始批量推理...")

    # ROS保持运行，批量处理所有图片（处理完后发布Marker）
    print(f"\n🚀 开始批量推理，待处理: {len(glob.glob(os.path.join(node.img_raw_dir, '*.jpg')))} 张")

    # 批量处理循环
    processed = 0
    while True:
        img_files = sorted(glob.glob(os.path.join(node.img_raw_dir, "*.jpg")))
        if len(img_files) == 0:
            break

        img_file = img_files[0]
        filename = os.path.basename(img_file)
        img_timestamp = float(os.path.splitext(filename)[0])

        # 匹配位姿（从日志文件找）
        pose = node.find_pose_for_image(img_timestamp)
        if pose is None:
            print(f"⚠️  找不到位姿，跳过: {filename}")
            os.remove(img_file)
            continue

        robot_x, robot_y, robot_yaw = pose

        # 读取图像
        img = cv2.imread(img_file)
        if img is None:
            print(f"⚠️  图像损坏: {filename}")
            os.remove(img_file)
            continue

        h, w = img.shape[:2]

        # 只取中间1/3宽度 + 下面1/3高度区域
        v_start = int(h * 2 / 3)
        u_start = int(w * 1 / 3)
        u_end = int(w * 2 / 3)

        mask = np.zeros((h, w), dtype=np.uint8)
        prob_masked = np.zeros((h, w), dtype=np.float32)

        if node.debug_no_infer:
            # 调试模式：直接把1/3区域全部当作地面
            mask[v_start:, u_start:u_end] = 255
            prob_masked[v_start:, u_start:u_end] = 1.0
        else:
            # 正常推理模式
            img_tensor = node.normalize_img(img).to(node.device)
            with torch.no_grad():
                logits = node.model(img_tensor)
                # 添加地面先验
                B, C, H, W = logits.shape
                pb = torch.zeros_like(logits)
                pb[:, :, int(H*2/3):, :] += 2.0
                logits = logits + pb
                prob_full = torch.sigmoid(logits)[0, 0].cpu().numpy()

            mask[v_start:, u_start:u_end] = ((prob_full[v_start:, u_start:u_end] > 0.5) * 255).astype(np.uint8)
            prob_masked[v_start:, u_start:u_end] = prob_full[v_start:, u_start:u_end]

        # 可视化叠加
        vis = img.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 127] = [0, 200, 0]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # 概率图
        prob_vis = (prob_masked * 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        h1, w1 = vis.shape[:2]
        h2 = int(h1 * 0.2)
        prob_color = cv2.resize(prob_color, (w1, h2))
        vis[-h2:, :] = cv2.addWeighted(vis[-h2:, :], 0.5, prob_color, 0.5, 0)

        # 投影到世界坐标
        node.project_to_world_with_pose(mask, img, robot_x, robot_y, robot_yaw)

        # 保存结果
        result_file = os.path.join(node.img_done_dir, filename)
        cv2.imwrite(result_file, vis)
        os.remove(img_file)

        processed += 1
        print(f"✅ 处理进度: {processed}/{len(img_files)+processed}, 最新: {filename}")

    print(f"\n🎉 全部完成！共处理 {processed} 张图片")
    print(f"📍 地面网格大小: {np.sum(node.ground_grid > 0)} 个网格")

    # 发布最终累积的Marker
    print("\n🗺️  发布最终地面Marker到RViz...")
    node.publish_accumulated_marker()
    # 保持ROS运行一小段时间，确保消息发送
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()
    print("✅ 完成！")


if __name__ == "__main__":
    main()
