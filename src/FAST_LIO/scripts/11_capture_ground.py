#!/usr/bin/env python3
"""
11_capture_ground.py — RTSP 流采集 + 地面图像保存一体化
功能:
  - 从 RTSP 拉流并发布到 /rtsp_image
  - 自动每 5 秒保存一张图像到数据集目录
  - 无需手动输入参数，直接运行
"""

import rclpy
import os
import cv2
import math
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime


class RTSPStreamPublisher(Node):
    def __init__(self, rtsp_url, width=1280, height=720, topic_name="/rtsp_image"):
        super().__init__('rtsp_stream_publisher')
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = None
        self.use_gstreamer = self._check_gstreamer_support()
        self.reconnect_interval = 2
        self.frame_count = 0

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, topic_name, 10)
        self.timer = self.create_timer(0.03, self.timer_callback)

        self.get_logger().info(f"RTSP流发布节点已启动")
        self.get_logger().info(f"RTSP地址: {self.rtsp_url}")
        self.get_logger().info(f"发布话题: {topic_name}")
        self.get_logger().info(f"使用模式: {'GStreamer（硬件解码）' if self.use_gstreamer else 'OpenCV原生'}")

        self._init_capture()

    def _check_gstreamer_support(self):
        try:
            if cv2.CAP_GSTREAMER:
                test_cap = cv2.VideoCapture("videotestsrc num-buffers=1 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
                if test_cap.isOpened():
                    test_cap.release()
                    return True
        except Exception as e:
            self.get_logger().warning(f"GStreamer不可用: {e}，切换到OpenCV原生模式")
        return False

    def _init_gstreamer_capture(self):
        pipeline = (
            f'rtspsrc location={self.rtsp_url} latency=200 tcp-timeout=5000000 ! '
            'rtph264depay ! h264parse ! nvv4l2decoder ! '
            'nvvidconv ! video/x-raw,format=BGRx ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink sync=false drop=true max-buffers=1 emit-signals=true'
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _init_opencv_capture(self):
        tcp_rtsp_url = f"{self.rtsp_url}?tcp"
        cap = cv2.VideoCapture(tcp_rtsp_url)

        if not cap.isOpened():
            cap = cv2.VideoCapture(self.rtsp_url)

        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION') and hasattr(cv2, 'VIDEO_ACCELERATION_NONE'):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        return cap

    def _init_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

        if self.use_gstreamer:
            self.cap = self._init_gstreamer_capture()
        else:
            self.cap = self._init_opencv_capture()

        if not self.cap or not self.cap.isOpened():
            self.get_logger().error(f"无法打开RTSP流，{self.reconnect_interval}秒后重试")
            self.create_timer(self.reconnect_interval, self._init_capture)
            return

    def _read_latest_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None

        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.retrieve()
        if ret:
            self.frame_count += 1
            if frame is not None and (frame.shape[0] != self.height or frame.shape[1] != self.width):
                frame = cv2.resize(frame, (self.width, self.height))
        else:
            self.frame_count = 0
            self.get_logger().warning("帧读取失败，尝试重新初始化捕获")
            self._init_capture()
        return ret, frame

    def timer_callback(self):
        ret, frame = self._read_latest_frame()
        if ret and frame is not None:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_init"
                self.image_pub.publish(ros_image)

                if self.frame_count % 100 == 0:
                    self.get_logger().info(f"已发布 {self.frame_count} 帧图像")
            except CvBridgeError as e:
                self.get_logger().error(f"图像格式转换失败: {e}")

    def cleanup(self):
        self.get_logger().info("正在关闭RTSP流发布节点...")
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass


class GroundDataCapturer(Node):
    def __init__(self, output_dir: str, auto_interval: float = 5.0):
        super().__init__("ground_data_capturer")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.current_img = None
        self.last_save_time = 0.0
        self.auto_interval = auto_interval
        self.latest_imu = None
        self.ANGULAR_VEL_THRESHOLD = 0.1  # 角速度阈值，小于此值才保存

        self.create_subscription(Image, "/rtsp_image", self.img_cb, 10)
        self.create_subscription(Imu, "/livox/imu", self.imu_cb, 10)

        if auto_interval > 0:
            self.create_timer(0.1, self.auto_save_check)

        self.get_logger().info(
            f"✅ 数据采集启动，输出目录: {os.path.abspath(output_dir)}\n"
            f"自动保存间隔: {auto_interval}s/张\n"
            f"静止检测: IMU角速度 < {self.ANGULAR_VEL_THRESHOLD} rad/s 才保存"
        )

    def img_cb(self, msg):
        try:
            self.current_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"转换图像失败: {e}")

    def imu_cb(self, msg: Imu):
        """保存最新IMU数据"""
        self.latest_imu = msg

    def save_current(self):
        if self.current_img is None:
            self.get_logger().warn("还没有收到图像！")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = os.path.join(self.output_dir, f"img_{ts}.jpg")
        cv2.imwrite(img_path, self.current_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.get_logger().info(f"✅ 已保存: {img_path}")
        self.last_save_time = self.get_clock().now().nanoseconds / 1e9

    def auto_save_check(self):
        if self.auto_interval <= 0:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        # 只在IMU检测到静止时保存
        if self.latest_imu is None:
            return
        wx = self.latest_imu.angular_velocity.x
        wy = self.latest_imu.angular_velocity.y
        wz = self.latest_imu.angular_velocity.z
        angular_vel_norm = math.sqrt(wx*wx + wy*wy + wz*wz)
        if angular_vel_norm > self.ANGULAR_VEL_THRESHOLD:
            # 机器人在移动/转动，不保存
            return
        if now - self.last_save_time >= self.auto_interval and self.current_img is not None:
            self.save_current()


def main():
    rclpy.init()

    # === 配置参数（修改这里适配你的 RTSP）===
    RTSP_URL = "rtsp://192.168.234.1:8554/test"
    STREAM_WIDTH = 1280
    STREAM_HEIGHT = 720
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_data", "images")
    AUTO_SAVE_INTERVAL = 5.0  # 默认每 5 秒保存一张

    # 创建节点
    rtsp_node = RTSPStreamPublisher(RTSP_URL, STREAM_WIDTH, STREAM_HEIGHT, "/rtsp_image")
    capture_node = GroundDataCapturer(OUTPUT_DIR, AUTO_SAVE_INTERVAL)

    # 多线程执行
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(rtsp_node)
    executor.add_node(capture_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rtsp_node.cleanup()
        capture_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
