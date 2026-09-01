import cv2
import time
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class RTSPStreamReaderNode(Node):
    def __init__(self, rtsp_url, width=1280, height=720, topic_name="/rtsp_image"):
        # 初始化ROS2节点
        super().__init__('rtsp_stream_publisher')
        
        # RTSP配置
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = None
        self.use_gstreamer = self._check_gstreamer_support()
        self.reconnect_interval = 2
        self.frame_count = 0
        
        # ROS2发布器配置
        self.bridge = CvBridge()  # OpenCV和ROS2图像格式转换
        self.image_pub = self.create_publisher(Image, topic_name, 10)  # 队列大小10
        self.timer = self.create_timer(0.03, self.timer_callback)  # 约30Hz发布
        
        # 日志输出
        self.get_logger().info(f"RTSP流发布节点已启动")
        self.get_logger().info(f"RTSP地址: {self.rtsp_url}")
        self.get_logger().info(f"发布话题: {topic_name}")
        self.get_logger().info(f"使用模式: {'GStreamer（硬件解码）' if self.use_gstreamer else 'OpenCV原生'}")
        
        # 初始化视频捕获
        self._init_capture()

    def _check_gstreamer_support(self):
        """检测ARM环境下GStreamer是否可用"""
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
        """初始化GStreamer捕获（Jetson硬件解码）"""
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
        """初始化OpenCV原生捕获（兼容模式）"""
        # URL指定TCP协议，适配低版本OpenCV
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
        """初始化视频捕获对象"""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

        if self.use_gstreamer:
            self.cap = self._init_gstreamer_capture()
            # GStreamer 打开失败，尝试 fallback 到 OpenCV
            if not self.cap or not self.cap.isOpened():
                self.get_logger().warning("GStreamer 打开失败，切换到 OpenCV 原生模式")
                self.use_gstreamer = False
                self.cap = self._init_opencv_capture()
        else:
            self.cap = self._init_opencv_capture()

        if not self.cap or not self.cap.isOpened():
            self.get_logger().error(f"无法打开RTSP流，{self.reconnect_interval}秒后重试")
            self.create_timer(self.reconnect_interval, self._init_capture)
            return

    def _read_latest_frame(self):
        """读取最新帧"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        # 清空缓冲区
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
        """定时器回调，发布图像到ROS2"""
        ret, frame = self._read_latest_frame()
        if ret and frame is not None:
            try:
                # 将OpenCV图像转换为ROS2 Image消息
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                # 设置时间戳（使用当前ROS2时间）
                ros_image.header.stamp = self.get_clock().now().to_msg()
                # 设置帧ID（可根据你的FAST-LIO配置修改）
                ros_image.header.frame_id = "camera_init"  # 请匹配你的TF树中的相机帧ID
                # 发布图像
                self.image_pub.publish(ros_image)
                
                # 每100帧输出一次日志
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f"已发布 {self.frame_count} 帧图像")
                    
            except CvBridgeError as e:
                self.get_logger().error(f"图像格式转换失败: {e}")

    def cleanup(self):
        """清理资源"""
        self.get_logger().info("正在关闭RTSP流发布节点...")
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 配置参数（根据你的实际情况修改）
    RTSP_URL = "rtsp://192.168.234.1:8554/test"
    STREAM_WIDTH = 1280
    STREAM_HEIGHT = 720
    TOPIC_NAME = "/rtsp_image"  # RViz订阅的话题名
    
    # 创建节点
    node = RTSPStreamReaderNode(RTSP_URL, STREAM_WIDTH, STREAM_HEIGHT, TOPIC_NAME)
    
    try:
        # 运行节点
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("手动停止节点")
    except Exception as e:
        node.get_logger().error(f"节点运行异常: {e}")
    finally:
        node.cleanup()

if __name__ == "__main__":
    main()

