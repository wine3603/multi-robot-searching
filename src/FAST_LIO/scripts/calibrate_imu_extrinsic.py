#!/usr/bin/env python3
import rclpy
import numpy as np
import yaml
import os
import time
from rclpy.node import Node
from sensor_msgs.msg import Imu

class IMUTiltCalculator(Node):
    def __init__(self, save_path):
        super().__init__('imu_tilt_calculator')
        # 参考你的代码：初始化参数
        self.save_path = save_path  # 明确命名，避免混淆
        self.acc_buffer = []
        self.imu_msg_count = 0
        self.calib_duration = 3.0
        self.start_collect_time = None

        # 订阅MID360 IMU话题（和你的代码一致）
        self.imu_sub = self.create_subscription(
            Imu, '/livox/imu', self.imu_callback, 100)

        # 参考你的日志风格
        self.get_logger().info("="*60)
        self.get_logger().info("🔥 IMU倾斜角计算（精准写入YAML）")
        self.get_logger().info("⚠️  操作：保持雷达静止，收集3秒数据")
        self.get_logger().info("="*60)

    def imu_callback(self, msg):
        """参考你的代码：仅收集有效IMU加速度数据"""
        self.imu_msg_count += 1
        if self.start_collect_time is None:
            self.start_collect_time = time.time()

        # 只收集3秒内的数据（和你的代码一致）
        if time.time() - self.start_collect_time > self.calib_duration:
            return

        # 读取加速度（和你的代码一致）
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        
        # 参考你的过滤逻辑（0.8-1.2）
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        if 0.8 < mag < 1.2:
            self.acc_buffer.append([ax, ay, az])

    def compute_tilt(self):
        """纯计算：只返回数值，不写入，确保计算值纯净"""
        # 1. 数据校验（参考你的逻辑）
        if len(self.acc_buffer) < 10:
            self.get_logger().error("❌ 有效数据不足，使用单位矩阵基准")
            mean_acc = np.array([0.0, 0.0, -1.0])
        else:
            # 2. 平均并归一化（和你的代码完全一致）
            mean_acc = np.mean(self.acc_buffer, axis=0)
            mean_acc = mean_acc / np.linalg.norm(mean_acc)
            self.get_logger().info(f"\n📏 IMU静止加速度方向：{mean_acc.round(4)}")

        # 3. 计算倾斜角（和你的代码完全一致）
        gx, gy, gz = mean_acc
        roll = np.arctan2(gy, gz) * 180/np.pi
        pitch = -np.arctan2(gx, np.sqrt(gy**2 + gz**2)) * 180/np.pi

        # 4. 修正角（负的倾斜角）
        roll_correct = -roll
        pitch_correct = -pitch

        # 5. 强制固定精度（1位小数，和日志显示一致）
        roll = round(roll, 1)
        pitch = round(pitch, 1)
        roll_correct = round(roll_correct, 1)
        pitch_correct = round(pitch_correct, 1)

        # 6. 打印原始计算值（无任何修改）
        self.get_logger().info(f"\n📊 原始计算值（未写入）：")
        self.get_logger().info(f"   raw_roll_deg: {roll}°")
        self.get_logger().info(f"   raw_pitch_deg: {pitch}°")
        self.get_logger().info(f"   roll_correct_deg: {roll_correct}°")
        self.get_logger().info(f"   pitch_correct_deg: {pitch_correct}°")

        return {
            "raw_roll_deg": roll,
            "raw_pitch_deg": pitch,
            "roll_correct_deg": -roll_correct,
            "pitch_correct_deg": -pitch_correct,
            "raw_acc": mean_acc.round(4).tolist()
        }

    def write_tilt_to_yaml(self, tilt_data):
        """精准写入：确保写入值=计算值"""
        # 1. 写入前校验数据类型（强制浮点数）
        for key in tilt_data:
            if isinstance(tilt_data[key], float):
                # 强制保留1位小数，避免yaml自动转换
                tilt_data[key] = float("{0:.1f}".format(tilt_data[key]))
            elif isinstance(tilt_data[key], list):
                # 加速度向量保留4位小数
                tilt_data[key] = [float("{0:.4f}".format(x)) for x in tilt_data[key]]

        # 2. 写入yaml（禁用流式输出，固定格式）
        with open(self.save_path, 'w', encoding='utf-8') as f:
            # 手动构造yaml内容，避免序列化误差
            yaml_content = f"""# IMU倾斜角计算结果（计算值=写入值）
raw_roll_deg: {tilt_data['raw_roll_deg']}
raw_pitch_deg: {tilt_data['raw_pitch_deg']}
roll_correct_deg: {tilt_data['roll_correct_deg']}
pitch_correct_deg: {tilt_data['pitch_correct_deg']}
raw_acc: {tilt_data['raw_acc']}
"""
            f.write(yaml_content)

        # 3. 写入后读取验证（关键：确认写入值和计算值一致）
        self.verify_written_data(tilt_data)

    def verify_written_data(self, original_data):
        """验证写入的数值是否和计算值一致"""
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                written_data = yaml.safe_load(f)
            
            # 逐字段校验
            self.get_logger().info(f"\n✅ 写入后校验结果：")
            for key in ['raw_roll_deg', 'raw_pitch_deg', 'roll_correct_deg', 'pitch_correct_deg']:
                original = original_data[key]
                written = written_data[key]
                if abs(original - written) < 1e-6:
                    self.get_logger().info(f"   ✔️ {key}: {original}° (写入值：{written}°) ✔️")
                else:
                    self.get_logger().error(f"   ❌ {key}: 计算值{original}° ≠ 写入值{written}° ❌")
            
            # 加速度向量校验
            original_acc = original_data['raw_acc']
            written_acc = written_data['raw_acc']
            if all(abs(o - w) < 1e-6 for o, w in zip(original_acc, written_acc)):
                self.get_logger().info(f"   ✔️ raw_acc: {original_acc} (写入值：{written_acc}) ✔️")
            else:
                self.get_logger().error(f"   ❌ raw_acc: 计算值{original_acc} ≠ 写入值{written_acc} ❌")

        except Exception as e:
            self.get_logger().error(f"❌ 校验失败：{str(e)}")

    def run(self):
        """完整流程：收集→计算→写入→校验"""
        # 1. 收集3秒数据（和你的代码一致）
        start = time.time()
        while time.time() - start < self.calib_duration + 1:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 2. 纯计算（无副作用）
        tilt_data = self.compute_tilt()
        
        # 3. 精准写入
        self.write_tilt_to_yaml(tilt_data)
        
        # 4. 日志总结
        self.get_logger().info(f"\n🎉 完整流程结束！YAML文件路径：{self.save_path}")

def main():
    # 保存到仓库内 config/（相对脚本定位，任意部署位置可运行）
    SAVE_PATH = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "config", "mid360_tilt.yaml"))
    
    # 检查路径是否可写
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
        print(f"📁 创建目录：{os.path.dirname(SAVE_PATH)}")

    rclpy.init()
    node = IMUTiltCalculator(SAVE_PATH)
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

