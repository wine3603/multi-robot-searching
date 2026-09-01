#!/bin/bash
# ===================== 第一步：定位仓库根（脚本任意位置可运行） =====================
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SELF_DIR"
SCRIPTS_DIR="$ROOT/src/FAST_LIO/scripts"

pkill -f fast_lio 2>/dev/null
pkill -f livox_ros_driver2 2>/dev/null
# ===================== 第二步：定义全局变量（存储PID） =====================
declare -a PIDS=()
# 校准配置路径
ORIGINAL_YAML="$ROOT/src/FAST_LIO/config/mid360.yaml"
CALIBRATED_YAML="$ROOT/src/FAST_LIO/config/mid360_calibrated.yaml"
CALIBRATE_SCRIPT="$SCRIPTS_DIR/calibrate_imu_extrinsic.py"

# ===================== 第三步：进程清理函数（精准杀死PID） =====================
cleanup_all_processes() {
    echo -e "\n[INFO] 开始清理所有相关进程..."
    
    # 1. 强制杀死通过PID跟踪的进程
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "[INFO] 强制杀死跟踪的进程PID：${PIDS[*]}"
        kill -9 "${PIDS[@]}" 2>/dev/null
        PIDS=()
    fi
    
    # 杀死校准脚本进程
    pkill -9 -f "calibrate_imu_extrinsic.py" 2>/dev/null
    
    # 2. 兜底杀死静态TF发布器
    killall -9 static_transform_publisher 2>/dev/null
    
    # 3. 杀死ROS2相关进程
    ros2 daemon stop 2>/dev/null
    pkill -9 -f "ros2 " 2>/dev/null
    pkill -9 -f "python3.*ros" 2>/dev/null
    pkill -9 -f "fastlio_mapping" 2>/dev/null
    pkill -9 -f "fast_lio" 2>/dev/null
    pkill -9 -f "livox_ros_driver2" 2>/dev/null
    pkill -9 -f "ros2 launch" 2>/dev/null
    
    # 4. 精准杀死所有自定义Python脚本
    pkill -9 -f "grid_map_generator.py" 2>/dev/null
    
    # 5. 清理ROS2缓存
    rm -rf /tmp/ros2* /tmp/RMW* 2>/dev/null
    
    echo "[INFO] 所有进程清理完成！"
}

# ===================== 第四步：捕获退出信号 =====================
trap cleanup_all_processes SIGINT EXIT SIGTERM

# ===================== 第五步：校准结果检查函数 =====================
check_calibration_file() {
    if [ ! -f "${CALIBRATED_YAML}" ]; then
        echo "[WARNING] 校准文件不存在，复制原始配置"
        cp "${ORIGINAL_YAML}" "${CALIBRATED_YAML}"
    fi
}

# ===================== 第六步：启动逻辑（捕获每个进程PID） =====================
echo "[INFO] 开始启动所有节点..."

# 启动Livox MID360驱动（捕获PID）
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
PIDS+=($!)
echo "[INFO] 开始IMU外参水平校准..."
python3 "${CALIBRATE_SCRIPT}" --target "${ORIGINAL_YAML}"
# 检查校准文件（容错）
check_calibration_file
echo "[INFO] IMU校准脚本执行完成"
sleep 3
ros2 launch fast_lio mapping.launch.py &
PIDS+=($!)
sl
# 启动自定义Python脚本（逐个捕获PID）
python3 "${SCRIPTS_DIR}/grid_map_generator.py" > /dev/null 2>&1 &
PIDS+=($!)
python3 "${SCRIPTS_DIR}/video.py" > /dev/null 2>&1 &
PIDS+=($!)
#python3 "${SCRIPTS_DIR}/11_capture_ground.py" > /dev/null 2>&1 &
#PIDS+=($!)
#sleep 100
#python3 "${SCRIPTS_DIR}/04_inference.py" > /dev/null 2>&1 &
#PIDS+=($!)

echo "[INFO] 所有节点启动完成！跟踪的进程PID：${PIDS[*]}"
echo "[INFO] 按 Ctrl+C 退出并强制清理所有进程"

# ===================== 第七步：保持脚本运行 =====================
wait

