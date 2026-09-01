#!/bin/bash
# 启动fast_lio并运行livox监测

# 定位仓库根（脚本任意位置可运行）
ROOT="$(cd "$(dirname "$0")" && pwd)"

# 源ROS2环境
source /opt/ros/humble/setup.bash

# 进入工作空间
cd "$ROOT"
source install/setup.bash

# 启动监测脚本在后台
echo "Starting livox monitor..."
python3 "$ROOT/src/FAST_LIO/scripts/livox_monitor.py" &
MONITOR_PID=$!

# 等待一小会儿
sleep 2

# 启动fast_lio mapping
echo "Starting fast_lio mapping..."
ros2 launch fast_lio mapping.launch.py

# 退出时杀掉监测进程
kill $MONITOR_PID
echo "Monitor stopped"
