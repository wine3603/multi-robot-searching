# AGIBOTNAV — 四足机器人实时语义 SLAM 快速探索与全覆盖

本仓库是论文《Fast Exploration and Coverage of Unknown Indoor Environments Using
Real-Time Semantic SLAM on Quadruped Robots》的完整软件包，运行于 **Agibot D1 Ultra
四足机器人**（Jetson Orin）上，搭载 **Livox MID360 激光雷达** 与 FPV 摄像头，
基于 **ROS2 Humble / JetPack R36.5 / Ubuntu 22.04**。

## 系统要求

| 项目 | 配置 |
|---|---|
| 机器人 | Agibot D1 Ultra（四足） |
| 计算平台 | NVIDIA Jetson Orin，ARMv8，29GB RAM，CUDA 12.6 |
| 系统 / 中间件 | Ubuntu 22.04，ROS2 Humble，JetPack R36.5 |
| 激光雷达 | Livox MID360 |
| 摄像头 | FPV 摄像头（RTSP 流） |

## 仓库结构

```
agibotnav/
├── run.sh                        # 一键启动: Livox 驱动 → IMU 校准 → FAST_LIO → 栅格地图 + 视频
├── run_monitor.sh                # livox_monitor + fast_lio mapping（故障自动重启）
├── run_all_experiments.sh        # 批量运行 8 种探索策略对比实验 → experiments/
├── default.rviz                  # RViz2 显示配置
├── experiments/                  # 8 种探索算法对比结果（覆盖率曲线、RRT 树、性能指标）
├── src/
│   ├── FAST_LIO/                 # Fast LiDAR-Inertial Odometry（C++）+ 全部管线/训练脚本
│   │   ├── src/ include/ launch/ config/ msg/ rviz/ ...
│   │   └── scripts/
│   │       ├── 01-05 系列        # 语义分割: 地图保存、标注、U-Net 训练、评估、推理
│   │       ├── 11-14 系列        # 地面检测: 数据采集、标注、训练、实时推理
│   │       ├── autonomous_explore.py / explore_Frontier.py / grid_map_generator.py
│   │       ├── calibrate_imu_extrinsic.py / calib_time_offset.py
│   │       ├── eval_baseline_vs_ours.py + eval_results/ + exploration_result/
│   │       ├── map_data_all/     # 已标注栅格地图（房间/走廊/墙壁/障碍物）
│   │       ├── ground_data/      # 地面训练数据集（图像 + 掩码）
│   │       └── checkpoints_ground/  # 训练好的地面检测权重（ground_best.pth）
│   ├── PathPlanning/             # 探索实验用 RRT 系列规划器（仅 rrt_2D）
│   └── agibot_D1_Edu-Ultra/      # Agibot D1 平台 SDK（高低层 API 头文件、示例、文档）
├── StixelNExT/                   # Stixel 地面检测模型（训练 + 推理）
├── footprints/                   # 可通行空间估计模块（CVPR 2020）
└── GANav-offroad/                # 基于 MMSegmentation 的语义分割框架
```

> 说明：所有 shell 脚本与 Python 节点都基于自身位置（`$(dirname "$0")` /
> `os.path.dirname(__file__)`）定位仓库根目录，无需修改硬编码路径即可部署到任意位置
> （例如 `/home/orin-001/sda/agibotnav/`）。

## 安装

```bash
# 1. 编译 ROS2 工作空间（fast_lio + livox_ros_driver2）
cd <仓库根>
colcon build --symlink-install
source install/setup.bash

# 2. Python 依赖（节点使用 rclpy、numpy、cv2、torch、scipy、matplotlib）
pip install numpy opencv-python torch scipy matplotlib sensor-msgs-py bridge

# 3. 放入训练好的权重（未提交，gitignore 忽略）：
#    - src/FAST_LIO/scripts/checkpoints_ground/ground_best.pth   （地面检测）
#    - src/FAST_LIO/scripts/checkpoints/best.pth                 （语义分割）
```

## 快速使用

### 1. 完整系统（推荐）

```bash
./run.sh
```

依次启动：Livox MID360 驱动 → IMU 外参校准（`calibrate_imu_extrinsic.py`）→ FAST_LIO
建图 → `grid_map_generator.py`（栅格地图 + 摄像头地面分割）→ `video.py`（RTSP 推流）。
按 `Ctrl+C` 可一键清理所有进程。

### 2. 带监控的建图（故障自动重启）

```bash
./run_monitor.sh
```

启动 `livox_monitor.py`（监测激光雷达健康状态、FAST_LIO 异常自动重启、拉起倾斜检测 /
栅格地图 / 延迟 TF 发布器），随后启动 FAST_LIO 建图。

### 3. 探索对比实验（基于实时地图的仿真）

```bash
./run_all_experiments.sh
```

对当前 `/map` 话题运行全部 8 种探索策略（frontier_greedy、rrt、rrt_connect、
extended_rrt、dynamic_rrt、rrt_star_smart、informed_rrt_star、dijkstra_info_gain），
并把 `result.json`、`coverage_curve.png`、`explored_map.png`、`rrt_tree.png` 保存到
`experiments/`。

## 脚本说明（src/FAST_LIO/scripts/）

### 语义分割（01-05 系列）

| 脚本 | 用途 |
|---|---|
| `01_save_map.py` | 将 `/map` 话题保存为 2D 栅格地图到 `map_data_all/map_data_N` |
| `02_annotate.py` | 人工标注工具（房间/走廊/墙壁/障碍物四类） |
| `03_train.py` | 训练 4 类语义分割 U-Net |
| `04_eval.py` | 模型评估（mIoU、准确率、各类别指标）→ `eval_results/` |
| `04_inference.py` | 实时语义分割推理（支持离线模式） |

### 地面检测（11-14 系列）

| 脚本 | 用途 |
|---|---|
| `11_capture_ground.py` | 从 RTSP 摄像头采集地面训练图像 |
| `12_annotate_ground.py` | 地面掩码交互标注工具 |
| `13_train_ground.py` | 训练地面分割 U-Net → `checkpoints_ground/ground_best.pth` |
| `14_infer_ground.py` | 实时地面检测推理节点 |

### 探索与标定

| 脚本 | 用途 |
|---|---|
| `autonomous_explore.py` | 动态 RRT 探索控制器（实物），结果写入 `exploration_result/` |
| `explore_Frontier.py` | 前沿探索节点 |
| `grid_map_generator.py` | LiDAR → 栅格地图 + 相机地面分割，发布 `/map` |
| `calibrate_imu_extrinsic.py` | IMU-LiDAR 外参水平校准 → `config/mid360_tilt.yaml` |
| `calib_time_offset.py` | LiDAR-IMU 时间偏移标定（rosbag） |
| `eval_baseline_vs_ours.py` | U-Net vs Footprints vs DeepLabV3+ vs StixelNExT 对比评估 |
| `simulation_explore_framework.py` | 8 策略探索对比框架（run_all_experiments.sh 使用） |
| `livox_monitor.py` | 激光雷达健康监测，FAST_LIO 自动重启 |
| `video.py` | RTSP 摄像头推流节点（`/rtsp_image`） |

## 实验数据

`experiments/` 包含 8 种探索算法的对比结果：覆盖率曲线、RRT 树可视化与性能指标
（覆盖率、效率、运动时间、CPU 时间、行驶距离）。运行 `./run_all_experiments.sh`
可复现全部实验。

## 致谢

- [FAST_LIO](https://github.com/hku-mars/FAST_LIO) — 激光惯性里程计
- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) — RRT 系列规划器（改编）
- [Footprints](https://github.com/nianticlabs/... footprints) — 可通行空间估计（CVPR 2020）
- [GANav-offroad](https://github.com/rayguan97/GANav-offroad) — 基于 MMSegmentation 的语义分割
- 智元 AgibotTech — D1 Edu-Ultra SDK

## 许可证

各子模块各自包含 LICENSE 文件（`src/FAST_LIO/LICENSE`、`src/PathPlanning/LICENSE`、
`footprints/LICENSE`、`GANav-offroad/LICENSE` 等）。