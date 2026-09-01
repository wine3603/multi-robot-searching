#!/bin/bash
# 批量运行所有对比实验
# 将结果保存在 experiments/ 目录下

# 定位仓库根（脚本任意位置可运行）
ROOT="$(cd "$(dirname "$0")" && pwd)"

# 创建输出目录
OUTPUT_DIR=$ROOT/experiments
mkdir -p "$OUTPUT_DIR"

# 所有可用策略列表 (rrt_sharp skipped - source file empty)
STRATEGIES=(
  # 基于前沿检测
  "frontier_greedy"
  "rrt"
  "rrt_connect"
  "extended_rrt"
  "dynamic_rrt"
  "rrt_star_smart"
  "informed_rrt_star"
  "dijkstra_info_gain"
)

TOTAL=${#STRATEGIES[@]}
CURRENT=0

echo "===================================================="
echo "🤖 自主探索对比实验 - 所有算法"
echo "🧮 总共 $TOTAL 种策略将被运行:"
for i in "${!STRATEGIES[@]}"; do
  s=${STRATEGIES[$i]}
  echo "   $((i+1))/$TOTAL: $s"
done
echo "📂 结果保存到: $OUTPUT_DIR"
echo "===================================================="
echo ""

# 顺序运行所有实验
for strat in "${STRATEGIES[@]}"; do
  CURRENT=$((CURRENT + 1))
  # 生成时间戳用于目录名
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  # 检查是否已有该策略的实验结果
  EXISTING=$(find "$OUTPUT_DIR" -type d -name "${strat}_*" | head -1)
  if [ -n "$EXISTING" ]; then
    echo "----------------------------------------------------"
    echo "⏭️  [$CURRENT/$TOTAL] 跳过 $strat"
    echo "   发现已有实验结果: $EXISTING"
    echo "----------------------------------------------------"
    echo ""
    continue
  fi

  echo "----------------------------------------------------"
  echo "▶️  [$CURRENT/$TOTAL] 当前实验: $strat"
  echo "🕒 开始时间: $(date)"
  echo "----------------------------------------------------"

  # 运行实验 (需要先启动map服务)
  # 假设已经有map话题发布，这里直接运行节点
  # 使用速度: linear 1.0 m/s, angular 1.0 rad/s
  python3 "$ROOT/src/FAST_LIO/scripts/simulation_explore_framework.py" --strategy "$strat" --output-dir "$OUTPUT_DIR" --coverage-threshold 0.95 --max-steps 500 --linear-speed 1.0 --angular-speed 1.0

  echo ""
  echo "✅ [$CURRENT/$TOTAL] 实验完成: $strat"
  echo "🕒 结束时间: $(date)"
  echo "⏳ 等待5秒开始下一个..."
  sleep 5
  echo ""
done

echo ""
echo "===================================================="
echo "🎉 所有实验完成! 总共运行 $CURRENT 种算法"
echo "📊 所有结果都在: $OUTPUT_DIR"
echo "   每个实验文件夹包含:"
echo "   - result.json        (完整统计数据)"
echo "   - coverage_curve.png (覆盖率曲线图)"
echo "   - explored_map.png   (最终探索地图可视化)"
echo "===================================================="
