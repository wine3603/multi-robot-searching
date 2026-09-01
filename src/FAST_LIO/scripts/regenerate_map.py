#!/usr/bin/env python3
"""重新生成探索地图可视化，避免直线穿过障碍物的视觉错觉"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 读取结果
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploration_result/result.json")

with open(result_path, 'r') as f:
    result = json.load(f)

trajectory = result['trajectory']
goals = result['exploration_goals']
robot = result['current_robot_pose']

print(f"加载数据: {len(trajectory)} 个轨迹点, {len(goals)} 个目标点")

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(14, 14), dpi=150)

# 1. 只画轨迹点，不画直线连接，避免穿过障碍物的视觉错觉
traj_x = [p[0] for p in trajectory]
traj_y = [p[1] for p in trajectory]

# 轨迹：用渐变色的点表示时间顺序
colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
for i, (x, y) in enumerate(trajectory):
    ax.scatter(x, y, c=[colors[i]], s=30, alpha=0.7, zorder=3)

# 2. 画轨迹顺序（箭头表示移动方向，只每隔几个点画一个避免太乱）
step = max(1, len(trajectory) // 10)
for i in range(0, len(trajectory) - step, step):
    x1, y1 = trajectory[i]
    x2, y2 = trajectory[i + step]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.5, lw=1.5),
                zorder=4)

# 3. 目标点（红色五角星）
goal_x = [p[0] for p in goals]
goal_y = [p[1] for p in goals]
ax.scatter(goal_x, goal_y, c='red', s=80, marker='*', label='Exploration Goals', zorder=5)

# 4. 标记目标点的序号，展示探索顺序
for i, (x, y) in enumerate(goals):
    ax.text(x + 0.1, y + 0.1, f"{i+1}", fontsize=8, color='darkred', zorder=6)

# 5. 机器人当前位置
ax.scatter(robot['x'], robot['y'], c='green', s=150, marker='o', edgecolors='black', label='Robot', zorder=7)
ax.add_patch(Circle((robot['x'], robot['y']), 0.5, color='green', alpha=0.2, zorder=6))

# 6. 起始位置
ax.scatter(trajectory[0][0], trajectory[0][1], c='blue', s=100, marker='s', label='Start', zorder=5)

ax.set_title(f"Exploration Trajectory (Coverage: {result['current_coverage']:.1%})", fontsize=14)
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 保存
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploration_result/exploration_map_fixed.png")
fig.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"✅ 改进的可视化已保存到: {output_path}")
print(f"   - 用彩色点和箭头展示真实轨迹，避免直线穿过障碍物的错觉")
print(f"   - 目标点标注了序号，展示探索顺序")
