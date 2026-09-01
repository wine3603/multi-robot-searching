#!/usr/bin/env python3
"""
01_save_map.py  —  把 /map 话题保存到本地

用法:
  python3 01_save_map.py [--output_base map_data_all]

  首张地图到达后自动保存到 map_data_all/map_data_N，N自动递增
  之后按 Enter 手动触发保存新地图
"""
import rclpy, threading, os, json, argparse
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from datetime import datetime

WALL_THR = 50


def occ_to_vis(occ: np.ndarray) -> np.ndarray:
    vis = np.full((*occ.shape, 3), 100, dtype=np.uint8)   # 灰=未知
    vis[occ == 0]         = 220                             # 白=自由
    vis[occ >= WALL_THR]  = 30                              # 深=障碍
    m = (occ > 0) & (occ < WALL_THR)
    v = np.clip(220 - occ[m].astype(np.float32) * 2, 60, 220).astype(np.uint8)
    vis[m] = np.stack([v, v, v], axis=-1)
    return vis


def find_next_index(base_dir):
    """找到下一个可用的地图编号"""
    if not os.path.exists(base_dir):
        return 1
    max_idx = 0
    for entry in os.listdir(base_dir):
        if entry.startswith("map_data_"):
            try:
                idx = int(entry.split("_")[-1])
                if idx > max_idx:
                    max_idx = idx
            except:
                pass
    return max_idx + 1


class MapSaver(Node):
    def __init__(self, output_base: str):
        super().__init__("map_saver")
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)
        self.current_idx = find_next_index(output_base)
        self.current_dir = os.path.join(output_base, f"map_data_{self.current_idx}")
        os.makedirs(self.current_dir, exist_ok=True)
        self.occ   = None
        self.meta  = None
        self._lock = threading.Lock()
        self._first = True
        self.create_subscription(OccupancyGrid, "/map", self._cb, 10)
        self.get_logger().info(
            f"等待 /map…  按 Enter 手动保存新地图  Ctrl-C 退出\n"
            f"新地图将保存到: {os.path.abspath(self.current_dir)}"
        )

    def _cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        occ = np.array(msg.data, np.int16).reshape(h, w)
        meta = dict(
            resolution=msg.info.resolution, width=w, height=h,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
        )
        with self._lock:
            self.occ, self.meta = occ, meta
        if self._first:
            self._first = False
            self.get_logger().info(f"收到首张地图 {w}×{h}，自动保存到 map_data_{self.current_idx}…")
            self.save()

    def save(self):
        with self._lock:
            occ, meta = self.occ, self.meta
        if occ is None:
            self.get_logger().warn("尚未收到地图！")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.current_dir, f"map_{ts}")
        # 原始数据
        np.save(f"{base}.npy", occ)
        # 元数据
        with open(f"{base}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        # 预览 PNG（flipud：Grid原点在左下 → 图像左上）
        cv2.imwrite(f"{base}.png", cv2.flip(occ_to_vis(occ), 0))
        self.get_logger().info(f"✅  {base}.npy / .png / _meta.json 保存完成")
        # 为下一次保存递增编号
        self.current_idx += 1
        self.current_dir = os.path.join(self.output_base, f"map_data_{self.current_idx}")
        os.makedirs(self.current_dir, exist_ok=True)
        self.get_logger().info(f"👉 下一张地图将保存到 map_data_{self.current_idx}")

def _kbd_thread(node: MapSaver):
    while rclpy.ok():
        try:
            input()
        except EOFError:
            break
        node.get_logger().info("手动触发保存新地图…")
        node.save()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_base", default="map_data_all")
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = MapSaver(args.output_base)
    threading.Thread(target=_kbd_thread, args=(node,), daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
