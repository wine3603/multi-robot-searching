#!/usr/bin/env python3
"""
02_annotate.py — Quad semantic map annotation tool (matplotlib backend)

Labels:
  0: Clear   1: Room  2: Corridor  3: Wall  4: Other

Controls:
  Left click   - Add vertex (4th point auto-finishes)
  Right click  - Finish quad (>=3 pts) or delete clicked quad
  Scroll       - Zoom in/out (centered on mouse)
  Middle drag  - Pan view
  1/2/3/4      - Select label (Room/Corridor/Wall/Other)
  0            - Clear mode
  F            - Insert full-map wall quad
  Z            - Undo (remove last point or last quad)
  C            - Clear all quads
  S            - Save (_quads.json + _label.npy)
  N / P        - Next / Previous map (auto-save)
  Q / Esc      - Quit (auto-save)
"""
import numpy as np
import cv2
import os, glob, json, argparse
from typing import List, Tuple, Optional
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PatchCollection

WIN_W, WIN_H = 1280, 800
WALL_THR     = 50
RECT_ALPHA   = 0.40

LABELS = {0: "Clear", 1: "Room", 2: "Corridor", 3: "Wall", 4: "Other"}
COLORS_RGB = {
    0: (0.4, 0.4, 0.4),     # gray
    1: (0.0, 0.78, 0.0),    # green
    2: (0.0, 0.86, 0.86),   # cyan
    3: (0.86, 0.24, 0.24),  # red
    4: (1.0, 0.55, 0.0),    # orange
}


def occ_to_rgb(occ: np.ndarray) -> np.ndarray:
    """Occupancy grid → RGB float [0,1] for matplotlib"""
    rgb = np.full((*occ.shape, 3), 0.4, dtype=np.float32)
    rgb[occ == 0] = 0.86
    rgb[occ >= WALL_THR] = 0.14
    m = (occ > 0) & (occ < WALL_THR)
    v = np.clip(0.86 - occ[m].astype(np.float32) * 1.8 / 255.0, 0.24, 0.86)
    rgb[m] = np.stack([v, v, v], axis=-1)
    return rgb


class QuadAnnotator:
    def __init__(self, data_dir: str):
        # 支持平铺 map_*.npy 与 map_data_*/ 子目录两种结构
        all_npy = sorted(glob.glob(os.path.join(data_dir, "map_*.npy")))
        for entry in sorted(os.listdir(data_dir)):
            subdir = os.path.join(data_dir, entry)
            if os.path.isdir(subdir):
                all_npy += sorted(glob.glob(os.path.join(subdir, "map_*.npy")))
        self.files = [f for f in all_npy if "_label" not in f]
        if not self.files:
            raise FileNotFoundError(f"No map_*.npy found in {data_dir}")

        self.idx = 0
        self._current_label = 1
        self.zoom = 1.0
        self.cx = self.cy = 0.0

        # Per-label quad lists
        self.quads = {1: [], 2: [], 3: [], 4: []}

        # Current drawing points
        self._current_points: List[Tuple[int, int]] = []

        # Pan state
        self._pan = False
        self._pan_start = (0.0, 0.0)
        self._pan_cx = 0.0
        self._pan_cy = 0.0

        # Mouse position (in data coords)
        self._mouse_x = 0.0
        self._mouse_y = 0.0

        self.occ = None
        self.base_rgb = None
        self.H = self.W = 0
        self.path = ""

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(WIN_W/100, WIN_H/100))
        self.fig.canvas.manager.set_window_title("QuadAnnotator")
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._load(0)
        print(f"Loaded {len(self.files)} maps")
        print(f"Current: {os.path.basename(self.path)}")

    def _json_path(self, p=None):
        return (p or self.path).replace(".npy", "_quads.json")

    def _label_path(self, p=None):
        return (p or self.path).replace(".npy", "_label.npy")

    def _load(self, idx: int):
        self.idx = idx % len(self.files)
        p = self.files[self.idx]
        self.occ = np.flipud(np.load(p).astype(np.int16))
        self.H, self.W = self.occ.shape
        self.base_rgb = occ_to_rgb(self.occ)

        # Load quads
        jp = self._json_path(p)
        if os.path.exists(jp):
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            self.quads = {1: [], 2: [], 3: [], 4: []}
            for label, items in data.items():
                lid = int(label)
                if lid in self.quads:
                    for item in items:
                        if len(item) == 8:
                            self.quads[lid].append(tuple(item))
            print(f"Loaded: {os.path.basename(p)} - Room:{len(self.quads[1])} Corr:{len(self.quads[2])} Wall:{len(self.quads[3])} Other:{len(self.quads[4])}")
        else:
            self.quads = {1: [], 2: [], 3: [], 4: []}
            print(f"New: {os.path.basename(p)}")

        self._current_points = []
        self.zoom = min(WIN_W / self.W, WIN_H / self.H) * 0.90
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.path = p
        self._update_view()
        self._redraw()

    def _save(self):
        # Save JSON
        data = {str(k): v for k, v in self.quads.items()}
        with open(self._json_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Save multi-channel label
        lbl = np.zeros((self.H, self.W), dtype=np.uint8)
        for label_id, quad_list in self.quads.items():
            for quad in quad_list:
                x0, y0, x1, y1, x2, y2, x3, y3 = quad
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                mask = np.zeros((self.H, self.W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                lbl[mask == 1] |= (1 << (label_id - 1))
        np.save(self._label_path(), np.flipud(lbl))
        total = sum(len(v) for v in self.quads.values())
        print(f"Saved: {self._json_path()}")
        print(f"Saved: {self._label_path()} ({total} quads)")

    def _auto_wall(self):
        self.quads[3].append((0, 0, self.W-1, 0, self.W-1, self.H-1, 0, self.H-1))
        print("Inserted full-map wall quad")

    def _data_to_map(self, dx, dy):
        """matplotlib data coords → map pixel coords (clipped)"""
        return (int(np.clip(dx, 0, self.W - 1)),
                int(np.clip(dy, 0, self.H - 1)))

    def _hit_test(self, mx, my) -> Tuple[int, int]:
        """Return (label_id, quad_index) or (-1, -1)"""
        for label_id in [4, 3, 2, 1]:
            for i in range(len(self.quads[label_id]) - 1, -1, -1):
                x0, y0, x1, y1, x2, y2, x3, y3 = self.quads[label_id][i]
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                if cv2.pointPolygonTest(pts, (mx, my), False) >= 0:
                    return label_id, i
        return -1, -1

    def _finish_quad(self):
        if len(self._current_points) >= 3:
            pts = self._current_points[:4]
            while len(pts) < 4:
                pts.append(pts[-1])
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            x3, y3 = pts[3]
            self.quads[self._current_label].append((x0, y0, x1, y1, x2, y2, x3, y3))
            print(f"+ {LABELS.get(self._current_label)}: ({x0},{y0})({x1},{y1})({x2},{y2})({x3},{y3})")
        self._current_points = []

    def _update_view(self):
        """Set matplotlib axis limits based on zoom/pan"""
        hw = WIN_W / (2 * self.zoom) / self.W * self.W
        hh = WIN_H / (2 * self.zoom) / self.H * self.H
        hw = WIN_W / (2 * self.zoom)
        hh = WIN_H / (2 * self.zoom)
        self.ax.set_xlim(self.cx - hw, self.cx + hw)
        self.ax.set_ylim(self.cy + hh, self.cy - hh)  # inverted y

    def _redraw(self):
        """Full redraw"""
        self.ax.clear()

        if self.occ is None:
            self.fig.canvas.draw()
            return

        # Draw base map
        self.ax.imshow(self.base_rgb, extent=[0, self.W, self.H, 0], aspect='auto')

        # Draw filled quads (semi-transparent overlay using alpha)
        for label_id in [1, 2, 3, 4]:
            color = COLORS_RGB[label_id]
            for quad in self.quads[label_id]:
                x0, y0, x1, y1, x2, y2, x3, y3 = quad
                poly = Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
                              closed=True, facecolor=color, alpha=RECT_ALPHA,
                              edgecolor=color, linewidth=2)
                self.ax.add_patch(poly)

        # Draw current points
        if self._current_points:
            xs = [p[0] for p in self._current_points]
            ys = [p[1] for p in self._current_points]
            self.ax.plot(xs, ys, 'o-', color=COLORS_RGB[self._current_label],
                        markersize=5, linewidth=2)

        # Set limits
        self._update_view()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Title
        stats = (f"[{self.idx+1}/{len(self.files)}] {os.path.basename(self.path)} "
                f"R:{len(self.quads[1])} C:{len(self.quads[2])} "
                f"W:{len(self.quads[3])} O:{len(self.quads[4])} "
                f"zoom:{self.zoom:.2f}x | Label: {self._current_label}-{LABELS[self._current_label]}")
        self.ax.set_title(stats, fontsize=9)

        # Help text at bottom
        self.fig.text(0.5, 0.01,
                     "1:Room 2:Corridor 3:Wall 4:Other 0:Clear | Left:vertex Right:finish/delete | "
                     "F:full-wall Z:undo S:save N/P:prev/next Q:quit",
                     ha='center', fontsize=7, color='gray',
                     transform=self.fig.transFigure)

        self.fig.canvas.draw()

    # ── Matplotlib event handlers ──

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        self._mouse_x = event.xdata
        self._mouse_y = event.ydata

        if event.button == MouseButton.MIDDLE:
            self._pan = True
            self._pan_start = (event.xdata, event.ydata)
            self._pan_cx = self.cx
            self._pan_cy = self.cy
            return

        if event.button == MouseButton.LEFT:
            pt = self._data_to_map(event.xdata, event.ydata)
            if len(self._current_points) < 4:
                self._current_points.append(pt)
            if len(self._current_points) == 4:
                self._finish_quad()
            self._redraw()
            return

        if event.button == MouseButton.RIGHT:
            if len(self._current_points) >= 3:
                self._finish_quad()
            else:
                mx, my = event.xdata, event.ydata
                label_id, idx = self._hit_test(mx, my)
                if label_id >= 1:
                    self.quads[label_id].pop(idx)
                    print(f"- Deleted {LABELS.get(label_id)} quad")
            self._redraw()
            return

    def _on_release(self, event):
        if event.button == MouseButton.MIDDLE:
            self._pan = False

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return
        self._mouse_x = event.xdata
        self._mouse_y = event.ydata

        if self._pan:
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            self.cx = self._pan_cx - dx
            self.cy = self._pan_cy - dy
            self._update_view()
            self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        mx0, my0 = event.xdata, event.ydata
        factor = 1.15 if event.button == 'up' else 1.0 / 1.15
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 40.0))
        # Adjust pan to zoom centered on mouse
        mx1 = self.cx + (mx0 - self.cx)  # no change needed for center zoom
        # Recalc: keep mouse position stable
        hw = WIN_W / (2 * self.zoom)
        hh = WIN_H / (2 * self.zoom)
        # After zoom, mouse should still point to same map coord
        self.cx = mx0 - (event.x - WIN_W/2) / self.zoom if False else self.cx
        # Simpler: just zoom and keep center
        self._update_view()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ['q', 'escape']:
            self._save()
            plt.close(self.fig)
        elif event.key == 's':
            self._save()
        elif event.key == 'z':
            if self._current_points:
                self._current_points.pop()
                self._redraw()
            elif any(len(v) > 0 for v in self.quads.values()):
                for lid in [4, 3, 2, 1]:
                    if self.quads[lid]:
                        q = self.quads[lid].pop()
                        print(f"Undo: {LABELS.get(lid)}")
                        self._redraw()
                        break
        elif event.key == 'f':
            self._auto_wall()
            self._redraw()
        elif event.key == 'c':
            self.quads = {1: [], 2: [], 3: [], 4: []}
            self._current_points = []
            print("Cleared all")
            self._redraw()
        elif event.key == 'enter':
            self._finish_quad()
            self._redraw()
        elif event.key == 'n':
            self._save()
            self._load(self.idx + 1)
        elif event.key == 'p':
            self._save()
            self._load(self.idx - 1)
        elif event.key == '0':
            self._current_label = 0
            print("Label: Clear mode (right-click to delete)")
        elif event.key in ['1', '2', '3', '4']:
            self._current_label = int(event.key)
            print(f"Label: {self._current_label} - {LABELS[self._current_label]}")

    def run(self):
        print("\n=== Controls ===")
        print("Left click   - Add vertex (4th auto-finishes)")
        print("Right click  - Finish quad (>=3 pts) or delete clicked quad")
        print("Scroll       - Zoom in/out")
        print("Middle drag  - Pan view")
        print("1/2/3/4      - Room/Corridor/Wall/Other")
        print("0            - Clear mode")
        print("F            - Full-map wall")
        print("Z            - Undo")
        print("C            - Clear all")
        print("S            - Save")
        print("N / P        - Next / Previous map")
        print("Q / Esc      - Quit (auto-save)")
        print("==================\n")

        plt.ioff()
        plt.show()
        print("Annotation tool closed")


def main():
    ap = argparse.ArgumentParser(description="Multi-label quad annotation tool")
    ap.add_argument("--data_dir", default="map_data_all")
    args, _ = ap.parse_known_args()
    QuadAnnotator(args.data_dir).run()


if __name__ == "__main__":
    main()
