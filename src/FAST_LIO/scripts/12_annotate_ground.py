#!/usr/bin/env python3
"""
12_annotate_ground.py — Interactive Ground Annotation Tool (using matplotlib)
Controls:
  Left drag   - Draw ground/drivable area
  Right drag  - Erase
  Scroll      - Adjust brush size
  Middle drag - Pan view
  Space       - Toggle original/annotated view
  S           - Save annotation
  N / P       - Next / Previous image
  C           - Clear current annotation
  Q / Esc     - Quit (auto-save)

Output:
  Images:     images/img_*.jpg
  Masks:      masks/img_*.png  (0=background/obstacle, 255=ground/drivable)
"""

import numpy as np
import cv2
import os
import glob
import argparse
from typing import Tuple
import matplotlib
# Try to use available GUI backend
try:
    matplotlib.use('Qt5Agg')
except:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')  # fallback, non-interactive

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backend_bases import MouseButton

# Configure font - use English to avoid font issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

WIN_W, WIN_H = 1280, 720
BRUSH_SIZE = 15
DISPLAY_SCALE = 1.0


class GroundAnnotator:
    def __init__(self, img_dir: str, mask_dir: str):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        os.makedirs(mask_dir, exist_ok=True)

        # Find all jpg/png images
        exts = ["*.jpg", "*.jpeg", "*.png"]
        self.files = []
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.files = sorted(self.files, key=lambda x: os.path.basename(x))

        if not self.files:
            raise FileNotFoundError(f"No image files found in {img_dir}")

        self.idx = 0
        self.brush_size = BRUSH_SIZE
        self.drawing = False
        self.erasing = False
        self.last_pt = None
        self.show_original = False
        self.zoom = 1.0
        self.cx = self.cy = 0.0  # Pan offset
        self.panning = False
        self.pan_start = None
        self.pan_cxcy_start = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        self.current_img = None
        self.current_mask = None
        self.H = self.W = 0

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(WIN_W/100, WIN_H/100))
        self.fig.canvas.manager.set_window_title("GroundAnnotator")
        self.fig.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Brush cursor circle
        self.brush_cursor = Circle((0, 0), self.brush_size, fill=False, color='white', linewidth=2)
        self.ax.add_patch(self.brush_cursor)

        self.img_display = None

        self._load(self.idx)
        print(f"Loaded {len(self.files)} images")
        print(f"Current: {os.path.basename(self.files[self.idx])}")

    def _mask_path(self, img_path: str) -> str:
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.mask_dir, f"{base}.png")

    def _load(self, idx: int):
        """Load image at specified index"""
        self.idx = idx % len(self.files)
        img_path = self.files[self.idx]
        self.current_img = cv2.imread(img_path)
        if self.current_img is None:
            print(f"Failed to read image: {img_path}")
            return
        self.H, self.W = self.current_img.shape[:2]

        # Try to load existing annotation
        mask_path = self._mask_path(img_path)
        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded existing annotation: {mask_path}")
        else:
            self.current_mask = np.zeros((self.H, self.W), dtype=np.uint8)
            print(f"New image annotation: {os.path.basename(img_path)}")

        # Reset view
        self.zoom = min(WIN_W / self.W, WIN_H / self.H) * 0.9
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.drawing = False
        self.erasing = False
        self.last_pt = None
        self._update_display()

    def _save(self):
        if self.current_mask is None:
            return
        img_path = self.files[self.idx]
        mask_path = self._mask_path(img_path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, self.current_mask)
        ground_ratio = np.sum(self.current_mask > 127) / (self.H*self.W)
        print(f"Saved: {mask_path}  Ground ratio: {ground_ratio:.1%}")

    def _data_to_image(self, data_x: float, data_y: float) -> Tuple[int, int]:
        """matplotlib data coords -> image pixel coords"""
        # extent=[0, W, H, 0] makes data coords match image coords directly
        ix = int(data_x)
        iy = int(data_y)
        return int(np.clip(ix, 0, self.W-1)), int(np.clip(iy, 0, self.H-1))

    def _draw_line(self, x0, y0, x1, y1, val):
        """Draw line using bresenham algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cv2.circle(self.current_mask, (x0, y0), self.brush_size, val, -1)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _on_button_press(self, event):
        if event.inaxes != self.ax:
            return
        self._last_mouse_x = event.xdata
        self._last_mouse_y = event.ydata

        if event.button == MouseButton.MIDDLE:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.pan_cxcy_start = (self.cx, self.cy)
        elif event.button == MouseButton.RIGHT:
            self.drawing = True
            self.erasing = True
            ix, iy = self._data_to_image(event.xdata, event.ydata)
            self.last_pt = (ix, iy)
            cv2.circle(self.current_mask, (ix, iy), self.brush_size + 2, 0, -1)
            self._update_display()
        elif event.button == MouseButton.LEFT:
            self.drawing = True
            self.erasing = False
            ix, iy = self._data_to_image(event.xdata, event.ydata)
            self.last_pt = (ix, iy)
            cv2.circle(self.current_mask, (ix, iy), self.brush_size, 255, -1)
            self._update_display()

    def _on_button_release(self, event):
        if event.button in [MouseButton.LEFT, MouseButton.RIGHT]:
            self.drawing = False
            self.erasing = False
            self.last_pt = None
        elif event.button == MouseButton.MIDDLE:
            self.panning = False

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        self._last_mouse_x = event.xdata
        self._last_mouse_y = event.ydata

        if self.panning:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.cx = self.pan_cxcy_start[0] - dx
            self.cy = self.pan_cxcy_start[1] + dy
            self._update_display()
        elif self.drawing:
            ix, iy = self._data_to_image(event.xdata, event.ydata)
            if self.last_pt is not None:
                val = 0 if self.erasing else 255
                self._draw_line(self.last_pt[0], self.last_pt[1], ix, iy, val)
            self.last_pt = (ix, iy)
            self._update_display()

        # Update brush cursor
        self.brush_cursor.set_center((event.xdata, event.ydata))
        self.brush_cursor.set_radius(self.brush_size)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.button == 'up':
            if self.brush_size < 80:
                self.brush_size += 2
        else:
            if self.brush_size > 3:
                self.brush_size -= 2
        print(f"Brush size: {self.brush_size}")
        self.brush_cursor.set_radius(self.brush_size)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ['q', 'escape']:
            self._save()
            plt.close(self.fig)
        elif event.key == 's':
            self._save()
        elif event.key == ' ':
            self.show_original = not self.show_original
            self._update_display()
        elif event.key == 'n':
            self._save()
            self._load(self.idx + 1)
            print(f"Current: {os.path.basename(self.files[self.idx])}")
        elif event.key == 'p':
            self._save()
            self._load(self.idx - 1)
            print(f"Current: {os.path.basename(self.files[self.idx])}")
        elif event.key == 'c':
            if self.current_mask is not None:
                self.current_mask.fill(0)
                print("Cleared current annotation")
                self._update_display()

    def _update_display(self):
        """Update matplotlib display"""
        self.ax.clear()

        if self.current_img is None:
            self.ax.text(0.5, 0.5, "No Image", ha='center', va='center')
            self.fig.canvas.draw()
            return

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)

        if self.show_original:
            display_img = img_rgb
        else:
            # Overlay mask
            overlay = np.zeros_like(img_rgb)
            overlay[self.current_mask > 127] = [0, 255, 0]  # RGB green
            display_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

        # Display image (matplotlib coords: origin='upper' y-axis down)
        self.ax.imshow(display_img, extent=[0, self.W, self.H, 0])

        # Set limits
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)

        # Title info
        stats = (f"[{self.idx+1}/{len(self.files)}] {os.path.basename(self.files[self.idx])} "
                 f"| Brush: {self.brush_size} | Ground: {np.sum(self.current_mask > 127) / (self.H*self.W):.1%}")
        self.ax.set_title(stats, fontsize=10)

        # Hide axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Re-add brush cursor
        self.brush_cursor = Circle((self._last_mouse_x, self._last_mouse_y),
                                   self.brush_size, fill=False, color='white', linewidth=2)
        self.ax.add_patch(self.brush_cursor)

        self.fig.canvas.draw()

    def run(self):
        print("\n=== Controls ===")
        print("Left drag    - Draw ground/drivable area")
        print("Right drag   - Erase")
        print("Scroll       - Adjust brush size")
        print("Middle drag  - Pan view")
        print("Space        - Toggle original/annotated view")
        print("S            - Save annotation")
        print("N / P        - Next / Previous image")
        print("C            - Clear current annotation")
        print("Q / Esc      - Quit (auto-save)")
        print("==================\n")

        # Use blocking display
        plt.ioff()
        plt.show()
        print("Annotation tool closed")


def main():
    ap = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_img_dir = os.path.join(script_dir, "ground_data", "images")
    default_mask_dir = os.path.join(script_dir, "ground_data", "masks")
    ap.add_argument("--img_dir", default=default_img_dir, help="Image directory")
    ap.add_argument("--mask_dir", default=default_mask_dir, help="Annotation mask output directory")
    args, _ = ap.parse_known_args()
    GroundAnnotator(args.img_dir, args.mask_dir).run()


if __name__ == "__main__":
    main()
