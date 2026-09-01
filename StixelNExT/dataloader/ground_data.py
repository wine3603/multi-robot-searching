"""
GroundData dataset adapter for StixelNExT
Convert full-image binary segmentation masks (0=bg, 255=ground)
to StixelNExT grid format (2 channels: occupancy, cut)
"""

import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import cv2
import yaml
import os

# Load config from project root
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path) as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def resize_mask_to_grid(full_mask: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Resize full mask to stixel grid"""
    grid_mask = cv2.resize(full_mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    # Threshold: if more than 50% is ground, mark as ground
    grid_mask = (grid_mask > 127.5).astype(np.float32)
    return grid_mask


def compute_cut_channel(occupancy: np.ndarray) -> np.ndarray:
    """Compute cut boundary from occupancy
    cut is where occupancy changes (ground -> non-ground)
    """
    grid_h, grid_w = occupancy.shape
    cut = np.zeros_like(occupancy)

    # Find vertical boundaries per column
    for col in range(grid_w):
        col_data = occupancy[:, col]
        # Find all transitions
        for row in range(1, grid_h):
            if col_data[row] != col_data[row - 1]:
                cut[row, col] = 1.0
                cut[row - 1, col] = 1.0

    return cut


class GroundDataStixel(Dataset):
    """
    Load from ground_data structure:
      root/images/  - images (.jpg/.png)
      root/masks/  - masks (.png) 0=background, 255=ground
    Output: (image, target)
      image: (3, 376, 1248) - resized to Kitti size
      target: (2, 94, 312) - stixel grid
        channel 0: occupancy (1=ground)
        channel 1: cut boundaries
    """
    def __init__(self, data_dir, phase, transform=None, target_transform=None,
                 return_original_image=False, return_name=False):
        root_dir = data_dir
        self.root_dir = root_dir
        self.phase = phase
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        # Get all image files
        exts = ['.jpg', '.jpeg', '.png']
        self.samples = []
        for f in sorted(os.listdir(self.img_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                base = os.path.splitext(f)[0]
                mask_path = os.path.join(self.mask_dir, base + '.png')
                if os.path.exists(mask_path):
                    self.samples.append((
                        os.path.join(self.img_dir, f),
                        mask_path,
                        base
                    ))

        # Split into train/val (80% / 20%)
        n_total = len(self.samples)
        n_train = int(0.8 * n_total)
        if phase == 'training' or phase == 'train':
            self.samples = self.samples[:n_train]
        elif phase == 'validation' or phase == 'val':
            self.samples = self.samples[n_train:]
        elif phase == 'testing' or phase == 'test':
            # Keep all for testing
            pass

        self.transform = transform
        self.target_transform = target_transform
        self.return_original_image = return_original_image
        self.return_name = return_name

        # Fixed size for Kitti compatibility (376x1248 -> grid_step=4 -> 94x312)
        self.img_height = 376
        self.img_width = 1248
        self.grid_step = 4
        self.grid_h = self.img_height // self.grid_step  # 94
        self.grid_w = self.img_width // self.grid_step   # 312

        self.img_size = {'height': self.img_height, 'width': self.img_width}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, base_name = self.samples[idx]

        # Read image
        feature_image = read_image(img_path, ImageReadMode.RGB).to(torch.float32)

        # Read mask (0=bg, 255=ground)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize image and mask to 376x1248 (expected by StixelNExT)
        _, h, w = feature_image.shape
        if (h, w) != (self.img_height, self.img_width):
            feature_image = feature_image.unsqueeze(0)
            feature_image = torch.nn.functional.interpolate(
                feature_image, size=(self.img_height, self.img_width),
                mode='bilinear', align_corners=False
            )
            feature_image = feature_image.squeeze(0)
            mask_np = cv2.resize(mask_np, (self.img_width, self.img_height),
                               interpolation=cv2.INTER_NEAREST)

        # Convert full mask to stixel grid
        grid_occupancy = resize_mask_to_grid(mask_np, self.grid_h, self.grid_w)
        grid_cut = compute_cut_channel(grid_occupancy)

        # Combine into target
        target = np.zeros((2, self.grid_h, self.grid_w), dtype=np.float32)
        target[0] = grid_occupancy
        target[1] = grid_cut
        target = torch.from_numpy(target)

        if self.transform:
            feature_image = self.transform(feature_image)
        if self.target_transform:
            target = self.target_transform(target)

        if self.return_original_image and self.return_name:
            cv2_original = cv2.imread(img_path)
            cv2_original = cv2.resize(cv2_original, (self.img_width, self.img_height))
            return feature_image, target, cv2_original, base_name
        elif self.return_original_image:
            cv2_original = cv2.imread(img_path)
            cv2_original = cv2.resize(cv2_original, (self.img_width, self.img_height))
            return feature_image, target, cv2_original
        elif self.return_name:
            return feature_image, target, base_name
        else:
            return feature_image, target
