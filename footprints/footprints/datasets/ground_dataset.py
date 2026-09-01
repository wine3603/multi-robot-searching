# Copyright Niantic 2020. Patent Pending. All rights reserved.
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class GroundDataset(Dataset):
    """
    Adapted for your ground segmentation dataset:
    - root/images/  - RGB images (.jpg/.png)
    - root/masks/  - binary masks (0=bg, 255=ground/free-space)

    Returns: image, ground_mask (same size as input)
    """

    def __init__(self, data_dir, split='train', height=192, width=640, transform=None, **kwargs):
        """
        Args:
            data_dir: path to dataset directory containing images/ and masks/
            split: 'train' or 'val' (80/20 split)
            height: target height for input images
            width: target width for input images
        """
        self.data_dir = data_dir
        self.split = split
        self.height = height
        self.width = width
        self.transform = transform

        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')

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
                    ))

        # Split into train/val (80% / 20%)
        n_total = len(self.samples)
        n_train = int(0.8 * n_total)
        if split == 'train':
            self.samples = self.samples[:n_train]
        elif split == 'val':
            self.samples = self.samples[n_train:]

        print(f"==> {split} split: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Read image - convert to RGB 0-1
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Read mask - 0=bg, 255=ground -> 0/1 float
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127.5).astype(np.float32)

        # Convert to torch tensor (C, H, W)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Resize to expected height and width (requested by options)
        # Ensure it's multiple of 32 for Footprints encoder
        new_h = ((self.height + 31) // 32) * 32
        new_w = ((self.width + 31) // 32) * 32

        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(new_h, new_w), mode='nearest'
        ).squeeze(0)

        # Footprints expects a dict with:
        # - 'image': (3, H, W) float tensor 0-1
        # - 'mask': (1, H, W) float tensor 0-1 (1=ground)
        return {
            'image': img,
            'mask': mask,
        }
