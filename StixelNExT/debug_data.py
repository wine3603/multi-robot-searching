"""Debug script to check if data conversion is correct"""

import cv2
import numpy as np
import torch
from dataloader.ground_data import GroundDataStixel

# Load a sample
dataset = GroundDataStixel(data_dir="/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data", phase='training')
print(f"Dataset size: {len(dataset)}")

# Check first sample
img, target = dataset[0]
print(f"\nImage shape: {img.shape}")
print(f"Target shape: {target.shape}")
print(f"Image min/max: {img.min():.2f} / {img.max():.2f}")
print(f"Target[0] (occupancy) min/max: {target[0].min():.2f} / {target[0].max():.2f}")
print(f"Target[1] (cut) min/max: {target[1].min():.2f} / {target[1].max():.2f}")
print(f"Number of ground pixels (channel 0): {(target[0] > 0.5).sum()}")
print(f"Number of cut pixels (channel 1): {(target[1] > 0.5).sum()}")

# Visualize occupancy
occ = target[0].numpy()
print(f"\nOccupancy grid shape: {occ.shape}")
print(f"Unique values in occupancy: {np.unique(occ)}")

# Save visualization
occ_vis = (occ * 255).astype(np.uint8)
cv2.imwrite('debug_occupancy.png', occ_vis)
print(f"\nSaved debug visualization to: debug_occupancy.png")

cut_vis = (target[1].numpy() * 255).astype(np.uint8)
cv2.imwrite('debug_cut.png', cut_vis)
print(f"Saved cut visualization to: debug_cut.png")
