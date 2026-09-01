"""Test the trained StixelNExT model to see what output it gives"""

import torch
import cv2
import numpy as np
import os
from models.ConvNeXt import ConvNeXt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = ConvNeXt(in_channels=3, out_channels=2).to(device)
ckpt_path = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/StixelNExT_ground_best.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Load a sample image
img_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/images"
mask_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/masks"

sample_img = sorted(os.listdir(img_dir))[0]
img_path = os.path.join(img_dir, sample_img)
mask_path = os.path.join(mask_dir, os.path.splitext(sample_img)[0] + '.png')

print(f"Testing on: {sample_img}")

# Preprocess - same as training
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
h, w = img_tensor.shape[1:]
# Resize to 376x1248
img_tensor = img_tensor.unsqueeze(0)
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
img_tensor = img_tensor.to(device)

print(f"Input shape: {img_tensor.shape}")

# Inference
with torch.no_grad():
    output = model(img_tensor)

print(f"Output shape: {output.shape}")
print(f"Output[0, 0] min: {output[0, 0].min().item():.4f}, max: {output[0, 0].max().item():.4f}")
print(f"Output[0, 0] mean: {output[0, 0].mean().item():.4f}")
print(f"Number of pixels > 0.5: {(output[0, 0] > 0.5).sum().item()} / {output[0, 0].numel()}")

# Visualize the occupancy
occ = output[0, 0].cpu().numpy()
occ_vis = (occ * 255).astype(np.uint8)
cv2.imwrite('test_occupancy.png', occ_vis)
print(f"\nSaved occupancy visualization to: test_occupancy.png")

# Compare with GT
gt_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
print(f"\nGT: {gt_full.shape}, number of ground pixels: {(gt_full > 127.5).sum()} / {gt_full.size}")
