import torch
import cv2
import numpy as np
import os
from models.ConvNeXt import ConvNeXt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = ConvNeXt(in_channels=3, out_channels=2).to(device)
ckpt_path = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/StixelNExT_ground_best.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
model.eval()

# Load a sample
img_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/images"
mask_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/masks"

sample_img = sorted(os.listdir(img_dir))[0]
img_path = os.path.join(img_dir, sample_img)
mask_path = os.path.join(mask_dir, os.path.splitext(sample_img)[0] + '.png')

img = cv2.imread(img_path)
print(f"Original image: {img.shape}")

# Preprocess - same as evaluation
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
h, w = img_tensor.shape[1:]
img_tensor = img_tensor.unsqueeze(0)
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
img_tensor = img_tensor.to(device)

print(f"Input shape: {img_tensor.shape}")

with torch.no_grad():
    output = model(img_tensor)

print(f"\nOutput shape: {output.shape}")
print(f"output[0, 0] min: {output[0, 0].min().item():.6f}")
print(f"output[0, 0] max: {output[0, 0].max().item():.6f}")
print(f"output[0, 0] mean: {output[0, 0].mean().item():.6f}")
print(f"output[0, 0] > 0.0: {(output[0, 0] > 0.0).sum().item()}/{output[0,0].numel()}")
print(f"output[0, 0] > 0.1: {(output[0, 0] > 0.1).sum().item()}/{output[0,0].numel()}")
print(f"output[0, 0] > 0.3: {(output[0, 0] > 0.3).sum().item()}/{output[0,0].numel()}")
print(f"output[0, 0] > 0.5: {(output[0, 0] > 0.5).sum().item()}/{output[0,0].numel()}")

# Show histogram
print(f"\nHistogram of output values:")
vals = output[0, 0].cpu().numpy().flatten()
print(f"  0.0-0.1: {np.sum((vals > 0.0) & (vals < 0.1))}")
print(f"  0.1-0.3: {np.sum((vals >= 0.1) & (vals < 0.3))}")
print(f"  0.3-0.5: {np.sum((vals >= 0.3) & (vals < 0.5))}")
print(f"  0.5-0.7: {np.sum((vals >= 0.5) & (vals < 0.7))}")
print(f"  0.7-1.0: {np.sum((vals >= 0.7) & (vals <= 1.0))}")
