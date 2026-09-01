import torch
import cv2
import numpy as np
import os
from models.ConvNeXt import ConvNeXt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = ConvNeXt(in_channels=3, out_channels=2).to(device)
ckpt_path = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/StixelNExT_ground_best.pth"
state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded, {sum(p.numel() for p in model.parameters())} params")

# Load a sample image
img_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/images"
img_path = os.path.join(img_dir, sorted(os.listdir(img_dir))[0])
img = cv2.imread(img_path)
print(f"Image: {img_path}, shape {img.shape}")

# Preprocess
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
img_tensor = img_tensor.to(device)

print(f"Input tensor shape: {img_tensor.shape}")

with torch.no_grad():
    output = model(img_tensor)

print(f"Output shape: {output.shape}")
print(f"output[0, :, :] min: {output[0, :, :].min().item():.6f}")
print(f"output[0, :, :] max: {output[0, :, :].max().item():.6f}")
print(f"output[0, :, :] mean: {output[0, :, :].mean().item():.6f}")
print(f"output[0, :, :] > 0.1: {(output[0, :, :] > 0.1).sum().item()} / {output[0, :, :].numel()}")
print(f"output[0, :, :] > 0.3: {(output[0, :, :] > 0.3).sum().item()} / {output[0, :, :].numel()}")
print(f"output[0, :, :] > 0.5: {(output[0, :, :] > 0.5).sum().item()} / {output[0, :, :].numel()}")

# Check output[1, :, :]
print(f"\noutput[1, :, :] min: {output[1, :, :].min().item():.6f}")
print(f"output[1, :, :] max: {output[1, :, :].max().item():.6f}")
