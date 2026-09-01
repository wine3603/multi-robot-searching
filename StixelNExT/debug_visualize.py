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

print(f"Model loaded")

# Load a sample image and mask
img_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/images"
mask_dir = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/ground_data/masks"
img_path = os.path.join(img_dir, sorted(os.listdir(img_dir))[0])
mask_path = os.path.join(mask_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
img = cv2.imread(img_path)
mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print(f"Image: {img_path}, shape {img.shape}")
print(f"Mask: {mask_path}, shape {mask_gt.shape}")

# Preprocess
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
img_tensor = img_tensor.to(device)

with torch.no_grad():
    output = model(img_tensor)

print(f"\nOutput shape: {output.shape}")
print(f"output[0, :, :] (occupancy) min: {output[0, :, :].min().item():.6f}")
print(f"output[0, :, :] (occupancy) max: {output[0, :, :].max().item():.6f}")
print(f"output[0, :, :] (occupancy) mean: {output[0, :, :].mean().item():.6f}")

# Save heatmap
occ_np = output[0, :, :].cpu().numpy()
# Scale to 0-255 for visualization
occ_vis = (occ_np * 255).astype(np.uint8)
occ_vis_color = cv2.applyColorMap(occ_vis, cv2.COLORMAP_JET)
cv2.imwrite('debug_occupancy.png', occ_vis_color)
print(f"\nSaved visualization to debug_occupancy.png")

# Count how many exceed 0.1, 0.3, 0.5
print(f"\nStatistics:")
for thresh in [0.05, 0.1, 0.2, 0.3, 0.5]:
    count = (occ_np > thresh).sum()
    total = occ_np.size
    print(f"  > {thresh:.2f}: {count} / {total} ({100*count/total:.2f}%)")

# Show top 10 values
flat = occ_np.flatten()
top_indices = flat.argsort()[-10:][::-1]
top_values = flat[top_indices]
print(f"\nTop 10 predictions: {top_values}")
