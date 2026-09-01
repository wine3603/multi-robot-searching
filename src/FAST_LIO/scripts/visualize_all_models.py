import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 直接导入eval脚本里的所有加载函数
from eval_baseline_vs_ours import (
    load_unet_model, load_baseline_model,
    load_stixelnext_model, load_footprints_model
)

# 路径设置
IMG_PATH = "ground_data/images/img_20260402_182507_176.jpg"
MASK_PATH = "ground_data/masks/img_20260402_182507_176.png"
MODEL_DIR = "model_checkpoints/"
OUTPUT_DIR = "visualization_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5

# ========== 推理 ==========

# 读取图像和GT
img = cv2.imread(IMG_PATH)
gt_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
gt_bin = (gt_mask > 127).astype(np.uint8) * 255
h, w = img.shape[:2]

print(f"图像尺寸: {w}x{h}")

# 归一化函数
def normalize_img(img):
    from eval_baseline_vs_ours import IMG_MEAN, IMG_STD
    img_float = img.astype(np.float32) / 255.0
    img_norm = (img_float - IMG_MEAN) / IMG_STD
    img_norm = img_norm.transpose(2, 0, 1)
    return torch.from_numpy(img_norm).unsqueeze(0).to(device)

# 保存叠加图
def save_overlay(img, pred_bin, name, iou=None):
    pred_color = np.zeros_like(img)
    pred_color[pred_bin > 0] = [0, 255, 0]  # 绿色
    overlay = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
    text = name if iou is None else f"{name} (IoU={iou})"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.jpg"), overlay)
    print(f"✓ 保存: {OUTPUT_DIR}/{name}.jpg")

# 1. 保存原图和GT
cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original_image.jpg"), img)
gt_overlay = np.zeros_like(img)
gt_overlay[gt_bin > 0] = [0, 255, 0]
gt_overlay = cv2.addWeighted(img, 0.6, gt_overlay, 0.4, 0)
cv2.putText(gt_overlay, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "1_ground_truth.jpg"), gt_overlay)
print("✓ 保存: 原图和GT")

# 2. UNet
print("\n=== UNet (IoU 0.91) ===")
unet, _ = load_unet_model(os.path.join(MODEL_DIR, "ground_best.pth"), device)
unet.eval()
with torch.no_grad():
    img_tensor = normalize_img(img)
    pb = torch.zeros_like(img_tensor)[:, :1, :, :]
    pb[:, :, int(h*2/3):, :] += 2.0
    pred = torch.sigmoid(unet(img_tensor) + pb)
    pred_np = (pred[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    save_overlay(img, pred_np, "2_UNet", "0.91")

# 3. Footprints
print("\n=== Footprints (IoU 0.65) ===")
footprints, _ = load_footprints_model(os.path.join(MODEL_DIR, "Footprints_ground_best.pth"), device)
footprints.eval()
with torch.no_grad():
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    # Resize to 32的倍数
    h_fp, w_fp = (h//32 +1)*32, (w//32 +1)*32
    img_fp = F.interpolate(img_tensor, size=(h_fp, w_fp), mode='bilinear', align_corners=False)
    pred = footprints(img_fp)
    pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
    pred_np = (pred[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    save_overlay(img, pred_np, "3_Footprints", "0.65")

# 4. DeepLabV3+
print("\n=== DeepLabV3+ (IoU 0.69) ===")
deeplab, _ = load_baseline_model(os.path.join(MODEL_DIR, "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"), device)
deeplab.eval()
with torch.no_grad():
    img_tensor = normalize_img(img)
    logits = deeplab(img_tensor)['out']
    probs = torch.softmax(logits, dim=1)
    prob_ground = torch.zeros_like(probs[:, 0:1, :, :])
    CITYSCAPES_GROUND_CLASSES = [0, 1]  # road, sidewalk
    for cls_idx in CITYSCAPES_GROUND_CLASSES:
        prob_ground += probs[:, cls_idx:cls_idx+1, :, :]
    pred_np = (prob_ground[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    save_overlay(img, pred_np, "4_DeepLabV3Plus", "0.69")

# 5. StixelNExT
print("\n=== StixelNExT (IoU 0.85) ===")
stixel, _ = load_stixelnext_model(os.path.join(MODEL_DIR, "StixelNExT_ground_best.pth"), device)
stixel.eval()
with torch.no_grad():
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    # Resize to 376x1248 for StixelNExT
    img_stixel = F.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
    output = stixel(img_stixel)
    prob_stixel = output[0, 0, :, :]
    # Resize back
    prob = F.interpolate(prob_stixel.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    pred_np = (prob[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
    save_overlay(img, pred_np, "5_StixelNExT", "0.85")

# 创建对比图
print("\n=== 创建对比图 ===")
images = [
    cv2.imread(os.path.join(OUTPUT_DIR, "0_original_image.jpg")),
    cv2.imread(os.path.join(OUTPUT_DIR, "1_ground_truth.jpg")),
    cv2.imread(os.path.join(OUTPUT_DIR, "2_UNet.jpg")),
    cv2.imread(os.path.join(OUTPUT_DIR, "3_Footprints.jpg")),
    cv2.imread(os.path.join(OUTPUT_DIR, "4_DeepLabV3Plus.jpg")),
    cv2.imread(os.path.join(OUTPUT_DIR, "5_StixelNExT.jpg"))
]

# 2行3列
row1 = np.hstack(images[:3])
row2 = np.hstack(images[3:])
comparison = np.vstack([row1, row2])
cv2.imwrite(os.path.join(OUTPUT_DIR, "ALL_MODELS_COMPARISON.jpg"), comparison)

print(f"\n✅ 所有可视化结果已保存到: {OUTPUT_DIR}/")
print(f"   对比大图: {OUTPUT_DIR}/ALL_MODELS_COMPARISON.jpg")
