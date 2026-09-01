#!/usr/bin/env python3
"""
13_train_ground.py — 地面分割模型训练
输入:  ground_data/images/ 原始图像
        ground_data/masks/  标注掩码
输出:  checkpoints_ground/ground_best.pth 训练好的模型

类别: 二分割  0=障碍物/背景, 1=地面/可通行

训练策略: 参考 03_train.py 梯度上升方式
  - 每轮依次训练每张标注图像
  - 停止阈值梯度上升: 从 0.3 → 逐步升到 0.98
  - 每张图达到 IoU 阈值后自动进入下一张
"""

import os
import glob
import argparse
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # 如果没装tqdm，提供简单的进度打印
    def tqdm(it, **kw):
        total = len(it) if hasattr(it, '__len__') else None
        for i, item in enumerate(it):
            if total is not None and (i % max(1, total // 10) == 0 or i == total - 1):
                print(f"  Progress: {i+1}/{total}", end='\r')
            yield item
        if total is not None:
            print()


# ============================================================================
# 模型定义 - UNet 二分割
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c_in, c_out))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, 2)
        self.conv = DoubleConv(c_in//2 + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.shape[2]-x.shape[2], skip.shape[3]-x.shape[3]
        x = F.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self, c_in=3, num_classes=1, base=32):
        super().__init__()
        b = base
        self.e1 = DoubleConv(c_in, b)
        self.e2 = Down(b, b*2)
        self.e3 = Down(b*2, b*4)
        self.e4 = Down(b*4, b*8)
        self.bot = DoubleConv(b*8, b*8)
        self.d3 = Up(b*8, b*4, b*4)
        self.d2 = Up(b*4, b*2, b*2)
        self.d1 = Up(b*2, b, b)
        self.head = nn.Conv2d(b, num_classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        x = self.bot(e4)
        x = self.d3(x, e3)
        x = self.d2(x, e2)
        x = self.d1(x, e1)
        return self.head(x)


# ============================================================================
# 数据处理 & 数据集
# ============================================================================

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img: np.ndarray):
    """BGR uint8 → 归一化到 [0,1] → ImageNet标准归一化"""
    img = img.astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    return img


def random_augment(img: np.ndarray, mask: np.ndarray):
    """随机数据增强，增强亮度变化应对光照差异"""
    # 随机水平翻转
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()

    # 随机垂直翻转
    if random.random() < 0.15:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()

    # 随机亮度/对比度调整（增强，应对亮度差距大）
    if random.random() < 0.8:
        # 亮度变化范围 ±40%（更大范围，适应不同光照）
        factor = 1.0 + random.uniform(-0.4, 0.4)
        img = img * factor
        # 随机对比度
        if random.random() < 0.5:
            gamma = random.uniform(0.7, 1.3)
            img = np.power(img, gamma)
        img = np.clip(img, 0, 1)

    # 随机高斯模糊（少量）
    if random.random() < 0.15:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img, mask


def random_crop(img: np.ndarray, mask: np.ndarray, size: int):
    h, w = img.shape[:2]
    # pad 不够大
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
        h, w = img.shape[:2]

    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img[y:y+size, x:x+size], mask[y:y+size, x:x+size]


class GroundDataset(Dataset):
    def __init__(self, img_path: str, mask_path: str, crop_size=256, aug=True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.crop_size = crop_size
        self.aug = aug

        # 预加载
        self.img = cv2.imread(img_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None or self.mask is None:
            raise Exception(f"无法读取图像或掩码: {img_path}, {mask_path}")

        # 调整太大的图像，限制长边
        max_side = 1024
        h, w = self.img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            nh = int(h * scale)
            nw = int(w * scale)
            self.img = cv2.resize(self.img, (nw, nh), interpolation=cv2.INTER_AREA)
            self.mask = cv2.resize(self.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        self.mask = (self.mask > 127).astype(np.uint8)

    def __len__(self):
        return 50 if self.aug else 1

    def __getitem__(self, idx):
        img = self.img.copy()
        mask = self.mask.copy()
        mask = mask.astype(np.float32)

        if self.aug:
            img, mask = random_crop(img, mask, self.crop_size)
            img, mask = random_augment(img, mask)
        else:
            # 验证集不裁剪增强
            pass

        img_norm = normalize(img)
        # (H, W, C) → (C, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()  # 1xHxW

        return img_tensor, mask_tensor


def collect_pairs(img_dir: str, mask_dir: str):
    """收集所有已标注的图像-掩码对"""
    pairs = []
    exts = ["*.jpg", "*.jpeg", "*.png"]
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    img_files = sorted(img_files)

    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{base}.png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))

    if not pairs:
        raise Exception(f"找不到匹配的图像-掩码对，请检查:\n  图像: {img_dir}\n  掩码: {mask_dir}")

    print(f"✅ 找到了 {len(pairs)} 个已标注图像-掩码对")
    return pairs


# ============================================================================
# 评估指标
# ============================================================================

@torch.no_grad()
def compute_metrics(pred_logits, target, th=0.5):
    """计算 IoU, Pixel Accuracy, Precision, Recall, F1-score
    全部用像素级别平均，和评估脚本保持一致
    """
    pred = (torch.sigmoid(pred_logits) > th).float()
    # True Positives, False Positives, False Negatives
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    inter = tp
    union = (pred + target).clamp(0, 1).sum().item()
    if union < 1e-6:
        iou = 0.0
    else:
        iou = inter / union
    correct = (pred == target).sum().item()
    acc = correct / target.numel()
    # Precision
    if tp + fp < 1e-6:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    # Recall
    if tp + fn < 1e-6:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    # F1-score
    if precision + recall < 1e-6:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return iou, acc, precision, recall, f1


def visualize_sample(img, mask_gt, mask_pred, out_path):
    """可视化结果: 原图 | GT | 预测"""
    img = img.permute(1, 2, 0).numpy()
    # 反归一化
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # BGR → RGB
    img = img[..., ::-1]

    mask_gt = mask_gt[0].numpy()
    mask_pred = mask_pred[0] > 0.5

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5))
    a1.imshow(img)
    a1.set_title('Input Image')
    a1.axis('off')

    a2.imshow(mask_gt, cmap='gray', vmin=0, vmax=1)
    a2.set_title('Ground Truth (Ground=white)')
    a2.axis('off')

    a3.imshow(mask_pred, cmap='gray', vmin=0, vmax=1)
    a3.set_title('Prediction')
    a3.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# ============================================================================
# 训练循环 - 梯度上升方式（参考 03_train.py）
# ============================================================================

class FullGroundDataset(Dataset):
    """包含所有标注图像的数据集，每个样本随机从任意图像裁剪"""
    def __init__(self, all_pairs, crop_size=256, aug=True):
        self.all_pairs = all_pairs
        self.crop_size = crop_size
        self.aug = aug
        self.data = []

        # 预加载所有图像
        for img_path, mask_path in all_pairs:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                raise Exception(f"无法读取图像或掩码: {img_path}, {mask_path}")

            # 调整太大的图像，限制长边
            max_side = 1024
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                nh = int(h * scale)
                nw = int(w * scale)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

            mask = (mask > 127).astype(np.uint8)
            self.data.append((img, mask))

        print(f"✅ FullGroundDataset: 预加载了 {len(self.data)} 张图像")

    def __len__(self):
        # 每个epoch 50 * num_images 个样本
        return 50 * len(self.data)

    def __getitem__(self, idx):
        # 随机选一张图
        img, mask = random.choice(self.data)
        mask = mask.astype(np.float32)

        if self.crop_size:
            img, mask = random_crop(img, mask, self.crop_size)

        if self.aug:
            img, mask = random_augment(img, mask)

        img_norm = normalize(img)
        # (H, W, C) → (C, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()  # 1xHxW

        return img_tensor, mask_tensor


def train(args):
    # 收集所有已标注数据
    all_pairs = collect_pairs(args.img_dir, args.mask_dir)
    num_images = len(all_pairs)

    device = torch.device(args.device)
    model = UNet(c_in=3, num_classes=1, base=args.base_ch).to(device)

    # 添加先验偏置：下方1/3区域一般都是地面
    def add_ground_prior(logits):
        B, C, H, W = logits.shape
        prior_bias = torch.zeros_like(logits)
        # 下方 1/3 区域增加偏置，鼓励预测为地面
        prior_bias[:, :, int(H*2/3):, :] += args.prior_bias
        return logits + prior_bias

    # Dice Loss + BCE混合，对类别不平衡更鲁棒 + 加入先验知识：下方1/3区域一般都是地面
    def criterion(logits, target):
        logits = add_ground_prior(logits)
        # BCE with pos weight
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))(logits, target)
        # Dice loss
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return bce + dice

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器：余弦退火，逐步降低学习率减少震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.ckpt_dir, "samples"), exist_ok=True)

    best_iou_global = 0.0
    total_epoch = 0

    # 改用混合所有图训练 - 解决遗忘问题
    print(f"\n📌 使用混合训练模式: 所有 {num_images} 张图混合，每个batch随机采样")
    print(f"   这可以解决'学完新图忘了旧图'的问题\n")

    # 是否从已有模型继续训练
    if args.resume.strip() and os.path.exists(args.resume):
        print(f"\n🔄 从已有模型继续训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"✅ 模型加载完成，继续训练\n")
    elif args.resume.strip():
        print(f"\n⚠️  模型文件不存在: {args.resume}，从头开始训练\n")

    # 增强亮度处理 - 应对亮度差异大
    print(f"训练参数: 批次 {args.batch_size}, 裁剪尺寸 {args.crop_size}, 基础通道 {args.base_ch}")
    print(f"         正样本权重 {args.pos_weight}, 初始学习率 {args.lr}, 下方先验偏置 {args.prior_bias}")
    print(f"         梯度累积 {args.grad_accum} (有效batch = {args.batch_size * args.grad_accum})")
    print(f"         最大epochs: {args.max_epochs}, 目标IoU: {args.target_iou:.1%}")
    print(f"         验证频率: 每 {args.val_freq} 个epoch验证一次")
    print(f"         使用余弦退火学习率衰减，训练后期自动降低学习率减少震荡")
    print("="*80)

    # 创建数据集 - 所有图混合
    ds_train = FullGroundDataset(all_pairs, crop_size=args.crop_size, aug=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)

    # 验证集：保留所有原图做整图验证
    val_datasets = []
    for img_path, mask_path in all_pairs:
        ds_val = GroundDataset(img_path, mask_path, crop_size=None, aug=False)
        val_datasets.append(ds_val)

    # 直接设置目标IoU，达到就提前停止
    target_iou = args.target_iou

    print("\nEpoch │ Train │  Val  │ Acc  │  IoU │ Prec │ Recall │  F1  │   LR   │ Status")
    print("─"*90)

    for epoch in range(1, args.max_epochs+1):
        total_epoch += 1
        model.train()
        t_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i, (img, tgt) in enumerate(tqdm(dl_train, leave=False)):
            img, tgt = img.to(device), tgt.to(device)
            logits = model(img)
            loss = criterion(logits, tgt)
            # 梯度累积
            if args.grad_accum > 1:
                loss = loss / args.grad_accum
            loss.backward()
            t_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 如果还有剩余的批次，做一次更新
        if n_batches % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_tloss = t_loss / n_batches

        # 更新学习率
        scheduler.step()

        # 每个epoch都打印训练损失，定期验证
        # 第一个epoch默认验证一次，让你立即看到初始性能
        if epoch == 1 or epoch % args.val_freq == 0:
            # 验证（在所有验证图上计算平均IoU）
            model.eval()
            v_loss = 0.0
            iou_sum = 0.0
            acc_sum = 0.0
            prec_sum = 0.0
            recall_sum = 0.0
            f1_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for ds_val in val_datasets:
                    img, tgt = ds_val[0]
                    img = img[None].to(device)
                    tgt = tgt[None].to(device)
                    logits = model(img)
                    logits = add_ground_prior(logits)
                    # 这里已经加过先验了，criterion不要再加（避免重复加）
                    # 直接计算loss
                    if args.prior_bias != 0:
                        # BCE with pos weight
                        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))(logits, tgt)
                        # Dice loss
                        pred = torch.sigmoid(logits)
                        intersection = (pred * tgt).sum()
                        dice = 1 - (2 * intersection + 1e-6) / (pred.sum() + tgt.sum() + 1e-6)
                        loss = bce + dice
                    else:
                        loss = criterion(logits, tgt)
                    v_loss += loss.item()
                    iou, acc, prec, recall, f1 = compute_metrics(logits, tgt)
                    iou_sum += iou
                    acc_sum += acc
                    prec_sum += prec
                    recall_sum += recall
                    f1_sum += f1
                    val_count += 1

            avg_vloss = v_loss / val_count
            iou = iou_sum / val_count
            accuracy = acc_sum / val_count
            precision = prec_sum / val_count
            recall = recall_sum / val_count
            f1 = f1_sum / val_count

            # 保存最好的模型
            improved = False
            if iou > best_iou_global:
                best_iou_global = iou
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "ground_best.pth"))
                # 保存可视化样例（最后一张）
                img_sample, tgt_sample = val_datasets[-1][0]
                logits = model(img_sample[None].to(device))
                logits = add_ground_prior(logits)
                out_path = os.path.join(args.ckpt_dir, "samples", f"current_best.png")
                visualize_sample(img_sample, tgt_sample, torch.sigmoid(logits[0]).cpu(), out_path)
                improved = True

            # 打印
            current_lr = optimizer.param_groups[0]['lr']
            if improved:
                print(f"{total_epoch:5d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {iou:.3f} │ {precision:.3f} │ {recall:.3f} │ {f1:.3f} │ {current_lr:.0e} │ ✅ NEW BEST")
            else:
                print(f"{total_epoch:5d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {iou:.3f} │ {precision:.3f} │ {recall:.3f} │ {f1:.3f} │ {current_lr:.0e}")

            # 如果达到目标IoU，提前停止
            if best_iou_global >= target_iou:
                print(f"\n🎉 全局验证：")
                print(f"   Mean IoU = {best_iou_global:.4f}")
                print(f"   Accuracy  = {accuracy:.4f}")
                print(f"   Precision = {precision:.4f}")
                print(f"   Recall    = {recall:.4f}")
                print(f"   F1-score  = {f1:.4f}")
                print(f"\n≥ 目标 {target_iou:.1%}，提前停止全部训练！")
                break

        else:
            # 非验证epoch，只打印训练损失
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{total_epoch:5d} │ {avg_tloss:6.3f} │    -   │   -   │   -   │ {current_lr:.0e} │")

        # end if val_freq
    # end for epoch

    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"最好模型保存在 {os.path.join(args.ckpt_dir, 'ground_best.pth')}")
    print(f"全局验证结果：")
    print(f"   Mean IoU:    {best_iou_global:.4f}")
    # Note: best_iou_global is the best epoch's IoU, the last epoch may be different
    print(f"   (其他指标看最后一次epoch打印)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="ground_data/images", help="原始图像目录")
    ap.add_argument("--mask_dir", default="ground_data/masks", help="标注掩码目录")
    ap.add_argument("--ckpt_dir", default="checkpoints_ground", help="检查点输出目录")
    ap.add_argument("--max_epochs", type=int, default=2000,
                    help="最大训练总epoch数")
    ap.add_argument("--val_freq", type=int, default=1,
                    help="每N个epoch验证一次")
    ap.add_argument("--batch_size", type=int, default=8, help="批次大小")
    ap.add_argument("--crop_size", type=int, default=256, help="随机裁剪尺寸")
    ap.add_argument("--base_ch", type=int, default=32, help="UNet基础通道数")
    ap.add_argument("--lr", type=float, default=1e-4, help="初始学习率（余弦退火自动衰减）")
    ap.add_argument("--pos_weight", type=float, default=6.0, help="BCE正样本权重，地面占比低时增大 (你的数据平均地面占比~15%，推荐6.0)")
    ap.add_argument("--prior_bias", type=float, default=2.0, help="下方1/3区域先验偏置，鼓励预测为地面，越大先验越强")
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    ap.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数，模拟更大batch size，比如 --grad_accum 4 相当于 batch_size*4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", nargs='?', const="checkpoints_ground/ground_best.pth", default="checkpoints_ground/ground_best.pth",
                    help="从已有模型继续训练（默认自动加载 best 模型，传 --resume '' 从头开始）")
    ap.add_argument("--target_iou", type=float, default=0.99,
                    help="目标IoU，达到就提前停止训练（默认99%）")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
