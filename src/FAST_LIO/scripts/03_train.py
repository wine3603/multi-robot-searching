#!/usr/bin/env python3
"""
03_train.py — 真正的训练/验证分离：用不同的地图
类别：房间、走廊、墙壁、其他（背景=全0）
"""
import os, glob, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ==============================================================================
# 配置
# ==============================================================================
NUM_CLASSES = 4
LABEL_NAMES = ["房间", "走廊", "墙壁", "其他"]

LABEL_COLORS = np.array([
    [0, 200, 0],        # 0 房间
    [0, 220, 220],      # 1 走廊
    [60, 60, 220],      # 2 墙壁
    [0, 140, 255],      # 3 其他
    [100, 100, 100]     # 背景（仅用于可视化）
], dtype=np.uint8)

# ==============================================================================
# 模型
# ==============================================================================
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
        x = F.pad(x, (0,dw,0,dh))
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self, c_in=1, num_classes=4, base=32):
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

# ==============================================================================
# 数据处理
# ==============================================================================
def norm_occ(occ):
    out = np.zeros_like(occ, np.float32)
    valid = occ >= 0
    out[valid] = occ[valid] / 100.0
    return out

def lbl_to_4channel(lbl):
    h, w = lbl.shape
    out = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    out[0] = ((lbl & 1) != 0)
    out[1] = ((lbl & 2) != 0)
    out[2] = ((lbl & 4) != 0)
    out[3] = ((lbl & 8) != 0)
    return out

def random_crop(img, target, size):
    """同时裁剪图像和目标，确保同一位置！"""
    h, w = img.shape
    # 先pad
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    h, w = img.shape
    # 随机选择裁剪位置
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    # 裁剪
    img_crop = img[y:y+size, x:x+size]
    target_crop = target[:, y:y+size, x:x+size]
    return img_crop, target_crop

class MapDataset(Dataset):
    def __init__(self, pairs, crop=256, aug=True):
        self.pairs = pairs
        self.crop = crop
        self.aug = aug
        self.data = []
        for occ_f, lbl_f in pairs:
            occ = np.flipud(np.load(occ_f)).astype(np.int16)
            lbl = np.flipud(np.load(lbl_f)).astype(np.uint8)
            self.data.append((occ, lbl))

    def __len__(self):
        return len(self.data) * 100 if self.aug else len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        occ, lbl = self.data[idx]

        img = norm_occ(occ)
        target = lbl_to_4channel(lbl)

        if self.crop:
            img, target = random_crop(img, target, self.crop)

        if self.aug:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                target = np.flip(target, axis=-1).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                target = np.flip(target, axis=-2).copy()

        img = torch.from_numpy(img.copy())[None].float()
        target = torch.from_numpy(target.copy()).float()
        return img, target

def collect_all_data(base_dir="map_data_all"):
    """自动从base_dir下收集所有子目录的数据"""
    all_pairs = []
    if not os.path.isdir(base_dir):
        print(f"警告: {base_dir} 目录不存在")
        return all_pairs

    # 如果base_dir下直接有map文件，也收集
    fs = sorted([f for f in glob.glob(os.path.join(base_dir, "map_*.npy")) if "_label" not in f])
    if fs:
        pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
        all_pairs.extend(pairs)
        print(f"从 {base_dir} 加载了 {len(pairs)} 张地图")

    # 收集所有子目录，按编号排序 map_data_1, map_data_2 ...
    subdirs = []
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir) and entry.startswith("map_data_"):
            try:
                idx = int(entry.split("_")[-1])
                subdirs.append((idx, entry, subdir))
            except:
                continue
    # 按编号从小到大排序
    subdirs.sort(key=lambda x: x[0])
    for idx, entry, subdir in subdirs:
        fs = sorted([f for f in glob.glob(os.path.join(subdir, "map_*.npy")) if "_label" not in f])
        if fs:
            pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
            all_pairs.extend(pairs)
            print(f"从 {subdir} 加载了 {len(pairs)} 张地图")
    return all_pairs

def collect_pairs_from_dirs(dirs):
    """从多个目录收集数据对"""
    all_pairs = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        fs = sorted([f for f in glob.glob(os.path.join(d, "map_*.npy")) if "_label" not in f])
        pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
        all_pairs.extend(pairs)
        print(f"从 {d} 加载了 {len(pairs)} 张地图")
    return all_pairs

# ==============================================================================
# 评估 & 可视化
# ==============================================================================
@torch.no_grad()
def compute_metrics(pred, target, th=0.5):
    """计算每个类别的 IoU, Pixel Accuracy, Precision, Recall, F1-score
    全部用像素级别平均，和eval脚本保持一致
    """
    pred_bin = (torch.sigmoid(pred) > th).float()
    metrics = []

    for c in range(NUM_CLASSES):
        p = pred_bin[:, c:c+1]
        t = target[:, c:c+1]
        # True Positives, False Positives, False Negatives
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()

        # IoU
        inter = tp
        union = (p + t).clamp(0, 1).sum().item()
        if union < 1e-6:
            iou = float('nan')
        else:
            iou = inter / union

        # Precision
        if tp + fp < 1e-6:
            precision = float('nan')
        else:
            precision = tp / (tp + fp)

        # Recall
        if tp + fn < 1e-6:
            recall = float('nan')
        else:
            recall = tp / (tp + fn)

        # F1-score
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) < 1e-6:
            f1 = float('nan')
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metrics.append((iou, precision, recall, f1))
    return metrics

@torch.no_grad()
def compute_accuracy(pred, target, th=0.5):
    """计算像素准确率：正确预测的像素比例"""
    pred_bin = (torch.sigmoid(pred) > th)
    correct = (pred_bin == target).sum().item()
    total = target.numel()
    return correct / total

def colorize(lbl):
    c, h, w = lbl.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    has_label = np.zeros((h, w), dtype=bool)
    for cls in range(NUM_CLASSES):
        mask = (lbl[cls] == 1)
        has_label |= mask
        if np.any(mask):
            out[mask] = np.clip(out[mask].astype(np.float32) * 0.5 + LABEL_COLORS[cls] * 0.5, 0, 255).astype(np.uint8)
    bg_mask = ~has_label
    out[bg_mask] = LABEL_COLORS[4]
    return out

@torch.no_grad()
def save_samples(model, val_datasets, device, out_dir):
    """保存所有验证图的可视化对比"""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i, ds_val in enumerate(val_datasets):
        img, tgt = ds_val[0]
        logits = model(img[None].to(device))
        pred = (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4))
        a1.imshow(img[0].numpy(), cmap='gray')
        a2.imshow(colorize(tgt.numpy()))
        a3.imshow(colorize(pred))
        a1.set_title('Input')
        a2.set_title('Label')
        a3.set_title('Pred')
        for a in [a1, a2, a3]:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'map_{i}.png'), dpi=100)
        plt.close()

# ==============================================================================
# 训练
# ==============================================================================
def train(args):
    # 收集所有数据
    if args.data_dirs.strip():
        # 用户指定了目录
        data_dirs = [d.strip() for d in args.data_dirs.split(',')]
        all_pairs = collect_pairs_from_dirs(data_dirs)
    else:
        # 自动发现 map_data_all 下所有数据
        all_pairs = collect_all_data("map_data_all")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: map_data_all/")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: {args.data_dirs}")

    print(f"总共 {len(all_pairs)} 张地图，将按顺序增量训练")
    print("="*80)

class FullMapDataset(Dataset):
    """包含所有地图的数据集，每个样本随机从任意地图裁剪"""
    def __init__(self, all_pairs, crop=256, aug=True):
        self.all_pairs = all_pairs
        self.crop = crop
        self.aug = aug
        self.data = []

        # 预加载所有地图
        for occ_f, lbl_f in all_pairs:
            occ = np.flipud(np.load(occ_f)).astype(np.int16)
            lbl = np.flipud(np.load(lbl_f)).astype(np.uint8)
            self.data.append((occ, lbl))

        print(f"✅ FullMapDataset: 预加载了 {len(self.data)} 张地图")

    def __len__(self):
        # 每个epoch 100 * num_maps 个样本
        return 100 * len(self.data)

    def __getitem__(self, idx):
        # 随机选一张地图
        occ, lbl = random.choice(self.data)
        img = norm_occ(occ)
        target = lbl_to_4channel(lbl)

        if self.crop:
            img, target = random_crop(img, target, self.crop)

        if self.aug:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                target = np.flip(target, axis=-1).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                target = np.flip(target, axis=-2).copy()

        img = torch.from_numpy(img.copy())[None].float()
        target = torch.from_numpy(target.copy()).float()
        return img, target


def train(args):
    # 收集所有数据
    if args.data_dirs.strip():
        # 用户指定了目录
        data_dirs = [d.strip() for d in args.data_dirs.split(',')]
        all_pairs = collect_pairs_from_dirs(data_dirs)
    else:
        # 自动发现 map_data_all 下所有数据
        all_pairs = collect_all_data("map_data_all")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: map_data_all/")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: {args.data_dirs}")

    num_maps = len(all_pairs)
    print(f"\n📌 使用混合训练模式: 所有 {num_maps} 张地图混合，每个batch随机采样")
    print(f"   解决'学完新图忘了旧图'和卡在90%上不去的问题\n")

    device = torch.device(args.device)
    model = UNet(num_classes=4, base=args.base_ch).to(device)

    # BCE + Dice 混合损失，对类别不平衡更鲁棒，更容易提升IoU
    def criterion(logits, target):
        # BCE
        bce = nn.BCEWithLogitsLoss()(logits, target)
        # Dice loss per class
        pred = torch.sigmoid(logits)
        dice_total = 0
        for c in range(NUM_CLASSES):
            p = pred[:, c:c+1]
            t = target[:, c:c+1]
            intersection = (p * t).sum()
            dice = 1 - (2 * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
            dice_total += dice
        return bce + dice_total / NUM_CLASSES

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    total_epoch = 0
    best_miou_global = 0.0

    # 是否从已有模型继续训练
    if args.resume.strip() and os.path.exists(args.resume):
        print(f"\n🔄 从已有模型继续训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"✅ 模型加载完成，继续训练\n")
        if args.initial_iou is not None:
            print(f"📌 使用用户指定的初始mIoU阈值: {args.initial_iou:.3f}")
    elif args.resume.strip():
        print(f"\n⚠️  模型文件不存在: {args.resume}，从头开始训练\n")

    # 打印参数
    print(f"训练参数: 批次 {args.batch_size}, 裁剪尺寸 {args.crop_size}, 基础通道 {args.base_ch}")
    print(f"         学习率 {args.lr}, 最大epochs: {args.max_epochs}")
    print(f"         验证频率: 每 {args.val_freq} 个epoch验证一次")
    if args.initial_iou is not None:
        print(f"         初始mIoU目标: {args.initial_iou:.3f} (用户指定)")
    print("="*80)

    # 创建数据集 - 所有图混合训练（解决遗忘+卡住）
    ds_train = FullMapDataset(all_pairs, crop=args.crop_size, aug=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)

    # 验证集：所有原图整图验证
    val_datasets = []
    for occ_f, lbl_f in all_pairs:
        ds_val = MapDataset([(occ_f, lbl_f)], crop=None, aug=False)
        val_datasets.append(ds_val)

    # 初始目标mIoU（梯度已禁用，目标固定为0.99）
    if args.initial_iou is not None:
        target_miou = args.initial_iou
    else:
        target_miou = 0.99

    print("\nEpoch │ Train │  Val  │ Acc  │ mIoU │ mPrec │ mRecall │ mF1 │ 房间 │ 走廊 │ 墙壁 │ 其他")
    print("─"*110)

    for epoch in range(1, args.max_epochs+1):
        total_epoch += 1
        model.train()
        t_loss = 0.0
        n_batches = 0

        for img, tgt in tqdm(dl_train, leave=False):
            img, tgt = img.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = criterion(logits, tgt)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            n_batches += 1

        avg_tloss = t_loss / n_batches

        # 定期验证，第一个epoch默认验证一次
        if epoch == 1 or epoch % args.val_freq == 0:
            model.eval()
            v_loss = 0.0
            iou_sum = [0.0]*4
            prec_sum = [0.0]*4
            recall_sum = [0.0]*4
            f1_sum = [0.0]*4
            cnt = [0]*4
            acc_sum = 0.0
            acc_cnt = 0

            with torch.no_grad():
                for ds_val in val_datasets:
                    img, tgt = ds_val[0]
                    img = img[None].to(device)
                    tgt = tgt[None].to(device)
                    logits = model(img)
                    loss = criterion(logits, tgt)
                    v_loss += loss.item()
                    metrics = compute_metrics(logits, tgt)
                    acc = compute_accuracy(logits, tgt)
                    acc_sum += acc
                    acc_cnt += 1
                    for c, (iou, prec, recall, f1) in enumerate(metrics):
                        if not np.isnan(iou):
                            iou_sum[c] += iou
                            prec_sum[c] += prec
                            recall_sum[c] += recall
                            f1_sum[c] += f1
                            cnt[c] += 1

            ious = [iou_sum[c]/max(cnt[c], 1) for c in range(4)]
            precs = [prec_sum[c]/max(cnt[c], 1) for c in range(4)]
            recalls = [recall_sum[c]/max(cnt[c], 1) for c in range(4)]
            f1s = [f1_sum[c]/max(cnt[c], 1) for c in range(4)]
            miou = float(np.nanmean(ious))
            mean_prec = float(np.nanmean(precs))
            mean_recall = float(np.nanmean(recalls))
            mean_f1 = float(np.nanmean(f1s))
            accuracy = acc_sum / max(acc_cnt, 1)
            avg_vloss = v_loss / len(val_datasets)

            # 保存最好的模型
            improved = False
            if miou > best_miou_global:
                best_miou_global = miou
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best.pth"))
                # 保存所有验证图的可视化效果
                save_samples(model, val_datasets, device,
                            os.path.join(args.ckpt_dir, "samples"))
                improved = True

            # 打印
            if improved:
                print(f"{total_epoch:4d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {miou:.3f} │ {mean_prec:.3f} │ {mean_recall:.3f} │ {mean_f1:.3f} │ "
                      f"{ious[0]:.3f} │ {ious[1]:.3f} │ {ious[2]:.3f} │ {ious[3]:.3f} │ ✅")
            else:
                print(f"{total_epoch:4d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {miou:.3f} │ {mean_prec:.3f} │ {mean_recall:.3f} │ {mean_f1:.3f} │ "
                      f"{ious[0]:.3f} │ {ious[1]:.3f} │ {ious[2]:.3f} │ {ious[3]:.3f}")

            # 梯度上升已禁用，目标固定
            # if miou >= target_miou and target_miou < 0.99:
            #     old_target = target_miou
            #     target_miou = min(target_miou + 0.03, 0.99)
            #     print(f"\n✅ 达到目标 mIoU {old_target:.1%} → 新目标 {target_miou:.1%}\n")

            # 提前停止
            if best_miou_global > 0.99:
                print(f"\n🎉 全局验证 mIoU = {best_miou_global:.4f} > 0.99，提前停止全部训练！")
                break

        else:
            # 非验证epoch，只打印训练损失
            print(f"{total_epoch:4d} │ {avg_tloss:6.3f} │    -   │   -   │   -   │    -   │    -   │    -   │    -   │    -   ")

    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"最好模型保存在 {args.ckpt_dir}/best.pth")
    print(f"全局最好 mIoU: {best_miou_global:.4f}")
    print(f"所有验证图都已经保存可视化到 {args.ckpt_dir}/samples/")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs",
                    default="",
                    help="数据目录，多个用逗号分隔，留空则自动发现 map_data_all 下所有数据")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--max_epochs", type=int, default=2000,
                    help="最大训练总epoch数")
    ap.add_argument("--val_freq", type=int, default=5,
                    help="每N个epoch验证一次")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", nargs='?', const="checkpoints/best.pth", default="checkpoints/best.pth",
                    help="从已有模型继续训练（默认自动加载 best 模型，传 --resume '' 从头开始）")
    ap.add_argument("--initial_iou", type=float, default=None,
                    help="初始mIoU目标阈值（默认0.3，达到后自动递增到0.98）")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
