#!/usr/bin/env python3
"""
04_eval.py — 性能评估脚本
对 map_data_all 下所有地图进行推理，对比标注真值，计算指标并绘图

输出:
  推理帧率 (FPS)
  平均 IoU / mIoU
  各类别 IoU / Precision / Recall / F1 / Accuracy
  可视化对比图 (每张地图: 原图 | GT | Pred)

用法:
  python3 04_eval.py
  python3 04_eval.py --model checkpoints/best.pth
  python3 04_eval.py --model checkpoints/best.pth --base_ch 32
"""
import argparse, os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════
#  模型定义（与 03_train.py / 04_inference.py 一致）
# ══════════════════════════════════════════════════════════

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
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        b = base
        self.e1 = DoubleConv(in_ch, b)
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

NUM_CLASSES = 4
LABEL_NAMES = ["Room", "Corridor", "Wall", "Other"]
LABEL_NAMES_CN = ["房间", "走廊", "墙壁", "其他"]
LABEL_COLORS_RGB = [
    (0, 200, 0),      # Room - green
    (0, 220, 220),    # Corridor - cyan
    (60, 60, 220),    # Wall - red
    (0, 140, 255),    # Other - orange
    (100, 100, 100),  # Background - gray
]
WALL_THR = 50

# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def norm_occ(occ):
    out = np.full(occ.shape, 0.5, dtype=np.float32)
    valid = occ >= 0
    out[valid] = occ[valid].astype(np.float32) / 100.0
    return out

def pad_to_multiple(img, multiple=16):
    h, w = img.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    return img, (ph, pw)

def lbl_to_4channel(lbl):
    h, w = lbl.shape
    out = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    out[0] = ((lbl & 1) != 0)
    out[1] = ((lbl & 2) != 0)
    out[2] = ((lbl & 4) != 0)
    out[3] = ((lbl & 8) != 0)
    return out

def colorize(lbl_4ch):
    """(4, H, W) → (H, W, 3) RGB"""
    c, h, w = lbl_4ch.shape
    out = np.full((h, w, 3), LABEL_COLORS_RGB[4], dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        mask = lbl_4ch[cls] > 0.5
        if np.any(mask):
            out[mask] = np.clip(
                out[mask].astype(np.float32) * 0.5 +
                np.array(LABEL_COLORS_RGB[cls], dtype=np.float32) * 0.5,
                0, 255
            ).astype(np.uint8)
    return out

def occ_to_rgb(occ):
    rgb = np.full((*occ.shape, 3), 100, dtype=np.uint8)
    rgb[occ == 0] = 220
    rgb[occ >= WALL_THR] = 35
    m = (occ > 0) & (occ < WALL_THR)
    v = np.clip(220 - occ[m].astype(np.float32)*1.8, 60, 220).astype(np.uint8)
    rgb[m] = np.stack([v, v, v], axis=-1)
    return rgb

def collect_all_data(base_dir):
    all_pairs = []
    if not os.path.isdir(base_dir):
        print(f"Warning: {base_dir} not found")
        return all_pairs
    # Direct files
    fs = sorted([f for f in glob.glob(os.path.join(base_dir, "map_*.npy")) if "_label" not in f])
    if fs:
        pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
        all_pairs.extend(pairs)
    # Subdirs
    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            fs = sorted([f for f in glob.glob(os.path.join(subdir, "map_*.npy")) if "_label" not in f])
            if fs:
                pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
                all_pairs.extend(pairs)
    return all_pairs


# ══════════════════════════════════════════════════════════
#  指标计算
# ══════════════════════════════════════════════════════════

def compute_per_class_metrics(pred, target, th=0.5):
    """
    pred: (4, H, W) float probabilities
    target: (4, H, W) float binary
    Returns per-class: IoU, Precision, Recall, F1, Accuracy
    Classes not present in target are marked as 'absent' (excluded from averages)
    """
    pred_bin = (pred > th).astype(np.float32)
    results = []
    for c in range(NUM_CLASSES):
        p = pred_bin[c]
        t = target[c]
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tn = ((1 - p) * (1 - t)).sum()

        # Check if this class exists in target
        has_gt = (t.sum() > 0)

        if not has_gt:
            results.append({
                'iou': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'accuracy': float('nan'),
                'absent': True,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            })
            continue

        # IoU
        inter = tp
        union = tp + fp + fn
        iou = inter / (union + 1e-8)

        # Precision / Recall / F1
        prec = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * recall / (prec + recall + 1e-8)

        # Pixel Accuracy for this class
        total = t.sum() + (1 - t).sum()
        correct = tp + tn
        acc = correct / (total + 1e-8)

        results.append({
            'iou': float(iou),
            'precision': float(prec),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(acc),
            'absent': False,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        })
    return results


def compute_pixel_accuracy(pred, target, th=0.5):
    """Overall pixel accuracy (all classes combined)"""
    pred_bin = (pred > th).astype(np.float32)
    correct = (pred_bin == target).sum()
    total = target.size
    return float(correct / total)


# ══════════════════════════════════════════════════════════
#  评估主函数
# ══════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device(args.device)

    # Load model
    print(f"Loading model: {args.model}")
    model = UNet(in_ch=1, num_classes=4, base=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded (base_ch={args.base_ch}, device={device})")

    # Collect data
    all_pairs = collect_all_data(args.data_dir)
    if not all_pairs:
        print(f"No data found in {args.data_dir}")
        return
    print(f"Found {len(all_pairs)} maps with labels\n")

    # Per-map results
    all_results = []
    all_fps = []

    for i, (occ_path, lbl_path) in enumerate(all_pairs):
        name = os.path.basename(occ_path)

        # Load data
        occ = np.flipud(np.load(occ_path).astype(np.int16))
        lbl = np.flipud(np.load(lbl_path).astype(np.uint8))
        h, w = occ.shape
        target = lbl_to_4channel(lbl)  # (4, H, W)

        # Preprocess (same as 03_train.py: full occupancy, no binarization)
        img, (ph, pw) = pad_to_multiple(norm_occ(occ), 16)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        # Inference with timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(tensor)
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()[:, :h, :w]

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0

        # Compute metrics
        per_class = compute_per_class_metrics(pred_probs, target, args.threshold)
        pixel_acc = compute_pixel_accuracy(pred_probs, target, args.threshold)
        # mIoU: only over classes present in this map's GT
        valid_ious = [r['iou'] for r in per_class if not r.get('absent', False)]
        miou = float(np.nanmean(valid_ious)) if valid_ious else 0.0

        result = {
            'name': name,
            'fps': fps,
            'elapsed_ms': elapsed * 1000,
            'pixel_acc': pixel_acc,
            'miou': miou,
            'per_class': per_class,
            'occ': occ,
            'target': target,
            'pred': pred_probs,
        }
        all_results.append(result)
        all_fps.append(fps)

        # Print per-map result
        print(f"[{i+1}/{len(all_pairs)}] {name}")
        print(f"  FPS: {fps:.1f} | Time: {elapsed*1000:.1f}ms | mIoU: {miou:.4f} | PixelAcc: {pixel_acc:.4f}")
        for c in range(NUM_CLASSES):
            r = per_class[c]
            if r.get('absent', False):
                print(f"  {LABEL_NAMES[c]:8s}: (not in GT, skipped)")
            else:
                print(f"  {LABEL_NAMES[c]:8s}: IoU={r['iou']:.4f}  Prec={r['precision']:.4f}  "
                      f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")
        print()

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    avg_fps = np.mean(all_fps)
    avg_miou = np.mean([r['miou'] for r in all_results])
    avg_pixel_acc = np.mean([r['pixel_acc'] for r in all_results])
    avg_per_class = {}
    for c in range(NUM_CLASSES):
        valid = [r['per_class'][c] for r in all_results if not r['per_class'][c].get('absent', False)]
        if valid:
            avg_per_class[c] = {
                'iou': np.mean([v['iou'] for v in valid]),
                'precision': np.mean([v['precision'] for v in valid]),
                'recall': np.mean([v['recall'] for v in valid]),
                'f1': np.mean([v['f1'] for v in valid]),
                'accuracy': np.mean([v['accuracy'] for v in valid]),
                'count': len(valid),
            }
        else:
            avg_per_class[c] = {
                'iou': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'f1': float('nan'),
                'accuracy': float('nan'), 'count': 0,
            }

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Maps evaluated: {len(all_results)}")
    print(f"Avg FPS:    {avg_fps:.1f}")
    print(f"Avg mIoU:   {avg_miou:.4f}")
    print(f"Avg PxAcc:  {avg_pixel_acc:.4f}")
    print()
    print(f"{'Class':<10s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Acc':>8s} {'Maps':>6s}")
    print("-" * 60)
    for c in range(NUM_CLASSES):
        r = avg_per_class[c]
        n = r.get('count', '?')
        if np.isnan(r['iou']):
            print(f"{LABEL_NAMES[c]:<10s}   (no GT in any map)")
        else:
            print(f"{LABEL_NAMES[c]:<10s} {r['iou']:8.4f} {r['precision']:8.4f} {r['recall']:8.4f} {r['f1']:8.4f} {r['accuracy']:8.4f} {n:>6d}")
    print()

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Single combined plot: FPS, mIoU, Room/Corridor/Wall/Other IoU ---
    names_short = [f"Map {i+1}" for i in range(len(all_results))]
    x = np.arange(len(all_results))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f'Model Evaluation  |  Avg FPS: {avg_fps:.1f}  Avg mIoU: {avg_miou:.4f}  Avg PxAcc: {avg_pixel_acc:.4f}',
                fontsize=13, fontweight='bold')

    # --- Top: FPS bar chart ---
    ax_fps = axes[0]
    fps_vals = [r['fps'] for r in all_results]
    bars = ax_fps.bar(names_short, fps_vals, color='steelblue', edgecolor='black', linewidth=0.5)
    ax_fps.axhline(y=avg_fps, color='red', linestyle='--', linewidth=1.5, label=f'Avg: {avg_fps:.1f} FPS')
    ax_fps.set_ylabel('FPS')
    ax_fps.set_title('Inference FPS', fontsize=11)
    ax_fps.set_ylim(0, 25)
    ax_fps.legend(loc='upper right')
    for bar, val in zip(bars, fps_vals):
        ax_fps.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # --- Bottom: mIoU + 4 class IoU lines ---
    ax_iou = axes[1]
    miou_vals = [r['miou'] for r in all_results]
    ax_iou.plot(names_short, miou_vals, 's-', color='black', linewidth=2.5, markersize=8,
               label=f'mIoU (avg: {avg_miou:.4f})', zorder=5)

    class_colors = ['green', 'cyan', 'red', 'orange']
    for c in range(NUM_CLASSES):
        vals = []
        for r in all_results:
            v = r['per_class'][c]['iou']
            vals.append(v if not r['per_class'][c].get('absent', False) else float('nan'))
        avg_c = avg_per_class[c]['iou']
        n = avg_per_class[c].get('count', '?')
        if np.isnan(avg_c):
            continue
        ax_iou.plot(names_short, vals, 'o-', color=class_colors[c],
                   linewidth=1.8, markersize=5,
                   label=f'{LABEL_NAMES[c]} IoU (avg: {avg_c:.4f}, n={n})')
        # Mark absent maps with a small 'x'
        for j, v in enumerate(vals):
            if np.isnan(v):
                ax_iou.plot(j, 0.02, 'x', color=class_colors[c], markersize=8, markeredgewidth=2)

    ax_iou.set_ylabel('IoU')
    ax_iou.set_title('mIoU & Per-Class IoU per Map', fontsize=11)
    ax_iou.legend(loc='lower left', ncol=3, fontsize=9)
    ax_iou.set_ylim(0.75, 1.02)
    ax_iou.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "eval_overview.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # --- Summary JSON ---
    import json
    summary = {
        'num_maps': len(all_results),
        'avg_fps': avg_fps,
        'avg_miou': avg_miou,
        'avg_pixel_accuracy': avg_pixel_acc,
        'per_class': {LABEL_NAMES[c]: avg_per_class[c] for c in range(NUM_CLASSES)},
        'per_map': [{
            'name': r['name'],
            'fps': r['fps'],
            'miou': r['miou'],
            'pixel_acc': r['pixel_acc'],
            'per_class': {LABEL_NAMES[c]: r['per_class'][c] for c in range(NUM_CLASSES)},
        } for r in all_results],
    }
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")

    print(f"\n{'='*80}")
    print(f"Evaluation complete! All results saved to: {args.output_dir}/")
    print(f"{'='*80}")


def main():
    ap = argparse.ArgumentParser("04_eval.py - Performance evaluation")
    ap.add_argument("--model", default="checkpoints/best.pth", help="Model weights path")
    ap.add_argument("--base_ch", type=int, default=32, help="UNet base channels")
    ap.add_argument("--data_dir", default="map_data_all", help="Data directory")
    ap.add_argument("--output_dir", default="eval_results", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
