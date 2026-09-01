#!/usr/bin/env python3
"""
eval_baseline_vs_ours.py — 对比评估多个模型在 ground_data 上的性能
支持从 model_checkpoints/ 文件夹自动加载所有模型进行对比
评估:
  - IoU, Pixel Accuracy, Precision, Recall, F1-score
输出:
  - 所有模型的性能对比柱状图
  - 汇总结果文本文件
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# 配置和常量
# ============================================================================

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes 标签中，哪些类别算作"地面"
# Cityscapes 19类: 0=road, 1=sidewalk, 都是地面类
CITYSCAPES_GROUND_CLASSES = [0, 1]

# ============================================================================
# 模型定义 - UNet (我们自己的模型结构)
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
        x = F.pad(x, (0,dw,0,dh))
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
# StixelNExT 模型定义 - Copied from StixelNExT project
# ============================================================================

from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth

class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(nn.Conv2d(in_features, out_features, kernel_size=4, stride=4), nn.BatchNorm2d(out_features))

class ConvNexStage(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, depth: int, **kwargs):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )
        self.outfeatures = out_features

class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """
    def __init__(self, in_features: int, out_features: int, kernel_size: int, norm=nn.BatchNorm2d, act=nn.ReLU, **kwargs):
        super().__init__(nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                   padding=kernel_size // 2, **kwargs), norm(out_features), act(),)

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dimensions),
                                  requires_grad=True)
    def forward(self, x):
        return self.gamma[None, ..., None, None] * x

class BottleNeckBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4, drop_p: float = .0,
                 layer_scaler_init_value: float = 1e-6,):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x

class ConvNextEncoder(nn.Module):
    def __init__(self, in_channels: int, stem_features: int, depths: List[int], widths: List[int], drop_p: float = .0,):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)
        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(in_out_widths, depths[1:], drop_probs[1:])
                ],
            ]
        )
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class Head(nn.Module):
    def __init__(self, out_features, out_channels, target_height, target_width):
        super().__init__()
        self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
        self.decoder = nn.Conv2d(out_features, out_channels, kernel_size=1, stride=1)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.upsample(x)
        x = self.decoder(x)
        return self.activation(x)

class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, stem_features=64, depths=[6, 3], widths=[96, 192, 384, 768],
                 drop_p: float = 0.0, out_channels: int = 2, target_height: int = 94, target_width: int = 312):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.decoder = Head(self.encoder.stages[-1].outfeatures, out_channels, target_height, target_width)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # [B, 2, 94, 312] -> return first channel (ground occupancy probability)
        return x

def stixelnext_model():
    """Return a StixelNExT ConvNeXt model with default settings"""
    return ConvNeXt(
        in_channels=3,
        out_channels=2,
        target_height=94,
        target_width=312
    )

# ============================================================================
# Footprints 模型定义 (Niantic CVPR 2020) - Copied from footprints/network.py
# Predicts free space / ground traversable area from a single RGB image
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class FootprintNetwork(nn.Module):

    def __init__(self, pretrained=False):
        super(FootprintNetwork, self).__init__()
        self.encoder = ResnetEncoder(pretrained=pretrained)
        self.mask_decoder = SkipDecoder(apply_sigmoid=False)  # for stability when using BCE loss
        self.depth_decoder = SkipDecoder(apply_sigmoid=True)

    def forward(self, input_image):
        features = self.encoder(input_image)
        mask_outputs = self.mask_decoder(features)
        # Return full resolution segmentation (channel 1 is all ground/free space)
        return mask_outputs['1/1'][:, 1:2, :, :].sigmoid()


class ResnetEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super(ResnetEncoder, self).__init__()

        encoder = resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        del encoder

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.layer0(x)
        self.features.append(x)
        self.features.append(self.layer1(self.features[-1]))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features


class SkipDecoder(nn.Module):

    def __init__(self, apply_sigmoid=True):
        super(SkipDecoder, self).__init__()

        num_ch = [512, 256, 128, 64, 64]

        self.block1 = ConvUpsampleAndConcatBlock(in_ch=512, out_ch=256, use_elu=True, use_bn=False)
        self.block2 = ConvUpsampleAndConcatBlock(in_ch=256, out_ch=128, use_elu=True, use_bn=False)
        self.block3 = ConvUpsampleAndConcatBlock(in_ch=128, out_ch=64, use_elu=True, use_bn=False)
        self.block4 = ConvUpsampleAndConcatBlock(in_ch=64, out_ch=64, use_elu=True, use_bn=False)

        self.outconv1 = OutConvBlock(in_ch=128, out_ch=2, scale=8, apply_sigmoid=apply_sigmoid)
        self.outconv2 = OutConvBlock(in_ch=64, out_ch=2, scale=4, apply_sigmoid=apply_sigmoid)
        self.outconv3 = OutConvBlock(in_ch=64, out_ch=2, scale=2, apply_sigmoid=apply_sigmoid)

        self.outconv4 = nn.Sequential(ConvBlock(in_ch=64, out_ch=32, use_elu=True, use_bn=False),
                                   OutConvBlock(in_ch=32, out_ch=2, scale=1,
                                                apply_sigmoid=apply_sigmoid))

    def forward(self, features):

        outputs = {}
        x = features[-1]

        x = self.block1(x, features[-2])

        x = self.block2(x, features[-3])
        outputs['1/8'] = self.outconv1(x)

        x = self.block3(x, features[-4])
        outputs['1/4'] = self.outconv2(x)

        x = self.block4(x, features[-5])
        outputs['1/2'] = self.outconv3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        outputs['1/1'] = self.outconv4(x)

        return outputs


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_elu=True, use_bn=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.pad = nn.ReflectionPad2d(1)

        if use_elu:
            self.non_lin = nn.ELU(inplace=True)
        else:
            self.non_lin = nn.ReLU(inplace=True)

        self.use_bn = use_bn

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.non_lin(x)

        x = self.pad(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.non_lin(x)

        return x


class ConvUpsampleAndConcatBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_elu=True, use_bn=False):
        super(ConvUpsampleAndConcatBlock, self).__init__()

        self.pre_concat_conv = ConvBlock(in_ch=in_ch, out_ch=out_ch, use_elu=use_elu,
                                         use_bn=use_bn)
        self.post_concat_conv = ConvBlock(in_ch=out_ch*2, out_ch=out_ch, use_elu=use_elu,
                                          use_bn=use_bn)

    def forward(self, x, cat_feats):
        x = self.pre_concat_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, cat_feats], 1)
        x = self.post_concat_conv(x)

        return x


class OutConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, scale, apply_sigmoid=True):
        super(OutConvBlock, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.pad = nn.ReflectionPad2d(1)
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3)

        if apply_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        if self.apply_sigmoid:
            x = self.sigmoid(x)

        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x


def footprints_model():
    """Return a Footprints model with default settings"""
    return FootprintNetwork(pretrained=False)

# ============================================================================
# DeepLabV3+ 模型定义 (for baseline) - Copied directly from VainF/DeepLabV3Plus-Pytorch
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = {}
        x = self.classifier(features)
        x = F.interpolate(x, input_shape, mode='bilinear', align_corners=False)
        result['out'] = x
        return result

# From VainF's mobilenetv2.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = (kernel_size - 1) // 2 * dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, output_stride=16, width_mult=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        if output_stride == 16:
            inverted_residual_setting[5][3] = 1
        elif output_stride == 8:
            inverted_residual_setting[4][3] = 1
            inverted_residual_setting[5][3] = 1

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation = 1
        previous_dilation = 1

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)

            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel

        # Do NOT add the final ConvBNReLU for segmentation - it's only for ImageNet classification and not in this checkpoint
        # Checkpoint counting: low_level_features has indices 0-3 (4 elements), high_level_features has indices 4-17 (14 elements)
        # Total features length = 18 (1 initial conv + 1+2+3+4+3+3+1 = 17 inverted residuals = 18 total)
        self.low_level_features = nn.Sequential(*features[0:4])
        self.high_level_features = nn.Sequential(*features[4:])

    def forward(self, x):
        low = self.low_level_features(x)
        out = self.high_level_features(low)
        return {'out': out, 'low_level': low}

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        for i, conv in enumerate(self.convs):
            if i == 4:
                x_gap = conv(x)
                x_gap = F.interpolate(x_gap, size=size, mode='bilinear', align_corners=False)
                features.append(x_gap)
            else:
                features.append(conv(x))
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[6, 12, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def deeplabv3plus_mobilenet(num_classes=19, output_stride=16):
    backbone = MobileNetV2(output_stride=output_stride)
    if output_stride == 16:
        aspp_dilate = [6, 12, 18]
    else:
        aspp_dilate = [12, 24, 36]
    classifier = DeepLabHeadV3Plus(320, 24, num_classes, aspp_dilate)
    model = DeepLabV3Plus(backbone, classifier)
    return model

# ============================================================================
# 数据处理
# ============================================================================

def normalize_img(img: np.ndarray):
    """BGR uint8 -> normalized tensor 1x3xHxW
    Keep BGR order same as training - cv2.imread outputs BGR, don't convert!
    """
    img_float = img.astype(np.float32) / 255.0
    img_norm = (img_float - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

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
    return pairs

def compute_metrics(pred_bin, target_bin):
    """计算 IoU, Pixel Accuracy, Precision, Recall, F1-score"""
    pred = pred_bin.astype(np.bool_)
    target = target_bin.astype(np.bool_)

    # True Positives, False Positives, False Negatives
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()

    # IoU
    inter = tp
    union = (pred | target).sum()
    if union == 0:
        iou = 0.0
    else:
        iou = inter / union

    # Pixel Accuracy
    correct = (pred == target).sum()
    total = pred.size
    acc = correct / total

    # Precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # F1-score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return float(iou), float(acc), float(precision), float(recall), float(f1)

# ============================================================================
# 模型加载
# ============================================================================

def load_baseline_model(model_path, device):
    """加载 DeepLabV3+ 基线模型"""
    print(f"  ↳ 加载 DeepLabV3+ (mobilenet): {model_path}")
    baseline_ckpt = torch.load(model_path, map_location=device, weights_only=False)
    baseline_model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)

    # Extract state dict
    if 'model_state' in baseline_ckpt:
        state_dict = baseline_ckpt['model_state']
    elif 'model' in next(iter(baseline_ckpt.keys())):
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in baseline_ckpt.items():
            name = k.replace('model.', '')
            state_dict[name] = v
    else:
        state_dict = baseline_ckpt

    # Fix key naming: in checkpoint, high_level_features indices start at 4 (continues from low_level_features)
    # in our model, high_level_features starts at 0, so need to subtract 4 from the index
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'backbone.high_level_features.' in k:
            # pattern: 'backbone.high_level_features.4.conv...' -> we need 'backbone.high_level_features.0.conv...'
            parts = k.split('.')
            idx = int(parts[2])
            new_idx = idx - 4
            parts[2] = str(new_idx)
            new_k = '.'.join(parts)
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    baseline_model.load_state_dict(new_state_dict)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    return baseline_model, "DeepLabV3+"

def load_unet_model(model_path, device, base_ch=32):
    """加载 UNet 模型"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"  ↳ 加载 UNet ({base_ch} base channels): {model_path}")
    model = UNet(c_in=3, num_classes=1, base=base_ch).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model, model_name

def load_stixelnext_model(model_path, device):
    """加载 StixelNExT 模型"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"  ↳ 加载 StixelNExT (ConvNeXt): {model_path}")
    model = stixelnext_model().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model, model_name

def load_footprints_model(model_path, device):
    """加载 Footprints 模型 (Niantic CVPR 2020)"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"  ↳ 加载 Footprints (ResNet34): {model_path}")
    model = footprints_model().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model, model_name

def discover_models(models_dir: str):
    """从文件夹发现所有需要评估的模型"""
    model_paths = []
    extensions = ["*.pth", "*.pt"]
    for ext in extensions:
        model_paths.extend(glob.glob(os.path.join(models_dir, ext)))
    # 按文件名排序
    model_paths = sorted(model_paths)
    return model_paths

# ============================================================================
# 评估单个模型
# ============================================================================

def evaluate_single_model(model, model_name, is_baseline, pairs, device, threshold=0.5, prior_bias=2.0):
    """评估单个模型在所有图像上的性能"""
    import time
    iou_sum = 0.0
    acc_sum = 0.0
    prec_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    count = 0
    results = []

    # 预热 (GPU需要先跑一次热身)
    if pairs:
        img = cv2.imread(pairs[0][0])
        if img is not None:
            h, w = img.shape[:2]
            with torch.no_grad():
                if 'stixel' in model_name.lower() or 'Stixel' in model_name:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    _ = torch.nn.functional.interpolate(img_tensor, size=(376, 1248), mode='bilinear', align_corners=False)
                    _ = model(_)
                elif 'footprint' in model_name.lower() or 'Footprints' in model_name:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    h_fp, w_fp = (h//32 +1)*32, (w//32 +1)*32
                    _ = torch.nn.functional.interpolate(img_tensor, size=(h_fp, w_fp), mode='bilinear', align_corners=False)
                    _ = model(_)
                else:
                    img_tensor = normalize_img(img).to(device)
                    _ = model(img_tensor)

    # 正式推理计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for img_path, mask_path in pairs:
        # 读取图像和GT
        img = cv2.imread(img_path)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue

        # 和训练保持一致：调整太大的图像，限制长边到 1024
        max_side = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            nh = int(h * scale)
            nw = int(w * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # GT: 0=背景, 255=地面 → 转为 0/1
        gt_bin = (gt > 127).astype(np.bool_)
        h, w = img.shape[:2]

        # 推理
        with torch.no_grad():
            if 'stixel' in model_name.lower() or 'Stixel' in model_name:
                # ==============================================
                # StixelNExT 模型 - 和训练时完全同分布处理
                # 1. 图像缩到 376x1248
                # 2. 输出是 94x312 网格
                # 3. GT 也缩到 376x1248 再下采样到 94x312 算IoU
                # ==============================================
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()  # 0-255 range
                img_tensor = img_tensor.unsqueeze(0).to(device)

                # 1. Resize input to StixelNExT training size (376x1248)
                img_tensor_stixel = torch.nn.functional.interpolate(
                    img_tensor, size=(376, 1248), mode='bilinear', align_corners=False
                )
                output = model(img_tensor_stixel)

                # 2. Get occupancy probability (channel 0) - 输出是 94x312
                if output.dim() == 3:  # [2, 94, 312]
                    prob_stixel = output[0, :, :]
                else:  # [1, 2, 94, 312]
                    prob_stixel = output[0, 0, :, :]

                # 3. GT 也做同样处理：缩到 376x1248 -> 下采样到 94x312
                gt_resized = cv2.resize(gt, (1248, 376), interpolation=cv2.INTER_NEAREST)
                gt_stixel = cv2.resize(gt_resized, (312, 94), interpolation=cv2.INTER_AREA)
                gt_bin_stixel = (gt_stixel > 127.5).astype(np.bool_)

                # 4. 直接在 94x312 网格上算IoU，和训练一致！
                pred_bin_stixel = (prob_stixel.cpu().numpy() > 0.1).astype(np.bool_)

                # 5. 仍然上采样用于可视化（不影响IoU计算）
                prob = torch.nn.functional.interpolate(
                    prob_stixel.unsqueeze(0).unsqueeze(0),
                    size=img.shape[:2],
                    mode='nearest'
                )[0, 0].cpu().numpy()
                pred_bin = (prob > 0.1).astype(np.bool_)

                # ========== IoU用网格版本计算，和训练时完全一致 ==========
                iou, acc, prec, recall, f1 = compute_metrics(pred_bin_stixel, gt_bin_stixel)
                results.append({
                    'filename': os.path.splitext(os.path.basename(img_path))[0],
                    'iou': iou,
                    'acc': acc,
                    'precision': prec,
                    'recall': recall,
                    'f1': f1
                })
                iou_sum += iou
                acc_sum += acc
                prec_sum += prec
                recall_sum += recall
                f1_sum += f1
                count += 1
                continue
            elif 'footprint' in model_name.lower() or 'footprints' in model_name.lower():
                # Footprints 模型 - ResNet34 encoder + skip decoder
                # Footprints expects: 0-1 normalized RGB, ImageNet statistics (0.45 mean, 0.225 std)
                # which is already handled inside the model
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()  # (H,W,3) -> (3,H,W)
                img_tensor = img_tensor / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                # Model outputs [1, 1, H, W] with sigmoid already applied
                prob = model(img_tensor)
                prob = prob[0, 0].cpu().numpy()
            elif is_baseline:
                # 基线模型 (DeepLabV3+) - Cityscapes 19类
                img_tensor = normalize_img(img).to(device)
                logits = model(img_tensor)['out']  # [1, 19, H, W]
                # 对cityscapes，road(0)和sidewalk(1)都是地面，累加概率
                probs = torch.softmax(logits, dim=1)
                prob = torch.zeros_like(probs[:, 0, :, :])
                for cls_idx in CITYSCAPES_GROUND_CLASSES:
                    prob += probs[:, cls_idx, :, :]
                prob = prob[0].cpu().numpy()
            else:
                # UNet 模型 - 单类分割
                img_tensor = normalize_img(img).to(device)
                logits = model(img_tensor)
                # 添加地面先验 - 和训练时保持一致
                B, C, H, W = logits.shape
                pb = torch.zeros_like(logits)
                pb[:, :, int(H*2/3):, :] += prior_bias
                logits = logits + pb
                prob = torch.sigmoid(logits)
                prob = prob[0, 0].cpu().numpy()

        pred_bin = (prob > threshold).astype(np.bool_)
        iou, acc, prec, recall, f1 = compute_metrics(pred_bin, gt_bin)

        results.append({
            'filename': os.path.splitext(os.path.basename(img_path))[0],
            'iou': iou,
            'acc': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1
        })

        iou_sum += iou
        acc_sum += acc
        prec_sum += prec
        recall_sum += recall
        f1_sum += f1
        count += 1

    # 结束计时，计算 FPS
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    fps = count / elapsed if elapsed > 0 else 0

    if count == 0:
        return None

    avg_iou = iou_sum / count
    avg_acc = acc_sum / count
    avg_prec = prec_sum / count
    avg_recall = recall_sum / count
    avg_f1 = f1_sum / count

    return {
        'model_name': model_name,
        'is_baseline': is_baseline,
        'avg_iou': avg_iou,
        'avg_acc': avg_acc,
        'avg_precision': avg_prec,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'fps': fps,
        'per_image': results,
        'count': count
    }

# ============================================================================
# 绘制多模型性能对比图
# ============================================================================

def plot_performance_comparison(all_results, out_dir):
    """绘制所有模型的性能对比折线图（更便于跨指标对比）"""
    metrics = [
        ('avg_iou', 'Mean IoU'),
        ('avg_acc', 'Pixel Acc'),
        ('avg_precision', 'Precision'),
        ('avg_recall', 'Recall'),
        ('avg_f1', 'F1-score'),
        ('fps', 'FPS')
    ]

    model_names = [r['model_name'] for r in all_results]
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, 1, figsize=(max(10, n_models * 1.8), 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']  # 不同形状的标记点
    ax2 = axes.twinx()  # 双Y轴用于显示 FPS（范围不同）

    x = np.arange(n_metrics)
    max_fps = max(r['fps'] for r in all_results)

    # 收集所有非FPS指标的值，用于动态设置Y轴范围
    all_non_fps_values = []
    for result in all_results:
        for m, name in metrics:
            if m != 'fps':
                all_non_fps_values.append(result[m])

    min_val = min(all_non_fps_values)
    max_val = max(all_non_fps_values)
    y_min = max(0, min_val - 0.08)  # 下限留出空间
    y_max = min(1, max_val + 0.08)  # 上限留出空间，不超过1

    for i, result in enumerate(all_results):
        values = []
        for m, name in metrics:
            if m == 'fps':
                # FPS归一化到y_min-y_max范围用于绘图，数值标签保留原始值
                val_norm = y_min + (result[m] / max_fps if max_fps > 0 else 0) * (y_max - y_min)
                values.append(val_norm)
            else:
                values.append(result[m])

        # 折线图 + 标记点
        axes.plot(x, values, marker=markers[i % len(markers)], linewidth=2.5,
                  markersize=9, label=result['model_name'], alpha=0.85)

        # 在每个数据点上方添加数值标签
        label_offset = (y_max - y_min) * 0.02
        for j, val in enumerate(values):
            if metrics[j][0] == 'fps':
                # FPS显示真实值
                label_val = result['fps']
                axes.text(x[j], val + label_offset, f'{label_val:.1f}',
                          ha='center', va='bottom', fontsize=9)
            else:
                axes.text(x[j], val + label_offset, f'{val:.3f}',
                          ha='center', va='bottom', fontsize=9)

    axes.set_ylabel('Score', fontsize=11)
    ax2.set_ylabel('FPS (actual value, right axis)', fontsize=11, color='gray')
    ax2.set_ylim(0, max_fps * 1.1)
    ax2.tick_params(axis='y', labelcolor='gray')
    axes.set_title('Performance Comparison of Different Models', fontsize=14, pad=15)
    axes.set_xticks(x)
    axes.set_xticklabels([name for _, name in metrics])
    axes.legend(loc='lower left', fontsize=11)
    axes.set_ylim(y_min, y_max)
    axes.grid(alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "performance_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 性能对比图保存到: {out_path}")

    # 另外单独绘制 IoU 对比柱状图（最重要的指标）
    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), 5))
    iou_values = [r['avg_iou'] for r in all_results]
    bars = ax.bar(model_names, iou_values, color=['#1f77b4' if not r['is_baseline'] else '#ff7f0e' for r in all_results])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.008,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_ylabel('Mean IoU', fontsize=12)
    ax.set_title('Mean IoU Comparison', fontsize=14, pad=15)
    ax.set_ylim(0, max(iou_values) * 1.15)
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    iou_out_path = os.path.join(out_dir, "iou_comparison.png")
    plt.savefig(iou_out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 IoU 对比图保存到: {iou_out_path}")

# ============================================================================
# 主评估函数
# ============================================================================

def evaluate(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # 收集数据
    pairs = collect_pairs(args.img_dir, args.mask_dir)
    print(f"✅ 找到 {len(pairs)} 个图像-GT对\n")

    if len(pairs) == 0:
        print("❌ 没有找到标注数据，请检查路径")
        return

    # 发现所有模型
    if args.models_dir:
        # 自动从文件夹加载所有模型
        model_paths = discover_models(args.models_dir)
        print(f"🔍 在 {args.models_dir} 中发现 {len(model_paths)} 个模型文件:\n")
        for p in model_paths:
            print(f"  - {os.path.basename(p)}")
        print()
    else:
        # 手动指定模型（兼容旧用法
        model_paths = []
        if args.baseline_model:
            model_paths.append(args.baseline_model)
        if args.our_model and args.our_model != args.baseline_model:
            model_paths.append(args.our_model)
        print(f"📋 手动指定 {len(model_paths)} 个模型:\n")
        for p in model_paths:
            print(f"  - {os.path.basename(p)}")
        print()

    if len(model_paths) == 0:
        print("❌ 没有找到任何模型文件")
        return

    # 逐个评估所有模型
    all_results = []
    for model_path in tqdm(model_paths, desc="评估模型进度"):
        print(f"\n🔄 正在评估: {os.path.basename(model_path)}")

        # 判断模型类型
        filename = os.path.basename(model_path).lower()
        is_baseline = 'deeplab' in filename or 'baseline' in filename
        is_stixel = 'stixel' in filename or 'stixelnext' in filename
        is_footprint = 'footprint' in filename or 'footprints' in filename

        if is_baseline:
            model, model_name = load_baseline_model(model_path, device)
        elif is_stixel:
            model, model_name = load_stixelnext_model(model_path, device)
        elif is_footprint:
            model, model_name = load_footprints_model(model_path, device)
        else:
            model, model_name = load_unet_model(model_path, device, args.base_ch)

        result = evaluate_single_model(
            model, model_name, is_baseline, pairs,
            device, args.threshold, args.prior_bias
        )

        if result:
            all_results.append(result)
            print(f"  ↳ 完成! 平均 IoU = {result['avg_iou']:.4f}")

    if len(all_results) == 0:
        print("❌ 没有成功评估任何模型")
        return

    # ========== 汇总结果 ==========
    print("\n" + "="*80)
    print("📊 EVALUATION RESULT (All Models)")
    print("="*80)
    print()
    print(f"{'Model Name':<25} {'Mean IoU':<10} {'Pixel Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'FPS':<8}")
    print("-"*80)
    for r in sorted(all_results, key=lambda x: -x['avg_iou']):
        print(f"{r['model_name']:<25} {r['avg_iou']:<10.4f} {r['avg_acc']:<10.4f} {r['avg_precision']:<10.4f} {r['avg_recall']:<10.4f} {r['avg_f1']:<10.4f} {r['fps']:<8.1f}")
    print()

    # 按 IoU 排序找出最佳模型
    best_result = max(all_results, key=lambda x: x['avg_iou'])
    print(f"🏆 最佳模型: {best_result['model_name']} (IoU = {best_result['avg_iou']:.4f}, FPS = {best_result['fps']:.1f})")
    print()

    # 保存汇总结果
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Multi-Model Evaluation Result\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {args.img_dir}\n")
        f.write(f"Number of images: {len(pairs)}\n")
        f.write(f"Number of models evaluated: {len(all_results)}\n\n")
        f.write(f"{'Model Name':<25} {'Mean IoU':<12} {'Pixel Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'FPS':<10}\n")
        f.write("-"*90 + "\n")
        for r in sorted(all_results, key=lambda x: -x['avg_iou']):
            f.write(f"{r['model_name']:<25} {r['avg_iou']:<12.6f} {r['avg_acc']:<12.6f} {r['avg_precision']:<12.6f} {r['avg_recall']:<12.6f} {r['avg_f1']:<12.6f} {r['fps']:<10.1f}\n")
        f.write("\n")
        f.write(f"Best model: {best_result['model_name']} (IoU = {best_result['avg_iou']:.6f}, FPS = {best_result['fps']:.1f})\n")
    print(f"💾 汇总结果保存到: {summary_path}")

    # 绘制性能对比图
    plot_performance_comparison(all_results, args.out_dir)

    print("\n✅ 评估完成!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="ground_data/images", help="图像目录")
    ap.add_argument("--mask_dir", default="ground_data/masks", help="GT掩码目录")
    ap.add_argument("--models_dir", default="model_checkpoints",
                help="模型存放目录，会自动加载该目录下所有 .pth/.pt 模型进行评估")
    ap.add_argument("--baseline_model", default=None,
                help="[兼容旧版] 单个基线模型路径 (不使用 models_dir 时用)")
    ap.add_argument("--our_model", default=None,
                help="[兼容旧版] 单个我们模型路径 (不使用 models_dir 时用)")
    ap.add_argument("--out_dir", default="eval_result", help="输出目录")
    ap.add_argument("--base_ch", type=int, default=32, help="UNet基础通道数")
    ap.add_argument("--threshold", type=float, default=0.5, help="分割阈值")
    ap.add_argument("--prior_bias", type=float, default=2.0, help="UNet下方区域先验偏置")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 兼容旧版本调用方式：如果没有指定 models_dir，但指定了 baseline_model，就使用单个模型模式
    if args.models_dir == "model_checkpoints" and args.baseline_model is None:
        # 默认检查是否存在模型
        if not glob.glob("model_checkpoints/*.pth") and not glob.glob("model_checkpoints/*.pt"):
            # 向后兼容：使用旧参数
            if os.path.exists("groundbasemodel/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"):
                args.baseline_model = "groundbasemodel/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
            if os.path.exists("checkpoints_ground/ground_best.pth"):
                args.our_model = "checkpoints_ground/ground_best.pth"
            if args.baseline_model or args.our_model:
                args.models_dir = None

    evaluate(args)

if __name__ == "__main__":
    main()
