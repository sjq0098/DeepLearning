# -*- coding: utf-8 -*-
"""
exp1 CIFAR10 模型定义。

四个老师要求的网络 + 扩展 Res2Net，全部按 CIFAR10 (3x32x32, 10 类) 适配：
  - BaselineCNN : 老师提供的原始 CNN（无跳跃连接），放在这里只是为了和其它模型用同一套训练协议公平对比
  - ResNet18    : 残差连接
  - DenseNet    : 密集连接 (DenseNet-BC, growth_rate=12)
  - MobileNet   : 深度可分离卷积 (MobileNet V1 风格)
  - Res2Net     : 块内多尺度分组残差 (扩展部分)

统一入口： build_model(name) -> nn.Module
所有分类头前都用 AdaptiveAvgPool2d(1)，因此不依赖具体空间尺寸，改输入大小也不会出错。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 0. Baseline CNN —— 老师原始版本（无跳跃连接）                                  #
# --------------------------------------------------------------------------- #
class BaselineCNN(nn.Module):
    """与 notebook 里老师给的 Net 结构完全一致：2 个卷积 + 3 个全连接。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------------------------------- #
# 1. ResNet (CIFAR 版 ResNet-18，BasicBlock)                                    #
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)          # 残差连接：恒等映射相加
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        # CIFAR stem：3x3 / stride1 / 不接 maxpool（图像只有 32x32，不能像 ImageNet 那样早早下采样）
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


# --------------------------------------------------------------------------- #
# 2. DenseNet (CIFAR 版 DenseNet-BC, growth_rate=12)                            #
# --------------------------------------------------------------------------- #
class _DenseBottleneck(nn.Module):
    """BN-ReLU-Conv1x1(4k) -> BN-ReLU-Conv3x3(k)，输出与输入在通道维拼接。"""
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)      # 密集连接：拼接而非相加
        return out


class _Transition(nn.Module):
    """过渡层：1x1 卷积压缩通道 + 2x2 平均池化下采样。"""
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, nblocks=(6, 6, 6, 6), growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, 3, padding=1, bias=False)

        self.dense1 = self._make_dense(num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = _Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense(num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = _Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense(num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = _Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense(num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense(self, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(_DenseBottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def DenseNetCifar(num_classes=10):
    return DenseNet(nblocks=(6, 6, 6, 6), growth_rate=12, reduction=0.5, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# 3. MobileNet (V1 风格，深度可分离卷积)                                         #
# --------------------------------------------------------------------------- #
class _DepthwiseSeparable(nn.Module):
    """Depthwise 3x3 (groups=in) + Pointwise 1x1，各跟 BN+ReLU。"""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (out_planes, stride)；int 表示 stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(_DepthwiseSeparable(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


# --------------------------------------------------------------------------- #
# 4. Res2Net (扩展部分，块内多尺度分组残差)                                       #
# --------------------------------------------------------------------------- #
class Bottle2neck(nn.Module):
    """Res2Net 基本块：把 bottleneck 中的 3x3 卷积拆成 scale 组，组间做层级残差，
    从而在单个块内获得多尺度感受野。移植自官方实现并按 CIFAR 适配。"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, 3, stride, 1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp = spx[0]
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]          # 层级残差：把前一组的输出加到当前组
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1:
            last = spx[self.nums]
            out = torch.cat((out, last if self.stype == 'normal' else self.pool(last)), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        baseWidth=self.baseWidth, scale=self.scale, stype='stage')]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def Res2NetCifar(num_classes=10):
    return Res2Net(Bottle2neck, [2, 2, 2, 2], baseWidth=26, scale=4, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# 统一入口                                                                      #
# --------------------------------------------------------------------------- #
_BUILDERS = {
    'cnn': BaselineCNN,
    'resnet': ResNet18,
    'densenet': DenseNetCifar,
    'mobilenet': MobileNet,
    'res2net': Res2NetCifar,
}

# notebook 里画图/做表用的展示名
DISPLAY_NAMES = {
    'cnn': 'BaselineCNN',
    'resnet': 'ResNet18',
    'densenet': 'DenseNet',
    'mobilenet': 'MobileNet',
    'res2net': 'Res2Net',
}


def build_model(name, num_classes=10):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"未知模型 '{name}'，可选：{list(_BUILDERS)}")
    return _BUILDERS[key](num_classes=num_classes)
