# -*- coding: utf-8 -*-
"""
exp1 "训练过程差异 + Res2Net 扩展" 可视化(免训练, 仅用已存 checkpoint + histories.json)。

产出 (exp1/images/):
  exp1_gradflow.png   逐层梯度流(初始化): 隔离 BatchNorm 与跳跃连接对梯度传播的贡献
                      —— 解释"为什么无跳跃的深层普通 CNN 难训练"(重点)
  exp1_efficiency.png 左: FLOPs vs 准确率气泡图(气泡=参数量) ; 右: 收敛速度(到达目标准确率所需 epoch)
  exp1_res2net.png    Res2Net - ResNet 逐类准确率增益(扩展部分: 多尺度在难类上的优势)

诚实性说明:
  老师的 BaselineCNN 很浅(2 卷积+3 全连接), 不足以体现深层梯度消失。为干净地说明跳跃连接的作用,
  左图采用 ResNet 论文式对照: 同样 18 层深度下, 比较 PlainNet(去掉残差相加) 与 ResNet, 并额外
  拆出 BatchNorm 的单独贡献 —— 这才是"无跳跃 vs 有跳跃"的公平对照。
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import build_model, DISPLAY_NAMES, DenseNetCifar
from train_utils import get_dataloaders, CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
ORDER = ["cnn", "resnet", "densenet", "mobilenet", "res2net"]
COLORS = {"cnn": "#7f7f7f", "resnet": "#1f77b4", "densenet": "#2ca02c",
          "mobilenet": "#ff7f0e", "res2net": "#d62728"}


# --------------------------------------------------------------------------- #
# 可控的 18 层网络: 同深度下开关 跳跃连接 / BatchNorm, 隔离各自对梯度流的作用
# --------------------------------------------------------------------------- #
class CtrlBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_skip, use_bn):
        super().__init__()
        self.use_skip, self.use_bn = use_skip, use_bn
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not use_bn)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not use_bn)
        self.b1 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.b2 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.sc = nn.Sequential()
        if use_skip and (stride != 1 or cin != cout):
            layers = [nn.Conv2d(cin, cout, 1, stride, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            self.sc = nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        if self.use_skip:
            out = out + self.sc(x)
        return F.relu(out)


class CtrlNet(nn.Module):
    """18 层: stem + 4 stage x 2 block, 与 ResNet18 同深度。"""
    def __init__(self, use_skip, use_bn, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=not use_bn)
        self.sbn = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        blocks = []
        cin = 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1),
                             (256, 2), (256, 1), (512, 2), (512, 1)]:
            blocks.append(CtrlBlock(cin, cout, stride, use_skip, use_bn)); cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.sbn(self.stem(x)))
        x = self.blocks(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.fc(x)


def layer_grad_profile(net, xb, yb):
    """一次前向+反向, 返回按前向顺序排列的各 Conv2d 权重梯度均值(绝对值)。"""
    net.to(DEVICE).train()
    net.zero_grad()
    loss = nn.CrossEntropyLoss()(net(xb.to(DEVICE)), yb.to(DEVICE))
    loss.backward()
    grads = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.weight.grad is not None:
            grads.append(m.weight.grad.abs().mean().item())
    return np.array(grads)


def fig_gradflow(xb, yb):
    torch.manual_seed(0)
    variants = [
        ("Plain, no BN, no skip", CtrlNet(False, False), "#d62728", "o-"),
        ("Plain + BN, no skip",   CtrlNet(False, True),  "#ff7f0e", "s-"),
        ("ResNet (BN + skip)",    CtrlNet(True,  True),  "#1f77b4", "^-"),
        ("DenseNet (BN + dense)", DenseNetCifar(),       "#2ca02c", "d-"),
    ]
    profiles = []
    for name, net, col, ls in variants:
        g = layer_grad_profile(net, xb, yb)
        profiles.append((name, g, col, ls))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, g, col, ls in profiles:
        xs = np.linspace(0, 1, len(g))
        ax[0].semilogy(xs, g + 1e-12, ls, color=col, ms=4, label=name)
    ax[0].set_xlabel("Relative depth  (0 = first layer / input side)")
    ax[0].set_ylabel("Mean |grad| of conv weights (log)")
    ax[0].set_title("(a) Gradient flow by depth (at init)")
    ax[0].legend(fontsize=8, loc="lower right")

    # (b) 概括: 梯度均匀度 = 各层梯度的 min/max (越接近 1 越均匀; 越小说明有层被"饿死")
    names = [p[0] for p in profiles]
    unif = [p[1].min() / (p[1].max() + 1e-12) for p in profiles]
    cols = [p[2] for p in profiles]
    y = np.arange(len(names))
    ax[1].barh(y, unif, color=cols)
    ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("gradient uniformity  min/max across layers  (log, →1 better)")
    ax[1].set_title("(b) How evenly gradient reaches all layers")
    for yi, r in zip(y, unif):
        ax[1].text(r, yi, f" {r:.1e}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp1_gradflow.png"), dpi=150)
    print("saved exp1_gradflow.png | min/max:",
          {n: f"{r:.1e}" for (n, _, _, _), r in zip(profiles, unif)})


def fig_efficiency(hist):
    try:
        from thop import profile
        have_thop = True
    except Exception:
        have_thop = False
    inp = torch.randn(1, 3, 32, 32).to(DEVICE)
    rows = []
    for k in ORDER:
        net = build_model(k).to(DEVICE).eval()
        params = hist[k]["params"] / 1e6
        acc = hist[k]["best_acc"]
        if have_thop:
            macs, _ = profile(net, inputs=(inp,), verbose=False)
            flops = macs * 2 / 1e6   # MFLOPs
        else:
            flops = np.nan
        rows.append((k, flops, acc, params))
        print(f"  {DISPLAY_NAMES[k]:11s} FLOPs={flops:8.1f}M  acc={acc:.2f}  params={params:.2f}M")

    loff = {"cnn": (8, -4), "resnet": (-20, 14), "densenet": (8, 4),
            "mobilenet": (8, -2), "res2net": (10, -16)}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for k, flops, acc, params in rows:
        ax[0].scatter(flops, acc, s=max(40, params * 55), color=COLORS[k],
                      alpha=0.7, edgecolors="k", linewidths=0.5)
        ax[0].annotate(DISPLAY_NAMES[k], (flops, acc),
                       textcoords="offset points", xytext=loff[k], fontsize=8)
    ax[0].set_xlabel("Compute  (MFLOPs, log)"); ax[0].set_xscale("log")
    ax[0].set_ylabel("Best test accuracy (%)"); ax[0].set_ylim(66, 93)
    ax[0].set_title("(a) Accuracy vs compute  (bubble = #params)")

    # (b) 收敛速度: 到达目标准确率所需 epoch
    target = 80.0
    names, epochs, cols = [], [], []
    for k in ORDER:
        acc_curve = hist[k]["history"]["test_acc"]
        reach = next((i + 1 for i, a in enumerate(acc_curve) if a >= target), None)
        names.append(DISPLAY_NAMES[k]); cols.append(COLORS[k])
        epochs.append(reach if reach else 0)
    x = np.arange(len(names))
    bars = ax[1].bar(x, epochs, color=cols)
    for xi, e in zip(x, epochs):
        ax[1].text(xi, e + 0.3 if e else 0.3,
                   f"{e}" if e else "never", ha="center", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20, fontsize=8)
    ax[1].set_ylabel(f"Epochs to reach {target:.0f}% acc")
    ax[1].set_title("(b) Convergence speed")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp1_efficiency.png"), dpi=150)
    print("saved exp1_efficiency.png")


def fig_res2net(hist):
    classes = list(CLASSES)
    pr = hist["resnet"]["per_class_acc"]; p2 = hist["res2net"]["per_class_acc"]
    delta = np.array([p2[c] - pr[c] for c in classes])
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cols = ["#d62728" if v >= 0 else "#7f7f7f" for v in delta[order]]
    ax.bar(np.arange(len(classes)), delta[order], color=cols)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([classes[i] for i in order], rotation=30, fontsize=9)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-9.5, 13)
    ax.set_ylabel("Per-class acc gain  Res2Net $-$ ResNet (pp)")
    ax.set_title("Res2Net vs ResNet: per-class accuracy is redistributed,\nnot uniformly improved")
    ov = hist["res2net"]["best_acc"] - hist["resnet"]["best_acc"]
    ax.text(0.03, 0.97,
            f"Overall: {ov:+.2f} pp  (ResNet {hist['resnet']['best_acc']:.1f}% "
            f"$\\to$ Res2Net {hist['res2net']['best_acc']:.1f}%)\n"
            f"+{delta.max():.1f} pp on '{classes[int(np.argmax(delta))]}'  /  "
            f"{delta.min():.1f} pp on '{classes[int(np.argmin(delta))]}'",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d0b000", alpha=0.9))
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp1_res2net.png"), dpi=150)
    print("saved exp1_res2net.png | mean gain=%.2fpp  best class gain=%.1fpp (%s)"
          % (delta.mean(), delta.max(), classes[int(np.argmax(delta))]))


def main():
    with open(os.path.join(HERE, "histories.json"), encoding="utf-8") as f:
        hist = json.load(f)
    _, testloader = get_dataloaders(batch_size=128, augment=False, num_workers=0, root=os.path.join(HERE, "data"))
    xb, yb = next(iter(testloader))
    fig_gradflow(xb, yb)
    fig_efficiency(hist)
    fig_res2net(hist)


if __name__ == "__main__":
    main()
