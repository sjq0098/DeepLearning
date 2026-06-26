# -*- coding: utf-8 -*-
"""
exp4 生成质量对比图(免训练, 用已存 G)。

类比 SOD 的"边界锐化对比": 用同一组噪声让 FC-GAN 与 DCGAN 生成, 并与真实图并排;
再对同一噪声做局部放大, 直接对比"边缘锐利度/噪点"(全连接 vs 卷积)。

产出: exp4/images/exp4_quality.png
"""
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import Z_DIM, FCGenerator, DCGenerator
from train_utils import set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"font.size": 11})


def load(G, ck):
    G = G.to(DEVICE)
    G.load_state_dict(torch.load(os.path.join(HERE, ck), map_location=DEVICE))
    G.eval(); return G


def grid_np(imgs, nrow):
    imgs = (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()
    return vutils.make_grid(imgs, nrow=nrow, padding=1, pad_value=1)[0].numpy()


@torch.no_grad()
def main():
    set_seed(7)
    Gfc = load(FCGenerator(), "ckpt_fc.pth")
    Gdc = load(DCGenerator(), "ckpt_dc.pth")
    z = torch.randn(16, Z_DIM, device=DEVICE)
    fc = Gfc(z); dc = Gdc(z)

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.FashionMNIST(os.path.join(HERE, "data"), train=True,
                                           download=True, transform=tf)
    real = torch.stack([ds[i][0] for i in range(16)])

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.5], hspace=0.18, wspace=0.08)
    for c, (title, imgs) in enumerate([("Real FashionMNIST", real),
                                       ("FC-GAN (fully-connected)", fc),
                                       ("DCGAN (convolutional)", dc)]):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(grid_np(imgs, 4), cmap="gray")
        ax.axis("off"); ax.set_title(title, fontsize=12)

    # --- 局部放大: 同 4 个噪声, FC vs DCGAN 的边缘细节 ---
    sel = [0, 3, 6, 9]
    def crop_zoom(t):                      # 取中心 16x16 放大
        x = (t.clamp(-1, 1) * 0.5 + 0.5)[0, 0]
        c = x[6:22, 6:22][None, None]
        up = F.interpolate(c, size=(64, 64), mode="nearest")[0, 0]
        return up.cpu().numpy()
    sub = fig.add_subplot(gs[1, :]); sub.axis("off")
    sub.set_title("Same noise, zoomed center crop —  FC-GAN (top) has grainy/noisy edges  vs  "
                  "DCGAN (bottom) sharp & smooth", fontsize=11)
    for j, idx in enumerate(sel):
        axf = fig.add_axes([0.10 + j * 0.20, 0.16, 0.085, 0.13])
        axf.imshow(crop_zoom(fc[idx:idx+1]), cmap="gray"); axf.axis("off")
        if j == 0:
            axf.text(-0.25, 0.5, "FC", transform=axf.transAxes, rotation=90,
                     va="center", fontsize=10, color="#d62728", fontweight="bold")
        axd = fig.add_axes([0.10 + j * 0.20, 0.02, 0.085, 0.13])
        axd.imshow(crop_zoom(dc[idx:idx+1]), cmap="gray"); axd.axis("off")
        if j == 0:
            axd.text(-0.25, 0.5, "DC", transform=axd.transAxes, rotation=90,
                     va="center", fontsize=10, color="#1a8c1a", fontweight="bold")
    fig.savefig(os.path.join(IMG, "exp4_quality.png"), dpi=150, bbox_inches="tight")
    print("saved exp4_quality.png")


if __name__ == "__main__":
    main()
