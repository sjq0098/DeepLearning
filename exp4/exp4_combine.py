# -*- coding: utf-8 -*-
"""
exp4 图表合并(美化排版): 把分散的小图合并成组图。
  gan_losses.png       : FC / DC 两个 GAN 的损失曲线 + D(x),D(G(z))->0.5 (2 panel)
  fc_traversal_full.png : 顶部=8 个自定义噪声样本; 下方=15x8 潜变量扰动(共用 8 列), 一张图
"""
import os, json
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from models import Z_DIM, FCGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"font.size": 11})


def fig_losses():
    hf = json.load(open(os.path.join(HERE, "hist_fc.json")))
    hd = json.load(open(os.path.join(HERE, "hist_dc.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, h, name in [(axes[0], hf, "FC-GAN"), (axes[1], hd, "DCGAN")]:
        ep = np.arange(1, len(h["G_loss"]) + 1)
        ax.plot(ep, h["G_loss"], "-", color="#1f77b4", label="G loss")
        ax.plot(ep, h["D_loss"], "-", color="#d62728", label="D loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ep, h["Dx"], "--", color="#2ca02c", lw=1.5, label="D(x)")
        ax2.plot(ep, h["DGz"], "--", color="#ff7f0e", lw=1.5, label="D(G(z))")
        ax2.axhline(0.5, color="gray", ls=":", lw=1)
        ax2.set_ylabel("D output"); ax2.set_ylim(0, 1)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="center right")
    fig.suptitle("Adversarial training is healthy: D(x), D(G(z)) both converge toward 0.5",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "gan_losses.png"), dpi=150)
    print("saved gan_losses.png")


@torch.no_grad()
def fig_traversal_full(dims=(5, 25, 45, 65, 85), values=(-2.5, 0.0, 2.5), n_base=8):
    G = FCGenerator().to(DEVICE)
    G.load_state_dict(torch.load(os.path.join(HERE, "ckpt_fc.pth"), map_location=DEVICE))
    G.eval()
    torch.manual_seed(123)
    base = torch.randn(n_base, Z_DIM, device=DEVICE)
    samples = G(base)                                   # (8,1,28,28)
    rows = []
    for d in dims:
        for v in values:
            z = base.clone(); z[:, d] = v
            rows.append(G(z))
    trav = torch.cat(rows, 0)                           # (120,1,28,28)

    def grid(t, nrow):
        t = (t.clamp(-1, 1) * 0.5 + 0.5).cpu()
        return vutils.make_grid(t, nrow=nrow, padding=1, pad_value=1)[0].numpy()

    fig = plt.figure(figsize=(7.2, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, len(dims) * len(values)], hspace=0.04)
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(grid(samples, n_base), cmap="gray"); ax0.axis("off")
    ax0.set_title("(a) 8 custom-noise samples (these 8 noises are the 8 columns below)", fontsize=10)
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(grid(trav, n_base), cmap="gray"); ax1.axis("off")
    ax1.set_title("(b) Latent traversal: 5 dims (groups of 3 rows) x values {-2.5, 0, +2.5}",
                  fontsize=10)
    # 在左侧标注每个维度组
    H = grid(trav, n_base).shape[0]
    grp = H / len(dims)
    for k, d in enumerate(dims):
        ax1.text(-3, grp * (k + 0.5), f"dim {d}", rotation=90, va="center", ha="right", fontsize=8)
    fig.savefig(os.path.join(IMG, "fc_traversal_full.png"), dpi=150, bbox_inches="tight")
    print("saved fc_traversal_full.png")


if __name__ == "__main__":
    fig_losses()
    fig_traversal_full()
