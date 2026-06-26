# -*- coding: utf-8 -*-
"""
exp4 模式崩溃可视化(用 mc_train.py 落盘的数据出图)。

产出 (exp4/images/):
  mc_collapse.png   3 联组图: (a)健康 GAN 样本  (b)崩溃 GAN 样本  (c)类别覆盖直方图
"""
import os, json
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"font.size": 11})


def grid_img(snap, n=32, nrow=8):
    x = (snap[:n].clamp(-1, 1) * 0.5 + 0.5)
    g = vutils.make_grid(x, nrow=nrow, padding=1, pad_value=1)[0].numpy()
    return g


def main():
    data = torch.load(os.path.join(HERE, "mc_data.pt"), weights_only=False)
    res = json.load(open(os.path.join(HERE, "mc_results.json")))
    classes = res["classes"]
    h_last = data["healthy_snaps"][-1]      # (64,1,28,28)
    c_last = data["collapsed_snaps"][-1]
    h_cov = np.array(res["healthy"]["coverage"], float)
    c_cov = np.array(res["collapsed"]["coverage"], float)
    h_cov /= h_cov.sum(); c_cov /= c_cov.sum()

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.18)

    ax0 = fig.add_subplot(gs[0]); ax0.imshow(grid_img(h_last), cmap="gray")
    ax0.set_title("(a) Healthy GAN\nall 10 classes appear"); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(grid_img(c_last), cmap="gray")
    ax1.set_title("(b) Mode collapse\nonly a few classes"); ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    x = np.arange(10); w = 0.4
    ax2.bar(x - w/2, h_cov * 100, w, color="#1f77b4", label=f"healthy (H={res['healthy']['cov_entropy']:.2f})")
    ax2.bar(x + w/2, c_cov * 100, w, color="#d62728", label=f"collapsed (H={res['collapsed']['cov_entropy']:.2f})")
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=90, fontsize=8)
    ax2.set_ylabel("Generated share (%)")
    ax2.set_title("(c) Class coverage of generated samples\n(classified by a FashionMNIST CNN)")
    ax2.axhline(10, color="gray", ls="--", lw=0.8)
    ax2.text(9.3, 11, "uniform 10%", fontsize=7, color="gray", ha="right")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "mc_collapse.png"), dpi=150, bbox_inches="tight")
    print("saved mc_collapse.png")
    nz_h = int((h_cov > 0.01).sum()); nz_c = int((c_cov > 0.01).sum())
    print(f"healthy covers {nz_h}/10 classes, collapsed covers {nz_c}/10 classes")
    print("collapsed dominant classes:",
          [classes[i] for i in np.argsort(-c_cov)[:3]])


if __name__ == "__main__":
    main()
