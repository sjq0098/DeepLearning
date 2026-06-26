# -*- coding: utf-8 -*-
"""
exp4 "随机数对生成结果的影响(重点)" 深化 + 卷积加分项深化(免训练, 仅用已存 G)。

产出 (exp4/images/):
  exp4_noise.png     (a) 潜空间球面插值(两个随机噪声之间连续过渡 -> 生成流形是连续映射)
                     (b) 截断技巧(噪声幅度 sigma 由小到大 -> 保真度↓多样性↑, 印证"低密度区"论点)
  exp4_featmaps.png  DCGAN 生成器中间特征图: z -> 7x7 -> 14x14 -> 28x28 的"由粗到细"合成过程
                     (展示对转置卷积上采样的理解)
"""
import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from models import Z_DIM, DCGenerator
from train_utils import set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"font.size": 11})


def load_dc():
    G = DCGenerator().to(DEVICE)
    G.load_state_dict(torch.load(os.path.join(HERE, "ckpt_dc.pth"), map_location=DEVICE))
    G.eval()
    return G


def slerp(a, b, t):
    """球面线性插值(GAN 潜空间插值的常用做法, 比线性更平滑)。"""
    a_n = a / a.norm(); b_n = b / b.norm()
    omega = torch.acos((a_n * b_n).sum().clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


@torch.no_grad()
def to_np_grid(imgs, nrow):
    imgs = (imgs.clamp(-1, 1) * 0.5 + 0.5).cpu()
    return vutils.make_grid(imgs, nrow=nrow, padding=1, pad_value=1)[0].numpy()


@torch.no_grad()
def fig_noise(G, n_steps=9, n_rows=4, scales=(0.2, 0.5, 1.0, 1.5, 2.0), n_col=8):
    # (a) 球面插值
    set_seed(2024)
    interp_imgs = []
    for _ in range(n_rows):
        z0 = torch.randn(Z_DIM, device=DEVICE); z1 = torch.randn(Z_DIM, device=DEVICE)
        zs = torch.stack([slerp(z0, z1, t) for t in np.linspace(0, 1, n_steps)])
        interp_imgs.append(G(zs))
    interp = to_np_grid(torch.cat(interp_imgs, 0), nrow=n_steps)

    # (b) 截断: 不同 sigma
    set_seed(7)
    base = torch.randn(n_col, Z_DIM, device=DEVICE)
    trunc_rows = [G(s * base) for s in scales]
    trunc = to_np_grid(torch.cat(trunc_rows, 0), nrow=n_col)

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_rows, len(scales)], hspace=0.22)
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(interp, cmap="gray"); ax0.axis("off")
    ax0.set_title("(a) Spherical interpolation between two random noise vectors "
                  "(left $\\to$ right): the generator is a smooth, continuous map", fontsize=11)
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(trunc, cmap="gray"); ax1.axis("off")
    ax1.set_title("(b) Truncation: scaling the noise by $\\sigma$ (top$\\to$bottom: "
                  + ", ".join(str(s) for s in scales)
                  + ") trades diversity for fidelity", fontsize=11)
    # 在截断图左侧标注 sigma
    h = trunc.shape[0] / len(scales)
    for i, s in enumerate(scales):
        ax1.text(-6, h * (i + 0.5), f"$\\sigma$={s}", ha="right", va="center", fontsize=9)
    fig.savefig(os.path.join(IMG, "exp4_noise.png"), dpi=150, bbox_inches="tight")
    print("saved exp4_noise.png")


@torch.no_grad()
def fig_featmaps(G, n_ch=8):
    """钩取 7x7 / 14x14 转置卷积(经 ReLU)的中间特征图。"""
    set_seed(123)
    z = torch.randn(1, Z_DIM, device=DEVICE)
    feats = {}
    h7 = G.net[2].register_forward_hook(lambda m, i, o: feats.__setitem__("7x7", o.detach()))
    h14 = G.net[5].register_forward_hook(lambda m, i, o: feats.__setitem__("14x14", o.detach()))
    out = G(z)
    h7.remove(); h14.remove()

    f7 = feats["7x7"][0].cpu().numpy()      # (256,7,7)
    f14 = feats["14x14"][0].cpu().numpy()   # (128,14,14)
    final = (out[0, 0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, n_ch + 2, wspace=0.15, hspace=0.25,
                          width_ratios=[1] * n_ch + [0.4, 2.4])
    for j in range(n_ch):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(f7[j], cmap="viridis"); ax.axis("off")
        if j == 0:
            ax.set_ylabel("7x7", rotation=0)
    for j in range(n_ch):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(f14[j], cmap="viridis"); ax.axis("off")
    fig.text(0.012, 0.72, "stage 1\n256 ch @ 7x7", fontsize=9, va="center")
    fig.text(0.012, 0.28, "stage 2\n128 ch @ 14x14", fontsize=9, va="center")
    axf = fig.add_subplot(gs[:, n_ch + 1]); axf.imshow(final, cmap="gray"); axf.axis("off")
    axf.set_title("output\n1 ch @ 28x28", fontsize=10)
    fig.suptitle("DCGAN generator: transpose-conv upsamples z from coarse 7x7 features "
                 "to a fine 28x28 image (8 of each stage's channels shown)", fontsize=11, y=1.02)
    fig.savefig(os.path.join(IMG, "exp4_featmaps.png"), dpi=150, bbox_inches="tight")
    print("saved exp4_featmaps.png | shapes 7x7:%s 14x14:%s" % (f7.shape, f14.shape))


def main():
    G = load_dc()
    fig_noise(G)
    fig_featmaps(G)


if __name__ == "__main__":
    main()
