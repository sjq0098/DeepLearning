# -*- coding: utf-8 -*-
"""
exp4 GAN 数据 / 训练 / 生成 / 可视化工具（FashionMNIST）。

- get_dataloader   : FashionMNIST，归一化到 [-1,1]（与生成器 Tanh 匹配），自动下载
- train_gan        : 标准 GAN 训练（BCE + Adam），记录 G/D loss、D(x)/D(G(z))，每 epoch 存固定噪声快照
- show_grid        : 把一批图像拼成网格显示/保存
- latent_traversal : 潜变量扰动实验——挑若干维度、每维若干取值，生成 (n_dims*n_vals) x n_base 网格
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from models import Z_DIM


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_dataloader(batch_size=128, root="data", num_workers=0):
    """FashionMNIST，ToTensor + Normalize 到 [-1,1]。
    注意：Windows + Jupyter 下 num_workers 必须为 0，否则 DataLoader 多进程会卡死 notebook。"""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True)
    return loader


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_gan(G, D, loader, n_epochs=20, lr=2e-4, betas=(0.5, 0.999), z_dim=Z_DIM,
              device="cpu", fixed_noise=None, label_smooth=0.0, verbose=True):
    """
    标准 GAN 训练。返回 (history, snapshots)：
      history = {'G_loss','D_loss','Dx','DGz'}  每个 epoch 一个均值
      snapshots = [每个 epoch 末用 fixed_noise 生成的图(CPU)]（fixed_noise 为 None 时为空）
    """
    G.to(device); D.to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCELoss()
    history = {'G_loss': [], 'D_loss': [], 'Dx': [], 'DGz': []}
    snapshots = []

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        gl = dl = dx = dgz = 0.0
        nb = 0
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)
            real_lab = torch.full((b, 1), 1.0 - label_smooth, device=device)
            fake_lab = torch.zeros((b, 1), device=device)

            # ---- 训练判别器 D：最大化 log D(x) + log(1 - D(G(z))) ----
            optD.zero_grad()
            out_real = D(real)
            loss_real = bce(out_real, real_lab)
            z = torch.randn(b, z_dim, device=device)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_fake = bce(out_fake, fake_lab)
            lossD = loss_real + loss_fake
            lossD.backward(); optD.step()

            # ---- 训练生成器 G：最大化 log D(G(z)) ----
            optG.zero_grad()
            out = D(fake)
            lossG = bce(out, torch.ones((b, 1), device=device))
            lossG.backward(); optG.step()

            gl += lossG.item(); dl += lossD.item()
            dx += out_real.mean().item(); dgz += out_fake.mean().item(); nb += 1

        history['G_loss'].append(gl / nb); history['D_loss'].append(dl / nb)
        history['Dx'].append(dx / nb); history['DGz'].append(dgz / nb)
        if fixed_noise is not None:
            G.eval()
            with torch.no_grad():
                snapshots.append(G(fixed_noise.to(device)).cpu())
        if verbose:
            print(f"epoch {epoch:2d}/{n_epochs}  lossD={dl/nb:.3f}  lossG={gl/nb:.3f}  "
                  f"D(x)={dx/nb:.3f}  D(G(z))={dgz/nb:.3f}")
    return history, snapshots


@torch.no_grad()
def generate(G, z, device="cpu"):
    """用噪声 z 生成图像，返回 CPU 张量 (N,1,28,28)。"""
    G.eval()
    return G(z.to(device)).cpu()


def show_grid(imgs, nrow=8, title=None, path=None, figsize=None):
    """imgs: (N,1,28,28) in [-1,1]。拼成网格显示，可保存。"""
    imgs = (imgs.detach().cpu().clamp(-1, 1) * 0.5 + 0.5)   # -> [0,1]
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2, pad_value=1)
    npimg = grid[0].numpy()                                  # 灰度取一个通道
    if figsize is None:
        figsize = (nrow * 0.9, (imgs.size(0) / nrow) * 0.9 + 0.5)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, cmap='gray'); plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    return plt.gcf()


def plot_gan_loss(history, title="GAN Loss", path=None):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history['G_loss'], label='G loss')
    plt.plot(history['D_loss'], label='D loss')
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    return plt.gcf()


def latent_traversal(G, base_z, dims, values, device="cpu"):
    """
    潜变量扰动：对每个维度 d in dims、每个取值 v in values，把 base_z 的第 d 维整列改成 v，
    生成 base_z.size(0) 张图。返回 (len(dims)*len(values)*n_base, 1,28,28)，
    按“每个(d,v)一行、共 len(dims)*len(values) 行、每行 n_base 列”排列（配合 show_grid(nrow=n_base)）。
    """
    G.eval()
    rows = []
    for d in dims:
        for v in values:
            z = base_z.clone()
            z[:, d] = float(v)
            with torch.no_grad():
                rows.append(G(z.to(device)).cpu())
    return torch.cat(rows, dim=0)
