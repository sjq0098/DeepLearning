# -*- coding: utf-8 -*-
"""
exp4 GAN 模型定义（FashionMNIST，28x28 单通道）。

统一约定：
  - 生成器 forward(z) -> (B, 1, 28, 28)，输出经 Tanh 落在 [-1, 1]
  - 判别器 forward(x) -> (B, 1)，x 形状 (B, 1, 28, 28)，输出经 Sigmoid 为“真”的概率

  - FCGenerator / FCDiscriminator : 全连接 GAN（老师原始版本风格，主线）
  - DCGenerator / DCDiscriminator : 卷积实现（DCGAN，加分项）
"""
import torch
import torch.nn as nn

Z_DIM = 100
IMG_SIZE = 28
IMG_DIM = IMG_SIZE * IMG_SIZE   # 784


# --------------------------- 全连接 GAN（主线）--------------------------- #
class FCGenerator(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, IMG_DIM),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, IMG_SIZE, IMG_SIZE)


class FCDiscriminator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# --------------------------- 卷积 DCGAN（加分项）--------------------------- #
class DCGenerator(nn.Module):
    """转置卷积：z(100,1,1) -> 7x7 -> 14x14 -> 28x28。"""
    def __init__(self, z_dim=Z_DIM, ngf=64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 7, 1, 0, bias=False),   # -> 7x7
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # -> 14x14
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),       # -> 28x28
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)


class DCDiscriminator(nn.Module):
    """卷积：28x28 -> 14x14 -> 7x7 -> 1。"""
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),            # -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),      # -> 7x7
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),        # -> 1x1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def weights_init(m):
    """DCGAN 论文推荐的权重初始化。"""
    cn = m.__class__.__name__
    if 'Conv' in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def build_gan(kind, z_dim=Z_DIM):
    """返回 (G, D)。kind: 'fc' 或 'dc'。"""
    if kind == 'fc':
        return FCGenerator(z_dim), FCDiscriminator()
    elif kind == 'dc':
        G, D = DCGenerator(z_dim), DCDiscriminator()
        G.apply(weights_init); D.apply(weights_init)
        return G, D
    raise ValueError(f"未知类型 '{kind}'，可选 'fc' / 'dc'")
