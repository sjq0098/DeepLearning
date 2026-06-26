# -*- coding: utf-8 -*-
"""
exp4 模式崩溃(mode collapse)对照实验。

训练三样东西并把数据落盘,供 mc_figs.py 出图:
  1) 一个小 FashionMNIST 分类器(用于度量生成样本的"类别覆盖率")
  2) 健康 GAN     : FC 生成器(带 BN), 平衡 1:1 训练, 正常 lr  -> 多样性好
  3) 崩溃 GAN     : FC 生成器(去掉 BN), D 学得慢 / G 多更新(过强), 触发模式崩溃

每个 GAN 都记录:
  - 每个 epoch 末用同一组 fixed_noise 生成的快照(看演化)
  - 每个 epoch 的 batch 内多样性(mean pairwise L2, 越低越崩溃)
  - G/D loss, D(x), D(G(z))
最终用分类器统计 2000 张生成图的类别分布(覆盖率/熵)。

落盘: mc_data.pt(快照+指标), mc_results.json(标量指标),
      ckpt_mc_healthy.pth / ckpt_mc_collapsed.pth / ckpt_classifier.pth
"""
import os, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from models import Z_DIM, FCDiscriminator
from train_utils import get_dataloader, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- 生成器(可选 BN) ----------------------------- #
class FCGen(nn.Module):
    """与 models.FCGenerator 同构,但可关闭 BatchNorm(崩溃组用)。"""
    def __init__(self, z_dim=Z_DIM, hidden=256, use_bn=True):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2, True),
                  nn.Linear(hidden, hidden * 2)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden * 2))
        layers += [nn.LeakyReLU(0.2, True), nn.Linear(hidden * 2, 784), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


# ----------------------------- 分类器 ----------------------------- #
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),  # 7
        )
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128),
                               nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.c(self.f(x))


def train_classifier(epochs=3):
    ck = os.path.join(HERE, "ckpt_classifier.pth")
    net = SmallCNN().to(DEVICE)
    if os.path.exists(ck):
        net.load_state_dict(torch.load(ck, map_location=DEVICE))
        print("[clf] loaded existing classifier")
        return net
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.FashionMNIST(os.path.join(HERE, "data"), train=True,
                                           download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    opt = optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        net.train(); correct = total = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); out = net(x); loss = lossf(out, y)
            loss.backward(); opt.step()
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
        print(f"[clf] epoch {ep+1}/{epochs} acc={correct/total:.4f}")
    torch.save(net.state_dict(), ck)
    return net


@torch.no_grad()
def batch_diversity(G, n=64):
    """batch 内平均两两 L2 距离(像素空间, [0,1])。越低=越崩溃。"""
    G.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    x = (G(z).clamp(-1, 1) * 0.5 + 0.5).view(n, -1).cpu().numpy()
    d = 0.0; k = 0
    for i, j in itertools.combinations(range(n), 2):
        d += float(np.linalg.norm(x[i] - x[j])); k += 1
    return d / k


@torch.no_grad()
def class_coverage(G, clf, n=2000):
    """生成 n 张图,用分类器预测类别,返回 10 维计数。"""
    G.eval(); clf.eval()
    counts = np.zeros(10, dtype=int)
    done = 0
    while done < n:
        b = min(256, n - done)
        z = torch.randn(b, Z_DIM, device=DEVICE)
        pred = clf(G(z)).argmax(1).cpu().numpy()
        for p in pred:
            counts[p] += 1
        done += b
    return counts


def train_gan_custom(tag, use_bn, lrG, lrD, g_steps, fixed_noise, clf,
                     n_epochs=20, batch_size=128):
    """训练一个 GAN, 记录快照/多样性/loss。g_steps=每个 batch 里 G 更新几次。"""
    set_seed(42)
    loader = get_dataloader(batch_size=batch_size, root=os.path.join(HERE, "data"))
    G = FCGen(use_bn=use_bn).to(DEVICE)
    D = FCDiscriminator().to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=lrG, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lrD, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    hist = {"G_loss": [], "D_loss": [], "Dx": [], "DGz": [], "diversity": []}
    snaps = []
    for ep in range(1, n_epochs + 1):
        G.train(); D.train()
        gl = dl = dx = dgz = 0.0; nb = 0
        for real, _ in loader:
            real = real.to(DEVICE); b = real.size(0)
            # D step
            optD.zero_grad()
            out_real = D(real); loss_real = bce(out_real, torch.ones(b, 1, device=DEVICE))
            z = torch.randn(b, Z_DIM, device=DEVICE); fake = G(z)
            out_fake = D(fake.detach()); loss_fake = bce(out_fake, torch.zeros(b, 1, device=DEVICE))
            lossD = loss_real + loss_fake; lossD.backward(); optD.step()
            # G step(s)
            for _ in range(g_steps):
                optG.zero_grad()
                z = torch.randn(b, Z_DIM, device=DEVICE); fake = G(z)
                out = D(fake); lossG = bce(out, torch.ones(b, 1, device=DEVICE))
                lossG.backward(); optG.step()
            gl += lossG.item(); dl += lossD.item()
            dx += out_real.mean().item(); dgz += out_fake.mean().item(); nb += 1
        hist["G_loss"].append(gl / nb); hist["D_loss"].append(dl / nb)
        hist["Dx"].append(dx / nb); hist["DGz"].append(dgz / nb)
        hist["diversity"].append(batch_diversity(G))
        G.eval()
        with torch.no_grad():
            snaps.append(G(fixed_noise.to(DEVICE)).cpu())
        print(f"[{tag}] ep{ep:2d} lossD={dl/nb:.3f} lossG={gl/nb:.3f} "
              f"Dx={dx/nb:.3f} DGz={dgz/nb:.3f} div={hist['diversity'][-1]:.3f}")
    cov = class_coverage(G, clf)
    torch.save(G.state_dict(), os.path.join(HERE, f"ckpt_mc_{tag}.pth"))
    return G, hist, torch.stack(snaps), cov


def main():
    set_seed(42)
    fixed_noise = torch.randn(64, Z_DIM)
    clf = train_classifier()
    print("\n==== 训练健康 GAN ====")
    _, h_hist, h_snaps, h_cov = train_gan_custom(
        "healthy", use_bn=True, lrG=2e-4, lrD=2e-4, g_steps=1,
        fixed_noise=fixed_noise, clf=clf)
    print("\n==== 训练崩溃 GAN ====")
    _, c_hist, c_snaps, c_cov = train_gan_custom(
        "collapsed", use_bn=False, lrG=4e-4, lrD=5e-5, g_steps=3,
        fixed_noise=fixed_noise, clf=clf)

    def entropy(c):
        p = c / c.sum(); p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    results = {
        "healthy": {"hist": h_hist, "coverage": h_cov.tolist(),
                    "cov_entropy": entropy(h_cov),
                    "final_diversity": h_hist["diversity"][-1]},
        "collapsed": {"hist": c_hist, "coverage": c_cov.tolist(),
                      "cov_entropy": entropy(c_cov),
                      "final_diversity": c_hist["diversity"][-1]},
        "classes": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "max_entropy": float(np.log(10)),
    }
    with open(os.path.join(HERE, "mc_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save({"fixed_noise": fixed_noise,
                "healthy_snaps": h_snaps, "collapsed_snaps": c_snaps,
                "healthy_cov": h_cov, "collapsed_cov": c_cov},
               os.path.join(HERE, "mc_data.pt"))
    print("\n==== DONE ====")
    print(f"healthy : entropy={results['healthy']['cov_entropy']:.3f} "
          f"div={results['healthy']['final_diversity']:.3f} cov={h_cov.tolist()}")
    print(f"collapsed: entropy={results['collapsed']['cov_entropy']:.3f} "
          f"div={results['collapsed']['final_diversity']:.3f} cov={c_cov.tolist()}")


if __name__ == "__main__":
    main()
