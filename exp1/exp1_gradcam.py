# -*- coding: utf-8 -*-
"""
exp1 Grad-CAM 对比图(免训练, 用 5 个已训练 checkpoint)。

类比 SOD 的"边界对比": 在同一批测试图上, 把五个模型"看哪里"(类激活图)并排可视化,
并标注各自的预测类别(绿=对/红=错)。直观展示不同结构在难样本上关注区域的差异。

产出: exp1/images/exp1_gradcam.png
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

from models import build_model, DISPLAY_NAMES
from train_utils import get_dataloaders, CLASSES, CIFAR10_MEAN, CIFAR10_STD

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORDER = ["cnn", "resnet", "densenet", "mobilenet", "res2net"]
CKPT = {"cnn": "ckpt_cnn.pth", "resnet": "ckpt_resnet.pth", "densenet": "ckpt_densenet.pth",
        "mobilenet": "ckpt_mobilenet.pth", "res2net": "ckpt_res2net.pth"}
MEAN = np.array(CIFAR10_MEAN); STD = np.array(CIFAR10_STD)


def target_module(model, key):
    attr = {"cnn": "conv2", "resnet": "layer4", "densenet": "dense4",
            "mobilenet": "layers", "res2net": "layer4"}[key]
    return getattr(model, attr)


def gradcam(model, x, tmod, class_idx=None):
    store = {}
    def fhook(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))
    h = tmod.register_forward_hook(fhook)
    out = model(x)
    cls = out.argmax(1).item() if class_idx is None else class_idx
    prob = out.softmax(1)[0, cls].item()
    model.zero_grad(); out[0, cls].backward()
    h.remove()
    A, G = store["a"][0], store["g"][0]                  # (C,H,W)
    w = G.mean(dim=(1, 2))
    cam = F.relu((w[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(32, 32), mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.detach().cpu().numpy(), cls, prob


def denorm(x):
    img = x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)


def overlay(img, cam):
    heat = cm.jet(cam)[..., :3]
    return np.clip(0.45 * heat + 0.55 * img, 0, 1)


def pick_images(testset, wanted=("cat", "bird", "dog", "ship", "deer")):
    """每个目标类取第一张测试图。"""
    idxs = {}
    for i in range(len(testset)):
        _, y = testset[i]
        c = CLASSES[y]
        if c in wanted and c not in idxs:
            idxs[c] = i
        if len(idxs) == len(wanted):
            break
    return [idxs[c] for c in wanted]


def main():
    _, testloader = get_dataloaders(batch_size=1, augment=False, num_workers=0,
                                    root=os.path.join(HERE, "data"))
    testset = testloader.dataset
    sel = pick_images(testset)
    models = {}
    for k in ORDER:
        m = build_model(k).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(HERE, CKPT[k]), map_location=DEVICE))
        m.eval(); models[k] = m

    ncol = len(ORDER) + 1
    fig, axes = plt.subplots(len(sel), ncol, figsize=(ncol * 1.7, len(sel) * 1.85))
    for r, idx in enumerate(sel):
        x, y = testset[idx]
        x = x.unsqueeze(0).to(DEVICE)
        img = denorm(x[0])
        ax = axes[r, 0]; ax.imshow(img); ax.axis("off")
        if r == 0:
            ax.set_title("input", fontsize=10)
        ax.set_ylabel(CLASSES[y], fontsize=10, rotation=90)
        ax.text(-0.18, 0.5, f"GT: {CLASSES[y]}", transform=ax.transAxes,
                rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
        for c, k in enumerate(ORDER, start=1):
            cam, cls, prob = gradcam(models[k], x.clone(), target_module(models[k], k))
            ax = axes[r, c]; ax.imshow(overlay(img, cam)); ax.axis("off")
            ok = (cls == y)
            ax.set_title(f"{CLASSES[cls]} {prob*100:.0f}%",
                         fontsize=8.5, color=("#1a8c1a" if ok else "#d62728"))
            if r == 0:
                ax.text(0.5, 1.32, DISPLAY_NAMES[k], transform=ax.transAxes,
                        ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Grad-CAM: where each model looks (title = prediction, green=correct / red=wrong)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "exp1_gradcam.png"), dpi=150, bbox_inches="tight")
    print("saved exp1_gradcam.png | images:", [CLASSES[testset[i][1]] for i in sel])


if __name__ == "__main__":
    main()
