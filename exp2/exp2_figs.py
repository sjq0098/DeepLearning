# -*- coding: utf-8 -*-
"""
exp2 "为什么 LSTM 优于 RNN" 可视化(全部免训练, 仅用已存 checkpoint + histories.json)。

产出 (exp2/images/):
  exp2_mech.png      左: 梯度沿时间反传(初始化, BPTT 梯度消失/高速路) ; 右: 真实任务准确率 vs 名字长度
  exp2_perclass.png  逐类召回率增益 (LSTM - RNN), 横向条形
  exp2_confdiff.png  混淆矩阵差值图 (LSTM - RNN), 纯 histories.json 后处理(可选)

诚实性说明:
  - 在【训练后】的短名字任务上, RNN 的循环权重谱范数较大, 两者梯度衰减相近, 故梯度消失
    并非该任务上 LSTM 取胜的主因。为说明【架构本身】的差异, 左图在【随机初始化】下度量
    ||d h_T/d h_t||: 普通 RNN 指数消失; 只有遗忘门打开(f≈1)时 LSTM 的细胞态才成"梯度高速路"。
  - 右图给出真实任务的行为证据: LSTM 相对 RNN 的优势随名字长度增大(+1.9pp -> +4.2pp),
    与"更好地保持长程信息"一致。
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import build_model
from train_utils import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
DEVICE = "cpu"
HIDDEN = 128
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def load_models(n_letters, n_classes):
    rnn = build_model("rnn", n_letters, HIDDEN, n_classes)
    rnn.load_state_dict(torch.load(os.path.join(HERE, "ckpt_rnn.pth"), map_location=DEVICE))
    lstm = build_model("lstm", n_letters, HIDDEN, n_classes)
    lstm.load_state_dict(torch.load(os.path.join(HERE, "ckpt_lstm.pth"), map_location=DEVICE))
    rnn.eval(); lstm.eval()
    return rnn, lstm


# --- (1) BPTT 梯度传播(随机初始化), ||d h_T/d h_t|| 随反传距离 d, 按 d=0 归一化 --- #
def grad_propagation_init(n_letters, L=20, n=250, seed=0):
    torch.manual_seed(seed); H = HIDDEN
    r = nn.RNN(n_letters, H)
    Wih, Whh = r.weight_ih_l0.detach(), r.weight_hh_l0.detach()
    bih, bhh = r.bias_ih_l0.detach(), r.bias_hh_l0.detach()

    def lstm_w(fbias):
        lin = nn.Linear(n_letters + H, 4 * H)
        with torch.no_grad():
            if fbias:
                lin.bias[H:2 * H] = fbias        # chunk 顺序 i,f,g,o -> 第二段=遗忘门
        return lin.weight.detach(), lin.bias.detach()

    def decay_rnn():
        acc = np.zeros(L)
        for _ in range(n):
            idx = torch.randint(0, n_letters, (L,)); X = torch.zeros(L, n_letters)
            X[torch.arange(L), idx] = 1.0; X.requires_grad_(True)
            h = torch.zeros(1, H); hs = []
            for t in range(L):
                h = torch.tanh(X[t:t+1] @ Wih.t() + bih + h @ Whh.t() + bhh)
                h.retain_grad(); hs.append(h)
            hs[-1].norm().backward()
            for d in range(L):
                acc[d] += hs[L-1-d].grad.norm().item()
        return acc / n

    def decay_lstm(fbias):
        W, b = lstm_w(fbias); acc = np.zeros(L)
        for _ in range(n):
            idx = torch.randint(0, n_letters, (L,)); X = torch.zeros(L, n_letters)
            X[torch.arange(L), idx] = 1.0; X.requires_grad_(True)
            h = torch.zeros(1, H); c = torch.zeros(1, H); hs = []
            for t in range(L):
                g = torch.cat([X[t:t+1], h], 1) @ W.t() + b
                i, f, gg, o = g.chunk(4, 1)
                i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o); gg = torch.tanh(gg)
                c = f * c + i * gg; h = o * torch.tanh(c)
                h.retain_grad(); hs.append(h)
            hs[-1].norm().backward()
            for d in range(L):
                acc[d] += hs[L-1-d].grad.norm().item()
        return acc / n

    dr = decay_rnn(); dl0 = decay_lstm(0.0); dl1 = decay_lstm(1.0)
    return dr / dr[0], dl0 / dl0[0], dl1 / dl1[0]


# --- (2) 真实任务: 准确率 vs 名字长度 --- #
@torch.no_grad()
def acc_vs_length(rnn, lstm, val_data, edges=((1, 4), (5, 6), (7, 8), (9, 10), (11, 30))):
    def bucket(model):
        accs, ns = [], []
        for lo, hi in edges:
            sub = [(lt, tt) for (lt, tt, _) in val_data if lo <= tt.size(0) <= hi]
            cor = sum(int(model(tt).topk(1)[1][0].item() == lt.item()) for lt, tt in sub)
            accs.append(100 * cor / len(sub)); ns.append(len(sub))
        return np.array(accs), ns
    a_r, n = bucket(rnn); a_l, _ = bucket(lstm)
    labels = [f"{lo}-{hi}" if hi < 30 else f"{lo}+" for lo, hi in edges]
    return labels, a_r, a_l, n


def main():
    train_data, val_data, classes, n_letters = load_data(root=os.path.join(HERE, "data"))
    n_classes = len(classes)
    print(f"classes={n_classes} val={len(val_data)} n_letters={n_letters}")
    rnn, lstm = load_models(n_letters, n_classes)

    # ====== 图1: 机制(梯度) + 行为(长度) ======
    g_rnn, g_l0, g_l1 = grad_propagation_init(n_letters)
    labels, a_r, a_l, ns = acc_vs_length(rnn, lstm, val_data)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    d = np.arange(len(g_rnn))
    ax[0].semilogy(d, g_rnn, "o-", color="#d62728", label="RNN")
    ax[0].semilogy(d, g_l0, "x--", color="#7f7f7f",
                   label="LSTM, forget gate closed ($f\\!\\approx\\!0.5$)")
    ax[0].semilogy(d, g_l1, "s-", color="#1f77b4",
                   label="LSTM, forget gate open ($f\\!\\approx\\!1$)")
    ax[0].set_xlabel("Back-prop distance from output  (0 = last char)")
    ax[0].set_ylabel("$\\|\\partial h_T/\\partial h_t\\|$  (normalised, log)")
    ax[0].set_title("(a) Gradient flow through time (at init)")
    ax[0].legend(fontsize=8)

    x = np.arange(len(labels)); w = 0.38
    ax[1].bar(x - w/2, a_r, w, color="#d62728", label="RNN")
    ax[1].bar(x + w/2, a_l, w, color="#1f77b4", label="LSTM")
    for xi, ar, al in zip(x, a_r, a_l):
        ax[1].text(xi, max(ar, al) + 1.2, f"+{al-ar:.1f}", ha="center",
                   fontsize=8, color="#1f77b4")
    for xi, n in zip(x, ns):
        ax[1].text(xi, 3, f"n={n}", ha="center", fontsize=7, color="white")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("Name length (characters)")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("(b) Accuracy vs name length: LSTM lead grows")
    ax[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp2_mech.png"), dpi=150)
    print("saved exp2_mech.png | grad@d=18 RNN=%.1e LSTM(f1)=%.1e (%.0fx) | gap %.1f->%.1fpp"
          % (g_rnn[18], g_l1[18], g_l1[18]/max(g_rnn[18], 1e-12), a_l[0]-a_r[0], a_l[-1]-a_r[-1]))

    # ====== 图2/3: 逐类召回增益 + 混淆差值 ======
    with open(os.path.join(HERE, "histories.json"), encoding="utf-8") as f:
        hist = json.load(f)
    conf_r = np.array(hist["rnn"]["confusion"]); conf_l = np.array(hist["lstm"]["confusion"])
    recall_delta = (np.diag(conf_l) - np.diag(conf_r)) * 100
    order = np.argsort(recall_delta)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cols = ["#1f77b4" if v >= 0 else "#d62728" for v in recall_delta[order]]
    ax.barh(np.arange(n_classes), recall_delta[order], color=cols)
    ax.set_yticks(np.arange(n_classes)); ax.set_yticklabels([classes[i] for i in order], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Recall gain  LSTM $-$ RNN  (pp)")
    ax.set_title("Per-class recall gain (LSTM $-$ RNN)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp2_perclass.png"), dpi=150)
    print("saved exp2_perclass.png | mean recall gain=%.2fpp, %d/%d classes improved"
          % (recall_delta.mean(), (recall_delta > 0).sum(), n_classes))

    diff = (conf_l - conf_r) * 100
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-30, vmax=30)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion difference (LSTM $-$ RNN), red = LSTM higher")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="prob. diff (pp)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "exp2_confdiff.png"), dpi=150)
    print("saved exp2_confdiff.png")


if __name__ == "__main__":
    main()
