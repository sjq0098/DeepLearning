# -*- coding: utf-8 -*-
"""
exp2 "token 输出图"(免训练): 把名字逐字符喂入分类器, 画出预测语言的概率如何随
每个字符演化 —— 直观展示 RNN vs LSTM 在"读到第几个字符时才看懂这是哪国名字"。

对每个前缀 name[:L] 跑一次模型, 得到 P(语言 | 前 L 个字符), 画成热力图
(行=候选语言, 列=字符位置, 颜色=概率), RNN 与 LSTM 并排对比。

产出: exp2/images/exp2_tokens.png
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import build_model
from train_utils import load_data, allowed_characters, lineToTensor

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
DEVICE = "cpu"; HID = 128
plt.rcParams.update({"font.size": 10})


def load_models(nl, nc):
    rnn = build_model("rnn", nl, HID, nc)
    rnn.load_state_dict(torch.load(os.path.join(HERE, "ckpt_rnn.pth"), map_location=DEVICE))
    lstm = build_model("lstm", nl, HID, nc)
    lstm.load_state_dict(torch.load(os.path.join(HERE, "ckpt_lstm.pth"), map_location=DEVICE))
    rnn.eval(); lstm.eval(); return rnn, lstm


def name_from_tensor(t):
    idx = t.squeeze(1).argmax(1).tolist()
    return "".join(allowed_characters[i] for i in idx)


@torch.no_grad()
def trajectory(model, name):
    """返回 (len, n_classes) 的每前缀概率。"""
    probs = []
    for L in range(1, len(name) + 1):
        out = model(lineToTensor(name[:L]).to(DEVICE))
        probs.append(out.exp()[0].cpu().numpy())
    return np.array(probs)                       # (L, C)


def pick_names(rnn, lstm, val_data, classes, n=3):
    """挑选: 长度 6-10、LSTM 预测正确、且 RNN 与 LSTM 轨迹差异较大的名字。"""
    cand = []
    seen_lang = set()
    for lt, tt, lab in val_data:
        L = tt.size(0)
        if not (6 <= L <= 10):
            continue
        name = name_from_tensor(tt)
        true = lt.item()
        pr = trajectory(rnn, name); pl = trajectory(lstm, name)
        if pl[-1].argmax() != true:               # 要求 LSTM 最终正确
            continue
        # 差异度量: LSTM 对真类的平均置信 - RNN 的
        score = pl[:, true].mean() - pr[:, true].mean()
        if lab in seen_lang:
            continue
        cand.append((score, name, true, lab))
        seen_lang.add(lab)
    cand.sort(reverse=True)
    return cand[:n]


def main():
    train_data, val_data, classes, nl = load_data(root=os.path.join(HERE, "data"))
    nc = len(classes)
    rnn, lstm = load_models(nl, nc)
    picks = pick_names(rnn, lstm, val_data, classes, n=3)
    print("picked:", [(p[1], p[3]) for p in picks])

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 3.0 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for r, (_, name, true, lab) in enumerate(picks):
        pr = trajectory(rnn, name); pl = trajectory(lstm, name)
        # 选取要显示的候选语言: 两模型轨迹里出现过的 top 概率类
        topk = np.argsort(-(np.r_[pr, pl].max(0)))[:6]
        if true not in topk:
            topk = np.r_[[true], topk[:5]]
        topk = list(dict.fromkeys(topk.tolist()))
        rows = [classes[i] for i in topk]
        for c, (mat, mname) in enumerate([(pr, "RNN"), (pl, "LSTM")]):
            ax = axes[r, c]
            im = ax.imshow(mat[:, topk].T, aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax.set_xticks(range(len(name))); ax.set_xticklabels(list(name), fontsize=9)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
            # 标出真类所在行
            trow = topk.index(true)
            ax.add_patch(plt.Rectangle((-0.5, trow - 0.5), len(name), 1, fill=False,
                                       edgecolor="#39ff14", lw=2))
            ax.set_title(f"{mname}: '{name}'  (true = {lab})", fontsize=10)
            if c == 0:
                ax.set_ylabel("candidate language", fontsize=9)
            if r == len(picks) - 1:
                ax.set_xlabel("character read so far", fontsize=9)
    fig.suptitle("Token-level output: predicted-language probability as each character is read "
                 "(green box = true language)", fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.45, top=0.93)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="P(language)")
    fig.savefig(os.path.join(IMG, "exp2_tokens.png"), dpi=150, bbox_inches="tight")
    print("saved exp2_tokens.png")


if __name__ == "__main__":
    main()
