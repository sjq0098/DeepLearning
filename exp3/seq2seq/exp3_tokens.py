# -*- coding: utf-8 -*-
"""
exp3 "token 输出图"(免训练): 解码轨迹。对一句源句, 在注意力解码器的每个输出步
画出 top-5 候选英文词及其概率(行=候选排名, 列=输出步, 颜色=概率, 单元格标注词)。
top-1 行从左到右读即为模型生成的译文; 颜色深浅反映该步的"把握程度"。

产出: exp3/images/exp3_tokens.png
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import EncoderRNN, AttnDecoderRNN, EOS_token
from train_utils import get_dataloader, tensorFromSentence

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.normpath(os.path.join(HERE, "..", "images"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HID = 128
plt.rcParams.update({"font.size": 10})


@torch.no_grad()
def decode_trace(enc, dec, src, ilang, olang, topk=5):
    inp = tensorFromSentence(ilang, src, DEVICE)
    enc_out, enc_hidden = enc(inp)
    dec_out, _, _ = dec(enc_out, enc_hidden)        # (1, T, V) log-probs
    probs = dec_out.exp()[0]                          # (T, V)
    rows_p, rows_w = [], []
    for t in range(probs.size(0)):
        v, i = probs[t].topk(topk)
        words = [olang.index2word[idx.item()] for idx in i]
        rows_p.append(v.cpu().numpy()); rows_w.append(words)
        if i[0].item() == EOS_token:
            break
    P = np.array(rows_p).T                            # (topk, steps)
    W = list(map(list, zip(*rows_w)))                 # (topk, steps)
    hyp = [rows_w[t][0] for t in range(len(rows_w))]
    hyp = [w for w in hyp if w != "<EOS>"]
    return P, W, hyp


def plot_trace(ax, P, W, src, hyp, topk=5):
    im = ax.imshow(P, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    for r in range(P.shape[0]):
        for c in range(P.shape[1]):
            ax.text(c, r, W[r][c], ha="center", va="center", fontsize=8,
                    color="white" if P[r, c] > 0.55 else "black")
    ax.set_xticks(range(P.shape[1]))
    ax.set_xticklabels([f"step {i+1}" for i in range(P.shape[1])], fontsize=8)
    ax.set_yticks(range(topk)); ax.set_yticklabels([f"top-{i+1}" for i in range(topk)], fontsize=8)
    ax.set_title(f"fr: {src}\n$\\to$ en: {' '.join(hyp)}", fontsize=10)
    return im


def main():
    ilang, olang, pairs, _ = get_dataloader(64, DEVICE, root=os.path.join(HERE, "data"))
    enc = EncoderRNN(ilang.n_words, HID).to(DEVICE)
    dec = AttnDecoderRNN(HID, olang.n_words).to(DEVICE)
    ck = torch.load(os.path.join(HERE, "ckpt_attn.pth"), map_location=DEVICE)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    enc.eval(); dec.eval()

    # 选两句: 一句较短、一句含语序重排的长句
    srcs = ["c est une bombe", "il perd tout le temps son parapluie"]
    srcs = [s for s in srcs if all(w in ilang.word2index for w in s.split())]

    fig, axes = plt.subplots(len(srcs), 1, figsize=(11, 3.4 * len(srcs)))
    if len(srcs) == 1:
        axes = [axes]
    for ax, src in zip(axes, srcs):
        P, W, hyp = decode_trace(enc, dec, src, ilang, olang)
        im = plot_trace(ax, P, W, src, hyp)
    fig.suptitle("Decoding trace: top-5 candidate English words at each output step "
                 "(top-1 row read left-to-right = the translation)", fontsize=12, y=1.0)
    fig.subplots_adjust(hspace=0.5, top=0.9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="P(word)")
    fig.savefig(os.path.join(IMG, "exp3_tokens.png"), dpi=150, bbox_inches="tight")
    print("saved exp3_tokens.png |", srcs)


if __name__ == "__main__":
    main()
