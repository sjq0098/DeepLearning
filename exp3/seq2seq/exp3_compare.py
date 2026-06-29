# -*- coding: utf-8 -*-
"""加性(Bahdanau) vs 缩放点积(Transformer) 注意力对比:
  1) 同口径重算两者 BLEU(n=1000) 供表格;
  2) 并排画两种注意力在相同句子上的对齐热力图 -> exp3_attn_compare.png
"""
import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import EncoderRNN, AttnDecoderRNN, EOS_token
from train_utils import get_dataloader, evaluate, corpus_bleu

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.normpath(os.path.join(HERE, "..", "images"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HID = 128
plt.rcParams.update({"font.size": 10})


def load(attn_type, ck, ilang, olang):
    enc = EncoderRNN(ilang.n_words, HID).to(DEVICE)
    dec = AttnDecoderRNN(HID, olang.n_words, attn_type=attn_type).to(DEVICE)
    c = torch.load(os.path.join(HERE, ck), map_location=DEVICE)
    enc.load_state_dict(c["enc"]); dec.load_state_dict(c["dec"])
    enc.eval(); dec.eval()
    return enc, dec


def main():
    ilang, olang, pairs, _ = get_dataloader(32, DEVICE, root=os.path.join(HERE, "data"))
    enc_a, dec_a = load("bahdanau", "ckpt_attn.pth", ilang, olang)
    enc_d, dec_d = load("dot", "ckpt_attn_dot.pth", ilang, olang)

    bleu_a = corpus_bleu(enc_a, dec_a, pairs, ilang, olang, DEVICE, n=1000, seed=0)
    bleu_d = corpus_bleu(enc_d, dec_d, pairs, ilang, olang, DEVICE, n=1000, seed=0)
    print(f"BLEU(n=1000): additive={bleu_a:.2f}  dot={bleu_d:.2f}")
    json.dump({"bleu_additive": round(bleu_a, 2), "bleu_dot": round(bleu_d, 2)},
              open(os.path.join(HERE, "compare_bleu.json"), "w"))

    srcs = ["il perd tout le temps son parapluie", "je ne suis pas decourage", "vous etes riches"]
    srcs = [s for s in srcs if all(w in ilang.word2index for w in s.split())]
    fig, axes = plt.subplots(len(srcs), 2, figsize=(11, 3.0 * len(srcs)))
    if len(srcs) == 1:
        axes = axes[None, :]
    for r, src in enumerate(srcs):
        for c, (enc, dec, name) in enumerate([(enc_a, dec_a, "Additive (Bahdanau)"),
                                              (enc_d, dec_d, "Scaled dot-product")]):
            words, attn = evaluate(enc, dec, src, ilang, olang, DEVICE)
            hyp = [w for w in words if w != "<EOS>"]
            a = attn[0].cpu().numpy()[:len(hyp), :len(src.split()) + 1]
            ax = axes[r, c]
            ax.matshow(a, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(src.split()) + 1))
            ax.set_xticklabels(src.split() + ["<eos>"], rotation=90, fontsize=8)
            ax.set_yticks(range(len(hyp))); ax.set_yticklabels(hyp, fontsize=8)
            ax.xaxis.set_ticks_position("bottom")
            ax.set_title(f"{name}\n{' '.join(hyp)}", fontsize=9)
    fig.suptitle("Additive attention (left) is sharper; scaled dot-product (right) is more "
                 "diffuse on this small RNN task", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "exp3_attn_compare.png"), dpi=150, bbox_inches="tight")
    print("saved exp3_attn_compare.png")


if __name__ == "__main__":
    main()
