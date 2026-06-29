# -*- coding: utf-8 -*-
"""训练"缩放点积注意力"变体(Transformer 所用的注意力), 与已训练的加性(Bahdanau)注意力
在完全相同的协议下对比。仅训练点积变体, 不动已有 ckpt_attn.pth。"""
import os, json, torch
from models import EncoderRNN, AttnDecoderRNN
from train_utils import get_dataloader, train, corpus_bleu, count_parameters

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

input_lang, output_lang, pairs, loader = get_dataloader(32, DEVICE, root=os.path.join(HERE, "data"))
enc = EncoderRNN(input_lang.n_words, 128).to(DEVICE)
dec = AttnDecoderRNN(128, output_lang.n_words, attn_type="dot").to(DEVICE)
print("training dot-product attention seq2seq ...")
hist = train(enc, dec, loader, n_epochs=30, lr=0.001, device=DEVICE, print_every=5)
bleu = corpus_bleu(enc, dec, pairs, input_lang, output_lang, DEVICE, n=1000, seed=0)
torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()},
           os.path.join(HERE, "ckpt_attn_dot.pth"))
json.dump({"bleu": round(bleu, 2), "final_loss": round(hist[-1], 4),
           "params": count_parameters(enc, dec)},
          open(os.path.join(HERE, "hist_dot.json"), "w"))
print(f"DOT done: bleu={bleu:.2f} final_loss={hist[-1]:.4f} params={count_parameters(enc,dec)}")
