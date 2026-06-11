# -*- coding: utf-8 -*-
"""
exp3 Seq2Seq 数据 / 训练 / 评估 / 可视化工具。

训练沿用老师 notebook 的写法（Adam + NLLLoss + teacher forcing），额外把每个 epoch 的
loss 收进 history 以便画曲线；并提供翻译评估、BLEU 对比、注意力热力图（用 Agg 后端 savefig，
不弹窗、便于在报告里引用）。
"""
import os
import re
import math
import time
import random
import unicodedata
import zipfile
import urllib.request
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models import SOS_token, EOS_token, MAX_LENGTH

DATA_URL = "https://download.pytorch.org/tutorial/data.zip"

eng_prefixes = (
    "i am ", "i m ", "he is", "he s ", "she is", "she s ",
    "you are", "you re ", "we are", "we re ", "they are", "they re ",
)


# ----------------------------- 数据 ----------------------------- #
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def maybe_download(root="data"):
    """确保 data/eng-fra.txt 存在，否则下载 data.zip 解压。"""
    if os.path.exists(os.path.join(root, "eng-fra.txt")):
        return
    print(f"未找到 {root}/eng-fra.txt，开始下载 {DATA_URL} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, "data.zip")
        with zipfile.ZipFile("data.zip") as z:
            z.extractall(".")
        print("下载并解压完成。")
    except Exception as e:
        raise RuntimeError(f"自动下载失败：{e}\n请手动下载 {DATA_URL} 解压到当前目录。")


def readLangs(lang1, lang2, reverse=False):
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang, output_lang = Lang(lang2), Lang(lang1)
    else:
        input_lang, output_lang = Lang(lang1), Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH \
        and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, verbose=True):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    if verbose:
        print(f"句对数: {len(pairs)} | {input_lang.name}: {input_lang.n_words} 词 | "
              f"{output_lang.name}: {output_lang.n_words} 词")
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, device, root="data"):
    maybe_download(root)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', reverse=True)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int64)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int64)
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp) + [EOS_token]
        tgt_ids = indexesFromSentence(output_lang, tgt) + [EOS_token]
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                         torch.LongTensor(target_ids).to(device))
    loader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    return input_lang, output_lang, pairs, loader


# ----------------------------- 训练 ----------------------------- #
def count_parameters(*modules):
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def _train_epoch(loader, encoder, decoder, enc_opt, dec_opt, criterion):
    total = 0.0
    for input_tensor, target_tensor in loader:
        enc_opt.zero_grad(); dec_opt.zero_grad()
        enc_out, enc_hidden = encoder(input_tensor)
        dec_out, _, _ = decoder(enc_out, enc_hidden, target_tensor)
        loss = criterion(dec_out.view(-1, dec_out.size(-1)), target_tensor.view(-1))
        loss.backward()
        enc_opt.step(); dec_opt.step()
        total += loss.item()
    return total / len(loader)


def train(encoder, decoder, loader, n_epochs, lr=0.001, device="cpu",
          print_every=5, verbose=True):
    """返回每个 epoch 的平均训练 loss 列表（history）。"""
    encoder.to(device); decoder.to(device)
    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    history = []
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        encoder.train(); decoder.train()
        loss = _train_epoch(loader, encoder, decoder, enc_opt, dec_opt, criterion)
        history.append(loss)
        if verbose and (epoch % print_every == 0 or epoch == 1):
            el = time.time() - start
            print(f"epoch {epoch:3d}/{n_epochs}  loss={loss:.4f}  ({el:.0f}s)")
    return history


# ----------------------------- 评估 ----------------------------- #
@torch.no_grad()
def evaluate(encoder, decoder, sentence, input_lang, output_lang, device):
    encoder.eval(); decoder.eval()
    input_tensor = tensorFromSentence(input_lang, sentence, device)
    enc_out, enc_hidden = encoder(input_tensor)
    dec_out, _, dec_attn = decoder(enc_out, enc_hidden)
    _, topi = dec_out.topk(1)
    ids = topi.squeeze()
    words = []
    for idx in ids:
        if idx.item() == EOS_token:
            words.append('<EOS>')
            break
        words.append(output_lang.index2word[idx.item()])
    return words, dec_attn


def evaluate_examples(encoder, decoder, pairs, input_lang, output_lang, device, n=8, seed=0):
    """打印 n 个随机样例的 (源/参考/译文)，返回列表。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        pair = rng.choice(pairs)
        words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        hyp = ' '.join(w for w in words if w != '<EOS>')
        rows.append((pair[0], pair[1], hyp))
    return rows


# ----------------------------- BLEU ----------------------------- #
def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4):
    """带 add-1 平滑的句级 BLEU-4。reference/hypothesis 为 token 列表。"""
    if len(hypothesis) == 0:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        ref_ng = _ngram_counts(reference, n)
        hyp_ng = _ngram_counts(hypothesis, n)
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        p = (overlap + 1.0) / (total + 1.0)          # add-1 平滑
        log_p += (1.0 / max_n) * math.log(p)
    bp = 1.0 if len(hypothesis) > len(reference) \
        else math.exp(1 - len(reference) / max(len(hypothesis), 1))
    return bp * math.exp(log_p)


@torch.no_grad()
def corpus_bleu(encoder, decoder, pairs, input_lang, output_lang, device, n=500, seed=0):
    """在 n 个随机样例上的平均句级 BLEU（百分制）。"""
    rng = random.Random(seed)
    sample = [rng.choice(pairs) for _ in range(n)]
    total = 0.0
    for src, ref in sample:
        words, _ = evaluate(encoder, decoder, src, input_lang, output_lang, device)
        hyp = [w for w in words if w != '<EOS>']
        total += sentence_bleu(ref.split(' '), hyp)
    return 100.0 * total / len(sample)


# ----------------------------- 绘图 ----------------------------- #
def plot_loss_curves(histories, labels, path=None):
    """histories: list of per-epoch loss lists；labels: 对应名称。"""
    plt.figure(figsize=(7, 4.5))
    for h, lb in zip(histories, labels):
        plt.plot(range(1, len(h) + 1), h, marker='o', ms=3, label=lb)
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('NLL Loss')
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    return plt.gcf()


def show_attention(input_sentence, output_words, attentions, path=None):
    """注意力热力图（Agg 后端，savefig，不弹窗）。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticks(range(len(input_sentence.split(' ')) + 1))
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
    ax.set_yticks(range(len(output_words)))
    ax.set_yticklabels(output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    return fig
