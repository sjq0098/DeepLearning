# -*- coding: utf-8 -*-
"""
exp2 名字识别（name classification）数据 / 训练 / 评估工具。

训练循环沿用老师 notebook 给出的写法（NLLLoss + SGD，按“伪 batch”累加单样本损失再更新、
梯度裁剪），只是额外在每个 epoch 末计算一次验证集准确率并记录，便于画准确率曲线。

字符表、lineToTensor、NamesDataset 与老师 notebook 保持一致（n_letters=58）。
"""
import os
import glob
import random
import string
import unicodedata
import zipfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn

# 与老师 notebook 一致：大小写字母 + " .,;'" + 一个 OOV 占位符 "_"
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)        # = 58
DATA_URL = "https://download.pytorch.org/tutorial/data.zip"


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in allowed_characters
    )


def letterToIndex(letter):
    if letter not in allowed_characters:
        return allowed_characters.find("_")     # OOV
    return allowed_characters.find(letter)


def lineToTensor(line):
    """名字 -> (len, 1, n_letters) 的 one-hot 张量。"""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def maybe_download_names(root="data"):
    """确保 root/names/*.txt 存在；不存在则下载 data.zip 解压。返回 names 目录路径。"""
    names_dir = os.path.join(root, "names")
    if glob.glob(os.path.join(names_dir, "*.txt")):
        return names_dir
    print(f"未找到 {names_dir}，开始下载 {DATA_URL} ...")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "data.zip")
    try:
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(".")            # data.zip 内含 data/names/...
        print("下载并解压完成。")
    except Exception as e:
        raise RuntimeError(
            f"自动下载失败：{e}\n"
            f"请手动下载 {DATA_URL}，解压到 exp2/ 下（得到 exp2/data/names/*.txt）。"
        )
    return names_dir


def load_data(root="data", val_frac=0.15, seed=2024):
    """
    读取所有 names/*.txt，返回 (train_data, val_data, classes, n_letters)。
    每个样本是 (label_tensor[long,shape=1], line_tensor[(len,1,n_letters)], label_str)。
    classes 为排序后的类别列表（保证可复现）。
    """
    names_dir = maybe_download_names(root)
    files = sorted(glob.glob(os.path.join(names_dir, "*.txt")))
    classes = sorted(os.path.splitext(os.path.basename(f))[0] for f in files)
    cls_index = {c: i for i, c in enumerate(classes)}

    samples = []
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        with open(f, encoding="utf-8") as fh:
            for line in fh.read().strip().split("\n"):
                name = unicodeToAscii(line.strip())
                if not name:
                    continue
                label_tensor = torch.tensor([cls_index[label]], dtype=torch.long)
                samples.append((label_tensor, lineToTensor(name), label))

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_frac)
    val_data, train_data = samples[:n_val], samples[n_val:]
    return train_data, val_data, classes, n_letters


@torch.no_grad()
def evaluate(model, data, device="cpu"):
    """返回验证集 top-1 准确率(%)。"""
    model.eval()
    correct = 0
    for label_tensor, text_tensor, _ in data:
        output = model(text_tensor.to(device))
        guess = output.topk(1)[1][0].item()
        correct += int(guess == label_tensor.item())
    return 100.0 * correct / len(data)


@torch.no_grad()
def compute_confusion(model, data, n_classes, device="cpu"):
    """返回按行归一化的混淆矩阵 (n_classes, n_classes)，行=真实类，列=预测类。"""
    model.eval()
    conf = torch.zeros(n_classes, n_classes)
    for label_tensor, text_tensor, _ in data:
        output = model(text_tensor.to(device))
        guess = output.topk(1)[1][0].item()
        conf[label_tensor.item()][guess] += 1
    for r in range(n_classes):
        s = conf[r].sum()
        if s > 0:
            conf[r] = conf[r] / s
    return conf


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_data, val_data, n_epoch=15, n_batch_size=64,
                lr=0.15, clip=3.0, device="cpu", report_every=1, verbose=True):
    """
    沿用老师写法：每个“伪 batch”累加 n_batch_size 个单样本的 NLLLoss，再一次性反向+裁剪+更新。
    每个 epoch 末在验证集上评估准确率。返回 history={'train_loss':[...], 'val_acc':[...]}。
    """
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}
    n = len(train_data)

    for epoch in range(1, n_epoch + 1):
        model.train()
        idxs = list(range(n))
        random.shuffle(idxs)
        batches = np.array_split(idxs, max(1, n // n_batch_size))

        running = 0.0
        for batch in batches:
            optimizer.zero_grad()
            batch_loss = 0.0
            for i in batch:
                label_tensor, text_tensor, _ = train_data[i]
                output = model(text_tensor.to(device))
                batch_loss = batch_loss + criterion(output, label_tensor.to(device))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += batch_loss.item() / len(batch)

        train_loss = running / len(batches)
        val_acc = evaluate(model, val_data, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        if verbose and epoch % report_every == 0:
            print(f"epoch {epoch:2d}/{n_epoch}  train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%")
    return history
