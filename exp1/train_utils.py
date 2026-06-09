# -*- coding: utf-8 -*-
"""
exp1 训练 / 评估 / 数据 工具。

训练循环刻意沿用老师 notebook 里给的写法
    criterion = CrossEntropyLoss
    optimizer = SGD(momentum)
    for epoch: for i, (inputs, labels) in trainloader: zero_grad -> forward -> loss -> backward -> step
只是在每个 epoch 末尾额外做一次验证集评估，把 train_loss / test_loss / test_acc 记进 history，
方便在 notebook 里画 loss 曲线、准确率曲线并做多模型对比。
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# CIFAR10 官方统计量（比老师 notebook 里的 0.5/0.5/0.5 更标准，收敛更好）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed=42):
    """固定随机种子，保证几个模型在同一套划分/初始化条件下对比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloaders(batch_size=128, augment=True, num_workers=2, root='./data'):
    """返回 (trainloader, testloader)。augment=True 时训练集加随机裁剪+翻转。"""
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = test_tf

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=train_tf)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=test_tf)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def count_parameters(net):
    """可训练参数量。"""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    """返回 (平均 loss, top-1 准确率%)。"""
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * labels.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def per_class_accuracy(net, loader, device, classes=CLASSES):
    """每个类别的准确率%，返回 {类名: acc}。"""
    net.eval()
    correct = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, preds = outputs.max(1)
        for l, p in zip(labels, preds):
            c = classes[l.item()]
            total[c] += 1
            if l.item() == p.item():
                correct[c] += 1
    return {c: 100.0 * correct[c] / max(total[c], 1) for c in classes}


def train_model(net, trainloader, testloader, epochs=20, lr=0.01, momentum=0.9,
                weight_decay=5e-4, device=None, log_every=200, verbose=True):
    """
    沿用老师给的训练流程（SGD + 交叉熵 + 标准 mini-batch 循环），
    每个 epoch 末做一次测试集评估并记录曲线。

    返回 history = {
        'train_loss': [...],   # 每个 epoch 的训练集平均 loss
        'test_loss' : [...],   # 每个 epoch 的测试集平均 loss
        'test_acc'  : [...],   # 每个 epoch 的测试集 top-1 准确率%
    }
    """
    device = device or get_device()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0           # 仅用于按 log_every 打印（保持老师那种打印手感）
        epoch_loss_sum, epoch_n = 0.0, 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss_sum += loss.item() * labels.size(0)
            epoch_n += labels.size(0)
            if verbose and i % log_every == log_every - 1:
                print(f'[epoch {epoch + 1}, iter {i + 1:5d}] loss: {running_loss / log_every:.3f}')
                running_loss = 0.0

        train_loss = epoch_loss_sum / epoch_n
        test_loss, test_acc = evaluate(net, testloader, criterion, device)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        if verbose:
            print(f'  => epoch {epoch + 1}/{epochs}  '
                  f'train_loss={train_loss:.3f}  test_loss={test_loss:.3f}  '
                  f'test_acc={test_acc:.2f}%')

    return history
