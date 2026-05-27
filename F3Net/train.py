"""
F3Net Training Script — Modern PyTorch (2.0+)

Changes from original:
  - apex.amp  → torch.cuda.amp (native AMP, available since PyTorch 1.6)
  - tensorboardX → torch.utils.tensorboard (native)
  - torch.compile() support (PyTorch 2.0+, optional)
  - Cleaner learning rate schedule
  - No dependency on NVIDIA Apex
"""

import os
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import F3Net, total_loss
from dataset import get_train_loader


def get_lr(epoch: int, max_epoch: int, base_lr: float) -> float:
    """Warm-up + linear decay (triangular schedule from the paper)."""
    progress = (epoch + 1) / (max_epoch + 1)
    return (1 - abs(2 * progress - 1)) * base_lr


def train(args):
    # ── Data ──
    loader = get_train_loader(args.datapath, batch_size=args.batch_size, num_workers=args.workers)
    print(f"Dataset: {args.datapath}, {len(loader.dataset)} images, {len(loader)} batches/epoch")

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = F3Net(pretrained=True).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── Optimizer: separate lr for backbone vs. head ──
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name:
            # Freeze stem (conv1 + bn1), only train layer1-4
            if "stem" in name:
                param.requires_grad = False
            else:
                backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},  # backbone: 1/10 lr
            {"params": head_params, "lr": args.lr},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # ── Native AMP (replaces apex) ──
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Logging ──
    os.makedirs(args.savepath, exist_ok=True)
    writer = SummaryWriter(args.savepath)
    global_step = 0

    # ── Training loop ──
    for epoch in range(args.epochs):
        model.train()

        # Update learning rate (triangular schedule)
        lr_backbone = get_lr(epoch, args.epochs, args.lr * 0.1)
        lr_head = get_lr(epoch, args.epochs, args.lr)
        optimizer.param_groups[0]["lr"] = lr_backbone
        optimizer.param_groups[1]["lr"] = lr_head

        for step, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward with AMP autocast
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = total_loss(outputs, masks, gamma=args.gamma)

            # Backward with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            global_step += 1
            if step % 10 == 0:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] epoch {epoch+1}/{args.epochs} "
                    f"step {step}/{len(loader)} "
                    f"lr={lr_head:.6f} loss={loss.item():.4f}"
                )

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr_head", lr_head, global_step)

        # Save checkpoint (last 1/3 of epochs, as in original)
        if epoch >= args.epochs * 2 // 3:
            ckpt_path = os.path.join(args.savepath, f"model_epoch{epoch+1}.pth")
            state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="F3Net Training")
    parser.add_argument("--datapath", type=str, default="./data/DUTS", help="Path to DUTS dataset")
    parser.add_argument("--savepath", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate for head")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=5.0, help="PPA loss hard-pixel weight")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
