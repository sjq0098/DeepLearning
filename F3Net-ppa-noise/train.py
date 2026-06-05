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

from model import F3Net, make_loss, total_loss
from dataset import get_train_loader


def get_lr(epoch: int, max_epoch: int, base_lr: float) -> float:
    """Warm-up + linear decay (triangular schedule from the paper)."""
    progress = (epoch + 1) / (max_epoch + 1)
    return (1 - abs(2 * progress - 1)) * base_lr


def train(args):
    # ── Data ──
    loader = get_train_loader(args.datapath, batch_size=args.batch_size, num_workers=args.workers,
                              split=args.train_split, mask_subdir=args.mask_subdir)
    print(f"Dataset: {args.datapath}, {len(loader.dataset)} images, {len(loader)} batches/epoch")

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cfm = not args.no_cfm
    use_mls = not args.no_mls
    use_bas = args.use_bas
    model = F3Net(
        pretrained=True,
        use_cfm=use_cfm,
        num_decoders=args.num_decoders,
        use_mls=use_mls,
        use_bas=use_bas,
    ).to(device)
    print(f"Model: use_cfm={use_cfm}, num_decoders={args.num_decoders}, use_mls={use_mls}, use_bas={use_bas}, loss={args.loss}")
    print(f"Sharpening: boundary_w={args.boundary_weight}, ssim_w={args.ssim_weight}, "
          f"edge_w={args.edge_weight}, bas_w={args.bas_weight}, k={args.boundary_kernel}")

    # Per-prediction loss function (BCE / IoU / PPA) — passed into total_loss
    loss_fn = make_loss(args.loss, gamma=args.gamma)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed weights from {args.resume} (start_epoch={args.start_epoch})")

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
    for epoch in range(args.start_epoch, args.epochs):
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
                loss = total_loss(
                    outputs, masks,
                    loss_fn=loss_fn,
                    num_decoders=args.num_decoders,
                    use_mls=use_mls,
                    boundary_weight=args.boundary_weight,
                    ssim_weight=args.ssim_weight,
                    edge_weight=args.edge_weight,
                    bas_weight=args.bas_weight,
                    boundary_kernel=args.boundary_kernel,
                    image=images,
                    use_bas=use_bas,
                )

            # Backward with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

        # Save checkpoint every `save_every` epochs, plus the final epoch.
        # (Original saves every epoch in the last 1/3, which produces ~67
        # files for a 200-epoch run — too many.)
        is_final = (epoch + 1 == args.epochs)
        is_periodic = args.save_every > 0 and (epoch + 1) % args.save_every == 0
        if is_final or is_periodic:
            ckpt_path = os.path.join(args.savepath, f"model_epoch{epoch+1}.pth")
            state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="F3Net Training")
    parser.add_argument("--datapath", type=str, default="./data/DUTS", help="Path to DUTS dataset")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Name of the <split>.txt training list (for the ② data-scaling study, e.g. train_175)")
    parser.add_argument("--mask_subdir", type=str, default="ground_truth_mask",
                        help="GT folder for TRAINING (set to a gt_noisy_* dir for the label-noise study; test stays clean)")
    parser.add_argument("--savepath", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate for head")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=5.0, help="PPA loss hard-pixel weight")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=25, help="Save a checkpoint every N epochs (final epoch is always saved)")
    parser.add_argument("--resume", type=str, default=None, help="Path to a .pth checkpoint to resume weights from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch index to start the training loop at (use with --resume; LR schedule still uses --epochs as the total)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    # Ablation flags (defaults reproduce the full F3Net baseline)
    parser.add_argument("--loss", type=str, default="ppa", choices=["bce", "iou", "ppa"],
                        help="Per-head loss function")
    parser.add_argument("--no_cfm", action="store_true",
                        help="Replace CrossFeatureModule with AdditiveFusion (multiplication -> addition)")
    parser.add_argument("--num_decoders", type=int, default=2, choices=[1, 2],
                        help="Number of cascaded sub-decoders (1 = no feedback, 2 = paper's CFD)")
    parser.add_argument("--no_mls", action="store_true",
                        help="Disable multi-level auxiliary supervision")
    # Edge-sharpening improvement flags (all 0 / off -> identical to baseline)
    parser.add_argument("--boundary_weight", type=float, default=0.0,
                        help="① weight for Boundary-IoU loss on the final prediction")
    parser.add_argument("--ssim_weight", type=float, default=0.0,
                        help="① weight for structural SSIM loss on the final prediction")
    parser.add_argument("--edge_weight", type=float, default=0.0,
                        help="A8 weight for salient-region-gated image-edge consistency loss")
    parser.add_argument("--use_bas", action="store_true",
                        help="③ add a boundary-aware auxiliary supervision head off f2")
    parser.add_argument("--bas_weight", type=float, default=0.5,
                        help="③ weight for the BAS head boundary BCE (only used with --use_bas)")
    parser.add_argument("--boundary_kernel", type=int, default=3,
                        help="Kernel size for morphological boundary extraction (① / ③ / A8)")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Max grad-norm for clipping (0 = off). Safety net for the sharpening losses.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
