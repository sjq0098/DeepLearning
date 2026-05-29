"""
F3Net evaluator on the test split.

Reports the metrics required by the assignment:
  - MAE                     (mean absolute error, lower is better)
  - F-measure (max  / mean) (over 256 thresholds, beta^2 = 0.3, higher is better)
  - F-measure (adaptive)    (threshold = min(1, 2 * mean(pred)) per image)

Predictions are produced at 352x352, then bilinearly upsampled back to each
image's native resolution before being compared against the un-resampled GT
(matches the PoolNet / standard SOD evaluation protocol).

Usage:
    python evaluate.py --checkpoint ./checkpoints/model_epoch32.pth
    python evaluate.py --checkpoint <...> --save_dir ./results
"""
import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import get_test_loader
from model import F3Net


def per_image_metrics(pred: np.ndarray, gt: np.ndarray, n_bins: int = 256, beta2: float = 0.3):
    """
    Vectorised single-image MAE + F-measure-at-256-thresholds.

    Args:
        pred: float in [0, 1], shape (H, W)
        gt:   float in [0, 1], shape (H, W)

    Returns:
        mae       (float)
        f_curve   (np.ndarray of length n_bins, F at each threshold)
        f_adapt   (float, F at adaptive threshold = min(1, 2*mean(pred)))
    """
    mae = float(np.abs(pred - gt).mean())

    gt_bin = gt >= 0.5
    sum_gt = float(gt_bin.sum())
    if sum_gt == 0:
        return mae, np.zeros(n_bins, dtype=np.float32), 0.0

    # ── Fast F-curve via cumulative histogram ──
    # Bin each prediction value into [0, n_bins-1]
    pred_q = np.clip((pred * n_bins).astype(np.int32), 0, n_bins - 1)
    h_fg = np.bincount(pred_q[gt_bin], minlength=n_bins).astype(np.float64)
    h_bg = np.bincount(pred_q[~gt_bin], minlength=n_bins).astype(np.float64)
    # At threshold index t, "predicted positive" pixels are those in bins >= t.
    # cumsum from the right gives those counts.
    tp_at_t = np.cumsum(h_fg[::-1])[::-1]
    pp_at_t = np.cumsum((h_fg + h_bg)[::-1])[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pp_at_t > 0, tp_at_t / pp_at_t, 0.0)
        recall = tp_at_t / sum_gt
        denom = beta2 * precision + recall
        f_curve = np.where(denom > 0, (1 + beta2) * precision * recall / denom, 0.0)

    # ── Adaptive threshold ──
    t_adapt = min(1.0, 2.0 * float(pred.mean()))
    pred_bin = pred >= t_adapt
    tp = float((pred_bin & gt_bin).sum())
    sum_pred = float(pred_bin.sum())
    if sum_pred > 0 and tp > 0:
        p_a = tp / sum_pred
        r_a = tp / sum_gt
        f_adapt = (1 + beta2) * p_a * r_a / (beta2 * p_a + r_a + 1e-10)
    else:
        f_adapt = 0.0

    return mae, f_curve.astype(np.float32), float(f_adapt)


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cfm = not args.no_cfm
    use_mls = not args.no_mls
    model = F3Net(
        pretrained=False,
        use_cfm=use_cfm,
        num_decoders=args.num_decoders,
        use_mls=use_mls,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Architecture: use_cfm={use_cfm}, num_decoders={args.num_decoders}, use_mls={use_mls}")

    loader = get_test_loader(args.datapath, size=args.input_size, num_workers=args.workers)
    print(f"Test set: {len(loader.dataset)} images")

    save_dir = None
    if args.save_dir:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving saliency maps to: {save_dir}")

    maes, f_curves, f_adapts = [], [], []
    for image, mask, shape, name in tqdm(loader, total=len(loader), ncols=80):
        image = image.to(device, non_blocking=True)
        H, W = int(shape[0].item()), int(shape[1].item())

        pred = model(image, out_size=(H, W))                 # (1, 1, H, W) logits
        pred = torch.sigmoid(pred[0, 0]).cpu().numpy()        # (H, W) in [0, 1]
        gt = mask[0].cpu().numpy()                            # (H, W) in [0, 1]

        mae, f_curve, f_adapt = per_image_metrics(pred, gt)
        maes.append(mae)
        f_curves.append(f_curve)
        f_adapts.append(f_adapt)

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, name[0] + ".png"), (pred * 255).astype(np.uint8))

    f_curve_mean = np.stack(f_curves).mean(axis=0)   # mean F at each threshold over dataset
    print()
    print(f"=== Results ({len(maes)} images) ===")
    print(f"  MAE                : {np.mean(maes):.4f}")
    print(f"  F-measure (max)    : {f_curve_mean.max():.4f}")
    print(f"  F-measure (mean)   : {f_curve_mean.mean():.4f}")
    print(f"  F-measure (adapt)  : {np.mean(f_adapts):.4f}")


def main():
    p = argparse.ArgumentParser(description="F3Net evaluator (MAE + F-measure)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth weights")
    p.add_argument("--datapath", type=str, default="./data", help="Dataset root (containing test.txt)")
    p.add_argument("--input_size", type=int, default=352, help="Model input size")
    p.add_argument("--save_dir", type=str, default=None, help="If set, also dump predicted saliency PNGs here")
    p.add_argument("--workers", type=int, default=2)
    # Architecture flags — must match how the checkpoint was trained
    p.add_argument("--no_cfm", action="store_true", help="Checkpoint was trained with AdditiveFusion instead of CFM")
    p.add_argument("--num_decoders", type=int, default=2, choices=[1, 2], help="Number of sub-decoders the checkpoint was trained with")
    p.add_argument("--no_mls", action="store_true", help="Checkpoint was trained without multi-level supervision")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
