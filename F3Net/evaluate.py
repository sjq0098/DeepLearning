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
import torch.nn.functional as F
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


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, dilate_width: int = 5, beta2: float = 0.3):
    """
    Boundary-band quality metrics, used to *quantify* edge sharpness (§7.1) and
    to validate the edge-sharpening improvements (boundary loss / BAS / A8).

    The "band" is the GT object contour dilated by `dilate_width` px. Metrics are
    computed only inside this band, where ordinary MAE/F-measure are dominated by
    the easy interior/background and cannot reveal boundary softness.

    Returns (boundary_MAE, boundary_F) or None if the image has no boundary
    (empty/full GT).

        boundary_MAE : mean |pred - gt| inside the band   (lower = sharper)
        boundary_F   : F-measure (beta^2=0.3) at threshold 0.5, restricted to band
    """
    gt_bin = (gt >= 0.5).astype(np.uint8)
    s = int(gt_bin.sum())
    if s == 0 or s == gt_bin.size:
        return None

    k3 = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(gt_bin, k3) - cv2.erode(gt_bin, k3)
    band = cv2.dilate(edge, np.ones((dilate_width, dilate_width), np.uint8)) > 0
    if not band.any():
        return None

    bmae = float(np.abs(pred - gt)[band].mean())

    pb = pred >= 0.5
    gtb = gt_bin.astype(bool)
    tp = float((pb & gtb & band).sum())
    pp = float((pb & band).sum())
    pg = float((gtb & band).sum())
    if pp > 0 and pg > 0 and tp > 0:
        prec, rec = tp / pp, tp / pg
        bf = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-10)
    else:
        bf = 0.0
    return bmae, float(bf)


def predict_tta(model, image, out_size, scales=(1.0,), flip=False):
    """
    Test-time augmentation (④): average the sigmoid prediction over several input
    scales and (optionally) a horizontal flip. Returns a (H, W) probability map.

    `image` is the fixed-size model input (1, 3, S, S); each scale resizes it and
    the logits are upsampled back to the native `out_size` before averaging, so
    all augmented views are combined in the original image geometry.
    """
    H, W = out_size
    acc = None
    n = 0
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            inp = image
        else:
            sz = max(32, int(round(image.shape[-1] * s)))
            inp = F.interpolate(image, size=(sz, sz), mode="bilinear", align_corners=False)
        views = [inp]
        flips = [False]
        if flip:
            views.append(torch.flip(inp, dims=[3]))
            flips.append(True)
        for v, is_flip in zip(views, flips):
            prob = torch.sigmoid(model(v, out_size=(H, W)))
            if is_flip:
                prob = torch.flip(prob, dims=[3])
            acc = prob if acc is None else acc + prob
            n += 1
    return (acc / n)[0, 0].cpu().numpy()


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
        use_bas=args.use_bas,
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
    b_maes, b_fs = [], []
    for image, mask, shape, name in tqdm(loader, total=len(loader), ncols=80):
        image = image.to(device, non_blocking=True)
        H, W = int(shape[0].item()), int(shape[1].item())

        if args.tta:
            scales = tuple(float(s) for s in args.tta_scales.split(","))
            pred = predict_tta(model, image, (H, W), scales=scales, flip=True)
        else:
            pred = torch.sigmoid(model(image, out_size=(H, W))[0, 0]).cpu().numpy()
        gt = mask[0].cpu().numpy()                            # (H, W) in [0, 1]

        mae, f_curve, f_adapt = per_image_metrics(pred, gt)
        maes.append(mae)
        f_curves.append(f_curve)
        f_adapts.append(f_adapt)

        bm = boundary_metrics(pred, gt)
        if bm is not None:
            b_maes.append(bm[0])
            b_fs.append(bm[1])

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, name[0] + ".png"), (pred * 255).astype(np.uint8))

    f_curve_mean = np.stack(f_curves).mean(axis=0)   # mean F at each threshold over dataset
    print()
    print(f"=== Results ({len(maes)} images) ===")
    print(f"  MAE                : {np.mean(maes):.4f}")
    print(f"  F-measure (max)    : {f_curve_mean.max():.4f}")
    print(f"  F-measure (mean)   : {f_curve_mean.mean():.4f}")
    print(f"  F-measure (adapt)  : {np.mean(f_adapts):.4f}")
    print(f"  --- boundary band (width=5, {len(b_maes)} imgs) ---")
    print(f"  Boundary MAE       : {np.mean(b_maes):.4f}")
    print(f"  Boundary F (0.5)   : {np.mean(b_fs):.4f}")


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
    p.add_argument("--use_bas", action="store_true", help="Checkpoint was trained with the BAS boundary head (③)")
    p.add_argument("--tta", action="store_true", help="④ multi-scale + hflip test-time augmentation")
    p.add_argument("--tta_scales", type=str, default="0.82,0.91,1.0",
                   help="Comma-separated input-scale multipliers for --tta. Keep <=1.0: the "
                        "model was trained on {224..352}px, so upscaling past 352 is OOD and hurts.")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
