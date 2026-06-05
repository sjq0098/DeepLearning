"""
Zero-training evidence for the "PPA amplifies where labels are noisiest" hypothesis.

PPA weights each pixel by w = 1 + gamma * alpha, alpha = |avgpool(GT,31) - GT|.
alpha peaks on object boundaries — which is *exactly* where human annotation is most
uncertain. This script quantifies, over the training set, how much of PPA's loss-weight
mass lands on the boundary band (the pixels a realistic annotation-noise model would
corrupt), and the per-pixel amplification factor there. It also breaks the analysis down
by object size, since small/thin objects concentrate the effect.

No training, no model — just GT masks. Run:
    python analyze_ppa_weight.py --datapath ./data --split train
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ppa_weight(mask: np.ndarray, gamma: float = 5.0, k: int = 31) -> np.ndarray:
    """PPA per-pixel weight, matching model.ppa_loss (reflect-padded avgpool)."""
    pad = k // 2
    m = torch.from_numpy(mask)[None, None].float()
    mp = F.pad(m, (pad, pad, pad, pad), mode="reflect")
    lm = F.avg_pool2d(mp, kernel_size=k, stride=1, padding=0)
    alpha = (lm - m).abs()
    return (1.0 + gamma * alpha)[0, 0].numpy()


def boundary_band(gt_bin: np.ndarray, dilate: int = 5) -> np.ndarray:
    """Pixels within `dilate` px of the GT contour — where annotation noise lives."""
    g = gt_bin.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(g, k3) - cv2.erode(g, k3)
    return cv2.dilate(edge, np.ones((dilate, dilate), np.uint8)) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", default="./data")
    p.add_argument("--split", default="train")
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--band", type=int, default=5)
    args = p.parse_args()

    with open(os.path.join(args.datapath, f"{args.split}.txt")) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    mask_dir = os.path.join(args.datapath, "ground_truth_mask")

    # Aggregators, overall and per size-bin (small <5%, medium 5-20%, large >20%).
    bins = {"small (<5%)": [], "medium (5-20%)": [], "large (>20%)": []}
    agg = {"band_px_frac": [], "band_wmass_frac": [], "amp_factor": [], "fg_frac": []}

    for name in names:
        m = cv2.imread(os.path.join(mask_dir, name + ".png"), 0)
        if m is None:
            continue
        mask = (m.astype(np.float32) / 255.0)
        gt_bin = mask >= 0.5
        fg = float(gt_bin.mean())
        w = ppa_weight(mask, gamma=args.gamma)
        band = boundary_band(gt_bin, dilate=args.band)
        if band.sum() == 0:
            continue

        band_px_frac = band.mean()
        band_wmass_frac = w[band].sum() / w.sum()
        amp = w[band].mean() / w[~band].mean()      # weight on noise pixels vs clean

        agg["band_px_frac"].append(band_px_frac)
        agg["band_wmass_frac"].append(band_wmass_frac)
        agg["amp_factor"].append(amp)
        agg["fg_frac"].append(fg)

        key = "small (<5%)" if fg < 0.05 else ("medium (5-20%)" if fg < 0.20 else "large (>20%)")
        bins[key].append((band_px_frac, band_wmass_frac, amp))

    print(f"=== PPA weight concentration on the boundary band ({args.split}, "
          f"{len(agg['amp_factor'])} imgs, gamma={args.gamma}, band={args.band}px) ===\n")
    bpf = np.mean(agg["band_px_frac"]) * 100
    bwf = np.mean(agg["band_wmass_frac"]) * 100
    amp = np.mean(agg["amp_factor"])
    print(f"  boundary band covers           : {bpf:5.1f}% of pixels")
    print(f"  ...but holds                   : {bwf:5.1f}% of total PPA weight mass")
    print(f"  weight-mass over-concentration : {bwf/bpf:4.2f}x   (>1 = weight piles onto the band)")
    print(f"  per-pixel amplification        : {amp:4.2f}x   (a band pixel gets this much more weight than a non-band pixel)")
    print()
    print("  --- broken down by object size (smaller object = noise more concentrated) ---")
    print(f"  {'size bin':16s} {'#img':>5s} {'band%px':>8s} {'band%wmass':>11s} {'over-conc':>10s} {'amp':>6s}")
    for key, vals in bins.items():
        if not vals:
            continue
        v = np.array(vals)
        bpf_b, bwf_b, amp_b = v[:, 0].mean() * 100, v[:, 1].mean() * 100, v[:, 2].mean()
        print(f"  {key:16s} {len(vals):5d} {bpf_b:7.1f}% {bwf_b:10.1f}% {bwf_b/bpf_b:9.2f}x {amp_b:5.2f}x")


if __name__ == "__main__":
    main()
