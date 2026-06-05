"""
Inject controlled boundary annotation noise into the TRAINING GT masks.

Real annotation noise is concentrated at object boundaries (that is where human
annotators disagree). We model it by flipping pixels *inside a band around the GT
contour*, which is exactly where PPA's alpha-weight is largest — letting us test
whether PPA's "amplify the boundary" design amplifies this noise.

The noise is FIXED per image (a corrupted copy on disk), not resampled per epoch,
matching real mislabeled data: the model sees the same wrong target every epoch
and memorises it. Only TRAIN names are corrupted; the test set stays clean.

Two knobs:
    --width d : boundary band half-width in px  (bigger = noise spreads wider)
    --prob  p : flip probability inside the band (bigger = noise stronger)

Output -> data/gt_noisy_d{d}p{int(p*100)}/<train name>.png   (0/255)

Usage:
    python make_noisy_gt.py --datapath ./data --width 3 --prob 0.5 --seed 42
"""
import argparse
import os

import cv2
import numpy as np


def boundary_band(gt_bin: np.ndarray, dilate: int) -> np.ndarray:
    g = gt_bin.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(g, k3) - cv2.erode(g, k3)
    return cv2.dilate(edge, np.ones((dilate, dilate), np.uint8)) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", default="./data")
    p.add_argument("--split", default="train", help="Which name list to corrupt (test stays clean!)")
    p.add_argument("--width", type=int, default=3, help="Boundary band half-width (px)")
    p.add_argument("--prob", type=float, default=0.5, help="Flip probability inside the band")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["flip", "dilate"], default="flip",
                   help="flip = random jitter in band; dilate = systematic outward bias")
    args = p.parse_args()

    with open(os.path.join(args.datapath, args.split + ".txt")) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    src = os.path.join(args.datapath, "ground_truth_mask")
    out = os.path.join(args.datapath, f"gt_noisy_d{args.width}p{int(args.prob * 100)}")
    os.makedirs(out, exist_ok=True)

    flipped_frac = []
    for i, name in enumerate(names):
        m = cv2.imread(os.path.join(src, name + ".png"), 0)
        gt = (m >= 128).astype(np.uint8)
        rng = np.random.default_rng(args.seed + i)        # deterministic per image

        if args.mode == "flip":
            band = boundary_band(gt, args.width)
            flip = (rng.random(gt.shape) < args.prob) & band
            noisy = gt.copy()
            noisy[flip] = 1 - noisy[flip]
        else:  # systematic outward dilation by `width` px on a fraction `prob` of images
            if rng.random() < args.prob:
                noisy = cv2.dilate(gt, np.ones((args.width * 2 + 1,) * 2, np.uint8))
            else:
                noisy = gt
            flip = noisy != gt

        flipped_frac.append(float(flip.mean()))
        cv2.imwrite(os.path.join(out, name + ".png"), (noisy * 255).astype(np.uint8))

    print(f"Wrote {len(names)} noisy masks -> {out}")
    print(f"  mode={args.mode}, width={args.width}, prob={args.prob}, seed={args.seed}")
    print(f"  avg fraction of pixels corrupted per image: {np.mean(flipped_frac)*100:.2f}%")


if __name__ == "__main__":
    main()
