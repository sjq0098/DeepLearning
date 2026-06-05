"""
Build a side-by-side montage to eyeball the injected boundary noise.
Columns: original image | clean GT | noisy GT | flipped pixels (red on image).
Rows: a few training images spanning small->large object size.

Usage:
    python viz_noisy_gt.py --noisy_dir gt_noisy_d3p50 --n 5 --out noise_viz/compare_d3p50.png
"""
import argparse
import os

import cv2
import numpy as np


def cell(img, size=256):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", default="./data")
    p.add_argument("--noisy_dir", default="gt_noisy_d3p50")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", default="noise_viz/compare.png")
    args = p.parse_args()

    with open(os.path.join(args.datapath, "train.txt")) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    # pick n images spread across object-size range
    fg = []
    for nm in names:
        m = cv2.imread(os.path.join(args.datapath, "ground_truth_mask", nm + ".png"), 0)
        fg.append((nm, (m >= 128).mean()))
    fg.sort(key=lambda x: x[1])
    idx = np.linspace(0, len(fg) - 1, args.n).astype(int)
    picks = [fg[i] for i in idx]

    rows = []
    for nm, frac in picks:
        image = cv2.imread(os.path.join(args.datapath, "images", nm + ".jpg"))
        clean = cv2.imread(os.path.join(args.datapath, "ground_truth_mask", nm + ".png"), 0)
        noisy = cv2.imread(os.path.join(args.datapath, args.noisy_dir, nm + ".png"), 0)

        # red overlay where noisy != clean, drawn on the original image
        flipped = (noisy >= 128) != (clean >= 128)
        overlay = image.copy()
        overlay[flipped] = (0, 0, 255)
        overlay = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)

        row = np.hstack([cell(image), cell(clean), cell(noisy), cell(overlay)])
        cv2.putText(row, f"{nm}  fg={frac*100:.1f}%  flipped={flipped.mean()*100:.2f}%",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)

    montage = np.vstack(rows)
    # column header
    hdr = np.full((24, montage.shape[1], 3), 30, np.uint8)
    for j, t in enumerate(["image", "clean GT", "noisy GT", "flipped (red)"]):
        cv2.putText(hdr, t, (j * 256 + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    montage = np.vstack([hdr, montage])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"Saved {args.out}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
