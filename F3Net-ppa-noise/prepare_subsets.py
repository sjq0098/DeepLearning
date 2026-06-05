"""
Generate *nested* training subsets for the "architecture is data-hungry" study (②).

Reads the existing 700-image train.txt and writes train_<n>.txt for each subset
size. The subsets are nested (train_175 ⊂ train_350 ⊂ train_525 ⊂ train_700), so
the only thing that changes between runs is how much data is added — a clean
controlled variable. The test set (test.txt) is untouched and shared by all runs.

Usage:
    python prepare_subsets.py --datapath ./data --sizes 175 350 525 700
"""
import argparse
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", default="./data")
    p.add_argument("--sizes", type=int, nargs="+", default=[175, 350, 525, 700])
    p.add_argument("--seed", type=int, default=42, help="Must match prepare_split.py for reproducibility")
    args = p.parse_args()

    train_txt = os.path.join(args.datapath, "train.txt")
    with open(train_txt) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(names)} training names from {train_txt}")

    max_size = max(args.sizes)
    if max_size > len(names):
        raise ValueError(f"requested subset size {max_size} > available {len(names)}")

    rng = random.Random(args.seed)
    shuffled = names.copy()
    rng.shuffle(shuffled)

    for n in sorted(args.sizes):
        subset = sorted(shuffled[:n])      # nested prefix of the shuffled order
        out = os.path.join(args.datapath, f"train_{n}.txt")
        with open(out, "w") as f:
            f.write("\n".join(subset) + "\n")
        print(f"train_{n}.txt -> {len(subset)} samples ({out})")

    print(f"\nSeed: {args.seed}. Subsets are nested (each is a prefix of the same shuffle).")


if __name__ == "__main__":
    main()
