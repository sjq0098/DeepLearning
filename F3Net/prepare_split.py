"""
Generate train/test split (700/300) for the assignment dataset.

The assignment specifies the ECSSD 1000-image set randomly split into
700 train / 300 test. We fix the RNG seed so the split is reproducible
across team members.

Usage:
    python prepare_split.py --datapath ./data
"""
import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="./data", help="Dataset root containing images/ and ground_truth_mask/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the 700/300 split")
    parser.add_argument("--n_train", type=int, default=700)
    args = parser.parse_args()

    images_dir = os.path.join(args.datapath, "images")
    masks_dir = os.path.join(args.datapath, "ground_truth_mask")

    image_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
    mask_stems = set(os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith(".png"))

    paired = [s for s in image_stems if s in mask_stems]
    missing = set(image_stems) - mask_stems
    if missing:
        print(f"WARN: {len(missing)} images have no matching mask, skipping (e.g. {sorted(missing)[:3]})")
    print(f"Total paired samples: {len(paired)}")

    rng = random.Random(args.seed)
    shuffled = paired.copy()
    rng.shuffle(shuffled)
    train_names = sorted(shuffled[: args.n_train])
    test_names = sorted(shuffled[args.n_train :])

    for split, items in [("train", train_names), ("test", test_names)]:
        path = os.path.join(args.datapath, f"{split}.txt")
        with open(path, "w") as f:
            f.write("\n".join(items) + "\n")
        print(f"{split}.txt -> {len(items)} samples ({path})")

    print(f"\nSeed: {args.seed} (change with --seed to regenerate)")


if __name__ == "__main__":
    main()
