"""
F3Net Inference Script — generates saliency maps for test datasets.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import F3Net
from dataset import get_test_loader


@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = F3Net(pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    for datapath in args.datasets:
        dataset_name = os.path.basename(datapath)
        loader = get_test_loader(datapath, size=352, num_workers=args.workers)
        save_dir = os.path.join(args.outdir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        for image, mask, shape, name in loader:
            image = image.to(device)
            # Pass original shape so output matches ground truth size
            pred = model(image, out_size=(shape[0].item(), shape[1].item()))
            pred = torch.sigmoid(pred[0, 0]).cpu().numpy()  # (H, W) in [0, 1]
            pred = (pred * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir, name[0] + ".png"), pred)

        print(f"  {dataset_name}: {len(loader)} images -> {save_dir}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="F3Net Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["./data/ECSSD", "./data/PASCAL-S", "./data/DUTS",
                                 "./data/HKU-IS", "./data/DUT-OMRON"],
                        help="Paths to test datasets")
    parser.add_argument("--outdir", type=str, default="./results", help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
