"""
Dataset for Salient Object Detection training/evaluation.

Modern PyTorch reimplementation:
  - Uses torchvision.transforms v2 style
  - Cleaner augmentation pipeline
  - Supports DUTS, ECSSD, PASCAL-S, HKU-IS, DUT-OMRON
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ImageNet mean/std (BGR order, matching cv2 loading)
MEAN = np.array([[[124.55, 118.90, 102.94]]], dtype=np.float32)
STD = np.array([[[56.77, 55.97, 57.50]]], dtype=np.float32)

# Multi-scale training sizes (paper: random from 5 scales)
TRAIN_SCALES = [224, 256, 288, 320, 352]


class SODDataset(Dataset):
    """
    Salient Object Detection dataset.

    Expected directory structure (matches the assignment ECSSD layout):
        datapath/
            images/              *.jpg
            ground_truth_mask/   *.png
            train.txt  (or test.txt)

    Each line in the txt file is a filename stem (no extension).
    Run prepare_split.py to generate train.txt / test.txt.
    """

    def __init__(self, datapath: str, mode: str = "train", size: int = 352):
        self.datapath = datapath
        self.mode = mode
        self.size = size

        txt_file = os.path.join(datapath, f"{mode}.txt")
        with open(txt_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        # Load image (keep BGR — MEAN/STD are in BGR order, matching the
        # original F3Net implementation; converting to RGB here would mis-align
        # per-channel normalization).
        image = cv2.imread(os.path.join(self.datapath, "images", name + ".jpg")).astype(np.float32)
        mask = cv2.imread(os.path.join(self.datapath, "ground_truth_mask", name + ".png"), 0).astype(np.float32)

        # Normalize
        image = (image - MEAN) / STD
        mask = mask / 255.0

        if self.mode == "train":
            image, mask = self._random_crop(image, mask)
            image, mask = self._random_flip(image, mask)
            return image, mask  # numpy arrays; collate_fn handles resize + tensor
        else:
            # Keep the ground-truth mask at its native resolution so the
            # evaluator can compare against the un-resampled labels. The
            # model still takes a fixed-size input but its prediction will
            # be bilinearly upsampled back to `orig_shape` in test.py.
            orig_shape = (mask.shape[0], mask.shape[1])  # (H, W) of original
            image_resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image_resized = torch.from_numpy(image_resized.copy()).permute(2, 0, 1).float()
            mask_orig = torch.from_numpy(mask.copy()).float()  # (H, W), native size
            return image_resized, mask_orig, orig_shape, name

    @staticmethod
    def _random_crop(image, mask):
        H, W, _ = image.shape
        rh = np.random.randint(H // 8) if H // 8 > 0 else 0
        rw = np.random.randint(W // 8) if W // 8 > 0 else 0
        oh = np.random.randint(rh) if rh > 0 else 0
        ow = np.random.randint(rw) if rw > 0 else 0
        return image[oh : H + oh - rh, ow : W + ow - rw, :], mask[oh : H + oh - rh, ow : W + ow - rw]

    @staticmethod
    def _random_flip(image, mask):
        if np.random.rand() < 0.5:
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy()
        return image, mask


def train_collate_fn(batch):
    """
    Custom collate: randomly pick one of 5 scales per batch,
    resize all images/masks to that size, stack into tensors.
    """
    size = TRAIN_SCALES[np.random.randint(len(TRAIN_SCALES))]
    images, masks = zip(*batch)
    resized_images, resized_masks = [], []
    for img, msk in zip(images, masks):
        resized_images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))
        resized_masks.append(cv2.resize(msk, (size, size), interpolation=cv2.INTER_LINEAR))
    images_t = torch.from_numpy(np.stack(resized_images)).permute(0, 3, 1, 2).float()
    masks_t = torch.from_numpy(np.stack(resized_masks)).unsqueeze(1).float()
    return images_t, masks_t


def get_train_loader(datapath: str, batch_size: int = 32, num_workers: int = 4):
    dataset = SODDataset(datapath, mode="train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def get_test_loader(datapath: str, size: int = 352, num_workers: int = 4):
    dataset = SODDataset(datapath, mode="test", size=size)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
