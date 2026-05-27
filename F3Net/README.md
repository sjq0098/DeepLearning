# F³Net — Modern PyTorch Reimplementation

A clean, modern PyTorch (2.0+) reimplementation of [F³Net: Fusion, Feedback and Focus for Salient Object Detection](https://arxiv.org/abs/1911.11445) (AAAI 2020).

## What Changed from the Original

| Aspect | Original (2019) | This Version |
|--------|-----------------|--------------|
| PyTorch | 1.3 | 2.0+ |
| Mixed Precision | `nvidia/apex` (O2) | `torch.cuda.amp` (native) |
| TensorBoard | `tensorboardX` | `torch.utils.tensorboard` (native) |
| ResNet weights | Manual download + `load_state_dict` | `torchvision.models` auto-download |
| BCE API | `reduce='none'` (deprecated) | `reduction='none'` |
| Compilation | N/A | Optional `torch.compile()` |
| Dependencies | apex, tensorboardX, cv2, numpy | **Only** torch, torchvision, cv2, numpy |

The core architecture (CFM, CFD) and loss function (PPA) are **identical** to the paper.

## Files

```
model.py    # F3Net model + PPA loss  (~230 lines)
dataset.py  # SOD dataset + augmentation
train.py    # Training with native AMP
test.py     # Inference / saliency map generation
```

## Quick Start

### Install

```bash
pip install torch torchvision opencv-python numpy
```

### Train

```bash
python train.py --datapath ./data/DUTS --epochs 32 --batch_size 32
# With torch.compile (PyTorch 2.0+):
python train.py --datapath ./data/DUTS --compile
```

### Test

```bash
python test.py --checkpoint ./checkpoints/model_epoch32.pth
```

## Architecture Overview

```
Input (352x352)
    │
    ▼
ResNet-50 Backbone ──► c2(256ch), c3(512ch), c4(1024ch), c5(2048ch)
    │
    ▼ (1x1 conv squeeze to 64ch each)
    │
    ▼
Decoder 1 (no feedback)
  CFM(f4, f5) → CFM(f3, f4') → CFM(f2, f3') → pred1
    │
    ▼ (feedback: pred1 features added back to f2~f5)
    │
Decoder 2 (with feedback)
  CFM(f4+fb, f5+fb) → CFM(f3+fb, f4') → CFM(f2+fb, f3') → pred2 ← final output
```

**CFM key idea**: `fused = transform(fl) * transform(fh)` — element-wise multiplication selects consensus, suppressing noise and sharpening boundaries.

**PPA loss key idea**: `weight = 1 + 5 * |avg_pool(gt, 31x31) - gt|` — boundary/thin-structure pixels get higher loss weight automatically.
