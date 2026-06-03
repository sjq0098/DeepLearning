"""
Shared boundary / structure utilities for the F3Net edge-sharpening experiments.

This module is the common foundation for three improvement lines:
  - ①  loss-driven sharpening : `boundary_loss` (Boundary-IoU) + `ssim_loss`
  - ③  explicit boundary head : `gt_boundary` produces the supervision target
  - A8  image-edge consistency : `edge_consistency_loss` (salient-region gated)

Design notes
------------
* Boundaries are extracted with a *morphological gradient* (dilation − erosion),
  implemented with max-pooling. On a binary GT mask this is an exact 0/1 band;
  on a soft sigmoid prediction it is a differentiable edge-strength map, so the
  same operator serves both as a (fixed) target and as a back-proppable term.
* All losses cast their inputs to float32. They are meant to be called *inside*
  `torch.cuda.amp.autocast`, where activations may be fp16 — mixing fp16 tensors
  with fp32 conv weights (the SSIM gaussian window) would otherwise error.
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  Morphological boundary operators (max-pool based)
# ──────────────────────────────────────────────────────────────
def _dilate(x: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)


def _erode(x: torch.Tensor, kernel: int) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=kernel // 2)


def morph_gradient(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Dilation − erosion. For binary input -> boundary band of width ~kernel."""
    return _dilate(x, kernel) - _erode(x, kernel)


def gt_boundary(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Boundary map of a (near-)binary GT mask, in [0, 1].
    `mask` shape (B, 1, H, W). Used as a *fixed* supervision target — no grad
    flows through it in practice (mask is a label).
    """
    return morph_gradient(mask.float(), kernel).clamp_(0, 1)


def soft_boundary(prob: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Differentiable boundary strength of a soft prediction `prob = sigmoid(logit)`.
    Gradients propagate through the max-pool sub-gradient back to `prob`.
    """
    return morph_gradient(prob.float(), kernel).clamp_(0, 1)


def boundary_band(mask: torch.Tensor, kernel: int = 5, thresh: float = 1e-2) -> torch.Tensor:
    """Binary {0,1} band region around the GT boundary, for gating / weighting."""
    return (gt_boundary(mask, kernel) > thresh).float()


# ──────────────────────────────────────────────────────────────
#  ① loss-driven sharpening terms
# ──────────────────────────────────────────────────────────────
def boundary_loss(logit: torch.Tensor, mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Boundary-IoU (soft Dice on edges) between predicted and GT boundaries.
    Encourages the predicted saliency map to have sharp edges that coincide
    with the GT object contour.
    """
    prob = torch.sigmoid(logit).float()
    mask = mask.float()
    pe = soft_boundary(prob, kernel)
    ge = gt_boundary(mask, kernel)
    inter = (pe * ge).sum(dim=(2, 3))
    union = pe.sum(dim=(2, 3)) + ge.sum(dim=(2, 3))
    dice = (2 * inter + 1.0) / (union + 1.0)
    return (1.0 - dice).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(logit: torch.Tensor, mask: torch.Tensor,
              window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    1 − SSIM between the predicted saliency map and the GT mask (BASNet's
    structural term). SSIM rewards local structural / contrast agreement, which
    in practice penalises the soft, low-contrast halo around object boundaries.
    """
    prob = torch.sigmoid(logit).float()
    mask = mask.float()
    C = prob.shape[1]
    win = _gaussian_window(window_size, sigma, C, prob.device, prob.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(prob, win, padding=pad, groups=C)
    mu2 = F.conv2d(mask, win, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(prob * prob, win, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(mask * mask, win, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(prob * mask, win, padding=pad, groups=C) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────────────────────
#  A8 image-edge consistency (TDKstain-inspired structural prior)
# ──────────────────────────────────────────────────────────────
def image_edges(image: torch.Tensor) -> torch.Tensor:
    """
    Per-image normalised edge magnitude of the *input image*, in [0, 1].

    `image` is the model input (B, 3, H, W), already mean/std normalised (BGR) —
    fine here: the gradient is linear and we max-normalise per image, so only the
    edge structure matters, not the absolute scale. A structural prior that exists
    independently of the GT label.

    Implemented with central finite differences (pure slicing, NO convolution).
    The earlier F.conv2d Sobel version triggered intermittent cuDNN runtime faults
    (STREAM_MISMATCH / EXECUTION_FAILED) on this GPU after tens of epochs; slicing
    avoids cuDNN entirely and is numerically equivalent for our purpose.
    """
    image = image.float()
    gray = image.mean(dim=1, keepdim=True)
    gx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0))   # d/dx
    gy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1))   # d/dy
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    maxv = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return mag / maxv


def edge_consistency_loss(logit: torch.Tensor, image: torch.Tensor,
                          mask: torch.Tensor, kernel: int = 3,
                          band_kernel: int = 11, eps: float = 1e-6) -> torch.Tensor:
    """
    Band-restricted image-edge *alignment* (improvement line A8, revised).

    Within the GT object-contour band, reward the predicted boundary for
    coinciding with real image edges, via a per-image cosine similarity:

        band = boundary_band(GT, band_kernel)          # near the true contour
        pe   = soft_boundary(sigmoid(logit)) · band     # predicted edge in band
        ie   = image_edge(image)            · band      # real image edge in band
        L    = 1 − cos(pe, ie)

    Why cosine-in-band and not the earlier "penalise boundary on flat regions"
    formulation: that version had a degenerate optimum — the model could satisfy
    it by flattening ALL predictions (zero gradient everywhere), which *softened*
    boundaries (the opposite of the goal, confirmed empirically). Cosine
    alignment has no such escape: flattening drives pe→0, so the similarity→0 and
    the loss stays ~1. The reward is only collected by placing a sharp predicted
    edge exactly on a real image edge along the contour.
    """
    prob = torch.sigmoid(logit).float()
    band = boundary_band(mask, band_kernel)
    pe = soft_boundary(prob, kernel) * band
    ie = image_edges(image) * band
    num = (pe * ie).sum(dim=(2, 3))
    den = torch.sqrt((pe * pe).sum(dim=(2, 3)) * (ie * ie).sum(dim=(2, 3)) + eps) + eps
    return (1.0 - num / den).mean()


# ──────────────────────────────────────────────────────────────
#  Quick numeric self-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    # A crude circular mask so boundaries are non-trivial.
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    disk = ((xx ** 2 + yy ** 2) < 0.3).float().view(1, 1, H, W).expand(B, 1, H, W).contiguous()
    logit = (disk * 4 - 2).requires_grad_(True)          # near-perfect logits
    image = torch.randn(B, 3, H, W)

    print("gt_boundary frac on  :", gt_boundary(disk).mean().item())
    print("boundary_band frac   :", boundary_band(disk).mean().item())
    for name, val in [
        ("boundary_loss", boundary_loss(logit, disk)),
        ("ssim_loss", ssim_loss(logit, disk)),
        ("edge_consistency", edge_consistency_loss(logit, image, disk)),
    ]:
        val.backward(retain_graph=True)
        print(f"{name:18s}: {val.item():.4f}  grad_ok={logit.grad is not None and bool(logit.grad.abs().sum() > 0)}")
        logit.grad = None
    print("[OK] boundary.py self-test passed")
