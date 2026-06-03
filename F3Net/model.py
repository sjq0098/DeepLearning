"""
F3Net: Fusion, Feedback and Focus for Salient Object Detection
Modern PyTorch (2.0+) reimplementation, with ablation hooks.

Architecture is configurable via three flags so we can run the paper's
Table 2 progression as a series of experiments:
  - use_cfm       : True  -> CrossFeatureModule (multiplicative fusion)
                    False -> AdditiveFusion     (same depth, '+' instead of '*')
  - num_decoders  : 1     -> single sub-decoder, no feedback
                    2     -> cascaded feedback decoder (paper's CFD)
  - use_mls       : True  -> multi-level auxiliary supervision (4 extra heads)
                    False -> only the final sub-decoder is supervised

Default flags reproduce the full F3Net used in the baseline.

Loss is selected at training time via `make_loss(name, gamma)`:
  - 'bce' : plain binary cross entropy
  - 'iou' : 1 - (inter + 1) / (union - inter + 1)
  - 'ppa' : pixel-position-aware weighted (BCE + IoU)   [paper's main loss]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from boundary import boundary_loss, ssim_loss, edge_consistency_loss, gt_boundary


# ──────────────────────────────────────────────────────────────
#  Weight Initialization
# ──────────────────────────────────────────────────────────────
def weight_init(module: nn.Module):
    """Kaiming initialization for Conv/Linear, ones/zeros for BN."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────
#  Backbone: ResNet-18 from torchvision (assignment-required backbone)
# ──────────────────────────────────────────────────────────────
class ResNet18Backbone(nn.Module):
    """
    Extract multi-level features from a pretrained ResNet-18.
    Returns features from layer1 ~ layer4 (stage 2 ~ stage 5 in the paper).
    Channels: 64, 128, 256, 512
    Strides:  4,  8,   16,  32
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # stage 1: conv1 + bn1 + relu + maxpool
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # stages 2-5
        self.layer1 = resnet.layer1   # ->  64 channels, stride 4
        self.layer2 = resnet.layer2   # -> 128 channels, stride 8
        self.layer3 = resnet.layer3   # -> 256 channels, stride 16
        self.layer4 = resnet.layer4   # -> 512 channels, stride 32

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c2 = self.layer1(x)   # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32
        return c2, c3, c4, c5


# ──────────────────────────────────────────────────────────────
#  Cross Feature Module (CFM)
# ──────────────────────────────────────────────────────────────
class CrossFeatureModule(nn.Module):
    """
    Selective feature fusion via element-wise multiplication.

    Given low-level features `fl` (clear boundaries, noisy background)
    and high-level features `fh` (clean background, coarse boundaries):
      1. Transform both through Conv-BN-ReLU
      2. Multiply to extract consensus (shared activation)
      3. Add consensus back to each branch (residual refinement)

    This suppresses noise in fl and sharpens boundaries in fh.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        C = channels

        # Branch for low-level (horizontal / "left")
        self.conv1h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv2h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv3h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv4h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))

        # Branch for high-level (vertical / "down")
        self.conv1v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv2v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv3v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv4v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))

        weight_init(self)

    def forward(self, fl: torch.Tensor, fh: torch.Tensor):
        if fh.shape[2:] != fl.shape[2:]:
            fh = F.interpolate(fh, size=fl.shape[2:], mode="bilinear", align_corners=False)

        # Transform
        h1 = self.conv1h(fl)
        h2 = self.conv2h(h1)
        v1 = self.conv1v(fh)
        v2 = self.conv2v(v1)

        # Cross: element-wise multiplication extracts consensus
        cross = h2 * v2

        # Refine each branch: consensus + skip connection
        h3 = self.conv3h(cross) + h1
        h4 = self.conv4h(h3)
        v3 = self.conv3v(cross) + v1
        v4 = self.conv4v(v3)

        return h4, v4


# ──────────────────────────────────────────────────────────────
#  Additive Fusion (ablation alternative to CFM)
# ──────────────────────────────────────────────────────────────
class AdditiveFusion(nn.Module):
    """
    Drop-in replacement for CFM that fuses via element-wise *addition*
    instead of multiplication. Same depth and channel count as CFM, so
    the only architectural change is `+` vs `*`. Used for the "± CFM"
    ablation row in the report.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        C = channels
        self.conv1h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv2h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv3h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv4h = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv1v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv2v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv3v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        self.conv4v = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(True))
        weight_init(self)

    def forward(self, fl: torch.Tensor, fh: torch.Tensor):
        if fh.shape[2:] != fl.shape[2:]:
            fh = F.interpolate(fh, size=fl.shape[2:], mode="bilinear", align_corners=False)
        h1 = self.conv1h(fl)
        h2 = self.conv2h(h1)
        v1 = self.conv1v(fh)
        v2 = self.conv2v(v1)
        fused = h2 + v2                       # ← only diff vs CFM
        h3 = self.conv3h(fused) + h1
        h4 = self.conv4h(h3)
        v3 = self.conv3v(fused) + v1
        v4 = self.conv4v(v3)
        return h4, v4


# ──────────────────────────────────────────────────────────────
#  Sub-Decoder (one stage of CFD)
# ──────────────────────────────────────────────────────────────
class SubDecoder(nn.Module):
    """
    One stage of the Cascaded Feedback Decoder.

    Bottom-up: aggregate features from stage5 -> stage2 via fusion module.
    Optionally accepts `feedback` from the previous decoder stage.
    """

    def __init__(self, channels: int = 64, use_cfm: bool = True):
        super().__init__()
        Fusion = CrossFeatureModule if use_cfm else AdditiveFusion
        # Names kept as `cfm*` for backward compatibility with the existing
        # baseline checkpoints (decoder1.cfm45.* etc.).
        self.cfm45 = Fusion(channels)
        self.cfm34 = Fusion(channels)
        self.cfm23 = Fusion(channels)
        weight_init(self)

    def forward(self, f2, f3, f4, f5, feedback=None):
        if feedback is not None:
            f5 = f5 + F.interpolate(feedback, size=f5.shape[2:], mode="bilinear", align_corners=False)
            f4 = f4 + F.interpolate(feedback, size=f4.shape[2:], mode="bilinear", align_corners=False)
            f3 = f3 + F.interpolate(feedback, size=f3.shape[2:], mode="bilinear", align_corners=False)
            f2 = f2 + F.interpolate(feedback, size=f2.shape[2:], mode="bilinear", align_corners=False)

        f4, f4v = self.cfm45(f4, f5)
        f3, f3v = self.cfm34(f3, f4v)
        f2, pred = self.cfm23(f2, f3v)
        return f2, f3, f4, f5, pred


# ──────────────────────────────────────────────────────────────
#  F3Net: Full Model (with ablation flags)
# ──────────────────────────────────────────────────────────────
class F3Net(nn.Module):
    """
    F3Net = ResNet-18 encoder + channel squeeze + N cascaded decoders.

    With default flags this is the full architecture matching the paper
    (R18 backbone substituted per assignment requirement).
    The flags `use_cfm`, `num_decoders`, `use_mls` enable the ablation
    progression: 'Bone -> +MLS -> +CFM -> +CFD -> +better loss'.

    Training output (a tuple) layout:
       (pred_1, ..., pred_N,  out2r, out3r, out4r, out5r)  if use_mls
       (pred_1, ..., pred_N)                                otherwise
    """

    def __init__(
        self,
        channels: int = 64,
        pretrained: bool = True,
        use_cfm: bool = True,
        num_decoders: int = 2,
        use_mls: bool = True,
        use_bas: bool = False,
    ):
        super().__init__()
        assert num_decoders in (1, 2), "num_decoders must be 1 or 2"
        self.use_cfm = use_cfm
        self.num_decoders = num_decoders
        self.use_mls = use_mls
        self.use_bas = use_bas

        # Backbone
        self.backbone = ResNet18Backbone(pretrained=pretrained)

        # Channel squeeze
        self.squeeze2 = nn.Sequential(nn.Conv2d( 64, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(128, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze5 = nn.Sequential(nn.Conv2d(512, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))

        # Sub-decoders
        self.decoder1 = SubDecoder(channels, use_cfm=use_cfm)
        self.head_p1 = nn.Conv2d(channels, 1, 3, padding=1)
        if num_decoders >= 2:
            self.decoder2 = SubDecoder(channels, use_cfm=use_cfm)
            self.head_p2 = nn.Conv2d(channels, 1, 3, padding=1)

        # Multi-level supervision aux heads
        if use_mls:
            self.head_r2 = nn.Conv2d(channels, 1, 3, padding=1)
            self.head_r3 = nn.Conv2d(channels, 1, 3, padding=1)
            self.head_r4 = nn.Conv2d(channels, 1, 3, padding=1)
            self.head_r5 = nn.Conv2d(channels, 1, 3, padding=1)

        # ③ Boundary-Aware auxiliary Supervision head (sibling of the saliency
        # head, off the final decoder feature). Training-only; predicts the
        # object contour, supervised by the GT morphological-gradient boundary.
        if use_bas:
            self.head_boundary = nn.Conv2d(channels, 1, 3, padding=1)

        # Initialize freshly-created layers (backbone is pretrained)
        weight_init(self.squeeze2)
        weight_init(self.squeeze3)
        weight_init(self.squeeze4)
        weight_init(self.squeeze5)
        weight_init(self.head_p1)
        if num_decoders >= 2:
            weight_init(self.head_p2)
        if use_mls:
            weight_init(self.head_r2)
            weight_init(self.head_r3)
            weight_init(self.head_r4)
            weight_init(self.head_r5)
        if use_bas:
            weight_init(self.head_boundary)

    def forward(self, x: torch.Tensor, out_size=None):
        size = x.shape[2:] if out_size is None else out_size

        # Encoder + squeeze
        c2, c3, c4, c5 = self.backbone(x)
        f2 = self.squeeze2(c2)
        f3 = self.squeeze3(c3)
        f4 = self.squeeze4(c4)
        f5 = self.squeeze5(c5)

        # Decoder cascade
        f2, f3, f4, f5, pred1_feat = self.decoder1(f2, f3, f4, f5)
        pred1 = F.interpolate(self.head_p1(pred1_feat), size=size, mode="bilinear", align_corners=False)
        preds = [pred1]
        final_feat = pred1_feat

        if self.num_decoders >= 2:
            f2, f3, f4, f5, pred2_feat = self.decoder2(f2, f3, f4, f5, pred1_feat)
            pred2 = F.interpolate(self.head_p2(pred2_feat), size=size, mode="bilinear", align_corners=False)
            preds.append(pred2)
            final_feat = pred2_feat

        if self.training:
            out = tuple(preds)
            if self.use_mls:
                out = out + (
                    F.interpolate(self.head_r2(f2), size=size, mode="bilinear", align_corners=False),
                    F.interpolate(self.head_r3(f3), size=size, mode="bilinear", align_corners=False),
                    F.interpolate(self.head_r4(f4), size=size, mode="bilinear", align_corners=False),
                    F.interpolate(self.head_r5(f5), size=size, mode="bilinear", align_corners=False),
                )
            if self.use_bas:
                # Boundary logit appended LAST (total_loss peels it off via use_bas).
                bnd = F.interpolate(self.head_boundary(final_feat), size=size, mode="bilinear", align_corners=False)
                out = out + (bnd,)
            return out

        return preds[-1]   # inference: final (refined) prediction


# ──────────────────────────────────────────────────────────────
#  Loss functions
# ──────────────────────────────────────────────────────────────
def bce_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, mask)


def iou_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Plain (un-weighted) IoU loss on the sigmoid'd prediction."""
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    return (1.0 - (inter + 1) / (union - inter + 1)).mean()


def ppa_loss(pred: torch.Tensor, mask: torch.Tensor, gamma: float = 5.0, kernel_size: int = 31):
    """
    Pixel Position Aware Loss = wBCE + wIoU.
    weight = 1 + gamma * |avg_pool(gt, k) - gt|  (boundaries get larger weight)
    """
    # Reflect-pad before pooling: avg_pool2d's own `padding` arg zero-pads and
    # counts those zeros in the mean (count_include_pad=True), which artificially
    # lowers local_mean at the image border -> spuriously inflates alpha there.
    # Reflect padding removes that border artifact.
    pad = kernel_size // 2
    mask_p = F.pad(mask, (pad, pad, pad, pad), mode="reflect")
    local_mean = F.avg_pool2d(mask_p, kernel_size=kernel_size, stride=1, padding=0)
    alpha = (local_mean - mask).abs()
    weight = 1.0 + gamma * alpha

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask * weight).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weight).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def make_loss(name: str, gamma: float = 5.0):
    """
    Returns a callable `(pred, mask) -> tensor`. Used by train.py to pick
    the per-prediction loss for an ablation.
    """
    name = name.lower()
    if name == "bce":
        return bce_loss
    if name == "iou":
        return iou_loss
    if name == "ppa":
        return lambda pred, mask: ppa_loss(pred, mask, gamma=gamma)
    raise ValueError(f"Unknown loss: {name!r} (expected bce/iou/ppa)")


def total_loss(outputs, mask, loss_fn=None, gamma: float = 5.0,
               num_decoders: int = 2, use_mls: bool = True,
               boundary_weight: float = 0.0, ssim_weight: float = 0.0,
               edge_weight: float = 0.0, bas_weight: float = 0.0,
               boundary_kernel: int = 3, image=None, use_bas: bool = False):
    """
    Aggregate the per-head losses, plus optional edge-sharpening terms.

    Base (unchanged):
        Total = mean(pred-head losses)
              + (1/2 L_r2 + 1/4 L_r3 + 1/8 L_r4 + 1/16 L_r5)   if use_mls

    Optional sharpening terms (all default to 0 -> exactly the old behaviour),
    applied to the *final* sub-decoder prediction:
        + boundary_weight * boundary_loss   (①  Boundary-IoU)
        + ssim_weight     * ssim_loss       (①  structural SSIM)
        + edge_weight     * edge_consistency_loss(image)   (A8, needs `image`)
        + bas_weight      * BCE(boundary_head, gt_boundary) (③, needs use_bas)

    Output tuple layout expected (matches F3Net.forward):
        (pred_1..pred_N, [out2r,out3r,out4r,out5r if mls], [bnd_logit if use_bas])
    """
    if loss_fn is None:
        loss_fn = lambda p, m: ppa_loss(p, m, gamma=gamma)

    outputs = list(outputs)
    boundary_logit = None
    if use_bas:
        boundary_logit = outputs[-1]          # BAS head is appended last
        outputs = outputs[:-1]

    # Predictions from sub-decoders
    pred_losses = [loss_fn(outputs[i], mask) for i in range(num_decoders)]
    total = sum(pred_losses) / len(pred_losses)

    if use_mls:
        aux = outputs[num_decoders:]
        assert len(aux) == 4, f"expected 4 aux heads with use_mls=True, got {len(aux)}"
        aux_weights = (1 / 2, 1 / 4, 1 / 8, 1 / 16)
        total = total + sum(loss_fn(a, mask) * w for a, w in zip(aux, aux_weights))

    # ── Edge-sharpening terms (applied to the final refined prediction) ──
    # Computed with autocast DISABLED (forced fp32). SSIM's windowed variance
    # terms are numerically fragile in fp16 — near-zero denominators produced a
    # sudden NaN at ~epoch 6 in the first A7 run. A plain `.float()` inside the
    # loss is not enough, because autocast re-casts conv2d back to fp16; only
    # this context actually keeps the ops in fp32.
    final_logit = outputs[num_decoders - 1]
    need_sharpen = (boundary_weight > 0 or ssim_weight > 0
                    or (edge_weight > 0 and image is not None)
                    or (use_bas and bas_weight > 0))
    if need_sharpen:
        with torch.autocast(device_type=final_logit.device.type, enabled=False):
            fl = final_logit.float()
            m = mask.float()
            if boundary_weight > 0:
                total = total + boundary_weight * boundary_loss(fl, m, boundary_kernel)
            if ssim_weight > 0:
                total = total + ssim_weight * ssim_loss(fl, m)
            if edge_weight > 0 and image is not None:
                total = total + edge_weight * edge_consistency_loss(fl, image.float(), m, boundary_kernel)
            if use_bas and bas_weight > 0:
                total = total + bas_weight * F.binary_cross_entropy_with_logits(
                    boundary_logit.float(), gt_boundary(m, boundary_kernel)
                )

    return total


# ──────────────────────────────────────────────────────────────
#  Quick sanity check (covers all 5 ablation variants)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (name, model kwargs, loss name, sharpening kwargs for total_loss)
    variants = [
        ("A1 bone (no CFM, N=1, no MLS, BCE)", dict(use_cfm=False, num_decoders=1, use_mls=False), "bce", {}),
        ("A2 +MLS                          ", dict(use_cfm=False, num_decoders=1, use_mls=True),  "bce", {}),
        ("A3 +CFM                          ", dict(use_cfm=True,  num_decoders=1, use_mls=True),  "bce", {}),
        ("A4 +CFD                          ", dict(use_cfm=True,  num_decoders=2, use_mls=True),  "bce", {}),
        ("A5 +PPA (full baseline)          ", dict(use_cfm=True,  num_decoders=2, use_mls=True),  "ppa", {}),
        ("A6 +BAS head (③)                 ", dict(use_cfm=True,  num_decoders=2, use_mls=True, use_bas=True), "ppa",
         dict(use_bas=True, bas_weight=0.5)),
        ("A7 +boundary+ssim (①)            ", dict(use_cfm=True,  num_decoders=2, use_mls=True), "ppa",
         dict(boundary_weight=1.0, ssim_weight=1.0)),
        ("A8 +edge-consistency (A8)        ", dict(use_cfm=True,  num_decoders=2, use_mls=True), "ppa",
         dict(edge_weight=1.0)),
    ]

    x = torch.randn(2, 3, 352, 352).to(device)
    mask = (torch.rand(2, 1, 352, 352) > 0.5).float().to(device)

    for name, kwargs, loss_name, extra in variants:
        model = F3Net(pretrained=False, **kwargs).to(device)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        loss_fn = make_loss(loss_name)
        model.train()
        outputs = model(x)
        loss = total_loss(outputs, mask, loss_fn=loss_fn,
                          num_decoders=kwargs["num_decoders"],
                          use_mls=kwargs["use_mls"],
                          image=x, boundary_kernel=3, **extra)
        loss.backward()
        model.eval()
        with torch.no_grad():
            pred = model(x)
        print(f"{name} | params {total_params:5.2f}M | n_train_outputs {len(outputs)} | "
              f"loss {loss.item():.3f} | infer_shape {tuple(pred.shape)}")
    print("[OK] All variants OK")
