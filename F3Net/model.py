"""
F3Net: Fusion, Feedback and Focus for Salient Object Detection
Modern PyTorch (2.0+) reimplementation.

Key components:
  - CFM  (Cross Feature Module):  element-wise multiplication based selective fusion
  - CFD  (Cascaded Feedback Decoder): multi-stage feedback refinement
  - PPA  (Pixel Position Aware Loss): structure-aware weighted BCE + wIoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        """
        Args:
            fl: low-level features  (B, C, H, W)
            fh: high-level features (B, C, H', W')  — will be upsampled to fl's size
        Returns:
            fl_refined, fh_refined: both at fl's spatial resolution
        """
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
        h3 = self.conv3h(cross) + h1   # residual from first transform
        h4 = self.conv4h(h3)
        v3 = self.conv3v(cross) + v1
        v4 = self.conv4v(v3)

        return h4, v4


# ──────────────────────────────────────────────────────────────
#  Sub-Decoder (one stage of CFD)
# ──────────────────────────────────────────────────────────────
class SubDecoder(nn.Module):
    """
    One stage of the Cascaded Feedback Decoder.

    Bottom-up: aggregate features from stage5 -> stage2 via CFM.
    Optionally accepts `feedback` from the previous decoder stage.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.cfm45 = CrossFeatureModule(channels)
        self.cfm34 = CrossFeatureModule(channels)
        self.cfm23 = CrossFeatureModule(channels)
        weight_init(self)

    def forward(self, f2, f3, f4, f5, feedback=None):
        """
        Args:
            f2..f5: multi-level features (B, C, Hi, Wi), resolution decreasing
            feedback: aggregated prediction from previous decoder (B, C, H2, W2) or None
        Returns:
            f2, f3, f4, f5: refined multi-level features
            pred: prediction features at f2's resolution
        """
        if feedback is not None:
            # Feedback: downsample aggregated features to each level's resolution
            f5 = f5 + F.interpolate(feedback, size=f5.shape[2:], mode="bilinear", align_corners=False)
            f4 = f4 + F.interpolate(feedback, size=f4.shape[2:], mode="bilinear", align_corners=False)
            f3 = f3 + F.interpolate(feedback, size=f3.shape[2:], mode="bilinear", align_corners=False)
            f2 = f2 + F.interpolate(feedback, size=f2.shape[2:], mode="bilinear", align_corners=False)

        # Bottom-up aggregation: high -> low
        f4, f4v = self.cfm45(f4, f5)   # fuse stage4 with stage5
        f3, f3v = self.cfm34(f3, f4v)  # fuse stage3 with fused-4
        f2, pred = self.cfm23(f2, f3v) # fuse stage2 with fused-3

        return f2, f3, f4, f5, pred


# ──────────────────────────────────────────────────────────────
#  F3Net: Full Model
# ──────────────────────────────────────────────────────────────
class F3Net(nn.Module):
    """
    F3Net = ResNet-50 encoder + channel squeeze + 2 cascaded feedback decoders.

    During training, returns 6 predictions for multi-level supervision:
      pred1, pred2 (decoder outputs) + out2r, out3r, out4r, out5r (auxiliary)

    During inference (eval mode), returns only pred2 (the final refined prediction).
    """

    def __init__(self, channels: int = 64, pretrained: bool = True):
        super().__init__()

        # Backbone (ResNet-18 per assignment requirement)
        self.backbone = ResNet18Backbone(pretrained=pretrained)

        # Channel squeeze: reduce backbone channels to uniform `channels`
        self.squeeze2 = nn.Sequential(nn.Conv2d( 64, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(128, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(256, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))
        self.squeeze5 = nn.Sequential(nn.Conv2d(512, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True))

        # Two cascaded feedback decoders (N=2 is optimal per ablation)
        self.decoder1 = SubDecoder(channels)
        self.decoder2 = SubDecoder(channels)

        # Prediction heads
        self.head_p1 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head_p2 = nn.Conv2d(channels, 1, 3, padding=1)

        # Auxiliary heads for multi-level supervision
        self.head_r2 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head_r3 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head_r4 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head_r5 = nn.Conv2d(channels, 1, 3, padding=1)

        # Initialize decoder + heads (backbone is pretrained)
        weight_init(self.squeeze2)
        weight_init(self.squeeze3)
        weight_init(self.squeeze4)
        weight_init(self.squeeze5)
        weight_init(self.head_p1)
        weight_init(self.head_p2)
        weight_init(self.head_r2)
        weight_init(self.head_r3)
        weight_init(self.head_r4)
        weight_init(self.head_r5)

    def forward(self, x: torch.Tensor, out_size=None):
        """
        Args:
            x: input image (B, 3, H, W)
            out_size: output spatial size, defaults to input size
        Returns:
            Training:  (pred1, pred2, out2r, out3r, out4r, out5r) — all (B,1,H,W) logits
            Inference: pred2 — (B, 1, H, W) logits
        """
        size = x.shape[2:] if out_size is None else out_size

        # Encoder
        c2, c3, c4, c5 = self.backbone(x)
        f2 = self.squeeze2(c2)
        f3 = self.squeeze3(c3)
        f4 = self.squeeze4(c4)
        f5 = self.squeeze5(c5)

        # Decoder 1 (no feedback)
        f2, f3, f4, f5, pred1_feat = self.decoder1(f2, f3, f4, f5)

        # Decoder 2 (with feedback from decoder 1)
        f2, f3, f4, f5, pred2_feat = self.decoder2(f2, f3, f4, f5, pred1_feat)

        # Upsample predictions to output size
        pred1 = F.interpolate(self.head_p1(pred1_feat), size=size, mode="bilinear", align_corners=False)
        pred2 = F.interpolate(self.head_p2(pred2_feat), size=size, mode="bilinear", align_corners=False)

        if self.training:
            # Auxiliary outputs for multi-level supervision
            out2r = F.interpolate(self.head_r2(f2), size=size, mode="bilinear", align_corners=False)
            out3r = F.interpolate(self.head_r3(f3), size=size, mode="bilinear", align_corners=False)
            out4r = F.interpolate(self.head_r4(f4), size=size, mode="bilinear", align_corners=False)
            out5r = F.interpolate(self.head_r5(f5), size=size, mode="bilinear", align_corners=False)
            return pred1, pred2, out2r, out3r, out4r, out5r
        else:
            return pred2


# ──────────────────────────────────────────────────────────────
#  Pixel Position Aware Loss (PPA)
# ──────────────────────────────────────────────────────────────
def ppa_loss(pred: torch.Tensor, mask: torch.Tensor, gamma: float = 5.0, kernel_size: int = 31):
    """
    Pixel Position Aware Loss = wBCE + wIoU

    Each pixel gets weight (1 + gamma * alpha), where:
        alpha(i,j) = |avg_pool(gt, neighborhood) - gt(i,j)|

    Pixels at boundaries / thin structures / holes get higher alpha.

    Args:
        pred: logits (B, 1, H, W) — NOT sigmoid'd
        mask: ground truth (B, 1, H, W) in [0, 1]
        gamma: controls hard-pixel emphasis (paper uses 5)
        kernel_size: neighborhood size for computing alpha (paper uses 31)
    """
    padding = kernel_size // 2

    # Compute per-pixel weight alpha = |local_mean(gt) - gt|
    # alpha is high at boundaries/thin structures, low at flat regions
    local_mean = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    alpha = (local_mean - mask).abs()
    weight = 1.0 + gamma * alpha  # (B, 1, H, W)

    # ── Weighted BCE ──
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    # ── Weighted IoU ──
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask * weight).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weight).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1) / (union - inter + 1)  # +1 for numerical stability

    return (wbce + wiou).mean()


def total_loss(outputs, mask, gamma: float = 5.0):
    """
    Compute the full F3Net loss with multi-level supervision.

    Loss = (L_pred1 + L_pred2) / 2
         + L_out2r / 2 + L_out3r / 4 + L_out4r / 8 + L_out5r / 16
    """
    pred1, pred2, out2r, out3r, out4r, out5r = outputs

    loss_p1 = ppa_loss(pred1, mask, gamma)
    loss_p2 = ppa_loss(pred2, mask, gamma)
    loss_r2 = ppa_loss(out2r, mask, gamma)
    loss_r3 = ppa_loss(out3r, mask, gamma)
    loss_r4 = ppa_loss(out4r, mask, gamma)
    loss_r5 = ppa_loss(out5r, mask, gamma)

    return (loss_p1 + loss_p2) / 2 + loss_r2 / 2 + loss_r3 / 4 + loss_r4 / 8 + loss_r5 / 16


# ──────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = F3Net(pretrained=False).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable / 1e6:.2f}M")

    # Forward pass test
    x = torch.randn(2, 3, 352, 352).to(device)
    mask = torch.rand(2, 1, 352, 352).to(device)

    model.train()
    outputs = model(x)
    loss = total_loss(outputs, mask)
    print(f"Training output shapes: {[o.shape for o in outputs]}")
    print(f"Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(x)
    print(f"Inference output shape: {pred.shape}")
    print("✓ All checks passed!")
