"""Fusion U-Net (after Moon & Lee 2022): a U-Net whose decoder outputs at every level are
upsampled to full resolution, concatenated ("fusion") and turned into the final mask.
Small channel widths so it fits a 4 GB GPU."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class FusionUNet(nn.Module):
    def __init__(self, in_ch=3, base=16, depth=5):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]               # e.g. 16,32,64,128,256
        self.enc = nn.ModuleList([conv_block(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([conv_block(chs[i - 1] * 2, chs[i - 1]) for i in range(depth - 1, 0, -1)])
        self.dec_ch = [chs[i - 1] for i in range(depth - 1, 0, -1)]
        # fusion module: bring every decoder level to `base` channels, upsample to full size, concatenate
        self.proj = nn.ModuleList([nn.Conv2d(c, base, 1) for c in self.dec_ch])
        n = len(self.dec_ch)
        self.fuse = nn.Sequential(nn.Conv2d(base * n, base, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base, 1, 1)
        self.feat_ch = base

    def forward(self, x, return_features=False):
        size = x.shape[-2:]
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        outs = []
        for up, dec, proj, skip in zip(self.up, self.dec, self.proj, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
            outs.append(F.interpolate(proj(x), size=size, mode="bilinear", align_corners=False))
        feat = self.fuse(torch.cat(outs, dim=1))
        logits = self.out(feat)
        return (logits, feat) if return_features else logits


def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


@torch.no_grad()
def dsc_jsc(logits, target, thr=0.5, eps=1e-6):
    """Per-image Dice and Jaccard at a threshold, averaged over the batch."""
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum((1, 2, 3))
    union_d = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dsc = (2 * inter + eps) / (union_d + eps)
    jsc = (inter + eps) / (union_d - inter + eps)
    return dsc.mean().item(), jsc.mean().item()
