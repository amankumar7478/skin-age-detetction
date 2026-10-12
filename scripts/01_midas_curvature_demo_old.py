"""Step 1: sanity-check the core idea.

Runs MiDaS-small on a few face images, converts the relative depth to a curvature map,
and saves a side-by-side figure:  image | depth | curvature.

Usage (from the project root):
    python scripts/01_midas_curvature_demo.py --images samples --out outputs/demo --n 8

Needs internet the first time (torch.hub downloads MiDaS code + weights from GitHub).
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.curvature import curvature_for_display, mean_curvature, normalize01  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, transforms.small_transform


@torch.no_grad()
def estimate_depth(model, transform, img_rgb, device):
    batch = transform(img_rgb).to(device)
    pred = model(batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()
    return pred.cpu().numpy()  # relative inverse depth (bigger = closer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples", help="folder with face images")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=2.0, help="smoothing before curvature")
    ap.add_argument("--size", type=int, default=256, help="resize longer side before MiDaS")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)[: args.n]
    if not paths:
        sys.exit(f"No images found in {args.images}. Put a few face images there first.")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model, transform = load_midas(device)

    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = args.size / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

        depth = estimate_depth(model, transform, img, device)
        curv = mean_curvature(depth, sigma=args.sigma)

        fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
        ax[0].imshow(img); ax[0].set_title("input")
        ax[1].imshow(normalize01(depth), cmap="magma"); ax[1].set_title("MiDaS depth (relative)")
        ax[2].imshow(curvature_for_display(curv), cmap="coolwarm", vmin=0, vmax=1)
        ax[2].set_title("mean curvature")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        out_file = Path(args.out) / f"{p.stem}_depth_curv.png"
        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        print("saved", out_file)


if __name__ == "__main__":
    main()
