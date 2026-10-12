"""Step 1: sanity-check the core idea.

Runs MiDaS on face images, converts relative depth to a curvature map, and saves a figure:
    input | depth | curvature (raw) | curvature (head-outline edges suppressed)

Usage (from the project root):
    python scripts/01_midas_curvature_demo.py --images samples --out outputs/demo
    python scripts/01_midas_curvature_demo.py --model MiDaS_small --out outputs/demo_small
    python scripts/01_midas_curvature_demo.py --sigma 1 --edge-frac 0.2

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
from models.curvature import (  # noqa: E402
    curvature_for_display, curvature_for_display_masked, mean_curvature,
    normalize01, silhouette_mask, suppress_edges,
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_midas(name, device):
    model = torch.hub.load("intel-isl/MiDaS", name, trust_repo=True)
    model.to(device).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = tf.small_transform if name == "MiDaS_small" else tf.dpt_transform
    return model, transform


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
    ap.add_argument("--model", default="DPT_Hybrid",
                    choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
    ap.add_argument("--sigma", type=float, default=2.0, help="smoothing before curvature")
    ap.add_argument("--size", type=int, default=256, help="resize longer side before MiDaS")
    ap.add_argument("--edge-frac", type=float, default=0.25,
                    help="depth-gradient threshold (fraction of max) for head-outline edges; lower = suppress more")
    ap.add_argument("--dilate", type=int, default=9, help="grow the suppressed edge band by this many pixels")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)[: args.n]
    if not paths:
        sys.exit(f"No images found in {args.images}. Put a few face images there first.")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model, transform = load_midas(args.model, device)

    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = args.size / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

        depth = estimate_depth(model, transform, img, device)   # full image keeps context
        curv = mean_curvature(depth, sigma=args.sigma)
        mask = silhouette_mask(depth, sigma=args.sigma, frac=args.edge_frac, dilate=args.dilate)
        curv_clean = suppress_edges(curv, mask)

        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(img); ax[0].set_title("input")
        ax[1].imshow(normalize01(depth), cmap="magma"); ax[1].set_title(f"depth ({args.model})")
        ax[2].imshow(curvature_for_display(curv), cmap="coolwarm", vmin=0, vmax=1)
        ax[2].set_title("curvature (raw)")
        ax[3].imshow(curvature_for_display_masked(curv_clean, mask), cmap="coolwarm", vmin=0, vmax=1)
        ax[3].set_title("curvature (outline suppressed)")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        out_file = Path(args.out) / f"{p.stem}_depth_curv.png"
        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        print("saved", out_file)


if __name__ == "__main__":
    main()
