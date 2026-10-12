"""Step 4: run the (frozen) depth model ONCE over all usable images and cache depth + curvature.

For every image it saves <out>/<id>.npz with
    depth    - relative depth scaled to 0..1 (float16)
    curv_s2, curv_s4 ... - mean curvature at each smoothing level in --sigmas, zero outside `keep` (float32)
    keep     - bool: inside the face AND away from head-outline depth jumps
    face     - bool: face region from the face-parsing labels
plus <out>/_stats.json (curvature percentiles per smoothing level) and <out>/_check.png (picture to eyeball).

Small sigma keeps finer structure but also the blocky depth-model artifacts; large sigma is smooth but
mostly face shape. We keep two scales and let the experiments show which is useful.

    python scripts/07_precompute_curvature.py --limit 20     # quick test
    python scripts/07_precompute_curvature.py                # full run (a few minutes; GPU must be free)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.curvature import curvature_for_display_masked, mean_curvature, normalize01, silhouette_mask  # noqa: E402

FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]   # skin, brows, eyes, nose, mouth, lips (CelebAMask-HQ ids)

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--photos", default="data/ffhq1000")
ap.add_argument("--parsed", default="face parsed label")
ap.add_argument("--out", default="outputs/curv")
ap.add_argument("--model", default="DPT_Hybrid", choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
ap.add_argument("--size", type=int, default=256, help="working resolution (same as the visual checks)")
ap.add_argument("--sigmas", default="2,4", help="comma-separated smoothing levels to store")
ap.add_argument("--edge-frac", type=float, default=0.35)
ap.add_argument("--dilate", type=int, default=5)
ap.add_argument("--erode", type=int, default=12, help="shrink the face region to stay off the jaw/cheek rim")
ap.add_argument("--limit", type=int, default=0, help="only process this many images (quick test)")
a = ap.parse_args()
sigmas = [float(s) for s in a.sigmas.split(",")]
names = [f"curv_s{int(s) if s == int(s) else s}" for s in sigmas]

ids = pd.read_csv(a.folds)["image_number"].tolist()
if a.limit:
    ids = ids[: a.limit]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, "| images:", len(ids), "| sigmas:", sigmas)
model = torch.hub.load("intel-isl/MiDaS", a.model, trust_repo=True).to(dev).eval()
tfs = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = tfs.small_transform if a.model == "MiDaS_small" else tfs.dpt_transform


@torch.no_grad()
def depth_of(img):
    pred = model(transform(img).to(dev))
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False)
    return pred.squeeze().cpu().numpy()


vals = {n: [] for n in names}
keep_frac, shown = [], {}
kern_c = np.ones((7, 7), np.uint8)
kern_e = np.ones((a.erode, a.erode), np.uint8)
for n, i in enumerate(ids):
    img = cv2.cvtColor(cv2.imread(str(Path(a.photos) / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
    depth = depth_of(img)
    sil = silhouette_mask(depth, sigma=sigmas[0], frac=a.edge_frac, dilate=a.dilate)

    P = np.load(Path(a.parsed) / f"{i:05d}.npy").astype(np.uint8)
    P = cv2.resize(P, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
    face = np.isin(P, FACE_CLASSES).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, kern_c)
    face = cv2.erode(face, kern_e)
    keep = face.astype(bool) & ~sil

    save = dict(depth=normalize01(depth).astype(np.float16), keep=keep, face=face.astype(bool))
    curvs = []
    for s, nm in zip(sigmas, names):
        c = np.where(keep, mean_curvature(depth, sigma=s), 0).astype(np.float32)
        save[nm] = c; curvs.append(c)
        if keep.any():
            vals[nm].append(np.abs(c[keep]))
    np.savez_compressed(out / f"{i:05d}.npz", **save)
    keep_frac.append(float(keep.sum() / max(1, face.sum())))
    if n % max(1, len(ids) // 8) == 0 and len(shown) < 6:
        shown[i] = (img, normalize01(depth), curvs, keep)
    if (n + 1) % 100 == 0:
        print(f"  {n + 1}/{len(ids)}")

stats = {nm: {f"p{q}": float(np.percentile(np.concatenate(v), q)) for q in (50, 90, 95, 99)} for nm, v in vals.items() if v}
stats["mean_fraction_of_face_kept"] = float(np.mean(keep_frac))
(out / "_stats.json").write_text(json.dumps(stats, indent=2))
print("stats:", json.dumps(stats))

rows = 2 + len(sigmas)
fig, ax = plt.subplots(rows, len(shown), figsize=(2.6 * len(shown), 2.6 * rows), squeeze=False)
for j, (i, (img, d, curvs, k)) in enumerate(shown.items()):
    ax[0, j].imshow(img); ax[0, j].set_title(f"{i:05d}")
    ax[1, j].imshow(d, cmap="magma")
    for r, (c, nm) in enumerate(zip(curvs, names), start=2):
        ax[r, j].imshow(curvature_for_display_masked(c, ~k, clip_pct=95), cmap="coolwarm", vmin=0, vmax=1)
        if j == 0:
            ax[r, j].set_ylabel(nm)
    for r in range(rows):
        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
fig.tight_layout(); fig.savefig(out / "_check.png", dpi=105)
print("saved", out / "_check.png", "and", len(ids), "files in", out)
