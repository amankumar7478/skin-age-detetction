"""Step A check: does the face-parsing model find the same face region we trained on?

Mode 1 (default): on ~50 dataset photos we KNOW the answer (the dataset's own parsing) -> overlap numbers.
Mode 2 (--folder): on any photos you point at (no answer known) -> pictures only.

    python -m pip install transformers          # once
    python scripts/08_check_face_region.py
    python scripts/08_check_face_region.py --folder samples
Outputs go to outputs/face_region_check/.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_region import FaceParser, center_square, region_from_labels  # noqa: E402

DATASET_FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]    # same list as scripts/07_precompute_curvature.py
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="data/ffhq1000")
ap.add_argument("--parsed", default="face parsed label")
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--folder", default="", help="check arbitrary photos in this folder instead")
ap.add_argument("--out", default="outputs/face_region_check")
ap.add_argument("--device", default=None)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def contour(img, mask, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    o = img.copy(); cv2.drawContours(o, cs, -1, color, 2); return o


def to512(rgb):
    rgb = center_square(rgb)
    return cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA if rgb.shape[0] > 512 else cv2.INTER_CUBIC)


fp = FaceParser(device=a.device)
print("device:", fp.dev)
print("\nmodel's own class list:", fp.id2label)
print("treated as FACE:", [(i, fp.id2label[i]) for i in fp.face_ids])
if len(fp.face_ids) != 9:
    print("!! expected 9 face classes (skin, nose, 2 eyes, 2 brows, mouth, 2 lips) - the name matching may be off. Tell me this list.")

# ------------------------------------------------------------------ mode 2: any photos
if a.folder:
    paths = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXTS)[: a.n]
    if not paths:
        sys.exit(f"no images in {a.folder}")
    show = paths[:8]
    fig, ax = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.4), squeeze=False)
    for j, p in enumerate(show):
        rgb = to512(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        lab = fp.labels(rgb)
        reg = region_from_labels(lab, fp.face_ids)
        small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        ax[0, j].imshow(contour(small, reg, (255, 0, 0))); ax[0, j].set_title(p.stem[:12], fontsize=8)
        ax[1, j].imshow(cv2.resize(lab, (256, 256), interpolation=cv2.INTER_NEAREST), cmap="tab20", interpolation="nearest")
        ax[0, j].axis("off"); ax[1, j].axis("off")
        print(f"{p.name}: face region = {100 * reg.mean():.1f}% of the 256x256 frame")
    fig.tight_layout(); fig.savefig(out / "folder_overlay.png", dpi=110)
    print("\nsaved", out / "folder_overlay.png", "(top: red outline = region we will use; bottom: model labels)")
    sys.exit(0)

# ------------------------------------------------------------------ mode 1: dataset photos with known answer
ids = pd.read_csv(a.folds)["image_number"].tolist()
ids = [i for i in ids if (Path(a.photos) / f"{i:05d}.jpg").exists() and (Path(a.parsed) / f"{i:05d}.npy").exists()]
ids = ids[:: max(1, len(ids) // a.n)][: a.n]
rows = []
for i in ids:
    rgb = cv2.cvtColor(cv2.imread(str(Path(a.photos) / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA) if rgb.shape[0] != 512 else rgb
    lab = fp.labels(rgb)
    ours = region_from_labels(lab, fp.face_ids).astype(bool)
    theirs = region_from_labels(np.load(Path(a.parsed) / f"{i:05d}.npy"), DATASET_FACE_CLASSES).astype(bool)
    inter = (ours & theirs).sum()
    rows.append(dict(id=i, iou=inter / max(1, (ours | theirs).sum()),
                     precision=inter / max(1, ours.sum()), recall=inter / max(1, theirs.sum()),
                     lab=lab, ours=ours, theirs=theirs, rgb=rgb))
df = pd.DataFrame([{k: v for k, v in r.items() if k in ("id", "iou", "precision", "recall")} for r in rows])
print(f"\nimages checked: {len(df)}")
print(df[["iou", "precision", "recall"]].describe().loc[["mean", "50%", "min"]].round(3).to_string())
print("\nworst 5:\n", df.sort_values("iou").head(5).round(3).to_string(index=False))
print("\n(IoU = overlap with the dataset's own face region, 1.0 = identical.  precision low = we cover too much,"
      " recall low = we miss parts of the face.)")
df.to_csv(out / "face_region_scores.csv", index=False)

order = df.sort_values("iou").index.tolist()
pick = order[:3] + order[len(order) // 2:: max(1, len(order) // 6)][:3]
fig, ax = plt.subplots(3, len(pick), figsize=(2.6 * len(pick), 8), squeeze=False)
for j, k in enumerate(pick):
    r = rows[k]
    small = cv2.resize(r["rgb"], (256, 256), interpolation=cv2.INTER_AREA)
    ov = contour(contour(small, r["theirs"], (0, 200, 0)), r["ours"], (255, 0, 0))
    ax[0, j].imshow(ov); ax[0, j].set_title(f"{r['id']:05d}  IoU {r['iou']:.2f}", fontsize=8)
    ax[1, j].imshow(cv2.resize(r["lab"], (256, 256), interpolation=cv2.INTER_NEAREST), cmap="tab20", interpolation="nearest")
    ax[2, j].imshow(cv2.resize(np.load(Path(a.parsed) / f"{r['id']:05d}.npy").astype(np.uint8), (256, 256),
                               interpolation=cv2.INTER_NEAREST), cmap="tab20", interpolation="nearest")
    for q in range(3):
        ax[q, j].axis("off")
ax[0, 0].set_title("green = dataset, red = ours\n" + ax[0, 0].get_title(), fontsize=8)
fig.tight_layout(); fig.savefig(out / "dataset_overlay.png", dpi=110)
print("saved", out / "dataset_overlay.png", "(rows: photo+outlines | model labels | dataset labels; first 3 columns = worst cases)")
