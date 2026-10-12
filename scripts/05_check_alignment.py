"""Step 3: confirm the FFHQ photos really match the wrinkle masks, and see what we have left.

Run AFTER unzipping ffhq1000.zip into data/ffhq1000 and after scripts/02_inspect_data.py.

    python scripts/05_check_alignment.py

Outputs (outputs/data_check):
  alignment_overlay.png  - photo with its wrinkle mask in red. Wrinkles must sit on real
                           creases/face features. If they look shifted or on the wrong face, STOP and tell me.
  meta_available.csv     - the images that have photo + mask + parsed label + age label
Prints the usable image count per age group and for the proposed 3-class split.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="data/ffhq1000")
ap.add_argument("--masks", default="manual_wrinkle_masks")
ap.add_argument("--meta", default="outputs/data_check/meta_1000.csv")
ap.add_argument("--out", default="outputs/data_check")
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

photos = {int(p.stem): p for p in Path(a.photos).glob("*.jpg")}
meta = pd.read_csv(a.meta, dtype={"age_group": str})
meta["has_photo"] = meta["image_number"].isin(photos)
print(f"photos on disk: {len(photos)} | in the 1,000 set: {int(meta.has_photo.sum())} | "
      f"missing: {int((~meta.has_photo).sum())}")
avail = meta[meta.has_photo].copy()
avail.to_csv(out / "meta_available.csv", index=False)

# --- usable counts
GROUP3 = {"10-14": "young", "15-19": "young", "20-29": "young",
          "30-39": "middle", "40-49": "middle",
          "50-69": "old", "70-120": "old"}          # 0-2, 3-6, 7-9 dropped
order = ["0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70-120"]
full = meta.age_group.value_counts().reindex(order).fillna(0).astype(int)
have = avail.age_group.value_counts().reindex(order).fillna(0).astype(int)
print("\nage group : all 1000 -> with photo")
for g in order:
    print(f"  {g:7s}: {full[g]:4d} -> {have[g]:4d}")
avail["group3"] = avail.age_group.map(GROUP3)
print("\nproposed 3 classes (photo available):\n", avail.group3.value_counts(dropna=False).to_string())

# --- overlay check
ids = sorted(avail.image_number)
pick = ids[:: max(1, len(ids) // a.n)][: a.n]
fig, ax = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.6), squeeze=False)
for j, i in enumerate(pick):
    img = cv2.cvtColor(cv2.imread(str(photos[i])), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(Path(a.masks) / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) > 40   # keep thin lines
    ov = img.copy(); ov[m] = (255, 0, 0)
    ax[0, j].imshow(img); ax[0, j].set_title(f"{i:05d}"); ax[1, j].imshow(ov)
    ax[0, j].axis("off"); ax[1, j].axis("off")
fig.tight_layout(); fig.savefig(out / "alignment_overlay.png", dpi=110); plt.close(fig)
print("\nsaved", out / "alignment_overlay.png", "- open it and check the red lines sit on real wrinkles")
