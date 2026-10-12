"""Pick out only the ~1,000 FFHQ images we need from a big FFHQ image folder.

Works on your laptop OR inside a Kaggle notebook. It searches --src recursively, matches files by
their 5-digit FFHQ number, resizes to --size (never upscales) and saves JPEG q95, then zips.

    # laptop
    python scripts/04_extract_subset.py --src "D:/ffhq_download" --ids ids.txt --out data/ffhq1000
    # kaggle notebook
    !python scripts/04_extract_subset.py --src /kaggle/input/<dataset-name> --ids /kaggle/input/<ids-dataset>/ids.txt --out /kaggle/working/ffhq1000 --zip
"""
import argparse
import os
import re
import shutil
from pathlib import Path

import cv2

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--ids", default="ids.txt")
ap.add_argument("--out", default="data/ffhq1000")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--zip", action="store_true", help="also make <out>.zip for downloading")
a = ap.parse_args()

need = {int(x) for x in Path(a.ids).read_text().split()}
found = {}
n_seen = 0
for root, _, files in os.walk(a.src):
    for f in files:
        p = Path(root) / f
        if p.suffix.lower() not in EXTS:
            continue
        n_seen += 1
        m = re.search(r"(\d{5})", p.stem)
        if m and int(m.group(1)) in need and int(m.group(1)) not in found:
            found[int(m.group(1))] = p
print(f"scanned {n_seen} image files; matched {len(found)} of {len(need)} wanted ids")

missing = sorted(need - set(found))
if missing:
    print(f"!! {len(missing)} ids not found, e.g. {missing[:10]}")
    print("   This dataset may be a subset of FFHQ. Tell me its name / how many images it has.")

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
shapes = set()
for i, p in sorted(found.items()):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    shapes.add((h, w))
    if max(h, w) > a.size:
        s = a.size / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("source image sizes seen:", sorted(shapes)[:5], "| saved to", out)

if a.zip:
    z = shutil.make_archive(str(out), "zip", str(out))
    print("zip:", z, f"({os.path.getsize(z)/1e6:.0f} MB)")
