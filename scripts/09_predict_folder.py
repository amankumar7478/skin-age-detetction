"""Predict the age GROUP for every photo in a folder, without and with depth curvature.

    python scripts/09_predict_folder.py                    # photos in  samples\
    python scripts/09_predict_folder.py --folder myphotos  # any other folder
    python scripts/09_predict_folder.py --no-crop          # photos already tightly cropped like FFHQ faces

Writes outputs/predictions/results.csv and one picture per photo (outputs/predictions/<name>.png).
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
from models.curvature import curvature_for_display_masked  # noqa: E402
from models.pipeline import SkinAgePipeline  # noqa: E402

LABEL = {"young": "young (10-29)", "middle": "middle (30-49)", "old": "old (50+)"}
ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="samples")
ap.add_argument("--ckpt", default="checkpoints/final_v1")
ap.add_argument("--out", default="outputs/predictions")
ap.add_argument("--no-crop", action="store_true")
a = ap.parse_args()

files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
if not files:
    sys.exit(f"no photos found in {a.folder}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print("loading models (first time takes a minute)...")
pipe = SkinAgePipeline(a.ckpt)
rows = []
for f in files:
    bgr = cv2.imread(str(f))
    if bgr is None:
        print(f"{f.name}: cannot read, skipped"); continue
    r = pipe.analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), autocrop=not a.no_crop)
    if r["error"]:
        print(f"{f.name}: {r['error']}"); rows.append(dict(photo=f.name, note=r["error"])); continue
    cl = r["classes"]; pw, pf = r["probs_wrinkle"], r["probs_fused"]
    gw, gf = cl[int(pw.argmax())], cl[int(pf.argmax())]
    print(f"{f.name:30s} without curvature: {LABEL[gw]:15s} ({pw.max():.0%}) | with curvature: {LABEL[gf]:15s} ({pf.max():.0%})"
          + ("" if gw == gf else "   <-- differ"))
    rows.append(dict(photo=f.name, without_curvature=gw, with_curvature=gf,
                     **{f"wrinkle_p_{c}": round(float(p), 3) for c, p in zip(cl, pw)},
                     **{f"fused_p_{c}": round(float(p), 3) for c, p in zip(cl, pf)}, note=""))

    img = r["img512"].copy()
    ov = img.copy(); ov[r["wrinkle512"] > 0.5] = (255, 0, 0)
    cur = curvature_for_display_masked(r["curv"], ~r["keep"], clip_pct=95)
    cur_rgb = (matplotlib.colormaps["coolwarm"](cur)[..., :3] * 255).astype(np.uint8)
    cur_rgb = np.where(r["keep"][..., None], cur_rgb, r["img256"] // 3)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    ax[0].imshow(img); ax[0].set_title("photo (face-cropped)")
    ax[1].imshow(ov); ax[1].set_title("predicted wrinkles")
    ax[2].imshow(cur_rgb); ax[2].set_title("depth curvature (face)")
    for x in ax[:3]:
        x.axis("off")
    y = np.arange(len(cl)); h = 0.38
    ax[3].barh(y - h / 2, pw, h, label=f"without curvature: {LABEL[gw]}", color="#8a8a8a")
    ax[3].barh(y + h / 2, pf, h, label=f"with curvature: {LABEL[gf]}", color="#1f77b4")
    ax[3].set_yticks(y); ax[3].set_yticklabels([LABEL[c] for c in cl]); ax[3].invert_yaxis(); ax[3].set_xlim(0, 1)
    ax[3].set_title("probability"); ax[3].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / f"{f.stem}.png", dpi=100); plt.close(fig)

pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
print("\nsaved", out / "results.csv", "and one picture per photo in", out)
