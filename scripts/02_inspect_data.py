"""Step 2: join FFHQ-Wrinkle (1,000 manual masks) with FFHQ-Aging age labels and inspect.

Needs NO face images yet. It tells us:
  * how many of the 1,000 images fall in each FFHQ-Aging age group (class balance)
  * whether wrinkle-mask density really rises with age (a first sanity check of the data)
  * that masks and face-parsed labels line up (saves a few overlays to look at)

Usage (from the project root, venv active):
    pip install pandas
    python scripts/02_inspect_data.py
    python scripts/02_inspect_data.py --masks manual_wrinkle_masks --parsed "face parsed label" --csv ffhq_aging_labels.csv
"""
import argparse
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGE_ORDER = ["0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70-120"]
# Spreadsheet programs turn "3-6" into a date and show "03-06". Undo the common cases.
FIXES = {"03-06": "3-6", "07-09": "7-9", "00-02": "0-2"}


def clean_age(v: str) -> str:
    v = str(v).strip()
    return FIXES.get(v, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="manual_wrinkle_masks")
    ap.add_argument("--parsed", default="face parsed label")
    ap.add_argument("--csv", default="ffhq_aging_labels.csv")
    ap.add_argument("--out", default="outputs/data_check")
    ap.add_argument("--face-classes", default="1,10",
                    help="face-parsing class ids treated as 'face' (skin=1, nose=10 in CelebAMask-HQ labels)")
    ap.add_argument("--n-overlays", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    face_ids = [int(x) for x in args.face_classes.split(",")]

    # --- 1. IDs available on disk
    mask_files = {int(p.stem): p for p in Path(args.masks).glob("*.png")}
    parsed_files = {int(p.stem): p for p in Path(args.parsed).glob("*.npy")}
    ids = sorted(set(mask_files) & set(parsed_files))
    print(f"manual masks: {len(mask_files)} | parsed labels: {len(parsed_files)} | both: {len(ids)}")
    if not ids:
        raise SystemExit("No matching files. Check --masks / --parsed folder names.")

    # --- 2. Age labels (read as text so nothing gets auto-converted)
    df = pd.read_csv(args.csv, dtype=str)
    df["image_number"] = df["image_number"].astype(int)
    df["age_group"] = df["age_group"].map(clean_age)
    unexpected = sorted(set(df["age_group"]) - set(AGE_ORDER))
    print("age_group values in CSV:", sorted(df["age_group"].unique()))
    if unexpected:
        print("\n!! WARNING: unexpected age_group values:", unexpected)
        print("   The CSV may have been altered by a spreadsheet program. Re-download the original")
        print("   ffhq_aging_labels.csv from GitHub and do NOT open-and-save it in Excel.\n")
    meta = df[df["image_number"].isin(ids)].copy()
    meta["age_confidence"] = meta["age_group_confidence"].astype(float)
    print(f"joined rows: {len(meta)} (expected {len(ids)})")

    # --- 3. wrinkle density inside the face region
    dens = {}
    for i in ids:
        m = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        P = np.load(parsed_files[i])
        P = cv2.resize(P.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        face = np.isin(P, face_ids)
        dens[i] = float((m[face] > 127).mean()) if face.any() else np.nan
    meta["wrinkle_density"] = meta["image_number"].map(dens)
    meta.to_csv(out / "meta_1000.csv", index=False)

    # --- 4. summary tables
    order = [a for a in AGE_ORDER if a in set(meta["age_group"])] + \
            [a for a in sorted(set(meta["age_group"])) if a not in AGE_ORDER]
    counts = meta["age_group"].value_counts().reindex(order).fillna(0).astype(int)
    summ = meta.groupby("age_group")["wrinkle_density"].agg(["count", "mean", "median"]).reindex(order)
    print("\nImages per age group:\n", counts.to_string())
    print("\nWrinkle density (fraction of face pixels) by age group:\n", summ.round(4).to_string())
    print("\nGender:\n", meta["gender"].value_counts().to_string())
    summ.to_csv(out / "age_group_summary.csv")

    # --- 5. plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(order, counts.values); ax[0].set_title("images per age group (1,000 manual)")
    ax[0].tick_params(axis="x", rotation=45)
    data = [meta.loc[meta.age_group == a, "wrinkle_density"].dropna().values for a in order]
    ax[1].boxplot(data); ax[1].set_xticklabels(order); ax[1].set_title("wrinkle density in face region")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out / "age_counts_and_density.png", dpi=130); plt.close(fig)

    # --- 6. overlays: mask + face-parsing, to check alignment
    pick = ids[:: max(1, len(ids) // args.n_overlays)][: args.n_overlays]
    fig, ax = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.4), squeeze=False)
    for j, i in enumerate(pick):
        m = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        P = np.load(parsed_files[i])
        P = cv2.resize(P.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        ax[0, j].imshow(m, cmap="gray"); ax[0, j].set_title(f"mask {i:05d}")
        ax[1, j].imshow(P, cmap="tab20"); ax[1, j].imshow(np.ma.masked_where(m <= 127, m), cmap="autumn", alpha=0.8)
        ax[1, j].set_title("parsing + wrinkles")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.tight_layout(); fig.savefig(out / "overlays.png", dpi=110); plt.close(fig)
    print("\nparsing class ids seen in first file:", np.unique(np.load(parsed_files[ids[0]])).tolist())
    print("saved to", out)


if __name__ == "__main__":
    main()
