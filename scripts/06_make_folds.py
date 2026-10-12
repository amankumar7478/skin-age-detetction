"""Assign each usable image to one of 5 folds, balanced by age group.

Why: the wrinkle U-Net must NEVER predict masks for images it was trained on, otherwise the later
age classifier would see unrealistically perfect masks. So we train 5 U-Nets (one per fold); each one
predicts the fold it did not see ("out-of-fold" masks).

    python scripts/06_make_folds.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="outputs/data_check/meta_available.csv")
ap.add_argument("--out", default="outputs/data_check/folds.csv")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

GROUP3 = {"10-14": "young", "15-19": "young", "20-29": "young",
          "30-39": "middle", "40-49": "middle",
          "50-69": "old", "70-120": "old"}          # 0-2, 3-6, 7-9 -> "drop" (kept for U-Net, not for age task)
df = pd.read_csv(a.meta, dtype={"age_group": str})
df["group3"] = df["age_group"].map(GROUP3).fillna("drop")
df["strat"] = df["age_group"].replace({"0-2": "0-9", "3-6": "0-9", "7-9": "0-9"})   # merge tiny groups

rng = np.random.default_rng(a.seed)
df["fold"] = -1
for _, idx in df.groupby("strat").groups.items():
    idx = np.array(list(idx)); rng.shuffle(idx)
    df.loc[idx, "fold"] = np.arange(len(idx)) % a.k

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df[["image_number", "age_group", "group3", "fold"]].to_csv(a.out, index=False)
print(f"{len(df)} images -> {a.out}")
print("\nimages per fold and class (group3):")
print(pd.crosstab(df["fold"], df["group3"]).to_string())
