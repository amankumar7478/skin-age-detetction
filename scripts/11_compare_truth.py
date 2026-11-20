"""Put the real age group next to the predictions.

  * Photos named like 00237.png are looked up in the dataset's own labels (outputs/data_check/folds.csv).
    NOTE: those photos were used to train the models, so agreement on them is optimistic - not a fair test.
  * For your own photos, make a small file  truth.csv  with two columns  photo,age  e.g.
        priya.jpg,24
        papa.jpg,52
    and pass it with --truth truth.csv  (these are fair, unseen tests).

    python scripts/11_compare_truth.py --pred outputs/predictions_v2/results.csv
    python scripts/11_compare_truth.py --pred outputs/predictions_v2/results.csv --truth truth.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="outputs/predictions_v2/results.csv")
ap.add_argument("--truth", default="")
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--labels", default="ffhq_aging_labels.csv")
a = ap.parse_args()


def group_of(age):
    return "young" if age < 30 else "middle" if age < 50 else "old"


p = pd.read_csv(a.pred)
p = p[p.get("without_curvature").notna()].copy()
p["stem"] = p.photo.map(lambda s: Path(s).stem)
folds = pd.read_csv(a.folds)
lookup = {f"{int(r.image_number):05d}": (r.age_group, r.group3) for r in folds.itertuples()}
full = {}
if Path(a.labels).exists():
    L = pd.read_csv(a.labels, dtype=str)
    idc = "image_number" if "image_number" in L.columns else L.columns[0]
    agc = "age_group" if "age_group" in L.columns else L.columns[1]
    full = {f"{int(r[idc]):05d}": str(r[agc]) for _, r in L.iterrows()}
user = {}
if a.truth:
    t = pd.read_csv(a.truth)
    user = {str(r.photo): r.age for r in t.itertuples()}

rows = []
for r in p.itertuples():
    if r.photo in user:
        age = user[r.photo]; true_g, src, true_txt = group_of(age), "your photo (unseen)", f"{age}"
    elif r.stem in lookup:
        true_txt, true_g = lookup[r.stem]; src = "dataset (was in training)"
        true_txt = str(true_txt)
    elif r.stem in full:
        true_txt = full[r.stem]
        lo = int(true_txt.replace("_", "-").split("-")[0].strip("+")) if true_txt[0].isdigit() else -1
        true_g = None if lo < 10 else group_of(lo)
        src = "dataset labels (not in training set)"
    else:
        true_txt, true_g, src = "?", None, "no label"
    rows.append(dict(photo=r.photo, true_age=true_txt, true_group=true_g, without_curvature=r.without_curvature,
                     with_curvature=r.with_curvature, source=src,
                     without_ok=None if true_g in (None, "nan") or pd.isna(true_g) else r.without_curvature == true_g,
                     with_ok=None if true_g in (None, "nan") or pd.isna(true_g) else r.with_curvature == true_g))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.to_string(index=False))
for src, g in d.groupby("source"):
    g = g.dropna(subset=["without_ok"])
    if len(g):
        print(f"\n{src}: {len(g)} photos | correct without curvature {g.without_ok.mean():.0%} | with curvature {g.with_ok.mean():.0%}")
out = Path(a.pred).with_name("truth_compare.csv")
d.to_csv(out, index=False)
print("\nsaved", out)
