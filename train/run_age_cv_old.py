"""Step 5: the core experiment - age-group classification with and without curvature (ablation).

Uses the SAME 5 folds as the wrinkle U-Net, and the U-Net's OUT-OF-FOLD wrinkle maps, so no image's
mask was produced by a model that trained on it.  Classes: young (10-29), middle (30-49), old (50+).

Configurations (each is the same small CNN, only the input channels differ):
  wrinkle        : wrinkle map only                    (baseline, like the base paper)
  curv_only      : curvature (sigma 4) only
  fused_s4       : wrinkle + curvature sigma 4         (our proposal)
  fused_s2       : wrinkle + curvature sigma 2
  fused_shuffled : wrinkle + curvature of a RANDOM OTHER image (control: should NOT help)

    python train/run_age_cv.py --smoke                      # quick check that it runs
    python train/run_age_cv.py                              # 1 seed, all configs (~1 h on RTX 2050)
    python train/run_age_cv.py --seeds 3                    # more reliable, ~3 h
    python train/run_age_cv.py --wrinkle-source human       # upper bound with the human masks
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

CLASSES = ["young", "middle", "old"]
S = 256
CONFIGS = {"wrinkle": ["w"], "curv_only": ["c4"], "fused_s4": ["w", "c4"], "fused_s2": ["w", "c2"],
           "fused_shuffled": ["w", "c4s"], "fused_multi": ["w", "c2", "c4"]}

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--oof", default="outputs/oof_masks")
ap.add_argument("--human", default="manual_wrinkle_masks")
ap.add_argument("--wrinkle-source", choices=["oof", "human"], default="oof")
ap.add_argument("--curv", default="outputs/curv")
ap.add_argument("--configs", default="wrinkle,curv_only,fused_s4,fused_s2,fused_shuffled")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--out", default="outputs/age_cv")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.epochs, a.seeds, a.configs = 1, 1, "wrinkle,fused_s4"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(a.folds)
df = df[df.group3.isin(CLASSES)].reset_index(drop=True)
if a.smoke:
    df = df.sample(60, random_state=0).reset_index(drop=True)
N = len(df)
stats = json.loads((Path(a.curv) / "_stats.json").read_text())
W, C2, C4 = (np.zeros((N, S, S), np.float32) for _ in range(3))
for k, i in enumerate(df.image_number):
    if a.wrinkle_source == "oof":
        w = cv2.imread(str(Path(a.oof) / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    else:
        w = cv2.imread(str(Path(a.human) / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    W[k] = cv2.resize(w, (S, S), interpolation=cv2.INTER_AREA)
    z = np.load(Path(a.curv) / f"{i:05d}.npz")
    for key, arr in (("curv_s2", C2), ("curv_s4", C4)):
        c = z[key].astype(np.float32)
        if c.shape[0] != S:
            c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)
        arr[k] = np.arcsinh(c / stats[key]["p90"]) / 2.5          # tame the huge range, keep the sign
perm = np.random.default_rng(123).permutation(N)                    # shuffled-curvature control
arrays = {"w": W, "c2": C2, "c4": C4, "c4s": C4[perm]}
for k, v in arrays.items():
    arrays[k] = (v - v.mean()) / (v.std() + 1e-8)
y = df.group3.map({c: j for j, c in enumerate(CLASSES)}).values.astype(np.int64)
fold = df.fold.values
print(f"majority-class baseline accuracy: {np.bincount(y).max() / N:.3f}")
print(f"device {dev} | images {N} | class counts {dict(zip(CLASSES, np.bincount(y).tolist()))} | wrinkle source: {a.wrinkle_source}")


# ---------------------------------------------------------------- model / training
def make_model(in_ch):
    m = mobilenet_v3_small(weights=None if a.no_pretrained else MobileNet_V3_Small_Weights.DEFAULT)
    conv = m.features[0][0]
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3.0 / in_ch)
    m.features[0][0] = new
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m


def train_predict(X, tr, va, seed):
    torch.manual_seed(seed)
    model = make_model(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cnt = np.bincount(y[tr], minlength=len(CLASSES))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)),
                                                    dtype=torch.float32, device=dev))
    rng = np.random.default_rng(seed)
    for _ in range(a.epochs):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), a.bs):
            idx = torch.from_numpy(order[s:s + a.bs])
            if len(idx) < 2:
                continue
            xb = X[idx].clone()
            flip = torch.rand(len(idx)) < 0.5
            xb[flip] = xb[flip].flip(-1)
            xb, yb = xb.to(dev), torch.from_numpy(y[idx.numpy()]).to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                o = model(xb)
            loss = lossf(o.float(), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    model.eval(); probs = []
    with torch.no_grad():
        for s in range(0, len(va), 64):
            xb = X[torch.from_numpy(va[s:s + 64])].to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                o = model(xb)
            probs.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(probs)                      # final-epoch model: no checkpoint picked on held-out data


def metrics(yt, yp):
    K = len(CLASSES)
    cm = np.zeros((K, K), int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return dict(acc=float(np.diag(cm).sum() / cm.sum()), macro_f1=float(f1.mean()), recall=rec, cm=cm)


# ---------------------------------------------------------------- run the ablation
results, pooled_probs = {}, {}
for cfg in a.configs.split(","):
    X = torch.from_numpy(np.stack([arrays[c] for c in CONFIGS[cfg]], axis=1)).float()
    seed_metrics, probs_sum = [], np.zeros((N, len(CLASSES)))
    t0 = time.time()
    for seed in range(a.seeds):
        P = np.zeros((N, len(CLASSES)))
        for k in range(5):
            tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
            if len(va) == 0:
                continue
            P[va] = train_predict(X, tr, va, seed * 100 + k)
        seed_metrics.append(metrics(y, P.argmax(1))); probs_sum += P
    pooled_probs[cfg] = probs_sum / a.seeds
    results[cfg] = seed_metrics
    m = seed_metrics
    print(f"{cfg:15s} acc {np.mean([r['acc'] for r in m]):.3f} | macro-F1 {np.mean([r['macro_f1'] for r in m]):.3f} | "
          f"recall y/m/o {np.round(np.mean([r['recall'] for r in m], 0), 2).tolist()} | {time.time() - t0:.0f}s")
    np.save(out / f"probs_{cfg}.npy", pooled_probs[cfg])
    pd.DataFrame(np.sum([r["cm"] for r in m], 0), index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{cfg}.csv")

# ---------------------------------------------------------------- summary + bootstrap vs. wrinkle-only
rows = []
for cfg, m in results.items():
    rows.append(dict(config=cfg, acc=np.mean([r["acc"] for r in m]), acc_sd=np.std([r["acc"] for r in m]),
                     macro_f1=np.mean([r["macro_f1"] for r in m]), macro_f1_sd=np.std([r["macro_f1"] for r in m]),
                     **{f"recall_{c}": float(np.mean([r["recall"][j] for r in m])) for j, c in enumerate(CLASSES)}))
summ = pd.DataFrame(rows)
if "wrinkle" in pooled_probs:
    rng = np.random.default_rng(0); base = pooled_probs["wrinkle"].argmax(1)
    for cfg in pooled_probs:
        if cfg == "wrinkle":
            continue
        pr, d = pooled_probs[cfg].argmax(1), []
        for _ in range(1000):
            ix = rng.integers(0, N, N)
            d.append(metrics(y[ix], pr[ix])["macro_f1"] - metrics(y[ix], base[ix])["macro_f1"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        summ.loc[summ.config == cfg, "dF1_vs_wrinkle"] = float(np.mean(d))
        summ.loc[summ.config == cfg, "dF1_95CI"] = f"[{lo:+.3f}, {hi:+.3f}]"

# compare every config with a reference, per seed (training randomness) and by bootstrap (test-set sampling)
def vs(ref, tag):
    if ref not in results:
        return
    base_pred = pooled_probs[ref].argmax(1)
    rng2 = np.random.default_rng(1)
    for cfg in results:
        if cfg == ref:
            continue
        dd = [r1["macro_f1"] - r2["macro_f1"] for r1, r2 in zip(results[cfg], results[ref])]
        pr, boots = pooled_probs[cfg].argmax(1), []
        for _ in range(1000):
            ix = rng2.integers(0, N, N)
            boots.append(metrics(y[ix], pr[ix])["macro_f1"] - metrics(y[ix], base_pred[ix])["macro_f1"])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        m_ = summ.config == cfg
        summ.loc[m_, f"dF1_vs_{tag}_seedmean"] = float(np.mean(dd))
        summ.loc[m_, f"dF1_vs_{tag}_seedsd"] = float(np.std(dd))
        summ.loc[m_, f"wins_vs_{tag}"] = f"{sum(d > 0 for d in dd)}/{len(dd)}"
        summ.loc[m_, f"dF1_vs_{tag}_95CI"] = f"[{lo:+.3f}, {hi:+.3f}]"


vs("fused_shuffled", "shuf")
summ.to_csv(out / "summary.csv", index=False)
print("\n", summ.round(3).to_string(index=False))
print("\nCurvature is only credited if fused beats fused_shuffled (the control), not just wrinkle-only: look at the *_vs_shuf columns.")
print("saved to", out)
