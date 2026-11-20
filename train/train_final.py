"""Step B: train the FINAL models used by the Streamlit app, on all the data (no held-out fold).

  stage 1 (~35 min): wrinkle U-Net on all images in folds.csv      -> checkpoints/final_v1/unet.pt
  stage 2 (~10 min): age classifiers on all 3-class images, using the out-of-fold wrinkle maps
                     (the maps an UNSEEN photo would get), 3 seeds each:
                       wrinkle   (wrinkle map only)     -> cls_wrinkle_s{seed}.pt
                       fused_s4  (wrinkle + curvature)  -> cls_fused_s4_s{seed}.pt
  plus settings.json: every constant the app needs to preprocess an uploaded photo exactly like training.

Nothing existing is touched; all output goes to a new folder.  Safe to stop and re-run: finished stages are skipped.

    python train/train_final.py --smoke        # 1-minute check that it runs
    python train/train_final.py                # both stages
    python train/train_final.py --stage cls    # only the classifiers (needs unet.pt only for the app, not here)
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
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import WrinkleDataset  # noqa: E402
from models.fusion_unet import FusionUNet, dice_bce_loss  # noqa: E402

CLASSES = ["young", "middle", "old"]
S = 256
ARCSINH_DIV = 2.5
CONFIGS = {"wrinkle": ["w"], "fused_s4": ["w", "c4"]}

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["all", "unet", "cls"], default="all")
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--photos", default="data/ffhq1000")
ap.add_argument("--masks", default="manual_wrinkle_masks")
ap.add_argument("--oof", default="outputs/oof_masks")
ap.add_argument("--curv", default="outputs/curv")
ap.add_argument("--out", default="checkpoints/final_v1")
ap.add_argument("--unet-epochs", type=int, default=40)
ap.add_argument("--unet-size", type=int, default=512)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--cls-epochs", type=int, default=30)
ap.add_argument("--cls-seeds", type=int, default=3)
ap.add_argument("--force", action="store_true", help="retrain even if the output file exists")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.unet_epochs, a.unet_size, a.cls_epochs, a.cls_seeds, a.out = 1, 64, 1, 1, "checkpoints/final_smoke"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
df_all = pd.read_csv(a.folds)
print(f"device {dev} | output folder {out}")


# ------------------------------------------------------------------ stage 1: U-Net
def stage_unet():
    ck = out / "unet.pt"
    if ck.exists() and not a.force:
        print("unet.pt already exists - skipping stage 1 (use --force to retrain)"); return
    torch.manual_seed(0); np.random.seed(0)
    ids = df_all.image_number.tolist()
    if a.smoke:
        ids = ids[:8]
    dl = DataLoader(WrinkleDataset(ids, a.photos, a.masks, a.unet_size, augment=True), batch_size=a.bs,
                    shuffle=True, num_workers=0, drop_last=len(ids) >= a.bs)
    model = FusionUNet(base=16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.unet_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"stage 1: U-Net on {len(ids)} images, {a.unet_epochs} epochs")
    for ep in range(1, a.unet_epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0
        for x, y, _ in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
            loss = dice_bce_loss(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        print(f"  epoch {ep:3d}/{a.unet_epochs} | loss {tot / len(dl.dataset):.4f} | {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), out / "unet_partial.pt")      # lets you see progress; final file below
    torch.save(model.state_dict(), ck)
    (out / "unet_partial.pt").unlink(missing_ok=True)
    print("saved", ck)


# ------------------------------------------------------------------ stage 2: classifiers
def make_model(in_ch):
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    conv = m.features[0][0]
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3.0 / in_ch)
    m.features[0][0] = new
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m


def stage_cls():
    df = df_all[df_all.group3.isin(CLASSES)].reset_index(drop=True)
    if a.smoke:
        df = df.sample(60, random_state=0).reset_index(drop=True)
    N = len(df)
    stats = json.loads((Path(a.curv) / "_stats.json").read_text())
    p90 = stats["curv_s4"]["p90"]
    W, C4 = np.zeros((N, S, S), np.float32), np.zeros((N, S, S), np.float32)
    for k, i in enumerate(df.image_number):
        w = cv2.imread(str(Path(a.oof) / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        W[k] = cv2.resize(w, (S, S), interpolation=cv2.INTER_AREA)
        c = np.load(Path(a.curv) / f"{i:05d}.npz")["curv_s4"].astype(np.float32)
        if c.shape[0] != S:
            c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)
        C4[k] = np.arcsinh(c / p90) / ARCSINH_DIV
    norm = {"w": [float(W.mean()), float(W.std() + 1e-8)], "c4": [float(C4.mean()), float(C4.std() + 1e-8)]}
    arrays = {"w": (W - norm["w"][0]) / norm["w"][1], "c4": (C4 - norm["c4"][0]) / norm["c4"][1]}
    y = df.group3.map({c: j for j, c in enumerate(CLASSES)}).values.astype(np.int64)
    print(f"stage 2: classifiers on {N} images, class counts {dict(zip(CLASSES, np.bincount(y).tolist()))}")
    cnt = np.bincount(y)
    cw = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    for cfg, chans in CONFIGS.items():
        X = torch.from_numpy(np.stack([arrays[c] for c in chans], axis=1)).float()
        for seed in range(a.cls_seeds):
            ck = out / f"cls_{cfg}_s{seed}.pt"
            if ck.exists() and not a.force:
                print("  exists, skipping", ck.name); continue
            torch.manual_seed(1000 + seed); rng = np.random.default_rng(1000 + seed)
            model = make_model(X.shape[1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.cls_epochs))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            lossf = nn.CrossEntropyLoss(weight=cw)
            t0 = time.time()
            for _ in range(a.cls_epochs):
                model.train(); order = rng.permutation(N)
                for s in range(0, N, 32):
                    idx = torch.from_numpy(order[s:s + 32])
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
            torch.save(model.state_dict(), ck)
            print(f"  saved {ck.name} ({time.time() - t0:.0f}s)", flush=True)
    settings = dict(classes=CLASSES, size=S, curv_key="curv_s4", curv_p90=p90, arcsinh_div=ARCSINH_DIV, norm=norm,
                    configs=CONFIGS, cls_seeds=a.cls_seeds, unet_size=a.unet_size, unet_base=16,
                    imagenet_mean=[0.485, 0.456, 0.406], imagenet_std=[0.229, 0.224, 0.225],
                    depth_model="DPT_Hybrid", erode=12, close=7, edge_frac=0.35, edge_dilate=5, edge_sigma=2,
                    curv_sigma=4, n_unet_images=len(df_all), n_cls_images=N,
                    cv_reference={"wrinkle_macroF1": "see outputs/age_cv/summary.csv", "note": "5-fold CV, 5 seeds"})
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print("saved", out / "settings.json")


if a.stage in ("all", "cls"):
    stage_cls()          # fast; fail early on data problems before the long U-Net run
if a.stage in ("all", "unet"):
    stage_unet()
print("done.")
