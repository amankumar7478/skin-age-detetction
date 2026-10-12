"""Train the fusion U-Net on FFHQ-Wrinkle manual masks, holding out one fold.

    python train/train_seg.py --smoke                 # 2 tiny batches: checks everything runs
    python train/train_seg.py --fold 0 --epochs 40    # real run, fold 0 is held out
Afterwards it saves the held-out fold's predicted wrinkle maps to outputs/oof_masks/ (used later by the age classifier).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import WrinkleDataset  # noqa: E402
from models.fusion_unet import FusionUNet, dice_bce_loss, dsc_jsc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="outputs/data_check/folds.csv")
ap.add_argument("--photos", default="data/ffhq1000")
ap.add_argument("--masks", default="manual_wrinkle_masks")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=2)      # raise to 8 on Kaggle; use 2 on a 4 GB GPU
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--base", type=int, default=16)
ap.add_argument("--workers", type=int, default=0)      # keep 0 on Windows
ap.add_argument("--select", choices=["last", "best"], default="last",
                help="checkpoint used for held-out predictions. 'last' avoids picking the epoch using the held-out fold itself")
ap.add_argument("--smoke", action="store_true", help="tiny quick test (64 px, 8 images, 1 epoch)")
a = ap.parse_args()

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(a.folds)
tr_ids = df.loc[df.fold != a.fold, "image_number"].tolist()
va_ids = df.loc[df.fold == a.fold, "image_number"].tolist()
if a.smoke:
    a.size, a.epochs, a.bs = 64, 1, 4
    tr_ids, va_ids = tr_ids[:8], va_ids[:4]
print(f"device {dev} | train {len(tr_ids)} | held-out fold {a.fold}: {len(va_ids)}")

tr = DataLoader(WrinkleDataset(tr_ids, a.photos, a.masks, a.size, augment=True), batch_size=a.bs,
                shuffle=True, num_workers=a.workers, drop_last=len(tr_ids) >= a.bs)
va = DataLoader(WrinkleDataset(va_ids, a.photos, a.masks, a.size), batch_size=a.bs, num_workers=a.workers)

model = FusionUNet(base=a.base).to(dev)
print("parameters (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
use_amp = dev.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

Path("checkpoints").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
ckpt = Path("checkpoints") / f"seg_fold{a.fold}.pt"
log = Path("outputs") / f"seg_fold{a.fold}_log.csv"
best = -1.0
with open(log, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_dsc", "val_jsc", "seconds"])
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0
        for x, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
            loss = dice_bce_loss(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        model.eval(); d = j = n = 0
        with torch.no_grad():
            for x, y, _ in va:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    lg = model(x)
                di, ji = dsc_jsc(lg.float(), y)
                d += di * x.size(0); j += ji * x.size(0); n += x.size(0)
        d, j = d / n, j / n
        w.writerow([ep, round(tot / len(tr.dataset), 4), round(d, 4), round(j, 4), round(time.time() - t0, 1)]); f.flush()
        print(f"epoch {ep:3d}/{a.epochs} | loss {tot/len(tr.dataset):.4f} | val DSC {d:.4f} JSC {j:.4f} | {time.time()-t0:.0f}s")
        if d > best:
            best = d
            if a.select == "best":
                torch.save(model.state_dict(), ckpt)
        if a.select == "last":
            torch.save(model.state_dict(), ckpt)
        last_d = d

# held-out predictions (out-of-fold) for the later age classifier
model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
out_dir = Path("outputs/oof_masks"); out_dir.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for x, _, ids in va:
        p = torch.sigmoid(model(x.to(dev)).float()).cpu().numpy()[:, 0]
        for pi, i in zip(p, ids.tolist()):
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), (pi * 255).astype(np.uint8))
print(f"final-epoch val DSC {last_d:.4f} | best-epoch val DSC {best:.4f} | used '{a.select}' checkpoint {ckpt} | held-out predictions -> {out_dir}")
