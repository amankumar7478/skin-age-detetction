"""Fit one 'temperature' per method on the cross-validated (held-out) probabilities, so the app's confidence
numbers are realistic.  The predicted group never changes; only how sure the model claims to be.
Writes checkpoints/final_v1/calibration.json.

    python scripts/12_calibrate_probs.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["young", "middle", "old"]
df = pd.read_csv("outputs/data_check/folds.csv")
df = df[df.group3.isin(CLASSES)].reset_index(drop=True)
y = df.group3.map({c: j for j, c in enumerate(CLASSES)}).values


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


out = {}
for cfg in ("wrinkle", "fused_s4"):
    p = np.load(f"outputs/age_cv/probs_{cfg}.npy")
    assert len(p) == len(y), f"{cfg}: {len(p)} probability rows vs {len(y)} labelled photos"
    logit = np.log(p + 1e-9)
    grid = np.linspace(0.3, 4.0, 371)
    T = float(grid[int(np.argmin([nll(softmax(logit / t), y) for t in grid]))])
    after = softmax(logit / T)
    out[cfg] = T
    print(f"{cfg:9s} T={T:.2f} | accuracy {np.mean(p.argmax(1) == y):.3f} (unchanged) | "
          f"NLL {nll(p, y):.3f} -> {nll(after, y):.3f} | ECE {ece(p, y):.3f} -> {ece(after, y):.3f}")
Path("checkpoints/final_v1/calibration.json").write_text(json.dumps(out))
print("saved checkpoints/final_v1/calibration.json")
