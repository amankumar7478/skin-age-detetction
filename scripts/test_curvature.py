"""Offline checks of the curvature code on synthetic surfaces (no MiDaS / GPU needed)."""
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.curvature import mean_curvature, silhouette_mask, suppress_edges

y, x = np.mgrid[-1:1:256j, -1:1:256j]

# 1) dome vs plane: a dome must be far more curved than a flat tilted plane
dome = np.sqrt(np.clip(1 - x**2 - y**2, 0, None))
plane = 0.3 * x + 0.1 * y
flat = float(np.abs(mean_curvature(plane)[20:-20, 20:-20]).mean())
assert abs(mean_curvature(dome)[128, 128]) > 10 * (flat + 1e-6)
print("dome vs plane: OK")

# 2) 'head against background': a raised disc (step edge) with small wrinkle ripples on top
r = np.sqrt(x**2 + y**2)
head = np.where(r < 0.7, 1.0, 0.0) + 0.02 * np.sin(40 * x) * (r < 0.6)
mask = silhouette_mask(head)
c = suppress_edges(mean_curvature(head), mask)
on_edge = mask[128, int(128 + 0.7 * 127)]
in_ripples = mask[128, 128]
inner = np.abs(c[(r < 0.5)]).max()
print("edge flagged:", bool(on_edge), "| ripple area flagged:", bool(in_ripples), "| ripple curvature kept:", inner > 0)
assert on_edge and not in_ripples and inner > 0
print("OK")
