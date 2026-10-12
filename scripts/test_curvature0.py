"""Offline check of the curvature code on a synthetic dome (no MiDaS / GPU needed)."""
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.curvature import mean_curvature

y, x = np.mgrid[-1:1:256j, -1:1:256j]
dome = np.sqrt(np.clip(1 - x**2 - y**2, 0, None))
plane = 0.3 * x + 0.1 * y

c_dome = mean_curvature(dome)
c_plane = mean_curvature(plane)
flat = float(np.abs(c_plane[20:-20, 20:-20]).mean())
print("dome  centre curvature :", float(c_dome[128, 128]))
print("plane mean |curvature| :", flat)
assert abs(c_dome[128, 128]) > 10 * (flat + 1e-6), "dome should be much more curved than a plane"
print("OK")
