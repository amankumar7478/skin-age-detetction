"""One-off (about 1 minute): measure how big the face is inside the FFHQ training photos, so the app's
auto-crop frames uploaded photos the same way.  Writes checkpoints/final_v1/crop_calib.json.

    python scripts/10_calibrate_crop.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.face_region import FaceParser  # noqa: E402
from models.pipeline import face_bbox  # noqa: E402

ids = pd.read_csv("outputs/data_check/folds.csv").image_number.sample(80, random_state=0).tolist()
parser = FaceParser()
ws, hs = [], []
for i in ids:
    img = cv2.cvtColor(cv2.imread(f"data/ffhq1000/{i:05d}.jpg"), cv2.COLOR_BGR2RGB)
    bb = face_bbox(np.isin(parser.labels(img), parser.face_ids))
    if bb:
        ws.append(bb[2] / img.shape[1]); hs.append(bb[3] / img.shape[0])
calib = {"w_frac": float(np.median(ws)), "h_frac": float(np.median(hs))}
print(f"typical face box in training photos: width {calib['w_frac']:.2f}, height {calib['h_frac']:.2f} of the frame")
Path("checkpoints/final_v1/crop_calib.json").write_text(json.dumps(calib))
print("saved checkpoints/final_v1/crop_calib.json")
