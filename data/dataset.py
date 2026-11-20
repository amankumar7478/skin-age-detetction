"""Photo + wrinkle-mask dataset for U-Net training."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class WrinkleDataset(Dataset):
    def __init__(self, ids, photos="data/ffhq1000", masks="manual_wrinkle_masks", size=512, augment=False):
        self.ids = list(ids)
        self.photos, self.masks = Path(photos), Path(masks)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.ids)

    def load(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.photos / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.masks / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # INTER_AREA keeps thin 1024px lines visible after shrinking; threshold turns them back into 0/1
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img, (m > 40).astype(np.float32)

    def __getitem__(self, k):
        i = self.ids[k]
        img, m = self.load(i)
        if self.augment:
            if np.random.rand() < 0.5:                       # horizontal flip
                img, m = img[:, ::-1], m[:, ::-1]
            a, b = np.random.uniform(0.85, 1.15), np.random.uniform(-15, 15)   # contrast / brightness
            img = np.clip(img.astype(np.float32) * a + b, 0, 255)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(m))[None]
        return x, y, i
