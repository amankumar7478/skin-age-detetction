"""Inference pipeline for the app: uploaded photo -> wrinkle map, depth curvature, age-group predictions
WITH and WITHOUT curvature.  Every preprocessing constant comes from checkpoints/final_v1/settings.json,
so an uploaded photo is prepared exactly like the training photos were."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.curvature import mean_curvature, normalize01, silhouette_mask  # noqa: E402
from models.face_region import FaceParser, center_square, region_from_labels  # noqa: E402


def limit_side(rgb, max_side=1024):
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    f = max_side / s
    return cv2.resize(rgb, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


def face_bbox(mask):
    """Bounding box (x, y, w, h, area) of the largest connected blob in a boolean mask, or None."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in st[k])
    return x, y, w, h, area


def square_crop_around(rgb, bbox, w_frac=0.50, h_frac=0.60):
    """Square crop centred on the face so the face fills about the same share of the frame as in FFHQ.
    Never zooms OUT and never pads: the crop is clamped to stay inside the photo (padding made smeared stripes)."""
    x, y, w, h, _ = bbox
    H, W = rgb.shape[:2]
    side = int(round(max(h / h_frac, w / w_frac)))
    side = max(32, min(side, H, W))
    x0 = int(np.clip(round(x + w / 2 - side / 2), 0, W - side))
    y0 = int(np.clip(round(y + h / 2 - side / 2), 0, H - side))
    return rgb[y0:y0 + side, x0:x0 + side]


def make_classifier(in_ch, n_classes):
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    conv = m.features[0][0]
    m.features[0][0] = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


class SkinAgePipeline:
    def __init__(self, ckpt_dir="checkpoints/final_v1", device=None):
        import torch
        from models.fusion_unet import FusionUNet
        self.torch = torch
        ck = Path(ckpt_dir)
        self.cfg = json.loads((ck / "settings.json").read_text())
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = self.cfg["classes"]
        calib = ck / "crop_calib.json"        # written by scripts/10_calibrate_crop.py (optional)
        self.crop_frac = json.loads(calib.read_text()) if calib.exists() else {"w_frac": 0.50, "h_frac": 0.60}

        self.unet = FusionUNet(base=self.cfg["unet_base"]).to(self.dev)
        self.unet.load_state_dict(torch.load(ck / "unet.pt", map_location=self.dev)); self.unet.eval()

        self.cls = {}
        for name, chans in self.cfg["configs"].items():
            models = []
            for f in sorted(ck.glob(f"cls_{name}_s*.pt")):
                m = make_classifier(len(chans), len(self.classes)).to(self.dev)
                m.load_state_dict(torch.load(f, map_location=self.dev)); m.eval(); models.append(m)
            if not models:
                raise FileNotFoundError(f"no classifier files cls_{name}_s*.pt in {ck}")
            self.cls[name] = (chans, models)

        self.parser = FaceParser(device=str(self.dev))
        self.depth_net = torch.hub.load("intel-isl/MiDaS", self.cfg["depth_model"], trust_repo=True).to(self.dev).eval()
        self.depth_tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform

    # ---------------------------------------------------------------- pieces
    def _depth(self, img256):
        torch = self.torch
        with torch.no_grad():
            pred = self.depth_net(self.depth_tf(img256).to(self.dev))
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img256.shape[:2], mode="bicubic",
                                                   align_corners=False)
        return pred.squeeze().cpu().numpy()

    def _wrinkles(self, img512):
        torch = self.torch
        c = self.cfg
        x = (img512.astype(np.float32) / 255.0 - np.array(c["imagenet_mean"], np.float32)) / np.array(c["imagenet_std"], np.float32)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.dev)
        with torch.no_grad():
            return torch.sigmoid(self.unet(x).float())[0, 0].cpu().numpy()

    def _predict(self, name, chan_maps):
        torch = self.torch
        chans, models = self.cls[name]
        X = torch.from_numpy(np.stack([chan_maps[c] for c in chans])[None]).float().to(self.dev)
        with torch.no_grad():
            return np.mean([torch.softmax(m(X).float(), 1)[0].cpu().numpy() for m in models], axis=0)

    # ---------------------------------------------------------------- main entry
    def analyze(self, rgb, autocrop=True):
        c, S = self.cfg, self.cfg["size"]
        rgb = limit_side(rgb, 1024)
        if autocrop:
            first = np.isin(self.parser.labels(rgb), self.parser.face_ids)
            bb = face_bbox(first)
            if bb is None or bb[4] < 0.004 * first.size:
                return {"error": "No face found. Try a clear, front-facing photo with the whole face visible."}
            sq = square_crop_around(rgb, bb, self.crop_frac["w_frac"], self.crop_frac["h_frac"])
        else:
            sq = center_square(rgb)
        img512 = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA)
        img256 = cv2.resize(img512, (S, S), interpolation=cv2.INTER_AREA)

        face = region_from_labels(self.parser.labels(img512), self.parser.face_ids, S, c["close"], c["erode"])
        face_frac = float(face.sum() / face.size)
        if face_frac < 0.04:
            return {"error": "The face area is too small or hidden (sideways, covered or too far away). "
                             "Try a closer, front-facing photo."}

        depth = self._depth(img256)
        sil = silhouette_mask(depth, sigma=c["edge_sigma"], frac=c["edge_frac"], dilate=c["edge_dilate"])
        keep = face.astype(bool) & ~sil
        curv = np.where(keep, mean_curvature(depth, sigma=c["curv_sigma"]), 0).astype(np.float32)

        w512 = self._wrinkles(img512)
        w256 = cv2.resize((np.round(w512 * 255) / 255).astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)

        cc = np.arcsinh(curv / c["curv_p90"]) / c["arcsinh_div"]
        maps = {"w": (w256 - c["norm"]["w"][0]) / c["norm"]["w"][1],
                "c4": (cc - c["norm"]["c4"][0]) / c["norm"]["c4"][1]}
        p_w, p_f = self._predict("wrinkle", maps), self._predict("fused_s4", maps)
        return dict(error=None, img512=img512, img256=img256, face=face.astype(bool), keep=keep, depth01=normalize01(depth),
                    curv=curv, wrinkle512=w512, probs_wrinkle=p_w, probs_fused=p_f, classes=self.classes,
                    face_frac=face_frac, keep_frac=float(keep.sum() / max(1, face.sum())),
                    wrinkle_density=float((w512[cv2.resize(face.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0] > 0.5).mean()))
