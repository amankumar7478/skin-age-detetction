"""Find the FACE REGION of any photo, the same way the training data was prepared.

In training (scripts/07_precompute_curvature.py) the face region came from the dataset's own
face-parsing labels:  classes skin, brows, eyes, nose, mouth, upper lip, lower lip  ->  close (7 px)
-> erode (12 px) at 256x256.  An uploaded photo has no such labels, so we run a pretrained
face-parsing model (SegFormer trained on CelebAMask-HQ, from Hugging Face) and apply the SAME recipe.

The model's class numbers are NOT the same as the dataset's, so we never hard-code numbers: we read
the model's own id->name list and pick the classes by NAME (see pick_face_ids).

    from models.face_region import FaceParser
    fp = FaceParser()                         # first run downloads the model (a few hundred MB)
    mask256 = fp.face_region(rgb_512)         # uint8 0/1, 256x256
"""
import cv2
import numpy as np

MODEL_ID = "jonathandinu/face-parsing"
INCLUDE = ("skin", "nose", "eye", "brow", "mouth", "lip")
EXCLUDE = ("glass", "eye_g", "ear", "hair", "hat", "neck", "cloth", "back")
# Recipe copied from scripts/07_precompute_curvature.py (keep identical!)
SIZE, CLOSE, ERODE = 256, 7, 12


def pick_face_ids(id2label):
    """Class ids (by NAME) that count as 'face': skin, nose, eyes, brows, mouth, lips. No glasses/ears/hair/neck."""
    ids = []
    for i, name in id2label.items():
        n = str(name).lower()
        if any(k in n for k in INCLUDE) and not any(k in n for k in EXCLUDE):
            ids.append(int(i))
    return sorted(ids)


def region_from_labels(labels, face_ids, size=SIZE, close=CLOSE, erode=ERODE, keep_largest=True):
    """label map (any size) -> face-region mask (uint8 0/1, size x size). Same steps as training."""
    p = cv2.resize(labels.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    face = np.isin(p, face_ids).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    face = cv2.erode(face, np.ones((erode, erode), np.uint8))
    if keep_largest and face.any():   # drop stray specks / thin spikes away from the main face
        n, cc, st, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
        if n > 2:
            face = (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return face


def center_square(rgb):
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]


class FaceParser:
    def __init__(self, model_id=MODEL_ID, device=None):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proc = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.dev).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.face_ids = pick_face_ids(self.id2label)

    def labels(self, rgb):
        """uint8 RGB image (H, W, 3) -> label map (H, W) in THIS MODEL's class ids."""
        torch = self.torch
        h, w = rgb.shape[:2]
        pix = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.dev)
        with torch.no_grad():
            logits = self.model(pixel_values=pix).logits
            up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy().astype(np.uint8)

    def face_region(self, rgb, size=SIZE, close=CLOSE, erode=ERODE):
        return region_from_labels(self.labels(rgb), self.face_ids, size, close, erode)
