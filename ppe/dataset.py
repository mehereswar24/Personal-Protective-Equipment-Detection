"""
PPE detection dataset with per-image class-presence masking.

The central fix. Only some source datasets annotate some classes: dataset4
labels helmets and heads but never vests; dataset2/3 never label heads. The old
pipeline fed every unannotated region to the loss as **background**, which
actively taught the model "an unlabelled bare head is not a head" — the single
biggest suppressor of violation detection, since violations are exactly what
those unlabelled regions represent.

Each sample therefore carries a `class_mask` derived from its source dataset.
The loss uses it to ignore classes that this image cannot testify about, so an
unlabelled helmet in a boots-only dataset is treated as *unknown*, not as a
negative.

Also implements:
  * letterbox resize (aspect preserved) — the old code squashed every image to
    a square, which distorted wide CCTV frames and made anchor/box statistics
    inconsistent between sources
  * repeat-factor sampling for the 18:1 class imbalance (LVIS-style), which
    replaces the loss class-weights that were entangled with the focal term
  * CCTV domain randomisation, since no real camera footage exists yet
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .taxonomy import CLASSES, CLASS_TO_IDX

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(img: np.ndarray, size: int, boxes: np.ndarray | None = None,
              pad_value: int = 114) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Resize preserving aspect ratio, pad to square.

    The old code did a bare cv2.resize to 300x300, so a 1920x1080 frame was
    squeezed horizontally by ~1.8x. Anything learnt about object shape was then
    source-resolution dependent.
    """
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad_value, dtype=img.dtype)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized

    out_boxes = None
    if boxes is not None and len(boxes):
        out_boxes = boxes.astype(np.float32).copy()
        out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]] * scale + dx
        out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]] * scale + dy
    return canvas, out_boxes, {"scale": scale, "dx": dx, "dy": dy}


def normalise_chw(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB → ImageNet-normalised float32 CHW, in one buffer.

    Deliberately not written as the obvious
    `((img.astype(f32)/255 - mean)/std).transpose(2, 0, 1).copy()`: that
    allocates four full-size float32 arrays in sequence and keeps two alive at
    once, so it peaks at twice the memory this does. Per sample the difference
    is a few MB, but it is paid on every worker's every prefetched sample at
    once, and this machine trains with ~1 GB of host RAM to spare — a dataloader
    worker died of MemoryError here on 2026-08-14.

    Kept module-level so the allocation behaviour can be tested without the
    dataset on disk; `tests/test_dataset.py` guards it.
    """
    h, w = img.shape[:2]
    out = np.empty((3, h, w), dtype=np.float32)
    for c in range(3):
        np.divide(img[:, :, c], 255.0, out=out[c], casting="unsafe")
        out[c] -= IMAGENET_MEAN[c]
        out[c] /= IMAGENET_STD[c]
    return out


def build_augmenter(size: int):
    """CCTV domain randomisation.

    We have no footage from the target cameras, so the training data (clean web
    photos) is systematically easier than deployment. These transforms simulate
    what a ceiling-mounted camera actually delivers: compression, motion blur,
    poor light, and — most importantly — low effective resolution on distant
    workers.
    """
    import albumentations as A

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.6, 1.5), translate_percent=(-0.1, 0.1),
                 rotate=(-7, 7), shear=(-5, 5), fit_output=False,
                 border_mode=cv2.BORDER_CONSTANT, fill=114, p=0.7),
        # distant-camera simulation — the single highest-value augmentation here
        A.OneOf([
            A.Downscale(scale_range=(0.3, 0.8), p=1.0),
            A.MotionBlur(blur_limit=(3, 11), p=1.0),
            A.Defocus(radius=(1, 5), p=1.0),
        ], p=0.45),
        A.ImageCompression(quality_range=(25, 75), p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.RandomGamma(gamma_limit=(60, 160), p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.6),
        A.GaussNoise(std_range=(0.03, 0.12), p=0.25),
        # Hue is deliberately barely touched: vest COLOUR is the signal for the
        # vest class, and destroying it would train the model out of its best cue.
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=25,
                             val_shift_limit=20, p=0.3),
        A.RandomShadow(p=0.15),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                               min_visibility=0.35, min_area=16))


class PPEDataset(Dataset):
    """Detection dataset over splits_v2.json.

    Each item: image tensor, target dict with boxes/labels, and `class_mask`
    marking which classes this image is allowed to supervise.
    """

    def __init__(self, split: str, size: int = 800, augment: bool = False,
                 splits_path: Path | None = None) -> None:
        self.size = size
        self.augment = augment
        path = splits_path or (ROOT / "data" / "ppe" / "splits_v2.json")
        data = json.loads(path.read_text(encoding="utf-8"))

        self.records: list[dict] = data["splits"][split]
        self.supervised: dict[str, list[str]] = data["supervised_classes_per_dataset"]
        self.classes = data.get("classes", CLASSES)
        self.split = split
        self._aug = build_augmenter(size) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def class_mask_for(self, dataset: str) -> torch.Tensor:
        """1 = this dataset annotates the class (loss applies), 0 = unknown."""
        allowed = set(self.supervised.get(dataset, self.classes))
        return torch.tensor([1.0 if c in allowed else 0.0 for c in self.classes],
                            dtype=torch.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = cv2.imread(str(ROOT / rec["img"]))
        if img is None:
            # Never fabricate a blank image with no boxes — the old code did
            # that, teaching the model "there is nothing here" from an IO error.
            return self[(idx + 1) % len(self)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = np.array([b[1:] for b in rec["boxes"]], dtype=np.float32).reshape(-1, 4)
        labels = np.array([CLASS_TO_IDX[b[0]] for b in rec["boxes"]], dtype=np.int64)

        if self._aug is not None and len(boxes):
            try:
                out = self._aug(image=img, bboxes=boxes.tolist(), labels=labels.tolist())
                img = out["image"]
                if out["bboxes"]:
                    boxes = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
                    labels = np.array(out["labels"], dtype=np.int64)
                else:
                    boxes = np.zeros((0, 4), np.float32)
                    labels = np.zeros((0,), np.int64)
            except Exception:
                pass       # a failed augmentation must not drop the sample

        img, boxes, _ = letterbox(img, self.size, boxes)
        if boxes is None:
            boxes = np.zeros((0, 4), np.float32)

        # drop degenerate boxes produced by cropping/affine
        if len(boxes):
            keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
            boxes, labels = boxes[keep], labels[keep]

        tensor = torch.from_numpy(normalise_chw(img))

        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
            "labels": torch.from_numpy(labels.astype(np.int64)).reshape(-1),
            "class_mask": self.class_mask_for(rec["dataset"]),
            "dataset": rec["dataset"],
            "image_id": rec["img"],
        }
        return tensor, target

    # ── imbalance handling ──
    def repeat_factors(self, threshold: float = 0.15) -> list[float]:
        """LVIS-style repeat-factor sampling.

        Images containing rare classes are sampled more often. This replaces the
        old approach of class weights inside the focal loss, which corrupted the
        focal modulation (`pt = exp(-weighted_ce)` is not p_t once the weight
        differs from 1), and the old "balance by capping images per class",
        which only deleted data without equalising anything.
        """
        img_freq: Counter = Counter()
        for rec in self.records:
            for c in {b[0] for b in rec["boxes"]}:
                img_freq[c] += 1
        n = max(1, len(self.records))
        cat_rf = {c: max(1.0, math.sqrt(threshold / (f / n)))
                  for c, f in img_freq.items() if f > 0}
        return [max([cat_rf.get(c, 1.0) for c in {b[0] for b in rec["boxes"]}] or [1.0])
                for rec in self.records]

    def class_counts(self) -> dict[str, int]:
        counts: Counter = Counter()
        for rec in self.records:
            for b in rec["boxes"]:
                counts[b[0]] += 1
        return dict(counts)


def collate(batch):
    """Detection batches have variable box counts, so targets stay a list."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


def make_sampler(dataset: PPEDataset, seed: int = 42):
    """WeightedRandomSampler driven by repeat factors."""
    from torch.utils.data import WeightedRandomSampler
    weights = dataset.repeat_factors()
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                 replacement=True, generator=g)
