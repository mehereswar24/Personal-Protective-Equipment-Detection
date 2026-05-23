import os
import cv2
import numpy as np
import glob
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

# ── Config ──────────────────────────────────────────────
INRIA_PATH   = "INRIAPerson"
OUTPUT_DIR   = "data/person"
PATCH_W      = 300          # SSD input size
PATCH_H      = 300
SEED         = 42
VAL_SPLIT    = 0.2
# ────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


def parse_xml(xml_path):
    """Parse VOC XML → (filename, list of [xmin, ymin, xmax, ymax])."""
    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    filename = root.find("filename").text
    size     = root.find("size")
    width    = int(size.find("width").text)
    height   = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        if obj.find("name").text.lower() != "person":
            continue
        bb   = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        # skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])

    return filename, width, height, boxes


def load_image(img_dir, filename):
    """Try multiple extensions when loading."""
    stem = Path(filename).stem
    for ext in [filename, stem+".jpg", stem+".png",
                stem+".JPG", stem+".JPEG"]:
        path = os.path.join(img_dir, ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    return None


def build_annotation_list(split="Train"):
    """Return list of dicts: {img_path, boxes, width, height}."""
    ann_dir  = os.path.join(INRIA_PATH, split, "Annotations")
    img_dir  = os.path.join(INRIA_PATH, split, "JPEGImages")
    xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))

    records = []
    skipped = 0

    for xml_path in xml_files:
        filename, w, h, boxes = parse_xml(xml_path)
        if not boxes:
            skipped += 1
            continue

        img = load_image(img_dir, filename)
        if img is None:
            skipped += 1
            continue

        # resolve actual image path
        stem = Path(filename).stem
        img_path = None
        for ext in [filename, stem+".jpg", stem+".png", stem+".JPG"]:
            p = os.path.join(img_dir, ext)
            if os.path.exists(p):
                img_path = p
                break

        records.append({
            "img_path" : img_path,
            "boxes"    : boxes,        # list of [xmin,ymin,xmax,ymax] absolute
            "width"    : w,
            "height"   : h,
        })

    print(f"  [{split}] loaded: {len(records)}  skipped: {skipped}")
    return records


class PersonDataset(Dataset):
    """
    Returns:
        image  : FloatTensor [3, 300, 300]  normalised
        target : dict with keys
                   boxes  — FloatTensor [N,4] normalised 0-1 (cx,cy,w,h)
                   labels — LongTensor  [N]   all 1 (person)
    """

    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        img  = cv2.imread(rec["img_path"])
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # ── augmentation ──────────────────────────────
        if self.augment:
            # horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                flipped = []
                for (x1,y1,x2,y2) in rec["boxes"]:
                    flipped.append([w-x2, y1, w-x1, y2])
                rec = dict(rec, boxes=flipped)

            # brightness / contrast jitter
            img = cv2.convertScaleAbs(
                img,
                alpha=random.uniform(0.8, 1.2),
                beta=random.randint(-20, 20)
            )

        # ── resize to 300x300 ─────────────────────────
        img_resized = cv2.resize(img, (PATCH_W, PATCH_H))

        # ── normalise boxes to cx,cy,w,h in [0,1] ────
        boxes_norm = []
        for (x1,y1,x2,y2) in rec["boxes"]:
            cx = ((x1+x2)/2) / w
            cy = ((y1+y2)/2) / h
            bw = (x2-x1) / w
            bh = (y2-y1) / h
            # clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.01, min(1.0, bw))
            bh = max(0.01, min(1.0, bh))
            boxes_norm.append([cx, cy, bw, bh])

        img_tensor = self.transform(img_resized)
        target = {
            "boxes"  : torch.tensor(boxes_norm, dtype=torch.float32),
            "labels" : torch.ones(len(boxes_norm), dtype=torch.long),
        }
        return img_tensor, target


def collate_fn(batch):
    """Custom collate — targets are variable length so we keep as list."""
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def prepare_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("STEP 1 — Building person detection dataset")
    print("=" * 55)

    # collect all records from both splits
    train_records = build_annotation_list("Train")
    test_records  = build_annotation_list("Test")
    all_records   = train_records + test_records

    # shuffle and split
    random.shuffle(all_records)
    split_idx   = int(len(all_records) * (1 - VAL_SPLIT))
    train_recs  = all_records[:split_idx]
    val_recs    = all_records[split_idx:]

    print(f"\n  Total images : {len(all_records)}")
    print(f"  Train        : {len(train_recs)}")
    print(f"  Val          : {len(val_recs)}")

    # save split paths to json so other scripts can load them
    split_info = {
        "train": train_recs,
        "val"  : val_recs,
    }
    json_path = os.path.join(OUTPUT_DIR, "splits.json")
    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Splits saved to: {json_path}")

    # quick sanity check — build datasets and one dataloader batch
    print("\n  Running sanity check on dataloader...")
    train_ds = PersonDataset(train_recs, augment=True)
    val_ds   = PersonDataset(val_recs,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=4,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    images, targets = next(iter(train_loader))
    print(f"  Batch images shape : {images.shape}")
    print(f"  Sample boxes       : {targets[0]['boxes']}")
    print(f"  Sample labels      : {targets[0]['labels']}")

    print("\n" + "=" * 55)
    print("DONE — dataset ready for SSD training")
    print("=" * 55)

    return train_loader, val_ds


if __name__ == "__main__":
    prepare_dataset()