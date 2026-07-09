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
PPE_DIRS     = [
    "data/ppe/dataset1", "data/ppe/dataset2", "data/ppe/dataset3",
    "data/ppe/dataset4", "data/ppe/dataset5", "data/ppe/dataset6",
]
OUTPUT_DIR   = "data/person"
PATCH_W      = 300          # SSD input size
PATCH_H      = 300
SEED         = 42
VAL_SPLIT    = 0.1
TEST_SPLIT   = 0.1
# ────────────────────────────────────────────────────────

# Labels we treat as "person" across both INRIA and the PPE datasets.
# Construction datasets sometimes label workers as "Human" / "Worker"; the
# bare-head boxes ("head", "no helmet") are tight head crops that are too
# small to use as person boxes, so we exclude them.
PERSON_LABELS = {"person", "people", "human", "worker", "pedestrian"}

random.seed(SEED)
np.random.seed(SEED)


def parse_xml(xml_path):
    """Parse VOC XML → (filename, width, height, list of [xmin,ymin,xmax,ymax])."""
    tree  = ET.parse(xml_path)
    root  = tree.getroot()

    fn_node = root.find("filename")
    filename = fn_node.text if fn_node is not None else Path(xml_path).stem
    size     = root.find("size")
    width    = int(float(size.find("width").text))
    height   = int(float(size.find("height").text))

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in PERSON_LABELS:
            continue
        bb   = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        # clamp + skip degenerate
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(width, xmax); ymax = min(height, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])

    return filename, width, height, boxes


def _find_image_alongside_xml(xml_path):
    """For PPE datasets where image sits next to its XML."""
    stem = Path(xml_path).stem
    folder = Path(xml_path).parent
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = folder / (stem + ext)
        if p.exists():
            return str(p)
    return None


def build_ppe_records():
    """Harvest person/worker/human annotations from all PPE datasets."""
    records = []
    skipped = 0
    for ds in PPE_DIRS:
        xmls = glob.glob(f"{ds}/**/*.xml", recursive=True)
        ds_imgs = 0
        for x in xmls:
            try:
                _, w, h, boxes = parse_xml(x)
            except Exception:
                skipped += 1
                continue
            if not boxes:
                continue
            img_path = _find_image_alongside_xml(x)
            if img_path is None:
                skipped += 1
                continue
            records.append({
                "img_path": img_path,
                "boxes":    boxes,
                "width":    w,
                "height":   h,
            })
            ds_imgs += 1
        print(f"  [PPE {ds}] images with person: {ds_imgs}")
    print(f"  PPE total : {len(records)} images, skipped {skipped}")
    return records


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
        if img is None:
            # Image missing on disk — return blank with empty boxes so the
            # batch survives. The dataloader will see zero positives.
            blank = np.zeros((PATCH_H, PATCH_W, 3), dtype=np.uint8)
            return (self.transform(blank),
                    {"boxes":  torch.zeros(0, 4),
                     "labels": torch.zeros(0, dtype=torch.long)})

        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = [list(b) for b in rec["boxes"]]

        # ── augmentation ──────────────────────────────
        if self.augment:
            # 1. Horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                boxes = [[w - x2, y1, w - x1, y2]
                         for (x1, y1, x2, y2) in boxes]

            # 2. Color jitter (brightness + contrast)
            img = cv2.convertScaleAbs(
                img,
                alpha=random.uniform(0.7, 1.3),
                beta=random.randint(-30, 30),
            )

            # 3. Random crop (50% chance) — covers cropped / zoomed views
            if random.random() < 0.5:
                crop_r = random.uniform(0.6, 1.0)
                cw = max(1, int(w * crop_r))
                ch = max(1, int(h * crop_r))
                x0 = random.randint(0, w - cw)
                y0 = random.randint(0, h - ch)
                x1c, y1c = x0 + cw, y0 + ch
                new_boxes = []
                for (bx1, by1, bx2, by2) in boxes:
                    nx1 = max(0, bx1 - x0)
                    ny1 = max(0, by1 - y0)
                    nx2 = min(cw, bx2 - x0)
                    ny2 = min(ch, by2 - y0)
                    if (nx2 - nx1) > 5 and (ny2 - ny1) > 5:
                        new_boxes.append([nx1, ny1, nx2, ny2])
                if new_boxes:
                    img   = img[y0:y1c, x0:x1c]
                    boxes = new_boxes
                    h, w  = img.shape[:2]

            # 4. Scale jitter (50% chance) — pad image to smaller relative
            # size to simulate distant workers. Place original onto a 1.5x
            # canvas in a random position.
            if random.random() < 0.5:
                scale = random.uniform(1.0, 1.8)
                new_w = int(w * scale)
                new_h = int(h * scale)
                canvas = np.full((new_h, new_w, 3),
                                 fill_value=random.randint(100, 160),
                                 dtype=np.uint8)
                ox = random.randint(0, new_w - w)
                oy = random.randint(0, new_h - h)
                canvas[oy:oy + h, ox:ox + w] = img
                img = canvas
                boxes = [[x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                         for (x1, y1, x2, y2) in boxes]
                h, w = img.shape[:2]

        # ── resize to 300x300 ─────────────────────────
        img_resized = cv2.resize(img, (PATCH_W, PATCH_H))

        # ── normalise boxes to cx,cy,w,h in [0,1] ────
        boxes_norm = []
        for (x1, y1, x2, y2) in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))
            bw = max(0.01, min(1.0,  bw))
            bh = max(0.01, min(1.0,  bh))
            boxes_norm.append([cx, cy, bw, bh])

        img_tensor = self.transform(img_resized)
        target = {
            "boxes":  torch.tensor(boxes_norm, dtype=torch.float32),
            "labels": torch.ones(len(boxes_norm), dtype=torch.long),
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
    print("STEP 1 - Building person detection dataset")
    print("=" * 55)

    # ── INRIA pedestrians (clean street-level) ───────────
    print("\n[INRIA]")
    inria_train = build_annotation_list("Train")
    inria_test  = build_annotation_list("Test")
    inria_all   = inria_train + inria_test
    print(f"  INRIA total: {len(inria_all)} images")

    # ── PPE-dataset persons (construction scenes) ────────
    print("\n[PPE datasets]")
    ppe_records = build_ppe_records()

    all_records = inria_all + ppe_records
    print(f"\n  Combined : {len(all_records)} images")
    total_boxes = sum(len(r["boxes"]) for r in all_records)
    print(f"  Boxes    : {total_boxes}")

    # ── Image-level shuffled split ───────────────────────
    random.shuffle(all_records)
    n        = len(all_records)
    n_test   = int(n * TEST_SPLIT)
    n_val    = int(n * VAL_SPLIT)
    n_train  = n - n_val - n_test

    train_recs = all_records[:n_train]
    val_recs   = all_records[n_train:n_train + n_val]
    test_recs  = all_records[n_train + n_val:]

    print(f"\n  Train : {len(train_recs)} images")
    print(f"  Val   : {len(val_recs)} images")
    print(f"  Test  : {len(test_recs)} images")

    # ── Save split ───────────────────────────────────────
    split_info = {
        "train": train_recs,
        "val":   val_recs,
        "test":  test_recs,
    }
    json_path = os.path.join(OUTPUT_DIR, "splits.json")
    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Splits saved to: {json_path}")

    # ── Sanity check ─────────────────────────────────────
    print("\n  Running sanity check on dataloader...")
    train_ds = PersonDataset(train_recs, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=4,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    images, targets = next(iter(train_loader))
    print(f"  Batch images shape : {tuple(images.shape)}")
    print(f"  Sample boxes       : {targets[0]['boxes']}")
    print(f"  Sample labels      : {targets[0]['labels']}")

    print("\n" + "=" * 55)
    print("DONE - person dataset ready")
    print("=" * 55)

    return train_loader, PersonDataset(val_recs, augment=False)


if __name__ == "__main__":
    prepare_dataset()