import os
import cv2
import glob
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# ── Config ──────────────────────────────────────────────
DATASET_DIRS = [
    "data/ppe/dataset1",
    "data/ppe/dataset2",
    "data/ppe/dataset3",
    "data/ppe/dataset4",
    "data/ppe/dataset5",
    "data/ppe/dataset6",
]
OUTPUT_DIR   = "data/ppe/merged"
IMG_SIZE     = 640
MAX_IMAGES_PER_CLASS = 2500   # cap number of IMAGES that contain each class
SEED         = 42
VAL_SPLIT    = 0.1
TEST_SPLIT   = 0.1
# ────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

# ── Class mapping ────────────────────────────────────────
# Maps raw dataset label → our unified label
# 'ignore' = skip this annotation
CLASS_MAP = {
    # helmet
    "hat"          : "helmet",
    "helmet"       : "helmet",
    "Helmet"       : "helmet",
    # no helmet
    "no hat"       : "no_helmet",
    "no helmet"    : "no_helmet",
    "no-helmet"    : "no_helmet",
    "head"         : "no_helmet",   # dataset4: bare head = no helmet
    # vest
    "vest"         : "vest",
    "Safety Vest"  : "vest",
    "Vest"         : "vest",
    # no vest
    "no vest"      : "no_vest",
    "no-vest"      : "no_vest",
    # gloves
    "gloves"       : "gloves",
    "Gloves"       : "gloves",
    "Glove"        : "gloves",
    # no gloves
    "no gloves"    : "ignore",
    # boots
    "boots"        : "boots",
    "Safety Boot"  : "boots",
    "Boots"        : "boots",
    # no boots
    "no boot"      : "ignore",
    "no boots"     : "ignore",
    # mask
    "Mask"         : "mask",
    "mask"         : "mask",
    # no mask
    "no-mask"      : "no_mask",
    # ignore
    "Human"        : "ignore",
    "Person"       : "ignore",
    "person"       : "ignore",
    "glasses"      : "ignore",
    "Glass"        : "ignore",
    "Ear-protection": "ignore",
}
# Final class list → integer label
CLASSES = [
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "gloves",
    "boots",
    "mask",
    "no_mask",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def parse_xml(xml_path):
    """Parse VOC XML → list of (unified_label, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size   = root.find("size")
    width  = int(size.find("width").text)
    height = int(size.find("height").text)

    filename = root.find("filename").text
    annotations = []

    for obj in root.findall("object"):
        raw_label = obj.find("name").text.strip()
        label     = CLASS_MAP.get(raw_label, "ignore")
        if label == "ignore":
            continue

        bb   = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))

        # clamp
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width,  xmax)
        ymax = min(height, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        annotations.append((label, xmin, ymin, xmax, ymax))

    return filename, width, height, annotations


def find_image(xml_path):
    """Find image file corresponding to an XML annotation."""
    stem    = Path(xml_path).stem
    folder  = Path(xml_path).parent
    for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
        p = folder / (stem + ext)
        if p.exists():
            return str(p)
    return None


def collect_all_records():
    """
    Walk all datasets, parse all XMLs, return flat list of:
    {img_path, label, xmin, ymin, xmax, ymax, width, height}
    """
    all_records = []
    class_counts = {c: 0 for c in CLASSES}

    for ds_dir in DATASET_DIRS:
        xml_files = glob.glob(f"{ds_dir}/**/*.xml", recursive=True)
        print(f"  {ds_dir}: {len(xml_files)} XMLs")

        for xml_path in xml_files:
            img_path = find_image(xml_path)
            if img_path is None:
                continue

            try:
                filename, w, h, anns = parse_xml(xml_path)
            except Exception as e:
                continue

            for (label, xmin, ymin, xmax, ymax) in anns:
                all_records.append({
                    "img_path": img_path,
                    "label"   : label,
                    "xmin"    : xmin,
                    "ymin"    : ymin,
                    "xmax"    : xmax,
                    "ymax"    : ymax,
                    "width"   : w,
                    "height"  : h,
                })
                class_counts[label] += 1

    return all_records, class_counts


def group_by_image(records):
    """Group flat annotation records by image path."""
    by_image = {}
    for r in records:
        by_image.setdefault(r["img_path"], []).append(r)
    return by_image


def balance_and_cap_by_image(by_image):
    """
    Cap by IMAGE not annotation: select up to MAX_IMAGES_PER_CLASS images
    that contain each class. An image is kept with ALL of its annotations
    intact (so the model never sees an unlabeled positive).
    """
    # For each class, list of images containing at least one annotation of it
    images_by_class = {c: [] for c in CLASSES}
    for img_path, recs in by_image.items():
        present = {r["label"] for r in recs}
        for c in present:
            images_by_class[c].append(img_path)

    # Cap per class, then union
    kept_images = set()
    for cls, imgs in images_by_class.items():
        random.shuffle(imgs)
        capped = imgs[:MAX_IMAGES_PER_CLASS]
        before = len(imgs)
        kept_images.update(capped)
        print(f"  {cls:15s}: {before:5d} images -> capped to {len(capped)}")

    # Reconstruct records — keep ALL annotations of every kept image
    balanced = []
    for img_path in kept_images:
        balanced.extend(by_image[img_path])

    print(f"  Kept {len(kept_images)} unique images, "
          f"{len(balanced)} annotations")
    return balanced


def split_records_by_image(records):
    """
    Split into train/val/test BY IMAGE, so the same image never appears
    in more than one split.
    """
    by_image = group_by_image(records)
    image_paths = list(by_image.keys())
    random.shuffle(image_paths)

    n        = len(image_paths)
    n_test   = int(n * TEST_SPLIT)
    n_val    = int(n * VAL_SPLIT)
    n_train  = n - n_test - n_val

    train_imgs = set(image_paths[:n_train])
    val_imgs   = set(image_paths[n_train:n_train+n_val])
    test_imgs  = set(image_paths[n_train+n_val:])

    train, val, test = [], [], []
    for r in records:
        if r["img_path"] in train_imgs:
            train.append(r)
        elif r["img_path"] in val_imgs:
            val.append(r)
        else:
            test.append(r)

    print(f"  Train : {len(train_imgs)} images, {len(train)} annotations")
    print(f"  Val   : {len(val_imgs)} images, {len(val)} annotations")
    print(f"  Test  : {len(test_imgs)} images, {len(test)} annotations")

    return train, val, test


def save_splits(train, val, test):
    """Save splits as JSON for use in training."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    path   = os.path.join(OUTPUT_DIR, "ppe_splits.json")
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\n  Splits saved to: {path}")
    return path


def verify_sample(records, n=5):
    """Load a few crops and confirm they look correct."""
    print(f"\n  Verifying {n} random crops...")
    sample = random.sample(records, min(n, len(records)))
    ok = 0
    for r in sample:
        img = cv2.imread(r["img_path"])
        if img is None:
            print(f"    [!] Could not load: {r['img_path']}")
            continue
        crop = img[r["ymin"]:r["ymax"], r["xmin"]:r["xmax"]]
        if crop.size == 0:
            print(f"    [!] Empty crop for label: {r['label']}")
            continue
        ok += 1
    print(f"  {ok}/{n} crops verified OK")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 - Preparing unified PPE dataset")
    print("=" * 60)

    print("\n[1/4] Collecting annotations from all datasets...")
    records, class_counts = collect_all_records()
    print(f"\n  Raw annotation counts:")
    for cls, cnt in class_counts.items():
        print(f"    {cls:15s}: {cnt}")
    print(f"\n  Total raw annotations: {len(records)}")

    print("\n[2/4] Balancing and capping per class (by IMAGE)...")
    by_image = group_by_image(records)
    print(f"  Unique images: {len(by_image)}")
    records = balance_and_cap_by_image(by_image)

    print("\n[3/4] Splitting train/val/test by IMAGE (80/10/10)...")
    train, val, test = split_records_by_image(records)

    print("\n[4/4] Saving splits...")
    save_splits(train, val, test)

    verify_sample(train)

    print("\n" + "=" * 60)
    print("DONE - PPE dataset ready")
    print("=" * 60)
    print(f"\nClasses ({len(CLASSES)}):")
    for i, c in enumerate(CLASSES):
        print(f"  {i}: {c}")