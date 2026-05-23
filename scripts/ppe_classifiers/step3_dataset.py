import torch
import cv2
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from step2_model import CLASS_TO_IDX, IDX_TO_CLASS

SPLITS_JSON = "data/ppe/merged/ppe_splits.json"
IMG_SIZE    = 300

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


class PPEDataset(Dataset):
    """
    Each record is a single annotated bounding box crop from the merged dataset.
    Returns:
        image  : FloatTensor [3, 300, 300]
        target : dict with boxes [N,4] cx,cy,w,h normalised and labels [N]
    """

    def __init__(self, records, augment=False):
        # group records by image so we load each image once
        self.augment = augment
        self.by_image = {}
        for r in records:
            key = r["img_path"]
            if key not in self.by_image:
                self.by_image[key] = []
            self.by_image[key].append(r)
        self.keys = list(self.by_image.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key     = self.keys[idx]
        records = self.by_image[key]

        img = cv2.imread(key)
        if img is None:
            # return blank if image missing
            return (torch.zeros(3, IMG_SIZE, IMG_SIZE),
                    {"boxes": torch.zeros(0,4),
                     "labels": torch.zeros(0, dtype=torch.long)})

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes  = []
        labels = []
        for r in records:
            boxes.append([r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
            labels.append(CLASS_TO_IDX[r["label"]])

        # ── Augmentation ──────────────────────────────────
        if self.augment:
            # horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                boxes = [[w-x2, y1, w-x1, y2]
                         for (x1,y1,x2,y2) in boxes]

            # brightness / contrast
            img = cv2.convertScaleAbs(
                img,
                alpha=random.uniform(0.7, 1.3),
                beta=random.randint(-30, 30)
            )

            # random crop (keep at least 80% of image)
            if random.random() > 0.5:
                crop_r = random.uniform(0.8, 1.0)
                x0 = random.randint(0, int(w*(1-crop_r)))
                y0 = random.randint(0, int(h*(1-crop_r)))
                x1c = x0 + int(w*crop_r)
                y1c = y0 + int(h*crop_r)
                img = img[y0:y1c, x0:x1c]
                new_boxes = []
                new_labels = []
                for (x1,y1,x2,y2), lbl in zip(boxes, labels):
                    nx1 = max(0, x1-x0)
                    ny1 = max(0, y1-y0)
                    nx2 = min(x1c-x0, x2-x0)
                    ny2 = min(y1c-y0, y2-y0)
                    if nx2 > nx1 and ny2 > ny1:
                        new_boxes.append([nx1, ny1, nx2, ny2])
                        new_labels.append(lbl)
                if new_boxes:
                    boxes  = new_boxes
                    labels = new_labels
                h, w = img.shape[:2]

        # ── Resize ────────────────────────────────────────
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ── Normalise boxes to cx,cy,w,h in [0,1] ────────
        boxes_norm = []
        for (x1,y1,x2,y2) in boxes:
            cx = ((x1+x2)/2) / w
            cy = ((y1+y2)/2) / h
            bw = (x2-x1) / w
            bh = (y2-y1) / h
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))
            bw = max(0.01, min(1.0,  bw))
            bh = max(0.01, min(1.0,  bh))
            boxes_norm.append([cx, cy, bw, bh])

        img_tensor = transform(img)
        target = {
            "boxes" : torch.tensor(boxes_norm, dtype=torch.float32),
            "labels": torch.tensor(labels,     dtype=torch.long),
        }
        return img_tensor, target


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def build_loaders(batch_size=16):
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    train_ds = PPEDataset(splits["train"], augment=True)
    val_ds   = PPEDataset(splits["val"],   augment=False)
    test_ds  = PPEDataset(splits["test"],  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing PPE dataset loader...")
    train_loader, val_loader, test_loader = build_loaders(batch_size=8)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    images, targets = next(iter(train_loader))
    print(f"Batch images shape : {images.shape}")
    print(f"Sample boxes       : {targets[0]['boxes']}")
    print(f"Sample labels      : {targets[0]['labels']}")
    print(f"Label names        : {[IDX_TO_CLASS[l] for l in targets[0]['labels'].tolist()]}")
    print("Dataset loader OK")
