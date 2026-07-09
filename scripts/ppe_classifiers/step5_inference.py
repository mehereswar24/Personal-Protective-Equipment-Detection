import torch
import cv2
import numpy as np
import sys
import os
sys.path.append("scripts/ppe_classifiers")
sys.path.append("scripts/person_detector")

from step2_model import (
    PPEDetector, AnchorGenerator, CLASSES, CLASS_TO_IDX,
    NUM_CLASSES, LEGACY_NUM_CLASSES, NUM_ANCHORS,
)
import torchvision.transforms as T

# ── Config ──────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PPE_MODEL_PATH  = "models/ppe_detector_best.pth"
CONF_THRESHOLD  = 0.50
NMS_THRESHOLD   = 0.3
MAX_DETECTIONS  = 30
INPUT_SIZE      = 300
# ────────────────────────────────────────────────────────
CLASS_THRESHOLDS = {
    "helmet"    : 0.45,
    "no_helmet" : 0.55,
    "vest"      : 0.45,
    "no_vest"   : 0.55,
    "gloves"    : 0.35,
    "boots"     : 0.35,
    "mask"      : 0.40,
    "no_mask"   : 0.45,
}

# colours per class
CLASS_COLORS = {
    "helmet"    : (0,   255, 0  ),   # green
    "no_helmet" : (0,   0,   255),   # red
    "vest"      : (0,   255, 128),   # light green
    "no_vest"   : (0,   0,   200),   # dark red
    "gloves"    : (255, 200, 0  ),   # yellow
    "boots"     : (255, 128, 0  ),   # orange
    "mask"      : (128, 255, 0  ),   # lime
    "no_mask"   : (0,   0,   180),   # red-blue
}

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _num_classes_from_checkpoint(ckpt):
    bias = ckpt["model"].get("cls_heads.0.bias")
    if bias is None:
        return NUM_CLASSES
    # bias shape = num_anchors * num_classes; back-compat with old 3-anchor
    # checkpoints by trying the current anchor count first.
    for n_anc in (NUM_ANCHORS, 3):
        if bias.numel() % n_anc == 0:
            candidate = bias.numel() // n_anc
            if candidate in (NUM_CLASSES, LEGACY_NUM_CLASSES):
                return candidate
    return NUM_CLASSES


def _class_index_for_model(cls_name, model_num_classes):
    if model_num_classes == len(CLASSES):
        return CLASSES.index(cls_name)
    return CLASS_TO_IDX[cls_name]


def load_ppe_model():
    ckpt  = torch.load(PPE_MODEL_PATH, map_location=DEVICE,
                       weights_only=False)
    ckpt_num_classes = _num_classes_from_checkpoint(ckpt)
    model = PPEDetector(num_classes=ckpt_num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"PPE model loaded from epoch {ckpt['epoch']} "
          f"(val_loss: {ckpt['val_loss']:.4f}, "
          f"classes: {ckpt_num_classes})")
    return model


def decode_boxes(box_preds, anchors, variances=(0.1, 0.2)):
    cx = box_preds[:,0]*variances[0]*anchors[:,2] + anchors[:,0]
    cy = box_preds[:,1]*variances[0]*anchors[:,3] + anchors[:,1]
    w  = torch.exp(box_preds[:,2]*variances[1]) * anchors[:,2]
    h  = torch.exp(box_preds[:,3]*variances[1]) * anchors[:,3]
    x1 = (cx - w/2).clamp(0, 1)
    y1 = (cy - h/2).clamp(0, 1)
    x2 = (cx + w/2).clamp(0, 1)
    y2 = (cy + h/2).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=1)


def nms(boxes, scores, iou_threshold=0.4):
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)
    
    # use torchvision's built-in NMS which is correct and fast
    from torchvision.ops import nms as tv_nms
    keep = tv_nms(boxes, scores, iou_threshold)
    return keep


@torch.no_grad()
def detect_ppe(model, anchors, frame_rgb):
    h, w = frame_rgb.shape[:2]
    img  = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    cls_logits, box_preds = model(tensor)
    cls_logits = cls_logits[0]
    box_preds  = box_preds[0]

    scores_all = torch.softmax(cls_logits, dim=1)

    all_boxes   = []
    all_scores  = []
    all_classes = []

    # per-class detection with individual thresholds
    model_num_classes = getattr(model, "num_classes", NUM_CLASSES)
    for cls_name in CLASSES:
        cls_idx = _class_index_for_model(cls_name, model_num_classes)
        threshold = CLASS_THRESHOLDS.get(cls_name, CONF_THRESHOLD)
        scores    = scores_all[:, cls_idx]
        mask      = scores > threshold

        if mask.sum() == 0:
            continue

        cls_boxes  = decode_boxes(box_preds[mask],
                                  anchors.to(DEVICE)[mask])
        cls_scores = scores[mask]

        # per-class NMS
        from torchvision.ops import nms as tv_nms
        keep = tv_nms(cls_boxes, cls_scores, NMS_THRESHOLD)
        keep = keep[:10]  # max 10 per class

        for k in keep:
            all_boxes.append(cls_boxes[k])
            all_scores.append(cls_scores[k])
            all_classes.append(cls_name)

    if not all_boxes:
        return []

    all_boxes   = torch.stack(all_boxes)
    all_scores  = torch.stack(all_scores)

    # final global NMS to remove cross-class overlaps
    from torchvision.ops import nms as tv_nms
    keep = tv_nms(all_boxes, all_scores, NMS_THRESHOLD)
    keep = keep[:MAX_DETECTIONS]

    results = []
    for k in keep:
        k = int(k.item())
        score   = all_scores[k].item()
        box     = all_boxes[k]
        label   = all_classes[k]
        x1 = int(box[0].item() * w)
        y1 = int(box[1].item() * h)
        x2 = int(box[2].item() * w)
        y2 = int(box[3].item() * h)
        results.append((label, x1, y1, x2, y2, score))

    return results


def draw_ppe(frame, detections):
    for (label, x1, y1, x2, y2, score) in detections:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame,
                      (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
    return frame


def compliance_status(detections):
    detected = set(d[0] for d in detections)
    status = {}
    pairs = [
        ("helmet",  "no_helmet",  "Helmet"),
        ("vest",    "no_vest",    "Vest"),
        ("gloves",  None,         "Gloves"),
        ("boots",   None,         "Boots"),
        ("mask",    "no_mask",    "Mask"),
    ]
    for pos, neg, name in pairs:
        if pos in detected and (neg is None or neg not in detected):
            status[name] = ("YES", (0, 255, 0))
        elif neg and neg in detected:
            status[name] = ("NO", (0, 0, 255))
        else:
            status[name] = ("?", (128, 128, 128))
    return status


def draw_compliance(frame, status):
    y = 30
    for name, (state, color) in status.items():
        cv2.putText(frame, f"{name}: {state}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        y += 25
    return frame


def test_on_image(image_path):
    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    model      = load_ppe_model()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load: {image_path}")
        return

    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detect_ppe(model, anchors, frame_rgb)

    print(f"Detected {len(detections)} PPE items:")
    for (label, x1, y1, x2, y2, score) in detections:
        print(f"  {label:12s} score={score:.3f} "
              f"box=({x1},{y1},{x2},{y2})")

    status = compliance_status(detections)
    print("\nCompliance status:")
    for name, state in status.items():
        print(f"  {name}: {state}")

    frame = draw_ppe(frame, detections)
    frame = draw_compliance(frame, status)

    out = "output/ppe_detection_test.jpg"
    cv2.imwrite(out, frame)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    import glob
    # test on a sample from the PPE dataset
    images = glob.glob("data/ppe/dataset1/train/*.jpg")
    if not images:
        images = glob.glob("data/ppe/dataset4/train/*.jpg")

    if images:
        import random
        test_img = random.choice(images[:50])
        print(f"Testing on: {test_img}")
        test_on_image(test_img)
    else:
        print("No test images found")
    
