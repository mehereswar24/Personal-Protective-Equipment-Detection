import torch
import cv2
import numpy as np
import sys
import os
import collections
import csv
import datetime
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
# Was `from sort import Sort` — a GPL-3.0 tracker that also imported
# matplotlib/TkAgg at module scope (so this script could not start headless).
# ppe.tracking.Tracker is an MIT reimplementation with the same call signature.
from ppe.tracking import Tracker as Sort

sys.path.append("scripts/person_detector")
sys.path.append("scripts/ppe_classifiers")

import importlib.util

# load person detector model
spec = importlib.util.spec_from_file_location(
    "person_step2", "scripts/person_detector/step2_model.py")
person_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(person_module)
MobileNetV2SSD  = person_module.MobileNetV2SSD
PersonAnchorGen = person_module.AnchorGenerator

# load PPE model
spec = importlib.util.spec_from_file_location(
    "ppe_step2", "scripts/ppe_classifiers/step2_model.py")
ppe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppe_module)
PPEDetector         = ppe_module.PPEDetector
PPEAnchorGen        = ppe_module.AnchorGenerator
PPE_CLASSES         = ppe_module.CLASSES
PPE_CLASS_TO_IDX    = ppe_module.CLASS_TO_IDX
PPE_NUM_CLASSES     = ppe_module.NUM_CLASSES
PPE_LEGACY_CLASSES  = ppe_module.LEGACY_NUM_CLASSES
PPE_NUM_ANCHORS     = ppe_module.NUM_ANCHORS

import torchvision.transforms as T
from torchvision.ops import nms as tv_nms

# ── Config ──────────────────────────────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERSON_MODEL_PATH = "models/person_detector_best.pth"
PPE_MODEL_PATH    = "models/ppe_detector_best.pth"
PERSON_CONF       = 0.45   # balanced: avoids background FPs while catching workers
PERSON_NMS        = 0.35   # slightly lower: avoids merging nearby workers into one box
PPE_NMS           = 0.20
MAX_PERSONS       = 10
TEMPORAL_WINDOW   = 7
# max_age was 2, set to "prevent predicted boxes from shooting off". That was a
# misdiagnosis: the tracker only ever emits boxes updated on the current frame,
# so it never showed prediction-only boxes in the first place. The low value
# just killed track continuity — a worker occluded for 3 frames came back with a
# NEW id, which orphaned their temporal-vote history AND bypassed the 10-second
# violation dedup (duplicate alerts for one person). 30 frames ≈ 1.2s at 25fps.
SORT_MAX_AGE      = 30
SORT_MIN_HITS     = 3      # debounce spurious single-frame detections
SORT_IOU_THRESH   = 0.3
INPUT_SIZE        = 300

PPE_THRESHOLDS = {
    "helmet"    : 0.45,
    "no_helmet" : 0.55,
    "vest"      : 0.45,
    "no_vest"   : 0.55,
    "gloves"    : 0.40,
    "boots"     : 0.40,
    "mask"      : 0.40,
    "no_mask"   : 0.45,
}

# FIX 2: max detections per class — helmet/vest=1, gloves/boots=2 (left+right)
MAX_DETS_PER_CLASS = {
    "helmet"    : 1,
    "no_helmet" : 1,
    "vest"      : 1,
    "no_vest"   : 1,
    "gloves"    : 2,   # FIX 2: was 1, now allows both left+right glove
    "boots"     : 2,   # FIX 2: was 1, now allows both left+right boot
    "mask"      : 1,
    "no_mask"   : 1,
}

# FIX 2: minimum pixel separation to accept two detections of same class
# prevents two anchors on the same glove both surviving NMS
MIN_SEPARATION_RATIO = 0.15   # 15% of crop width

PPE_BOX_LIMITS = {
    "helmet"    : (0.08, 0.05, 0.60, 0.28),
    "no_helmet" : (0.08, 0.05, 0.60, 0.28),
    "mask"      : (0.03, 0.02, 0.70, 0.40),  # relaxed: allow wider/taller face crops
    "no_mask"   : (0.03, 0.02, 0.70, 0.40),  # relaxed: same as mask
    "vest"      : (0.15, 0.15, 0.95, 0.80),
    "no_vest"   : (0.15, 0.15, 0.95, 0.80),
    "gloves"    : (0.04, 0.04, 0.35, 0.30),
    "boots"     : (0.05, 0.05, 0.55, 0.35),
}

MIN_DISPLAY_SIZE = {
    "helmet"    : (0.28, 0.12),
    "no_helmet" : (0.28, 0.12),
    "mask"      : (0.14, 0.08),
    "no_mask"   : (0.14, 0.08),
}

HEAD_CLASSES = {"helmet", "no_helmet", "mask", "no_mask"}

COLOR_PERSON  = (255, 255, 0)
COLOR_SAFE    = (0,   255, 0)
COLOR_UNSAFE  = (0,   0,   255)
COLOR_UNKNOWN = (128, 128, 128)

PPE_COLORS = {
    "helmet"    : (0,   255, 0  ),
    "no_helmet" : (0,   0,   255),
    "vest"      : (0,   255, 128),
    "no_vest"   : (0,   0,   200),
    "gloves"    : (255, 200, 0  ),
    "boots"     : (255, 165, 0  ),
    "mask"      : (128, 255, 0  ),
    "no_mask"   : (0,   0,   180),
}

# FIX 1: ASCII replacements for Unicode symbols that break OpenCV putText
STATUS_SYMBOLS = {
    "yes"     : "YES",
    "no"      : "NO",
    "unknown" : "?",
}
# ────────────────────────────────────────────────────────

person_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

ppe_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _num_classes_from_checkpoint(ckpt):
    bias = ckpt["model"].get("cls_heads.0.bias")
    if bias is None:
        return PPE_NUM_CLASSES
    # bias shape = num_anchors * num_classes; back-compat with old 3-anchor
    # checkpoints by trying the current anchor count first.
    for n_anc in (PPE_NUM_ANCHORS, 3):
        if bias.numel() % n_anc == 0:
            candidate = bias.numel() // n_anc
            if candidate in (PPE_NUM_CLASSES, PPE_LEGACY_CLASSES):
                return candidate
    return PPE_NUM_CLASSES


def _ppe_class_index_for_model(cls_name, model_num_classes):
    """
    Return the column index in the softmax output for cls_name.
    model_num_classes==9  → trained with background at index 0,
                            PPE_CLASS_TO_IDX gives 1-based indices (correct).
    model_num_classes==8  → legacy checkpoint without background class,
                            use 0-based CLASSES.index().
    """
    if model_num_classes == PPE_NUM_CLASSES:   # 9: background + 8 PPE classes
        return PPE_CLASS_TO_IDX[cls_name]      # 1-based, correct
    # Legacy: 8-class checkpoint (no background slot)
    return PPE_CLASSES.index(cls_name)         # 0-based


def load_models():
    person_model = MobileNetV2SSD(num_classes=2).to(DEVICE)
    ckpt = torch.load(PERSON_MODEL_PATH, map_location=DEVICE,
                      weights_only=False)
    person_model.load_state_dict(ckpt["model"])
    person_model.eval()
    print(f"Person detector loaded (epoch {ckpt['epoch']}, "
          f"val_loss {ckpt['val_loss']:.4f})")

    ckpt = torch.load(PPE_MODEL_PATH, map_location=DEVICE,
                      weights_only=False)
    ppe_num_classes = _num_classes_from_checkpoint(ckpt)
    ppe_model = PPEDetector(num_classes=ppe_num_classes).to(DEVICE)
    ppe_model.load_state_dict(ckpt["model"])
    ppe_model.eval()
    print(f"PPE detector loaded (epoch {ckpt['epoch']}, "
          f"val_loss {ckpt['val_loss']:.4f}, classes {ppe_num_classes})")

    return person_model, ppe_model


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


@torch.no_grad()
def detect_persons(model, anchors, frame_rgb):
    h, w   = frame_rgb.shape[:2]
    img    = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = person_transform(img).unsqueeze(0).to(DEVICE)

    cls_logits, box_preds = model(tensor)
    cls_logits = cls_logits[0]
    box_preds  = box_preds[0]

    scores = torch.softmax(cls_logits, dim=1)[:, 1]
    mask   = scores > PERSON_CONF
    if mask.sum() == 0:
        return []

    boxes  = decode_boxes(box_preds[mask], anchors.to(DEVICE)[mask])
    scores = scores[mask]
    keep   = tv_nms(boxes, scores, PERSON_NMS)[:MAX_PERSONS]

    results = []
    for k in keep:
        box   = boxes[k]
        score = scores[k].item()
        x1 = int(box[0].item() * w)
        y1 = int(box[1].item() * h)
        x2 = int(box[2].item() * w)
        y2 = int(box[3].item() * h)
        results.append((x1, y1, x2, y2, score))
    return results


def fast_detect_persons(model, anchors, frame_rgb, compute_w=1280):
    """
    Fast person detection for high-res frames.
    Downscales to a smaller resolution (e.g. 1280px wide) to run detection
    quickly, then scales the bounding boxes back to the original resolution.
    This replaces the extremely slow tiled detection.
    """
    h, w = frame_rgb.shape[:2]

    # If frame is small enough, just run normal detection
    if w <= compute_w:
        return detect_persons(model, anchors, frame_rgb)

    # Downscale for fast detection
    scale = compute_w / w
    compute_h = int(h * scale)
    small_img = cv2.resize(frame_rgb, (compute_w, compute_h))

    dets = detect_persons(model, anchors, small_img)

    results = []
    inv_scale = w / compute_w
    for (x1, y1, x2, y2, score) in dets:
        results.append((int(x1 * inv_scale), int(y1 * inv_scale),
                        int(x2 * inv_scale), int(y2 * inv_scale), score))
    return results


def _filter_spatially_separated(boxes_list, scores_list, max_dets):
    """
    FIX 2 helper: from a ranked list of boxes, greedily keep up to max_dets
    that are spatially separated by at least MIN_SEPARATION_RATIO.
    Prevents two anchors on the same object both surviving after NMS.
    """
    kept = []
    # cls_boxes are normalized 0..1 here, so keep the separation normalized too.
    min_sep = MIN_SEPARATION_RATIO
    for box, score in zip(boxes_list, scores_list):
        if len(kept) >= max_dets:
            break
        cx = (box[0] + box[2]) / 2
        too_close = any(abs(cx - (kb[0] + kb[2]) / 2) < min_sep
                        for kb, _ in kept)
        if not too_close:
            kept.append((box, score))
    return kept


def _is_reasonable_ppe_box(label, box):
    min_w, min_h, max_w, max_h = PPE_BOX_LIMITS.get(
        label, (0.01, 0.01, 1.0, 1.0))
    bw = max((box[2] - box[0]).item(), 0.0)
    bh = max((box[3] - box[1]).item(), 0.0)
    return min_w <= bw <= max_w and min_h <= bh <= max_h


def _candidate_quality(label, box, score, body_start):
    quality = float(score)

    if label in {"helmet", "no_helmet"}:
        cx = ((box[0] + box[2]) / 2).item()
        cy = ((box[1] + box[3]) / 2).item()
        bw = (box[2] - box[0]).item()
        bh = (box[3] - box[1]).item()
        target_y = max(0.08, body_start - 0.04)
        quality -= 0.6 * abs(cx - 0.5)
        quality -= 1.4 * abs(cy - target_y)
        quality += 0.12 * min(bw / 0.35, 1.0)
        quality += 0.06 * min(bh / 0.16, 1.0)

    elif label in HEAD_CLASSES:
        cx = ((box[0] + box[2]) / 2).item()
        cy = ((box[1] + box[3]) / 2).item()
        target_y = body_start if body_start > 0.05 else 0.16
        quality -= 0.6 * abs(cx - 0.5)
        quality -= 1.4 * abs(cy - target_y)

    return quality


def _rank_ppe_candidates(label, boxes, scores, keep, body_start):
    ranked = []
    for k in keep:
        idx = int(k.item())
        box = boxes[idx]
        if not _is_reasonable_ppe_box(label, box):
            continue
        quality = _candidate_quality(label, box, scores[idx].item(), body_start)
        ranked.append((quality, idx))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in ranked]


def _merge_helmet_candidates(boxes, scores, ranked_keep, body_start):
    if not ranked_keep:
        return []

    best_idx = ranked_keep[0]
    best_box = boxes[best_idx]
    best_quality = _candidate_quality(
        "helmet", best_box, scores[best_idx].item(), body_start)
    best_cx = ((best_box[0] + best_box[2]) / 2).item()
    best_cy = ((best_box[1] + best_box[3]) / 2).item()

    cluster = []
    cluster_scores = []
    for idx in ranked_keep:
        box = boxes[idx]
        score = scores[idx].item()
        quality = _candidate_quality("helmet", box, score, body_start)
        cx = ((box[0] + box[2]) / 2).item()
        cy = ((box[1] + box[3]) / 2).item()
        if (quality >= best_quality - 0.12 and
                abs(cx - best_cx) <= 0.24 and
                abs(cy - best_cy) <= 0.14):
            cluster.append(box)
            cluster_scores.append(score)

    if not cluster:
        return [(best_box, scores[best_idx].item())]

    cluster_t = torch.stack(cluster)
    merged = torch.stack([
        cluster_t[:, 0].min(),
        cluster_t[:, 1].min(),
        cluster_t[:, 2].max(),
        cluster_t[:, 3].max(),
    ])

    max_w = 0.55
    width = (merged[2] - merged[0]).item()
    if width > max_w:
        cx = ((merged[0] + merged[2]) / 2).item()
        merged[0] = max(0.0, cx - max_w / 2)
        merged[2] = min(1.0, cx + max_w / 2)

    return [(merged, max(cluster_scores))]


def _expand_to_min_display_size(label, bx1, by1, bx2, by2, crop_w, crop_h):
    min_w_ratio, min_h_ratio = MIN_DISPLAY_SIZE.get(label, (0.0, 0.0))
    min_w = int(min_w_ratio * crop_w)
    min_h = int(min_h_ratio * crop_h)

    bw = bx2 - bx1
    bh = by2 - by1
    if bw >= min_w and bh >= min_h:
        return bx1, by1, bx2, by2

    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    bw = max(bw, min_w, 1)
    bh = max(bh, min_h, 1)

    bx1 = max(0, int(round(cx - bw / 2)))
    by1 = max(0, int(round(cy - bh / 2)))
    bx2 = min(crop_w, int(round(cx + bw / 2)))
    by2 = min(crop_h, int(round(cy + bh / 2)))
    return bx1, by1, bx2, by2


def _tighten_helmet_box(label, bx1, by1, bx2, by2, body_start, crop_h):
    if label != "helmet":
        return bx1, by1, bx2, by2

    upper_head = max(0, int((body_start - 0.07) * crop_h))
    lower_head = min(crop_h, int((body_start + 0.035) * crop_h))
    by1 = max(by1, upper_head)
    by2 = min(by2, lower_head)
    return bx1, by1, bx2, by2


@torch.no_grad()
def detect_ppe_in_crop(model, anchors, crop_rgb, pad_top_ratio):
    """
    Run PPE detection on a person crop.
    pad_top_ratio: fraction of crop height that was added above the person box.
    """
    if crop_rgb.size == 0:
        return []

    h, w = crop_rgb.shape[:2]

    img    = cv2.resize(crop_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = ppe_transform(img).unsqueeze(0).to(DEVICE)
    cls_logits, box_preds = model(tensor)
    cls_logits = cls_logits[0]
    box_preds  = box_preds[0]
    scores_all = torch.softmax(cls_logits, dim=1)

    body_start = pad_top_ratio
    body_range = 1.0 - pad_top_ratio

    def zone(rel_start, rel_end):
        return (
        body_start + rel_start * body_range,
        body_start + rel_end   * body_range,
    )

    # Define zones as absolute fractions of the entire crop (0.0 to 1.0)
    # Mask sits on the lower face — needs a taller zone than helmet.
    head_zone  = (0.00, 0.40)   # helmet / no_helmet
    mask_zone  = (0.00, 0.50)   # mask / no_mask  (lower face can be up to 50%)
    torso_zone = (0.20, 0.65)
    hand_zone  = (0.35, 0.80)
    foot_zone  = (0.65, 1.00)

    VERTICAL_ZONES = {
        "helmet"    : head_zone,
        "no_helmet" : head_zone,
        "mask"      : mask_zone,
        "no_mask"   : mask_zone,
        "vest"      : torso_zone,
        "no_vest"   : torso_zone,
        "gloves"    : hand_zone,
        "boots"     : foot_zone,
    }

    all_boxes   = []
    all_scores  = []
    all_classes = []

    model_num_classes = getattr(model, "num_classes", PPE_NUM_CLASSES)
    for cls_name in PPE_CLASSES:
        cls_idx = _ppe_class_index_for_model(cls_name, model_num_classes)
        threshold = PPE_THRESHOLDS.get(cls_name, 0.5)
        max_dets  = MAX_DETS_PER_CLASS.get(cls_name, 1)   # FIX 2

        scores = scores_all[:, cls_idx]
        mask   = scores > threshold
        if mask.sum() == 0:
            continue

        cls_boxes  = decode_boxes(box_preds[mask], anchors.to(DEVICE)[mask])
        cls_scores = scores[mask]

        v_min, v_max = VERTICAL_ZONES.get(cls_name, (0.0, 1.0))
        cy     = (cls_boxes[:, 1] + cls_boxes[:, 3]) / 2
        v_mask = (cy >= v_min) & (cy <= v_max)
        if v_mask.sum() == 0:
            continue

        cls_boxes  = cls_boxes[v_mask]
        cls_scores = cls_scores[v_mask]

        # Merge helmet fragments before NMS; NMS can suppress useful partial
        # helmet boxes that should form one final hard-hat box.
        if cls_name == "helmet":
            keep = torch.arange(cls_boxes.shape[0], device=cls_boxes.device)
        else:
            keep = tv_nms(cls_boxes, cls_scores, PPE_NMS)
        ranked_keep = _rank_ppe_candidates(
            cls_name, cls_boxes, cls_scores, keep, body_start)
        if not ranked_keep:
            continue

        if cls_name == "helmet":
            kept_pairs = _merge_helmet_candidates(
                cls_boxes, cls_scores, ranked_keep, body_start)
        else:
            kept_pairs = _filter_spatially_separated(
                [cls_boxes[k] for k in ranked_keep],
                [cls_scores[k].item() for k in ranked_keep],
                max_dets
            )

        for box, score in kept_pairs:
            all_boxes.append(box)
            all_scores.append(score)
            all_classes.append(cls_name)

    if not all_boxes:
        return []

    results = []
    for i in range(len(all_boxes)):
        label = all_classes[i]
        score = float(all_scores[i]) 
        box   = all_boxes[i]
        
        bx1 = int(box[0].item() * w)
        by1 = int(box[1].item() * h)
        bx2 = int(box[2].item() * w)
        by2 = int(box[3].item() * h)
        bx1, by1, bx2, by2 = _expand_to_min_display_size(
            label, bx1, by1, bx2, by2, w, h)
        bx1, by1, bx2, by2 = _tighten_helmet_box(
            label, bx1, by1, bx2, by2, body_start, h)

        expand = {"helmet": (0.00, 0.00), "vest": (0.05, 0.05),
                  "gloves": (0.10, 0.10), "boots": (0.08, 0.08)}
        exp_x, exp_y = expand.get(label, (0.0, 0.0))
        bw  = max(bx2 - bx1, 1)
        bh  = max(by2 - by1, 1)
        bx1 = max(0, int(bx1 - exp_x * bw))
        by1 = max(0, int(by1 - exp_y * bh))
        bx2 = min(w, int(bx2 + exp_x * bw))
        by2 = min(h, int(by2 + exp_y * bh))

        results.append((label, bx1, by1, bx2, by2, score))
        
    return results


def _visibility_skip_classes(px1, py1, px2, py2, cx2, cy2, frame_w, frame_h):
    """
    Decide which PPE classes to skip based on which body parts are out of frame.

    boots  skipped if:
      - the padded crop hits the frame bottom (cy2 >= frame_h - 10), meaning
        the feet region was clipped; OR
      - the person box height is < 50% of the frame height (upper-body-only).
    gloves skipped if both left AND right sides of the person box are within
      8% of the frame edges (arms clipped on both sides).
    Returns a set of class names to treat as unknown.
    """
    skip = set()
    person_w = px2 - px1
    person_h = py2 - py1
    # Bottom truncation: person bounding box hits frame bottom (feet cut off)
    bottom_clipped  = py2 >= frame_h - 5
    upper_body_only = person_h < frame_h * 0.50
    
    if bottom_clipped or upper_body_only:
        skip.add("boots")
    # Side truncation on both sides → can't see hands
    left_clipped  = px1 <= frame_w * 0.08
    right_clipped = px2 >= frame_w * 0.92
    if left_clipped and right_clipped:
        skip.add("gloves")
    return skip


def compliance_for_person(ppe_detections, skip=None):
    """
    skip: set of PPE item names (e.g. {"boots", "gloves"}) whose status
          should be forced to unknown because they are out of frame.
    """
    if skip is None:
        skip = set()
    detected = set(d[0] for d in ppe_detections)
    status   = {}
    pairs = [
        ("helmet",  "no_helmet", "Helmet"),
        ("vest",    "no_vest",   "Vest"),
        ("gloves",  None,        "Gloves"),
        ("boots",   None,        "Boots"),
        ("mask",    "no_mask",   "Mask"),
    ]
    for pos, neg, name in pairs:
        if name.lower() in {s.lower() for s in skip}:
            status[name] = (STATUS_SYMBOLS["unknown"], COLOR_UNKNOWN)
        elif pos in detected and (neg is None or neg not in detected):
            status[name] = (STATUS_SYMBOLS["yes"], COLOR_SAFE)
        elif neg and neg in detected:
            status[name] = (STATUS_SYMBOLS["no"], COLOR_UNSAFE)
        else:
            status[name] = (STATUS_SYMBOLS["unknown"], COLOR_UNKNOWN)
    return status


def is_compliant(status):
    return (status.get("Helmet", ("?",))[0] == STATUS_SYMBOLS["yes"] and
            status.get("Vest",   ("?",))[0] == STATUS_SYMBOLS["yes"])


def draw_person(frame, px1, py1, px2, py2,
                ppe_detections, status, person_idx, sf=1.0):
    compliant = is_compliant(status)
    p_color   = COLOR_SAFE if compliant else COLOR_UNSAFE
    thick     = max(1, int(2 * sf))

    # person box
    cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_PERSON, thick)

    # PPE boxes
    for (label, bx1, by1, bx2, by2, score) in ppe_detections:
        fx1 = px1 + bx1
        fy1 = py1 + by1
        fx2 = px1 + bx2
        fy2 = py1 + by2
        color = PPE_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, thick)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (fx1, max(fy1 - int(4 * sf), 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sf, color, thick)

    # compliance panel — right of person box, clamped to frame
    panel_x = min(px2 + int(5 * sf), frame.shape[1] - int(160 * sf))
    panel_y = max(py1 + int(20 * sf), int(20 * sf))
    for name, (state, color) in status.items():
        # FIX 1: label is now e.g. "Helmet:YES" — all ASCII, renders correctly
        cv2.putText(frame, f"{name}:{state}",
                    (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sf, color, thick)
        panel_y += int(22 * sf)

    # badge
    badge = "SAFE" if compliant else "UNSAFE"
    badge_h = int(24 * sf)
    badge_w = int(100 * sf)
    cv2.rectangle(frame, (px1, py1 - badge_h), (px1 + badge_w, py1), p_color, -1)
    cv2.putText(frame, f"P{person_idx} {badge}",
                (px1 + int(3 * sf), py1 - int(7 * sf)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sf, (0, 0, 0), thick)

    return frame


class TemporalVoter:
    def __init__(self, window=5):
        self.window  = window
        self.history = collections.defaultdict(
            lambda: collections.deque(maxlen=window))

    def update(self, person_id, ppe_labels):
        for label in PPE_CLASSES:
            self.history[(person_id, label)].append(label in ppe_labels)

    def get_stable_labels(self, person_id):
        stable = set()
        for label in PPE_CLASSES:
            hist = self.history[(person_id, label)]
            if len(hist) > 0 and sum(hist) > len(hist) // 2:
                stable.add(label)
        return stable


def _scale_factor(frame):
    """Return a scale multiplier for text/line sizes based on frame width."""
    return max(1.0, frame.shape[1] / 1280.0)


def process_frame(frame, person_model, ppe_model,
                   person_anchors, ppe_anchors, voter, tracker):
    """
    Process a single frame with SORT-tracked person IDs.

    Args:
        tracker: sort.Sort instance for persistent person tracking
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w      = frame.shape[:2]
    sf        = _scale_factor(frame)  # scale text/lines for high-res

    # Use fast downscaled detection instead of slow tiling
    persons = fast_detect_persons(person_model, person_anchors, frame_rgb)

    # Filter tiny false-positive detections
    # Relaxed for CCTV: distant workers appear small
    frame_area = h * w
    persons = [
        (x1, y1, x2, y2, s) for x1, y1, x2, y2, s in persons
        if (x2 - x1) >= 25                                      # minimum width
        and (y2 - y1) >= 40                                     # minimum height
        and (x2 - x1) * (y2 - y1) >= 1600                      # min 40×40 px area
        and (y2 - y1) > (x2 - x1) * 0.6                        # taller than wide
        and (x2 - x1) * (y2 - y1) < frame_area * 0.40          # not a huge background blob
    ]

    # Feed detections into SORT tracker
    if persons:
        dets_np = np.array([[x1, y1, x2, y2, s]
                            for x1, y1, x2, y2, s in persons],
                           dtype=np.float64)
    else:
        dets_np = np.empty((0, 5), dtype=np.float64)

    tracked = tracker.update(dets_np)  # returns [[x1,y1,x2,y2,track_id], ...]

    num_tracked = 0
    for trk in tracked:
        px1, py1, px2, py2, track_id = trk
        # Clamp tracked boxes strictly to the image frame
        px1 = max(0, min(w - 1, int(px1)))
        py1 = max(0, min(h - 1, int(py1)))
        px2 = max(0, min(w - 1, int(px2)))
        py2 = max(0, min(h - 1, int(py2)))
        track_id = int(track_id)
        num_tracked += 1

        person_w = px2 - px1
        person_h = py2 - py1
        
        # Skip invalid or collapsed boxes from tracker overshoots
        if person_w < 10 or person_h < 10:
            continue

        # Proportional padding to capture extended limbs and helmet
        pad_sides  = int(person_w * 0.25)
        pad_bottom = int(person_h * 0.15)
        pad_top    = int(person_h * 0.20)

        cx1 = max(0, px1 - pad_sides)
        cy1 = max(0, py1 - pad_top)
        cx2 = min(w, px2 + pad_sides)
        cy2 = min(h, py2 + pad_bottom)

        crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
        if crop_rgb.size == 0:
            continue

        actual_pad_top  = py1 - cy1
        crop_h          = cy2 - cy1
        pad_top_ratio   = actual_pad_top / crop_h if crop_h > 0 else 0.0

        ppe_dets = detect_ppe_in_crop(
            ppe_model, ppe_anchors, crop_rgb, pad_top_ratio)

        # Determine which PPE items are out of frame
        skip_cls = _visibility_skip_classes(px1, py1, px2, py2, cx2, cy2, w, h)

        # Filter skipped classes from detections so they don't appear as boxes
        valid_ppe_dets = []
        for d in ppe_dets:
            label, bx1, by1, bx2, by2, score = d
            fx1, fy1, fx2, fy2 = cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2
            
            if label.replace("no_", "") in skip_cls:
                continue
                
            # Spatial filtering: eliminate gross false positives
            fx_center = (fx1 + fx2) / 2
            
            # Helmet and Mask must be in the top 40% and horizontally centered
            if "helmet" in label:
                if fy1 > py1 + person_h * 0.4 or fx_center < px1 or fx_center > px2:
                    continue
            if "mask" in label:
                if fy1 > py1 + person_h * 0.35 or fx_center < px1 or fx_center > px2:
                    continue
            # Boots must be in the bottom 40%
            if "boots" in label and fy2 < py1 + person_h * 0.6:
                continue
                
            valid_ppe_dets.append(d)
        ppe_dets = valid_ppe_dets

        # Use persistent track_id for temporal voting
        detected_labels = set(d[0] for d in ppe_dets)
        voter.update(track_id, detected_labels)
        stable_labels = voter.get_stable_labels(track_id)

        display_dets = ([d for d in ppe_dets if d[0] in stable_labels]
                        if stable_labels else ppe_dets)

        status = compliance_for_person(display_dets, skip=skip_cls)

        frame = draw_person(frame, cx1, cy1, cx2, cy2,
                            display_dets, status, track_id, sf)

    thick = max(1, int(2 * sf))
    cv2.putText(frame, f"Persons: {num_tracked}",
                (10, h - int(10 * sf)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * sf, (255, 255, 255), thick)
    return frame


def run_on_image(image_path, output_path="output/pipeline_test.jpg"):
    person_model, ppe_model = load_models()
    person_anchors = PersonAnchorGen()()
    ppe_anchors    = PPEAnchorGen()()
    voter          = TemporalVoter(TEMPORAL_WINDOW)
    tracker        = Sort(max_age=SORT_MAX_AGE, min_hits=1,
                          iou_threshold=SORT_IOU_THRESH)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot load: {image_path}")
        return

    result = process_frame(frame, person_model, ppe_model,
                           person_anchors, ppe_anchors, voter, tracker)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")


# Maximum width for the preview window (the saved video stays full-res)
DISPLAY_MAX_WIDTH = 1280


def _display_frame(window_name, frame, max_w=DISPLAY_MAX_WIDTH):
    """Show frame in a resizable window, scaled to fit the screen."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        display = cv2.resize(frame, (max_w, int(h * scale)))
    else:
        display = frame
    cv2.imshow(window_name, display)


def run_on_video(video_path, output_path="output/pipeline_output.avi",
                  display=True, skip_frames=0):
    """
    Run PPE detection on a video file.

    Args:
        video_path:   path to the input video file
        output_path:  path for the annotated output video
        display:      if True, show a live preview window (press 'q' to quit)
        skip_frames:  process every N-th frame (0 = process all). Skipped
                      frames still appear in the output but without new
                      detections (the last annotated frame is repeated).
    """
    import time

    person_model, ppe_model = load_models()
    person_anchors = PersonAnchorGen()()
    ppe_anchors    = PPEAnchorGen()()
    voter          = TemporalVoter(TEMPORAL_WINDOW)
    tracker        = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                          iou_threshold=SORT_IOU_THRESH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"XVID"),
                          fps, (fw, fh))

    frame_count = 0
    last_result = None
    print(f"Processing video: {video_path}")
    print(f"  Resolution: {fw}x{fh} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"  Output: {output_path}")
    if display:
        cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PPE Detection", min(fw, DISPLAY_MAX_WIDTH),
                         min(fh, int(fh * DISPLAY_MAX_WIDTH / fw)))
        print("  Press 'q' in the preview window to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Optionally skip frames for speed
        if skip_frames > 0 and (frame_count % (skip_frames + 1)) != 1:
            if last_result is not None:
                out.write(last_result)
            continue

        t0 = time.perf_counter()
        result = process_frame(frame, person_model, ppe_model,
                               person_anchors, ppe_anchors, voter, tracker)
        dt = time.perf_counter() - t0
        current_fps = 1.0 / dt if dt > 0 else 0.0

        # Draw FPS and progress
        cv2.putText(result, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        if total_frames > 0:
            pct = frame_count / total_frames * 100
            cv2.putText(result, f"Frame {frame_count}/{total_frames} ({pct:.0f}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

        last_result = result
        out.write(result)

        if display:
            _display_frame("PPE Detection", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  Stopped by user.")
                break

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames "
                  f"({current_fps:.1f} FPS)")

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Done - {frame_count} frames processed.")
    print(f"Output saved to: {output_path}")


# ── Violation Logger ────────────────────────────────────────
class ViolationLogger:
    """
    Logs PPE violations to a CSV file and saves snapshot images.
    Designed for CCTV / continuous operation.
    """
    def __init__(self, log_dir="output/violations"):
        self.log_dir = log_dir
        self.snap_dir = os.path.join(log_dir, "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "violations.csv")
        self._write_header()
        # Track last alert time per person to avoid log spam (1 alert / 10 s)
        self._last_alert: dict = {}

    def _write_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "source", "track_id",
                            "missing_ppe", "snapshot"])

    def log(self, frame, source, track_id, status):
        missing = [name for name, (state, _) in status.items()
                   if state == STATUS_SYMBOLS["no"]]
        if not missing:
            return
        now = time.time()
        if now - self._last_alert.get(track_id, 0) < 10:
            return
        self._last_alert[track_id] = now
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"{ts}_id{track_id}.jpg"
        snap_path = os.path.join(self.snap_dir, snap_name)
        cv2.imwrite(snap_path, frame)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().isoformat(),
                source, track_id,
                "|".join(missing), snap_name
            ])
        print(f"  [ALERT] ID {track_id} missing: {missing}")
# ────────────────────────────────────────────────────────────


def _draw_timestamp(frame):
    """Overlay current date-time on the frame (CCTV style)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    h, w = frame.shape[:2]
    cv2.putText(frame, ts, (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def _open_capture(source, max_retries=10):
    """Open a VideoCapture with exponential back-off retries."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"  Reconnect attempt {attempt}/{max_retries} "
              f"(waiting {delay:.0f}s)...")
        time.sleep(delay)
        delay = min(delay * 2, 30)
    return None


def process_frame_logged(frame, person_model, ppe_model,
                         person_anchors, ppe_anchors,
                         voter, tracker, logger, source):
    """
    process_frame + violation logging. Used by camera/video pipelines.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w      = frame.shape[:2]
    sf        = _scale_factor(frame)

    persons = fast_detect_persons(person_model, person_anchors, frame_rgb)
    frame_area = h * w
    persons = [
        (x1, y1, x2, y2, s) for x1, y1, x2, y2, s in persons
        if (x2 - x1) >= 25
        and (y2 - y1) >= 40
        and (x2 - x1) * (y2 - y1) >= 1600
        and (y2 - y1) > (x2 - x1) * 0.6
        and (x2 - x1) * (y2 - y1) < frame_area * 0.40
    ]

    dets_np = (np.array([[x1, y1, x2, y2, s] for x1, y1, x2, y2, s in persons],
                         dtype=np.float64)
               if persons else np.empty((0, 5), dtype=np.float64))
    tracked = tracker.update(dets_np)

    num_tracked = 0
    for trk in tracked:
        px1, py1, px2, py2, track_id = trk
        px1 = max(0, min(w - 1, int(px1)))
        py1 = max(0, min(h - 1, int(py1)))
        px2 = max(0, min(w - 1, int(px2)))
        py2 = max(0, min(h - 1, int(py2)))
        track_id = int(track_id)
        num_tracked += 1

        person_w = px2 - px1
        person_h = py2 - py1
        if person_w < 10 or person_h < 10:
            continue

        pad_sides  = int(person_w * 0.25)
        pad_bottom = int(person_h * 0.15)
        pad_top    = int(person_h * 0.20)
        cx1 = max(0, px1 - pad_sides)
        cy1 = max(0, py1 - pad_top)
        cx2 = min(w, px2 + pad_sides)
        cy2 = min(h, py2 + pad_bottom)

        crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
        if crop_rgb.size == 0:
            continue

        actual_pad_top = py1 - cy1
        crop_h         = cy2 - cy1
        pad_top_ratio  = actual_pad_top / crop_h if crop_h > 0 else 0.0

        ppe_dets = detect_ppe_in_crop(
            ppe_model, ppe_anchors, crop_rgb, pad_top_ratio)

        # Determine which PPE items are out of frame
        skip_cls = _visibility_skip_classes(px1, py1, px2, py2, cx2, cy2, w, h)

        # Filter skipped classes from detections
        valid_ppe_dets = []
        for d in ppe_dets:
            label, bx1, by1, bx2, by2, score = d
            fx1, fy1, fx2, fy2 = cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2
            
            if label.replace("no_", "") in skip_cls:
                continue
                
            # Spatial filtering: eliminate gross false positives
            fx_center = (fx1 + fx2) / 2
            
            # Helmet and Mask must be in the top 40% and horizontally centered
            if "helmet" in label:
                if fy1 > py1 + person_h * 0.4 or fx_center < px1 or fx_center > px2:
                    continue
            if "mask" in label:
                if fy1 > py1 + person_h * 0.35 or fx_center < px1 or fx_center > px2:
                    continue
            # Boots must be in the bottom 40%
            if "boots" in label and fy2 < py1 + person_h * 0.6:
                continue
                
            valid_ppe_dets.append(d)
        ppe_dets = valid_ppe_dets

        detected_labels = set(d[0] for d in ppe_dets)
        voter.update(track_id, detected_labels)
        stable_labels = voter.get_stable_labels(track_id)

        display_dets = ([d for d in ppe_dets if d[0] in stable_labels]
                        if stable_labels else ppe_dets)
        status = compliance_for_person(display_dets, skip=skip_cls)

        # Log violations if logger provided
        if logger is not None:
            logger.log(frame, source, track_id, status)

        frame = draw_person(frame, cx1, cy1, cx2, cy2,
                            display_dets, status, track_id, sf)

    thick = max(1, int(2 * sf))
    cv2.putText(frame, f"Persons: {num_tracked}",
                (10, h - int(10 * sf)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * sf, (255, 255, 255), thick)
    return frame


def run_on_camera(source=0, output_path="output/camera_output.avi",
                  display=True, skip_frames=0,
                  log_violations=True, log_dir="output/violations",
                  max_reconnect=10):
    """
    Run PPE detection on a live camera / CCTV / RTSP stream.

    Args:
        source:          camera index (0) or RTSP/HTTP URL
        output_path:     path for the recorded annotated video
        display:         show live preview window (press 'q' to quit)
        skip_frames:     process every N-th frame (0 = all)
        log_violations:  save violation CSV + snapshots
        log_dir:         directory for violation logs
        max_reconnect:   reconnect attempts before giving up
    """
    person_model, ppe_model = load_models()
    person_anchors = PersonAnchorGen()()
    ppe_anchors    = PPEAnchorGen()()
    voter          = TemporalVoter(TEMPORAL_WINDOW)
    tracker        = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                          iou_threshold=SORT_IOU_THRESH)
    logger = ViolationLogger(log_dir) if log_violations else None

    cap = _open_capture(source, max_reconnect)
    if cap is None:
        print(f"Cannot open camera source after retries: {source}")
        return

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"XVID"),
                          fps, (fw, fh))

    frame_count   = 0
    consec_fails  = 0
    MAX_CONSEC    = 30   # reconnect after 30 consecutive read failures
    last_result   = None

    print(f"Live feed from: {source}")
    print(f"  Resolution: {fw}x{fh}  |  Recording: {output_path}")
    if log_violations:
        print(f"  Violations logged to: {log_dir}")
    print("  Press 'q' to stop.")

    if display:
        cv2.namedWindow("PPE Detection - Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PPE Detection - Live", min(fw, DISPLAY_MAX_WIDTH),
                         min(fh, int(fh * DISPLAY_MAX_WIDTH / max(fw, 1))))

    while True:
        ret, frame = cap.read()
        if not ret:
            consec_fails += 1
            if consec_fails >= MAX_CONSEC:
                print("  Feed lost — reconnecting...")
                cap.release()
                cap = _open_capture(source, max_reconnect)
                if cap is None:
                    print("  Could not reconnect. Exiting.")
                    break
                consec_fails = 0
            time.sleep(0.05)
            continue
        consec_fails = 0
        frame_count += 1

        if skip_frames > 0 and (frame_count % (skip_frames + 1)) != 1:
            if last_result is not None:
                out.write(last_result)
            continue

        t0 = time.perf_counter()
        result = process_frame_logged(frame, person_model, ppe_model,
                                      person_anchors, ppe_anchors,
                                      voter, tracker, logger, str(source))
        dt = time.perf_counter() - t0
        current_fps = 1.0 / dt if dt > 0 else 0.0

        cv2.putText(result, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(result, "LIVE", (fw - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _draw_timestamp(result)

        last_result = result
        out.write(result)

        if display:
            _display_frame("PPE Detection - Live", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  Stopped by user.")
                break

        if frame_count % 100 == 0:
            print(f"  {frame_count} frames ({current_fps:.1f} FPS)")

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Done — {frame_count} frames.  Recording: {output_path}")


def run_on_multistream(sources, base_output_dir="output/multistream",
                       log_violations=True, skip_frames=1):
    """
    Run PPE detection on multiple CCTV streams concurrently.

    Args:
        sources:         list of camera indices or RTSP URLs
        base_output_dir: each stream writes to base_output_dir/stream_N.avi
        log_violations:  enable violation logging per stream
        skip_frames:     skip N frames between processed frames
    """
    os.makedirs(base_output_dir, exist_ok=True)
    threads = []
    for i, src in enumerate(sources):
        out_path = os.path.join(base_output_dir, f"stream_{i}.avi")
        log_dir  = os.path.join(base_output_dir, f"violations_stream_{i}")
        t = threading.Thread(
            target=run_on_camera,
            kwargs=dict(source=src, output_path=out_path,
                        display=False, skip_frames=skip_frames,
                        log_violations=log_violations, log_dir=log_dir),
            daemon=True, name=f"stream-{i}"
        )
        t.start()
        threads.append(t)
        print(f"Started stream {i}: {src}  ->  {out_path}")
    for t in threads:
        t.join()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
              ".m4v", ".mpg", ".mpeg", ".ts"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PPE Compliance Detection - image, video, or live camera")
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to image/video file, camera index (0,1,...), "
             "or RTSP/HTTP URL. Defaults to image mode on "
             "'output/test_known.jpg'.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: output/pipeline_test.jpg for images, "
             "output/pipeline_output.avi for videos)")
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live preview window (useful for headless servers)")
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Process every N-th frame for faster video processing "
             "(0 = process all)")
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable violation logging (CSV + snapshots)")
    parser.add_argument(
        "--log-dir", default="output/violations",
        help="Directory to save violation logs (default: output/violations)")
    parser.add_argument(
        "--multi", nargs="+", metavar="SOURCE",
        help="Run multiple streams concurrently: provide space-separated "
             "camera indices or RTSP URLs")
    args = parser.parse_args()

    # Multi-stream mode
    if args.multi:
        sources = [int(s) if s.isdigit() else s for s in args.multi]
        run_on_multistream(sources,
                           log_violations=not args.no_log,
                           skip_frames=args.skip_frames)
        sys.exit(0)

    source = args.input

    # No argument -> default image test
    if source is None:
        run_on_image("output/test_known.jpg")

    # Numeric string -> camera index
    elif source.isdigit():
        out = args.output or "output/camera_output.avi"
        run_on_camera(source=int(source), output_path=out,
                      display=not args.no_display,
                      skip_frames=args.skip_frames,
                      log_violations=not args.no_log,
                      log_dir=args.log_dir)

    # RTSP / HTTP stream URL
    elif source.startswith(("rtsp://", "http://", "https://")):
        out = args.output or "output/stream_output.avi"
        run_on_camera(source=source, output_path=out,
                      display=not args.no_display,
                      skip_frames=args.skip_frames,
                      log_violations=not args.no_log,
                      log_dir=args.log_dir)

    # File on disk -> detect type by extension
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTS:
            out = args.output or "output/pipeline_test.jpg"
            run_on_image(source, output_path=out)
        elif ext in VIDEO_EXTS:
            out = args.output or "output/pipeline_output.avi"
            run_on_video(source, output_path=out,
                         display=not args.no_display,
                         skip_frames=args.skip_frames)
        else:
            print(f"Unknown file type: {ext}")
            print(f"Supported images: {IMAGE_EXTS}")
            print(f"Supported videos: {VIDEO_EXTS}")

    else:
        print(f"Input not found: {source}")
        print("Provide an image path, video path, camera index (0), "
              "RTSP URL, or use --multi for multiple streams.")
