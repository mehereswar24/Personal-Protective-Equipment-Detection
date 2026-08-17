"""
Baseline the EXISTING checkpoints on the new leakage-free test split.

This is the number the project has never had. Everything after this is measured
against it.

Important caveat, stated up front so the output is not misread: the existing
PPE checkpoint was trained on the OLD 8-class taxonomy
(helmet/no_helmet/vest/no_vest/gloves/boots/mask/no_mask) while the new split
uses the positives-only taxonomy (person/head/helmet/vest/gloves/boots). We map
what corresponds:

    old helmet    -> helmet
    old no_helmet -> head        (that class WAS 87% dataset4 `head` boxes)
    old vest      -> vest
    old gloves    -> gloves
    old boots     -> boots
    old no_vest / mask / no_mask -> dropped (no counterpart)

`person` is not predicted by the PPE model at all, so it is reported as absent
rather than as a zero — the model was never asked to find it.

Usage:
    python tools/eval_baseline.py                    # test split
    python tools/eval_baseline.py --split val --limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.metrics import (  # noqa: E402
    Detection, GroundTruth, best_threshold, evaluate, format_report,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "ppe" / "splits_v2.json"
PPE_CKPT = ROOT / "models" / "ppe_detector_best.pth"
INPUT_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# old-model class index (1-based, background=0) -> new taxonomy name
OLD_TO_NEW = {"helmet": "helmet", "no_helmet": "head", "vest": "vest",
              "gloves": "gloves", "boots": "boots"}

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_ppe_model():
    """Load the legacy MobileNetV2-SSD PPE detector + its anchors."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ppe_step2", ROOT / "scripts" / "ppe_classifiers" / "step2_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = torch.load(PPE_CKPT, map_location="cpu", weights_only=False)
    # This project's checkpoints store weights under "model"; be tolerant of
    # the other common conventions too.
    state = None
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None:
        state = ckpt
    print(f"  checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")
    print(f"  checkpoint classes: {ckpt.get('classes')}")

    # Infer head width from the checkpoint rather than trusting the module
    # constant (the audit found a legacy 3-anchor variant on disk).
    cls_bias = state.get("cls_heads.0.bias")
    n_anchors, n_cls = mod.NUM_ANCHORS, mod.NUM_CLASSES
    if cls_bias is not None and cls_bias.numel() % n_anchors != 0:
        raise SystemExit(f"Head width {cls_bias.numel()} not divisible by "
                         f"NUM_ANCHORS={n_anchors}; this is the legacy 3-anchor "
                         f"checkpoint the audit flagged as unloadable.")
    if cls_bias is not None:
        n_cls = cls_bias.numel() // n_anchors

    model = mod.PPEDetector(num_classes=n_cls)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    anchors = mod.AnchorGenerator()().to(DEVICE)     # hoisted once, not per class
    return model, anchors, mod


def decode(box_preds: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """SSD variance decode (0.1, 0.2) → x1y1x2y2 in [0,1]."""
    cx = anchors[:, 0] + box_preds[:, 0] * 0.1 * anchors[:, 2]
    cy = anchors[:, 1] + box_preds[:, 1] * 0.1 * anchors[:, 3]
    w = anchors[:, 2] * torch.exp(box_preds[:, 2] * 0.2)
    h = anchors[:, 3] * torch.exp(box_preds[:, 3] * 0.2)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1).clamp(0, 1)


@torch.no_grad()
def predict(model, anchors, mod, img_bgr) -> list[tuple[str, float, tuple]]:
    import cv2
    h0, w0 = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = TRANSFORM(resized).unsqueeze(0).to(DEVICE)

    cls_logits, box_preds = model(tensor)
    cls_logits, box_preds = cls_logits[0], box_preds[0]
    scores_all = torch.softmax(cls_logits, dim=1)

    out: list[tuple[str, float, tuple]] = []
    for old_name, new_name in OLD_TO_NEW.items():
        idx = mod.CLASS_TO_IDX.get(old_name)
        if idx is None:
            continue
        scores = scores_all[:, idx]
        # Deliberately NO score threshold: AP integrates the whole PR curve, so
        # thresholding here would silently truncate recall and flatter the model.
        keep = scores > 0.01
        if keep.sum() == 0:
            continue
        boxes = decode(box_preds[keep], anchors[keep])
        sc = scores[keep]
        k = nms(boxes, sc, 0.45)[:100]
        for b, s in zip(boxes[k].cpu().numpy(), sc[k].cpu().numpy()):
            out.append((new_name, float(s),
                        (float(b[0] * w0), float(b[1] * h0),
                         float(b[2] * w0), float(b[3] * h0))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--out", default="reports/baseline.json")
    args = ap.parse_args()

    if not SPLITS.exists():
        print(f"Missing {SPLITS}. Run: python tools/build_splits.py")
        return 1
    data = json.loads(SPLITS.read_text(encoding="utf-8"))
    records = data["splits"][args.split]
    if args.limit:
        records = records[:args.limit]
    classes = data["classes"]

    print(f"Baseline: legacy PPE checkpoint on '{args.split}' "
          f"({len(records)} images, split hash {data.get('content_hash')})")
    print(f"Device: {DEVICE}")

    import cv2
    model, anchors, mod = load_ppe_model()

    dets: list[Detection] = []
    gts: list[GroundTruth] = []
    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        if i % 250 == 0 and i:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(records)}  ({i/el:.1f} img/s)")
        img_path = ROOT / rec["img"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        image_id = rec["img"]
        area = float(rec["w"] * rec["h"])
        for label, x1, y1, x2, y2 in rec["boxes"]:
            gts.append(GroundTruth(image_id, label, (x1, y1, x2, y2),
                                   dataset=rec["dataset"], img_area=area))
        for label, score, box in predict(model, anchors, mod, img):
            dets.append(Detection(image_id, label, score, box))

    dt = time.perf_counter() - t0
    print(f"  done in {dt:.1f}s ({len(records)/max(dt,1e-9):.1f} img/s)")

    res = evaluate(dets, gts, classes)
    print(format_report(res, f"BASELINE — legacy PPE checkpoint — {args.split} split"))

    print("\n  suggested per-class score thresholds (max F1 on this split):")
    for c in classes:
        m = res["per_class"][c]
        if m["n_gt"] == 0 or m["pr"] is None:
            continue
        thr, f1 = best_threshold(m["pr"])
        print(f"    {c:10s} thr={thr:.3f}  F1={f1:.3f}")

    print("\n  NOTE: `person` is absent because the legacy PPE model never had "
          "that class.\n  It is reported as no-GT-match rather than as a zero.")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "split": args.split, "n_images": len(records),
        "split_hash": data.get("content_hash"),
        "map50": res["map50"], "map50_95": res["map50_95"],
        "per_class": {c: {k: v for k, v in m.items() if k != "pr"}
                      for c, m in res["per_class"].items()},
        "per_size": res["per_size"],
        "per_class_dataset": res["per_class_dataset"],
    }
    out_path.write_text(json.dumps(serialisable, indent=1), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
