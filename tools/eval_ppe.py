"""Score a trained FCOS/RetinaNet checkpoint on a held-out split.

`tools/eval_baseline.py` scores the OLD checkpoints, which are a different
architecture entirely (custom anchors, `mod.CLASS_TO_IDX`), so it cannot read
these. This is the equivalent for the torchvision detectors trained by
`tools/train_ppe.py`.

Differences from the in-training eval, which are the reason this exists:

  * the FULL split, not the `--eval-batches` subsample the training loop uses
    to keep per-epoch cost down. Training-time mAP is a noisy estimate over
    ~240 images; this is the real number.
  * the `test` split, which training never touches, rather than `val`, which
    selected the checkpoint. Reporting a selection split as if it were held out
    is how projects fool themselves.

The +1/-1 class convention appears here too: torchvision emits 1-based labels
(0 = background), the rest of the project is 0-based. See MaskedLossWrapper.

Usage:
    python tools/eval_ppe.py                                  # test split
    python tools/eval_ppe.py --split val --limit 200          # quick check
    python tools/eval_ppe.py --ckpt models/ppe_fcos_epoch2_map0.4424.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.dataset import PPEDataset, collate  # noqa: E402
from ppe.metrics import Detection, GroundTruth, evaluate, format_report  # noqa: E402
from ppe.models import build_ppe_model  # noqa: E402
from ppe.taxonomy import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _stream in (sys.stdout, sys.stderr):        # see train_ppe.py - cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@torch.no_grad()
def collect(model, loader, limit_batches: int = 0):
    """Run the model over a split and return (detections, ground truths)."""
    model.eval()
    dets: list[Detection] = []
    gts: list[GroundTruth] = []
    n_images = 0
    t0 = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            outputs = model(images)

        for out, tgt, img in zip(outputs, targets, images):
            n_images += 1
            area = float(img.shape[-1]) * float(img.shape[-2])
            image_id = tgt["image_id"]
            for box, label in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()):
                gts.append(GroundTruth(image_id, CLASSES[int(label)], tuple(box),
                                       dataset=tgt["dataset"], img_area=area))
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, lab in zip(boxes, scores, labels):
                idx = int(lab) - 1                 # torchvision 1-based -> ours
                if 0 <= idx < len(CLASSES):
                    dets.append(Detection(image_id, CLASSES[idx], float(s),
                                          tuple(b)))

        if (i + 1) % 50 == 0:
            rate = n_images / (time.perf_counter() - t0)
            print(f"  {n_images} images  ({rate:.1f} img/s)", flush=True)

    return dets, gts, n_images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/ppe_fcos.pth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N batches (0 = whole split)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ckpt_path = ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"no such checkpoint: {ckpt_path}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "fcos")
    size = int(ckpt.get("size", 640))
    trained_epoch = ckpt.get("epoch", "?")
    val_map50 = ckpt.get("val_map50")

    print(f"checkpoint : {ckpt_path.name}")
    print(f"  arch {arch}  size {size}  epoch {trained_epoch}"
          + (f"  val mAP@0.5 {val_map50:.4f}" if isinstance(val_map50, float) else ""))
    if ckpt.get("classes") and list(ckpt["classes"]) != list(CLASSES):
        print(f"  ! checkpoint classes {ckpt['classes']} differ from current "
              f"taxonomy {CLASSES} - results will be meaningless",
              file=sys.stderr)
        return 2

    model = build_ppe_model(arch, pretrained=False, min_size=size, max_size=size)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)

    ds = PPEDataset(args.split, size=size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=True)
    print(f"split      : {args.split}  ({len(ds)} images"
          + (f", limited to {args.limit} batches)" if args.limit else ")"))

    t0 = time.perf_counter()
    dets, gts, n_images = collect(model, loader, args.limit)
    dt = time.perf_counter() - t0
    print(f"\n{n_images} images in {dt:.0f}s ({n_images / max(dt, 1e-9):.1f} img/s), "
          f"{len(dets)} detections vs {len(gts)} ground truths")

    res = evaluate(dets, gts, CLASSES)
    print(format_report(res, f"{ckpt_path.name} on {args.split}"))

    out = Path(args.out) if args.out else ROOT / "reports" / f"eval_{arch}_{args.split}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # `per_class[c]["pr"]` holds the raw PR curve as numpy arrays: not JSON
    # serialisable, and megabytes of it. Dropping it gives exactly the shape
    # reports/baseline.json already uses, so the two files stay comparable.
    per_class = {c: {k: v for k, v in m.items() if k != "pr"}
                 for c, m in res["per_class"].items()}
    payload = {
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_epoch": trained_epoch,
        "checkpoint_val_map50": val_map50,
        "split": args.split,
        "n_images": n_images,
        "arch": arch,
        "size": size,
        "map50": res["map50"],
        "map50_95": res["map50_95"],
        "per_class": per_class,
        "per_size": res["per_size"],
        "datasets": res["datasets"],
        "per_class_dataset": res.get("per_class_dataset", {}),
    }
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
