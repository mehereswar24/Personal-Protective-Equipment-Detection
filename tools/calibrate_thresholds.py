"""Derive per-class score thresholds from the PR curve, and verify them.

The detector emits every box above torchvision's default `score_thresh=0.2` -
47,504 detections for 11,186 objects on test. That is correct for AP, which
integrates the whole precision-recall curve, but it is not an operating point:
shipped as-is the pipeline would draw three false boxes for every true one.
Something has to pick a cutoff per class, and until now that was hand-picked
guesswork (see the docstring of `ppe.metrics.best_threshold`), with
`ppe/config.py` carrying values disconnected from any PR curve.

Method, and the ordering matters:

  1. calibrate on **val** - the split already used for checkpoint selection
  2. verify on **test** - never touched during training or selection

Choosing thresholds on test and then reporting test numbers would be tuning on
the held-out set: the resulting figures would be optimistic and unfalsifiable.
So test is only ever read *after* the thresholds are fixed.

`--beta` weights recall against precision (F-beta). The default is 2.0, not 1.0,
because this is a compliance system: a missed violation is a safety failure,
while a false positive costs a human a second of review. F1 would silently
trade away the more expensive error.

Usage:
    python tools/calibrate_thresholds.py                       # val -> test, F2
    python tools/calibrate_thresholds.py --beta 1.0            # balanced
    python tools/calibrate_thresholds.py --no-verify           # skip test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.dataset import PPEDataset, collate  # noqa: E402
from ppe.metrics import Detection, best_threshold, evaluate  # noqa: E402
from ppe.models import build_ppe_model  # noqa: E402
from ppe.taxonomy import CLASSES  # noqa: E402
from tools.eval_ppe import collect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def pr_at_threshold(pr_curve, thr: float) -> tuple[float, float]:
    """Precision and recall at the operating point `thr`.

    The curve is ordered by descending score, so the last index whose score is
    still >= thr is the point reached by keeping exactly those detections.
    """
    if pr_curve is None:
        return float("nan"), float("nan")
    precision, recall, scores = pr_curve
    keep = np.nonzero(np.asarray(scores) >= thr)[0]
    if len(keep) == 0:
        return float("nan"), 0.0
    i = int(keep[-1])
    return float(precision[i]), float(recall[i])


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "fcos")
    size = int(ckpt.get("size", 640))
    model = build_ppe_model(arch, pretrained=False, min_size=size, max_size=size)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE), arch, size, ckpt


def run_split(model, split: str, size: int, batch: int, workers: int,
              limit: int = 0):
    ds = PPEDataset(split, size=size, augment=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate, pin_memory=True)
    print(f"  {split}: {len(ds)} images")
    return collect(model, loader, limit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/ppe_fcos_v2.pth")
    ap.add_argument("--calibrate-split", default="val")
    ap.add_argument("--verify-split", default="test")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--beta", type=float, default=2.0,
                    help="F-beta; >1 favours recall (default 2.0, safety bias)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"no such checkpoint: {ckpt_path}", file=sys.stderr)
        return 2

    model, arch, size, ckpt = load_model(ckpt_path)
    print(f"checkpoint : {ckpt_path.name}  (arch {arch}, size {size}, "
          f"epoch {ckpt.get('epoch', '?')})")
    print(f"F-beta     : {args.beta}  "
          f"({'recall-weighted' if args.beta > 1 else 'balanced' if args.beta == 1 else 'precision-weighted'})")

    # ── calibrate ──
    print(f"\ncalibrating on {args.calibrate_split} (never on the verify split)")
    dets, gts, _ = run_split(model, args.calibrate_split, size, args.batch,
                             args.workers, args.limit)
    cal = evaluate(dets, gts, CLASSES)

    thresholds: dict[str, float] = {}
    rows = []
    for c, m in cal["per_class"].items():
        if m["n_gt"] == 0:
            thresholds[c] = 0.5
            rows.append((c, 0.5, float("nan"), float("nan"), float("nan"), 0))
            continue
        thr, fscore = best_threshold(m["pr"], beta=args.beta)
        prec, rec = pr_at_threshold(m["pr"], thr)
        thresholds[c] = round(float(thr), 4)
        rows.append((c, thr, prec, rec, fscore, m["n_gt"]))

    print(f"\n  {'class':10s} {'thr':>7s} {'prec':>8s} {'recall':>8s} "
          f"{f'F{args.beta:g}':>7s} {'#gt':>7s}")
    for c, thr, prec, rec, f, n in rows:
        print(f"  {c:10s} {thr:7.3f} {prec:8.4f} {rec:8.4f} {f:7.4f} {n:7d}")

    payload = {
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "calibrated_on": args.calibrate_split,
        "beta": args.beta,
        "thresholds": thresholds,
        "calibration": {c: {"threshold": t, "precision": p, "recall": r,
                            "f_beta": f, "n_gt": n}
                        for c, t, p, r, f, n in rows},
    }

    # ── verify ──
    if not args.no_verify:
        print(f"\nverifying on {args.verify_split} with those thresholds fixed")
        vdets, vgts, _ = run_split(model, args.verify_split, size, args.batch,
                                   args.workers, args.limit)
        kept = [d for d in vdets if d.score >= thresholds.get(d.label, 0.5)]
        ver = evaluate(kept, vgts, CLASSES)
        raw = evaluate(vdets, vgts, CLASSES)

        print(f"\n  {'class':10s} {'thr':>7s} {'prec':>8s} {'recall':>8s} "
              f"{'#det':>8s} {'was':>8s}")
        for c, m in ver["per_class"].items():
            if m["n_gt"] == 0:
                continue
            print(f"  {c:10s} {thresholds[c]:7.3f} {m['precision']:8.4f} "
                  f"{m['recall']:8.4f} {m['n_det']:8d} "
                  f"{raw['per_class'][c]['n_det']:8d}")
        print(f"\n  detections: {len(kept)} kept of {len(vdets)} "
              f"({100 * len(kept) / max(1, len(vdets)):.1f}%), "
              f"for {len(vgts)} ground truths")

        payload["verification"] = {
            "split": args.verify_split,
            "n_detections_raw": len(vdets),
            "n_detections_kept": len(kept),
            "n_ground_truth": len(vgts),
            "per_class": {c: {k: v for k, v in m.items() if k != "pr"}
                          for c, m in ver["per_class"].items()},
        }

    out = Path(args.out) if args.out else ROOT / "reports" / f"thresholds_{arch}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
