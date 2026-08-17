"""
Detection metrics — the thing this project has never had.

Implements VOC/COCO-style Average Precision from scratch (no pycocotools
dependency, so it runs anywhere) with the reporting the audit said we need:

  * AP@0.5 (headline — box conventions differ across the source datasets, so
    AP@[.5:.95] punishes convention drift more than real capability)
  * AP@[.50:.95]
  * per-class AP
  * per-SIZE AP using RELATIVE buckets (sqrt(box_area)/sqrt(img_area)), because
    absolute COCO size thresholds are meaningless across 375px–888px sources
  * a per-(class x source-dataset) matrix, because in this data each class is
    essentially single-source — a bare per-class number hides that the model
    may just be recognising "this looks like dataset4"

Everything takes plain Python/numpy structures so it can score any detector.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field, replace

import numpy as np

# relative size buckets: sqrt(box area) / sqrt(image area)
SIZE_BUCKETS = (("tiny", 0.0, 0.05), ("small", 0.05, 0.10),
                ("medium", 0.10, 0.25), ("large", 0.25, 1.01))


@dataclass
class Detection:
    image_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)


@dataclass
class GroundTruth:
    image_id: str
    label: str
    box: tuple[float, float, float, float]
    dataset: str = ""
    img_area: float = 1.0
    difficult: bool = False


@dataclass
class APResult:
    ap: float
    precision: float
    recall: float
    n_gt: int
    n_det: int
    # precision/recall arrays for threshold picking
    pr_curve: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4], b: [M,4] → [N,M] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    q = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, q, side="left")
    idx = np.clip(idx, 0, len(mpre) - 1)
    return float(np.mean(mpre[idx]))


def average_precision(dets: list[Detection], gts: list[GroundTruth],
                      iou_thr: float = 0.5) -> APResult:
    """Single-class AP with greedy highest-score-first matching."""
    n_gt = sum(1 for g in gts if not g.difficult)
    if n_gt == 0:
        return APResult(ap=float("nan"), precision=float("nan"),
                        recall=float("nan"), n_gt=0, n_det=len(dets))
    if not dets:
        return APResult(ap=0.0, precision=0.0, recall=0.0, n_gt=n_gt, n_det=0)

    gt_by_img: dict[str, list[GroundTruth]] = collections.defaultdict(list)
    for g in gts:
        gt_by_img[g.image_id].append(g)
    matched: dict[str, set[int]] = collections.defaultdict(set)

    dets = sorted(dets, key=lambda d: -d.score)
    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)

    for i, det in enumerate(dets):
        candidates = gt_by_img.get(det.image_id, [])
        if not candidates:
            fp[i] = 1
            continue
        ious = iou_matrix(np.array([det.box], dtype=np.float32),
                          np.array([g.box for g in candidates], dtype=np.float32))[0]
        order = np.argsort(-ious)
        hit = -1
        for j in order:
            if ious[j] < iou_thr:
                break
            if j in matched[det.image_id]:
                continue          # this GT already claimed by a higher-score det
            hit = int(j)
            break
        if hit >= 0:
            if candidates[hit].difficult:
                continue          # neither TP nor FP
            matched[det.image_id].add(hit)
            tp[i] = 1
        else:
            fp[i] = 1

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = _ap_from_pr(recall, precision)
    scores = np.array([d.score for d in dets], dtype=np.float32)
    return APResult(ap=ap,
                    precision=float(precision[-1]), recall=float(recall[-1]),
                    n_gt=n_gt, n_det=len(dets),
                    pr_curve=(precision, recall, scores))


def size_bucket(gt: GroundTruth) -> str:
    x1, y1, x2, y2 = gt.box
    rel = np.sqrt(max(0.0, (x2 - x1) * (y2 - y1))) / max(np.sqrt(gt.img_area), 1e-9)
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= rel < hi:
            return name
    return "large"


def evaluate(dets: list[Detection], gts: list[GroundTruth], classes: list[str]
             ) -> dict:
    """Full report: overall, per-class, per-size, per-(class x dataset)."""
    by_class_d: dict[str, list[Detection]] = collections.defaultdict(list)
    by_class_g: dict[str, list[GroundTruth]] = collections.defaultdict(list)
    for d in dets:
        by_class_d[d.label].append(d)
    for g in gts:
        by_class_g[g.label].append(g)

    iou_sweep = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    per_class: dict[str, dict] = {}
    for c in classes:
        r50 = average_precision(by_class_d[c], by_class_g[c], 0.5)
        aps = [average_precision(by_class_d[c], by_class_g[c], t).ap for t in iou_sweep]
        aps = [a for a in aps if not np.isnan(a)]
        per_class[c] = {
            "ap50": r50.ap,
            "ap50_95": float(np.mean(aps)) if aps else float("nan"),
            "precision": r50.precision, "recall": r50.recall,
            "n_gt": r50.n_gt, "n_det": r50.n_det,
            "pr": r50.pr_curve,
        }

    valid = [per_class[c]["ap50"] for c in classes if per_class[c]["n_gt"] > 0]
    valid95 = [per_class[c]["ap50_95"] for c in classes
               if per_class[c]["n_gt"] > 0 and not np.isnan(per_class[c]["ap50_95"])]

    # per size bucket (all classes pooled)
    per_size: dict[str, float] = {}
    for name, _, _ in SIZE_BUCKETS:
        g_sub = [g for g in gts if size_bucket(g) == name]
        if not g_sub:
            per_size[name] = float("nan")
            continue
        keep_imgs = {g.image_id for g in g_sub}
        # Out-of-bucket ground truth in the same images is marked `difficult`,
        # i.e. ignored, rather than dropped.
        #
        # Dropping it — which this did originally — makes a *correct* detection
        # of a large helmet count as a false positive against the tiny-helmet
        # bucket, purely because both appear in one image. Every bucket then
        # reads far below the real per-class AP: on the v2 test run all four
        # buckets sat at 0.25-0.46 while mAP@0.5 was 0.836, including `large`.
        # That is not a size effect, it is the metric scoring other sizes as
        # errors, and it made small objects look like the systemic weakness.
        #
        # `average_precision` already implements COCO-style ignore: a detection
        # matching a difficult GT is neither TP nor FP, and difficult GT are
        # excluded from n_gt. So recall stays over in-bucket objects only.
        gts_masked = [g if size_bucket(g) == name else replace(g, difficult=True)
                      for g in gts if g.image_id in keep_imgs]
        d_sub = [d for d in dets if d.image_id in keep_imgs]
        sub_aps = []
        for c in classes:
            gg = [g for g in gts_masked if g.label == c]
            if not any(not g.difficult for g in gg):
                continue
            dd = [d for d in d_sub if d.label == c]
            r = average_precision(dd, gg, 0.5)
            if not np.isnan(r.ap):
                sub_aps.append(r.ap)
        per_size[name] = float(np.mean(sub_aps)) if sub_aps else float("nan")

    # per (class x dataset) — exposes the class-is-domain confound
    datasets = sorted({g.dataset for g in gts if g.dataset})
    matrix: dict[str, dict[str, float]] = {}
    for c in classes:
        row: dict[str, float] = {}
        for ds in datasets:
            gg = [g for g in by_class_g[c] if g.dataset == ds]
            if not gg:
                row[ds] = float("nan")
                continue
            imgs = {g.image_id for g in gg}
            dd = [d for d in by_class_d[c] if d.image_id in imgs]
            row[ds] = average_precision(dd, gg, 0.5).ap
        matrix[c] = row

    return {
        "map50": float(np.mean(valid)) if valid else float("nan"),
        "map50_95": float(np.mean(valid95)) if valid95 else float("nan"),
        "per_class": per_class,
        "per_size": per_size,
        "per_class_dataset": matrix,
        "datasets": datasets,
    }


def best_threshold(pr_curve, beta: float = 1.0) -> tuple[float, float]:
    """Score threshold maximising F-beta, for calibrating per-class cutoffs.

    The current thresholds in the pipeline were hand-picked with no PR curve
    behind them; this is what replaces that guesswork.
    """
    if pr_curve is None:
        return 0.5, float("nan")
    precision, recall, scores = pr_curve
    denom = (beta * beta * precision) + recall
    f = np.where(denom > 0, (1 + beta * beta) * precision * recall / np.maximum(denom, 1e-9), 0.0)
    i = int(np.argmax(f))
    return float(scores[i]), float(f[i])


def format_report(res: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [f"\n{'=' * 70}", f" {title}", f"{'=' * 70}"]
    lines.append(f"  mAP@0.5      : {res['map50']:.4f}")
    lines.append(f"  mAP@[.5:.95] : {res['map50_95']:.4f}")

    lines.append("\n  per class:")
    lines.append(f"    {'class':10s} {'AP50':>7s} {'AP50-95':>8s} {'prec':>7s} {'recall':>7s} {'#gt':>7s} {'#det':>7s}")
    for c, m in res["per_class"].items():
        if m["n_gt"] == 0:
            lines.append(f"    {c:10s} {'—':>7s} {'—':>8s} {'—':>7s} {'—':>7s} {0:7d} {m['n_det']:7d}")
            continue
        lines.append(f"    {c:10s} {m['ap50']:7.4f} {m['ap50_95']:8.4f} "
                     f"{m['precision']:7.4f} {m['recall']:7.4f} {m['n_gt']:7d} {m['n_det']:7d}")

    lines.append("\n  per relative size (all classes):")
    for k, v in res["per_size"].items():
        lines.append(f"    {k:8s} AP50 {'—' if np.isnan(v) else f'{v:.4f}'}")

    if res["datasets"]:
        lines.append("\n  per class x source dataset (AP50) — exposes class≡domain:")
        header = "    " + f"{'class':10s}" + "".join(f"{d:>10s}" for d in res["datasets"])
        lines.append(header)
        for c, row in res["per_class_dataset"].items():
            cells = "".join(f"{'—':>10s}" if np.isnan(row[d]) else f"{row[d]:10.3f}"
                            for d in res["datasets"])
            lines.append(f"    {c:10s}{cells}")
    return "\n".join(lines)
