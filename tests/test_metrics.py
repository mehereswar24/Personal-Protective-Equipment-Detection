"""Unit tests for the AP implementation.

The whole project is about to be measured with this code, so it needs to be
right before any baseline number is trusted. These check the properties that
actually matter: perfect/empty predictions, duplicate suppression, IoU
threshold behaviour, ranking sensitivity, and the relative size bucketing.

Run:  venv\\Scripts\\python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.metrics import (  # noqa: E402
    Detection, GroundTruth, average_precision, best_threshold, evaluate,
    iou_matrix, size_bucket,
)


def gt(img, label, box, ds="ds1", area=10_000.0):
    return GroundTruth(img, label, box, dataset=ds, img_area=area)


def det(img, label, score, box):
    return Detection(img, label, score, box)


# ── IoU ──────────────────────────────────────────────────────────────────────

def test_iou_identical_is_one():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[50, 50, 60, 60]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_half_overlap():
    # 10x10 and 10x10 sharing a 5x10 strip → inter 50, union 150
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-6)


def test_iou_empty_inputs():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert iou_matrix(empty, np.array([[0, 0, 1, 1]], dtype=np.float32)).shape == (0, 1)


# ── AP basics ────────────────────────────────────────────────────────────────

def test_perfect_prediction_is_ap_one():
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    d = [det("a", "helmet", 0.9, (0, 0, 10, 10))]
    assert average_precision(d, g, 0.5).ap == pytest.approx(1.0)


def test_no_detections_is_ap_zero():
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    assert average_precision([], g, 0.5).ap == 0.0


def test_no_ground_truth_is_nan():
    """A class with no GT must not be scored as 0 — that would drag the mean
    down and make an untested class look like a failing one."""
    d = [det("a", "helmet", 0.9, (0, 0, 10, 10))]
    assert np.isnan(average_precision(d, [], 0.5).ap)


def test_duplicate_detection_counts_as_false_positive():
    """Two boxes on one GT: the best-scoring one is a TP, the other a FP.

    Note the AP is still 1.0 — that is correct interpolated-AP behaviour, not a
    bug. The duplicate arrives *after* recall is already 1.0, and interpolated
    precision takes the max over all recall >= r, so the trailing FP cannot pull
    the envelope down. It shows up in the raw precision instead.
    """
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    d = [det("a", "helmet", 0.9, (0, 0, 10, 10)),
         det("a", "helmet", 0.8, (0, 0, 10, 10))]
    r = average_precision(d, g, 0.5)
    assert r.n_det == 2
    assert r.precision == pytest.approx(0.5)      # 1 TP / 2 dets
    assert r.recall == pytest.approx(1.0)
    assert r.ap == pytest.approx(1.0)


def test_false_positive_before_full_recall_lowers_ap():
    """A FP ranked ABOVE a real detection does reduce AP — the case that matters
    for threshold picking, and the one the duplicate test above cannot cover."""
    g = [gt("a", "helmet", (0, 0, 10, 10)), gt("b", "helmet", (0, 0, 10, 10))]
    clean = [det("a", "helmet", 0.9, (0, 0, 10, 10)),
             det("b", "helmet", 0.8, (0, 0, 10, 10))]
    with_fp = [det("a", "helmet", 0.9, (0, 0, 10, 10)),
               det("a", "helmet", 0.85, (90, 90, 99, 99)),   # FP, ranked 2nd
               det("b", "helmet", 0.8, (0, 0, 10, 10))]
    assert average_precision(clean, g, 0.5).ap == pytest.approx(1.0)
    assert average_precision(with_fp, g, 0.5).ap < 1.0


def test_iou_threshold_rejects_loose_box():
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    d = [det("a", "helmet", 0.9, (6, 0, 16, 10))]   # IoU = 40/160 = 0.25
    assert average_precision(d, g, 0.5).ap == 0.0
    assert average_precision(d, g, 0.2).ap == pytest.approx(1.0)


def test_wrong_image_is_not_matched():
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    d = [det("b", "helmet", 0.9, (0, 0, 10, 10))]
    assert average_precision(d, g, 0.5).ap == 0.0


def test_ranking_matters():
    """Same boxes, better ordering → higher AP. Guards the score sort."""
    g = [gt("a", "helmet", (0, 0, 10, 10)), gt("b", "helmet", (0, 0, 10, 10))]
    good = [det("a", "helmet", 0.9, (0, 0, 10, 10)),
            det("b", "helmet", 0.8, (0, 0, 10, 10)),
            det("a", "helmet", 0.1, (50, 50, 60, 60))]      # low-score FP last
    bad = [det("a", "helmet", 0.9, (50, 50, 60, 60)),        # FP first
           det("b", "helmet", 0.8, (0, 0, 10, 10)),
           det("a", "helmet", 0.7, (0, 0, 10, 10))]
    assert average_precision(good, g, 0.5).ap > average_precision(bad, g, 0.5).ap


def test_difficult_gt_excluded_from_count():
    g = [gt("a", "helmet", (0, 0, 10, 10)),
         GroundTruth("a", "helmet", (20, 20, 30, 30), difficult=True)]
    r = average_precision([det("a", "helmet", 0.9, (0, 0, 10, 10))], g, 0.5)
    assert r.n_gt == 1
    assert r.ap == pytest.approx(1.0)


# ── size buckets ─────────────────────────────────────────────────────────────

def test_size_bucket_is_relative_not_absolute():
    """The same pixel box is 'large' in a small image and 'tiny' in a big one —
    this is why absolute COCO thresholds were wrong for this dataset."""
    small_img = gt("a", "gloves", (0, 0, 50, 50), area=100 * 100)   # rel 0.5
    big_img = gt("b", "gloves", (0, 0, 50, 50), area=4000 * 4000)   # rel 0.0125
    assert size_bucket(small_img) == "large"
    assert size_bucket(big_img) == "tiny"


def test_glove_scale_lands_in_small_bucket():
    """Median glove in this dataset is rel ~0.069 → 'small'."""
    g = gt("a", "gloves", (0, 0, 69, 69), area=1000 * 1000)
    assert size_bucket(g) == "small"


# ── full report ──────────────────────────────────────────────────────────────

def test_evaluate_reports_per_class_and_per_dataset():
    classes = ["helmet", "vest"]
    g = [gt("a", "helmet", (0, 0, 10, 10), ds="ds1"),
         gt("b", "vest", (0, 0, 10, 10), ds="ds2")]
    d = [det("a", "helmet", 0.9, (0, 0, 10, 10)),
         det("b", "vest", 0.9, (0, 0, 10, 10))]
    res = evaluate(d, g, classes)
    assert res["map50"] == pytest.approx(1.0)
    assert res["per_class"]["helmet"]["ap50"] == pytest.approx(1.0)
    # class x dataset matrix: helmet only exists in ds1
    assert res["per_class_dataset"]["helmet"]["ds1"] == pytest.approx(1.0)
    assert np.isnan(res["per_class_dataset"]["helmet"]["ds2"])


def test_map_ignores_classes_without_gt():
    """A class with no GT must not be averaged in as a zero."""
    classes = ["helmet", "gloves"]
    g = [gt("a", "helmet", (0, 0, 10, 10))]
    d = [det("a", "helmet", 0.9, (0, 0, 10, 10))]
    res = evaluate(d, g, classes)
    assert res["map50"] == pytest.approx(1.0)     # not 0.5
    assert res["per_class"]["gloves"]["n_gt"] == 0


def test_best_threshold_picks_a_sensible_cut():
    g = [gt("a", "helmet", (0, 0, 10, 10)), gt("b", "helmet", (0, 0, 10, 10))]
    d = [det("a", "helmet", 0.95, (0, 0, 10, 10)),
         det("b", "helmet", 0.90, (0, 0, 10, 10)),
         det("a", "helmet", 0.10, (50, 50, 60, 60))]   # junk FP
    r = average_precision(d, g, 0.5)
    thr, f1 = best_threshold(r.pr_curve)
    assert f1 == pytest.approx(1.0)
    assert thr > 0.10        # the junk detection is excluded


def test_best_threshold_handles_missing_curve():
    thr, f1 = best_threshold(None)
    assert thr == 0.5 and np.isnan(f1)


# ── per-size AP must ignore other sizes, not score them as errors ──

def test_per_size_ignores_out_of_bucket_detections():
    """The bug: an image holds one tiny and one large helmet, and the model
    detects BOTH correctly. The tiny bucket must read 1.0 - the large detection
    is out of bucket, so it is ignored, not counted as a false positive.

    Scoring it as an FP is what dragged every bucket (including `large`) to
    0.25-0.46 while mAP@0.5 was 0.836.
    """
    area = 1000.0 * 1000.0
    tiny = GroundTruth("img", "helmet", (0, 0, 30, 30), img_area=area)      # rel .03
    large = GroundTruth("img", "helmet", (100, 100, 500, 500), img_area=area)  # rel .40
    dets = [Detection("img", "helmet", 0.9, (0, 0, 30, 30)),
            Detection("img", "helmet", 0.8, (100, 100, 500, 500))]

    res = evaluate(dets, [tiny, large], ["helmet"])
    assert res["per_size"]["tiny"] == pytest.approx(1.0)
    assert res["per_size"]["large"] == pytest.approx(1.0)


def test_per_size_still_penalises_a_genuine_miss():
    """The guard above must not be satisfied by ignoring everything: a bucket
    with a real false positive and a real miss must still score below 1."""
    area = 1000.0 * 1000.0
    tiny = GroundTruth("img", "helmet", (0, 0, 30, 30), img_area=area)
    dets = [Detection("img", "helmet", 0.9, (600, 600, 630, 630))]   # matches nothing
    res = evaluate(dets, [tiny], ["helmet"])
    assert res["per_size"]["tiny"] == pytest.approx(0.0)
