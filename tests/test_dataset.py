"""Tests for the dataset layer.

The class-presence mask is the fix for the biggest accuracy bug in the project,
so it gets tested directly rather than only via training runs.
"""

from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.dataset import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, PPEDataset, letterbox, normalise_chw,
)
from ppe.taxonomy import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "ppe" / "splits_v2.json"
needs_data = pytest.mark.skipif(not SPLITS.exists(), reason="splits_v2.json not built")


# ── letterbox ──

def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((100, 400, 3), dtype=np.uint8)     # 4:1 wide
    out, _, meta = letterbox(img, 200)
    assert out.shape == (200, 200, 3)
    # 400->200 means scale 0.5, so content is 200x50 centred with padding
    assert meta["scale"] == pytest.approx(0.5)
    assert meta["dy"] == 75 and meta["dx"] == 0


def test_letterbox_transforms_boxes_consistently():
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 400, 100]], dtype=np.float32)
    _, out_boxes, meta = letterbox(img, 200, boxes)
    # full-extent box must still span the content area exactly
    assert out_boxes[0][0] == pytest.approx(0.0)
    assert out_boxes[0][2] == pytest.approx(200.0)
    assert out_boxes[0][1] == pytest.approx(meta["dy"])
    assert out_boxes[0][3] == pytest.approx(meta["dy"] + 50)


def test_letterbox_square_input_unpadded():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, _, meta = letterbox(img, 150)
    assert meta["dx"] == 0 and meta["dy"] == 0


# ── normalisation ──
#
# A dataloader worker died of MemoryError here (2026-08-14) because the
# normalisation allocated four full-size float32 arrays per sample. These tests
# pin both halves of the fix: the arithmetic is unchanged, and the allocation
# stays at one buffer. Without the second test the first would happily pass
# again on the version that OOM'd.

def _reference_normalise(img):
    """The obvious expression the fast path must stay bit-identical to."""
    return (((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
            .transpose(2, 0, 1).copy())


def test_normalise_is_bit_identical_to_reference():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(normalise_chw(img), _reference_normalise(img))


def test_normalise_handles_channel_extremes():
    """Per-channel means/stds differ, so a channel mix-up survives random data
    but not this: 0 and 255 land on exactly -mean/std and (1-mean)/std."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 1] = 255
    out = normalise_chw(img)
    for c, expected in enumerate([
        (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    ]):
        assert out[c] == pytest.approx(expected)


def test_normalise_returns_contiguous_float32_chw():
    """torch.from_numpy shares memory, so a non-contiguous or wrong-dtype
    result would either copy again or silently reinterpret the buffer."""
    out = normalise_chw(np.zeros((8, 12, 3), dtype=np.uint8))
    assert out.shape == (3, 8, 12)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_normalise_is_not_square_only():
    """The first version of the fix sized its buffer from the dataset's square
    `size` rather than the image, which is wrong for any non-square input."""
    assert normalise_chw(np.zeros((7, 19, 3), dtype=np.uint8)).shape == (3, 7, 19)


def test_normalise_allocates_a_single_buffer():
    """The regression guard. The reference expression peaks at two full-size
    arrays; this must peak at one."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    one_buffer = 256 * 256 * 3 * 4                      # bytes, float32 CHW

    tracemalloc.start()
    tracemalloc.reset_peak()
    normalise_chw(img)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < one_buffer * 1.5, (
        f"normalisation peaked at {peak / one_buffer:.1f}x one buffer; "
        "it should allocate exactly one")


def test_reference_normalise_would_fail_the_allocation_guard():
    """Proves the guard above discriminates — if this ever passes, the
    threshold has gone slack and the guard is no longer protecting anything."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    one_buffer = 256 * 256 * 3 * 4

    tracemalloc.start()
    tracemalloc.reset_peak()
    _reference_normalise(img)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak > one_buffer * 1.5


# ── class-presence masking ──

@needs_data
def test_class_mask_marks_unsupervised_classes_as_unknown():
    """dataset4 (Hard Hat Workers) annotates helmet/head/person but NEVER vest,
    gloves or boots. Those must be masked out, otherwise every unlabelled vest
    in that dataset trains the model that vests are background."""
    ds = PPEDataset("val", size=320)
    mask = ds.class_mask_for("dataset4")
    idx = {c: i for i, c in enumerate(ds.classes)}
    assert mask[idx["helmet"]] == 1.0
    assert mask[idx["head"]] == 1.0
    assert mask[idx["vest"]] == 0.0, "vest is not annotated in dataset4"
    assert mask[idx["gloves"]] == 0.0
    assert mask[idx["boots"]] == 0.0


@needs_data
def test_dataset1_supervises_everything():
    ds = PPEDataset("val", size=320)
    assert ds.class_mask_for("dataset1").sum() == len(ds.classes)


@needs_data
def test_unknown_dataset_defaults_to_full_supervision():
    ds = PPEDataset("val", size=320)
    assert ds.class_mask_for("does_not_exist").sum() == len(ds.classes)


# ── items ──

@needs_data
def test_item_shapes_and_types():
    ds = PPEDataset("val", size=320)
    img, target = ds[0]
    assert img.shape == (3, 320, 320)
    assert img.dtype.is_floating_point
    assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
    assert target["labels"].ndim == 1
    assert len(target["boxes"]) == len(target["labels"])
    assert target["class_mask"].shape == (len(ds.classes),)


@needs_data
def test_boxes_inside_canvas():
    ds = PPEDataset("val", size=320)
    for i in range(min(25, len(ds))):
        _, t = ds[i]
        if len(t["boxes"]):
            assert t["boxes"].min() >= -1.0
            assert t["boxes"].max() <= 321.0


@needs_data
def test_labels_within_class_range():
    ds = PPEDataset("val", size=320)
    for i in range(min(25, len(ds))):
        _, t = ds[i]
        if len(t["labels"]):
            assert int(t["labels"].min()) >= 0
            assert int(t["labels"].max()) < len(CLASSES)


# ── imbalance ──

@needs_data
def test_repeat_factors_favour_rare_classes():
    """gloves is the rarest class; images containing it must be sampled more
    often than images that only contain the most common class."""
    ds = PPEDataset("train", size=320)
    rf = ds.repeat_factors()
    assert len(rf) == len(ds)
    assert min(rf) >= 1.0

    counts = ds.class_counts()
    rarest = min(counts, key=counts.get)
    common = max(counts, key=counts.get)
    rare_rf = [r for r, rec in zip(rf, ds.records)
               if rarest in {b[0] for b in rec["boxes"]}]
    common_only_rf = [r for r, rec in zip(rf, ds.records)
                      if {b[0] for b in rec["boxes"]} == {common}]
    if rare_rf and common_only_rf:
        assert np.mean(rare_rf) > np.mean(common_only_rf)


@needs_data
def test_split_has_no_group_overlap():
    """Guards the leakage fix: no source photo may appear in two splits."""
    data = json.loads(SPLITS.read_text(encoding="utf-8"))
    groups = {s: {r["group"] for r in data["splits"][s]}
              for s in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])
