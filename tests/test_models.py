"""Tests for the model wrapper, chiefly the label-convention boundary.

This project indexes classes 0-based everywhere (`CLASS_TO_IDX`, the dataset,
the metrics, the class mask). TorchVision's detection heads use a *different*
convention: 0 is background and real classes start at 1. `MaskedLossWrapper` is
the single place those two meet, so it is the single place the +1 belongs.

Getting this wrong is close to undetectable from the training log: the loss
falls perfectly smoothly while mAP sits near zero, because the model is
learning a consistent labelling that simply is not the one being scored. It
cost a 24-epoch run on 2026-08-14, caught at epoch 2. Hence these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.models import MaskedLossWrapper  # noqa: E402
from ppe.taxonomy import CLASSES, CLASS_TO_IDX  # noqa: E402


class _Recorder(nn.Module):
    """Stands in for a torchvision detector and records what it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[int]] = []

    def forward(self, images, targets=None):
        if targets is None:                  # eval: torchvision returns dets
            return [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.int64)}
                    for _ in images]
        for t in targets:
            self.seen.append(t["labels"].tolist())
        return {"classification": torch.tensor(1.0),
                "bbox_regression": torch.tensor(1.0)}


def _targets(labels, mask=None):
    n = len(labels)
    return [{
        "boxes": torch.zeros((n, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "class_mask": mask if mask is not None else torch.ones(len(CLASSES)),
    }]


def _run(labels, mask=None):
    rec = _Recorder()
    wrapper = MaskedLossWrapper(rec)
    wrapper.train()
    wrapper([torch.zeros(3, 32, 32)], _targets(labels, mask))
    return rec


def test_labels_are_shifted_to_torchvision_convention():
    rec = _run([0, 1, 5])
    assert rec.seen == [[1, 2, 6]]


def test_background_index_is_never_handed_to_the_model():
    """The actual bug. `person` is index 0; unshifted it collides with
    torchvision's background and becomes structurally undetectable."""
    rec = _run([CLASS_TO_IDX["person"]])
    assert 0 not in rec.seen[0], (
        "label 0 reached the detector, where it means background - "
        "the class can never be predicted")


def test_every_class_maps_into_the_head_range():
    """num_classes is len(CLASSES)+1, so valid slots are 1..len(CLASSES).
    An off-by-one at either end silently kills the first or last class."""
    rec = _run(list(range(len(CLASSES))))
    got = rec.seen[0]
    assert min(got) == 1
    assert max(got) == len(CLASSES)
    assert sorted(got) == list(range(1, len(CLASSES) + 1))


def test_shift_survives_class_mask_grouping():
    """The shift happens inside the per-signature grouping loop, so it must
    apply to every group, not just the first."""
    rec = _Recorder()
    wrapper = MaskedLossWrapper(rec)
    wrapper.train()
    mask_a, mask_b = torch.ones(len(CLASSES)), torch.zeros(len(CLASSES))
    mask_b[0] = 1.0
    targets = [
        {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([0]), "class_mask": mask_a},
        {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([3]), "class_mask": mask_b},
    ]
    wrapper([torch.zeros(3, 32, 32), torch.zeros(3, 32, 32)], targets)
    assert len(rec.seen) == 2, "expected one forward per class-mask signature"
    assert sorted(rec.seen) == [[1], [4]]


def test_eval_mode_does_not_shift():
    """In eval the model returns detections, it is not given targets, so the
    wrapper must pass straight through and leave decoding to run_eval."""
    rec = _Recorder()
    wrapper = MaskedLossWrapper(rec)
    wrapper.eval()
    with torch.no_grad():
        wrapper([torch.zeros(3, 32, 32)])
    assert rec.seen == []


def test_decode_round_trips_with_run_eval():
    """run_eval decodes predictions with `idx = label - 1`. Composed with the
    wrapper's +1 that must be the identity, or training and scoring disagree."""
    for name, idx in CLASS_TO_IDX.items():
        shifted = _run([idx]).seen[0][0]
        assert CLASSES[shifted - 1] == name
