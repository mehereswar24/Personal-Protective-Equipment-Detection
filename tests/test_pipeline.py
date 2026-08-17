"""Tests for the single-stage inference pipeline's pure logic.

Association and person filtering are the parts that decide whether a helmet
gets credited to the right worker, so they are tested directly rather than only
through a video run. Nothing here loads a checkpoint or touches the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.pipeline import Detected, assign_items, containment, items_for_person  # noqa: E402


def _d(label, box, score=0.9):
    return Detected(label, score, box)


# ── containment ──

def test_containment_fully_inside():
    assert containment((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_containment_disjoint():
    assert containment((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_containment_half():
    assert containment((0, 0, 10, 10), (5, 0, 100, 100)) == pytest.approx(0.5)


def test_containment_is_not_iou():
    """The reason association uses containment: a glove inside a person box has
    a tiny IoU but containment 1.0. An IoU threshold would reject every item."""
    glove, person = (100, 100, 120, 120), (0, 0, 400, 800)
    inter = 20 * 20
    union = 400 * 800 + inter - inter
    assert inter / union < 0.01              # IoU would reject it
    assert containment(glove, person) == pytest.approx(1.0)


def test_containment_zero_area_is_safe():
    assert containment((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ── assignment ──

def test_item_goes_to_the_containing_person():
    people = [(0, 0, 100, 200), (200, 0, 300, 200)]
    items = [_d("helmet", (10, 10, 40, 40))]
    got = assign_items(people, items)
    assert [len(g) for g in got] == [1, 0]


def test_item_is_assigned_exclusively_to_the_best_container():
    """Two workers side by side with overlapping boxes must not both be
    credited with the same helmet - that would inflate compliance."""
    people = [(0, 0, 100, 200), (50, 0, 150, 200)]     # overlapping
    items = [_d("helmet", (110, 10, 140, 40))]         # inside the second only
    got = assign_items(people, items)
    assert [len(g) for g in got] == [0, 1]
    assert sum(len(g) for g in got) == 1


def test_unassigned_item_is_dropped_not_misattributed():
    people = [(0, 0, 100, 200)]
    items = [_d("helmet", (500, 500, 540, 540))]       # nobody's
    assert [len(g) for g in assign_items(people, items)] == [0]


def test_no_people_means_no_assignments():
    assert assign_items([], [_d("vest", (0, 0, 10, 10))]) == []


def test_items_for_person_respects_containment_floor():
    person = (0, 0, 100, 100)
    # 100x100 item overlapping one quadrant: 50*50 / (100*100) = 0.25 contained
    quarter_in = _d("vest", (50, 50, 150, 150))
    assert containment(quarter_in.box, person) == pytest.approx(0.25)
    assert items_for_person(person, [quarter_in], min_containment=0.5) == []
    assert len(items_for_person(person, [quarter_in], min_containment=0.2)) == 1
