"""Tests for the replacement tracker.

These pin the behaviours that were actually broken before: id churn under
occlusion (which silently defeated temporal voting and violation dedup),
NaN boxes crashing the frame loop, and process-global id collisions between
concurrent camera streams.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.tracking import Tracker, iou_batch  # noqa: E402


def box(x, y, w=40, h=100, score=0.9):
    return [x, y, x + w, y + h, score]


def test_iou_batch_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_batch(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert iou_batch(a, b)[0, 0] == pytest.approx(0.0)


def test_single_object_keeps_one_id():
    trk = Tracker(min_hits=1, max_age=30)
    ids = set()
    for i in range(20):
        out = trk.update(np.array([box(100 + i * 3, 50)], dtype=np.float32))
        assert len(out) == 1
        ids.add(int(out[0][4]))
    assert ids == {1}, f"id churned across frames: {ids}"


def test_id_survives_occlusion():
    """The bug that broke everything downstream: a worker briefly occluded came
    back with a NEW id, orphaning their vote history and re-firing alerts."""
    trk = Tracker(min_hits=1, max_age=30)
    for i in range(5):
        out = trk.update(np.array([box(100 + i * 2, 50)], dtype=np.float32))
    first_id = int(out[0][4])

    for _ in range(10):                     # fully occluded: no detections
        trk.update(np.empty((0, 5), dtype=np.float32))

    out = trk.update(np.array([box(125, 50)], dtype=np.float32))
    assert len(out) == 1
    assert int(out[0][4]) == first_id, "track id changed after occlusion"


def test_track_dies_after_max_age():
    trk = Tracker(min_hits=1, max_age=5)
    trk.update(np.array([box(100, 50)], dtype=np.float32))
    for _ in range(7):
        trk.update(np.empty((0, 5), dtype=np.float32))
    out = trk.update(np.array([box(400, 300)], dtype=np.float32))
    assert int(out[0][4]) != 1, "stale track should have been retired"


def test_two_objects_get_distinct_stable_ids():
    trk = Tracker(min_hits=1, max_age=30)
    for i in range(10):
        dets = np.array([box(50 + i, 50), box(400 - i, 60)], dtype=np.float32)
        out = trk.update(dets)
        assert len(out) == 2
    assert len({int(r[4]) for r in out}) == 2


def test_min_hits_debounces_new_tracks():
    trk = Tracker(min_hits=3, max_age=30)
    trk.update(np.array([box(0, 0)], dtype=np.float32))   # frame 1..3 grace
    for _ in range(4):
        trk.update(np.array([box(500, 500)], dtype=np.float32))
    out = trk.update(np.array([box(10, 10)], dtype=np.float32))
    # the brand-new box at (10,10) should not be reported on its first frame
    assert all(abs(r[0] - 10) > 1 for r in out)


def test_empty_input_is_safe():
    trk = Tracker()
    assert trk.update(None).shape == (0, 5)
    assert trk.update(np.empty((0, 5), dtype=np.float32)).shape == (0, 5)


def test_boxes_without_scores_accepted():
    trk = Tracker(min_hits=1)
    out = trk.update(np.array([[10, 10, 50, 110]], dtype=np.float32))
    assert len(out) == 1


def test_never_emits_nan():
    """A negative area state used to produce NaN via sqrt(), which crashed the
    caller at int(nan). Drive the filter hard and assert finiteness."""
    trk = Tracker(min_hits=1, max_age=50)
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(0, 4)
        dets = np.array([box(float(rng.integers(0, 600)), float(rng.integers(0, 400)),
                             w=float(rng.integers(5, 200)), h=float(rng.integers(5, 300)))
                         for _ in range(n)], dtype=np.float32) if n else np.empty((0, 5), np.float32)
        out = trk.update(dets)
        assert np.all(np.isfinite(out)), "tracker emitted NaN/inf"


def test_extreme_scale_change_is_not_matched():
    trk = Tracker(min_hits=1, max_age=30, max_scale_change=4.0)
    trk.update(np.array([box(100, 100, w=40, h=100)], dtype=np.float32))
    # same place, ~25x the area → must not be treated as the same object
    out = trk.update(np.array([box(100, 100, w=200, h=500)], dtype=np.float32))
    assert int(out[0][4]) != 1 or len(out) == 1 and trk._next_id > 2


def test_streams_do_not_share_ids():
    """The old tracker kept its counter on the CLASS, so two camera threads
    mutated one global and could hand out the same id."""
    a, b = Tracker(min_hits=1), Tracker(min_hits=1)
    oa = a.update(np.array([box(10, 10)], dtype=np.float32))
    ob = b.update(np.array([box(10, 10)], dtype=np.float32))
    assert int(oa[0][4]) == 1 and int(ob[0][4]) == 1   # independent counters
    for _ in range(5):
        a.update(np.array([box(10, 10)], dtype=np.float32))
    assert b._next_id == 2, "second tracker's ids advanced due to the first"


def test_reset_clears_state():
    trk = Tracker(min_hits=1)
    trk.update(np.array([box(10, 10)], dtype=np.float32))
    assert trk.active_ids
    trk.reset()
    assert not trk.active_ids and trk.frame_count == 0


def test_active_ids_tracks_liveness():
    trk = Tracker(min_hits=1, max_age=2)
    trk.update(np.array([box(10, 10)], dtype=np.float32))
    assert trk.active_ids == {1}
    for _ in range(5):
        trk.update(np.empty((0, 5), dtype=np.float32))
    assert trk.active_ids == set()


def test_no_gui_imports():
    """The old sort.py did matplotlib.use('TkAgg') + skimage at import time,
    which made the pipeline fail to start in a headless container.

    Checks real import statements via the AST — grepping the source text would
    also match the docstring above, which deliberately names those modules.
    """
    import ast
    import ppe.tracking as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    banned = {"matplotlib", "skimage", "cv2", "torch"}
    assert not (imported & banned), (
        f"tracker must stay headless/dependency-light, but imports {imported & banned}")
