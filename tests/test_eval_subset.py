"""Tests for the per-epoch eval subsample.

The training loop scores a subset of val each epoch and selects the checkpoint
on the result. That subset used to be "the first N batches", and splits_v2.json
is ordered by source dataset, so it was 100% dataset1 - 11% of val, while
dataset4 (47% of it) was never scored at all. Training reported 0.6385 where
the full split says 0.7910, and the epoch-to-epoch differences selection ran on
were smaller than the bias.

The bug was invisible from the training log: the numbers looked plausible, moved
in the right direction, and were simply measuring one domain. So the guard has
to assert the property directly.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_ppe import stratified_eval_subset  # noqa: E402


def _records(**counts):
    """Build records ordered by dataset, like splits_v2.json actually is."""
    out = []
    for name, n in counts.items():
        out.extend({"dataset": name, "img": f"{name}/{i}.jpg"} for i in range(n))
    return out


# Real val composition, which is what made the old behaviour so misleading.
VAL_LIKE = dict(dataset4=1053, dataset1=818, dataset5=185, dataset2=144, dataset3=61)


def test_subset_covers_every_source_dataset():
    """The actual bug: the old first-N slice returned one dataset out of five."""
    recs = _records(**VAL_LIKE)
    idxs = stratified_eval_subset(recs, 240, seed=42)
    seen = {recs[i]["dataset"] for i in idxs}
    assert seen == set(VAL_LIKE), f"missing sources: {set(VAL_LIKE) - seen}"


def test_first_n_slice_would_fail_this():
    """Proves the guard discriminates - the old behaviour must not pass it."""
    recs = _records(**VAL_LIKE)
    naive = list(range(240))                      # what the loader used to do
    assert len({recs[i]["dataset"] for i in naive}) == 1


def test_subset_is_roughly_proportional():
    recs = _records(**VAL_LIKE)
    idxs = stratified_eval_subset(recs, 240, seed=42)
    got = Counter(recs[i]["dataset"] for i in idxs)
    total = len(recs)
    for name, n in VAL_LIKE.items():
        expected = 240 * n / total
        # generous band: rounding plus the +1 floor for tiny sources
        assert abs(got[name] - expected) <= max(3.0, expected * 0.35), (
            f"{name}: got {got[name]}, expected about {expected:.1f}")


def test_subset_is_stable_across_epochs():
    """Selection compares epochs, so the sample must not move between them.
    A reshuffling subset would put sampling noise straight into the signal."""
    recs = _records(**VAL_LIKE)
    a = stratified_eval_subset(recs, 240, seed=42)
    b = stratified_eval_subset(recs, 240, seed=42)
    assert a == b


def test_different_seeds_give_different_samples():
    recs = _records(**VAL_LIKE)
    assert stratified_eval_subset(recs, 240, seed=1) != \
        stratified_eval_subset(recs, 240, seed=2)


def test_indices_are_valid_and_unique():
    recs = _records(**VAL_LIKE)
    idxs = stratified_eval_subset(recs, 240, seed=42)
    assert len(idxs) == len(set(idxs))
    assert all(0 <= i < len(recs) for i in idxs)
    assert idxs == sorted(idxs)


def test_never_exceeds_requested_size():
    """The +1 floor per source could overshoot when many sources are tiny."""
    recs = _records(**{f"d{i}": 5 for i in range(40)})
    assert len(stratified_eval_subset(recs, 10, seed=42)) <= 10


def test_request_larger_than_split_returns_everything_available():
    recs = _records(dataset1=6, dataset2=4)
    idxs = stratified_eval_subset(recs, 1000, seed=42)
    assert len(idxs) == len(recs)


def test_single_dataset_split_still_works():
    recs = _records(only=100)
    idxs = stratified_eval_subset(recs, 20, seed=42)
    assert len(idxs) == 20
    assert {recs[i]["dataset"] for i in idxs} == {"only"}
