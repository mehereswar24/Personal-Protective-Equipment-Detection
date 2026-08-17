"""Tests for temporal smoothing.

Each test here corresponds to one of the four defects found in the original
voter, so a regression on any of them fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.smoothing import State, TemporalVoter, compliance  # noqa: E402

CLASSES = ["helmet", "vest", "gloves"]


def voter(**kw) -> TemporalVoter:
    opts = dict(window=9, warmup=4, on_ratio=0.6, off_ratio=0.3)
    opts.update(kw)
    return TemporalVoter(CLASSES, **opts)


# ── defect 1: no warm-up ──

def test_single_frame_is_not_stable():
    """The old rule `sum > len//2` made one frame instantly 'stable'."""
    v = voter()
    v.update(1, {"helmet"})
    assert v.state_for(1, "helmet") is State.UNKNOWN
    assert v.stable_labels(1) == set()


def test_state_appears_only_after_warmup():
    v = voter(warmup=4)
    for i in range(3):
        v.update(1, {"helmet"})
        assert v.state_for(1, "helmet") is State.UNKNOWN, f"trusted at frame {i+1}"
    v.update(1, {"helmet"})
    assert v.state_for(1, "helmet") is State.PRESENT


# ── defect 3: intersection instead of voting ──

def test_single_missed_frame_does_not_drop_a_stable_label():
    """The core flicker bug: the old code intersected with the current frame,
    so one missed detection removed a consistently-present item."""
    v = voter()
    for _ in range(8):
        v.update(1, {"helmet", "vest"})
    assert v.stable_labels(1) == {"helmet", "vest"}

    v.update(1, {"helmet"})              # vest missed on this frame only
    assert "vest" in v.stable_labels(1), "one dropped frame removed a stable label"


def test_sustained_absence_does_flip_state():
    v = voter()
    for _ in range(9):
        v.update(1, {"helmet", "vest"})
    assert v.state_for(1, "vest") is State.PRESENT
    for _ in range(9):
        v.update(1, {"helmet"})
    assert v.state_for(1, "vest") is State.ABSENT


# ── hysteresis ──

def test_hysteresis_holds_state_in_the_band():
    """Between off_ratio and on_ratio the previous state persists, so a
    detector hovering near threshold does not make the badge flap."""
    v = voter(window=10, warmup=2, on_ratio=0.7, off_ratio=0.3)
    for _ in range(10):
        v.update(1, {"helmet"})
    assert v.state_for(1, "helmet") is State.PRESENT
    # drive to ~50% — inside the band, so PRESENT should hold
    for _ in range(5):
        v.update(1, set())
    assert v.confidence(1, "helmet") == pytest.approx(0.5)
    assert v.state_for(1, "helmet") is State.PRESENT


def test_dropping_below_off_ratio_flips_to_absent():
    v = voter(window=10, warmup=2, on_ratio=0.7, off_ratio=0.3)
    for _ in range(10):
        v.update(1, {"helmet"})
    for _ in range(8):
        v.update(1, set())
    assert v.state_for(1, "helmet") is State.ABSENT


# ── defect 4: unbounded memory ──

def test_retain_evicts_dead_tracks():
    v = voter()
    for tid in range(50):
        v.update(tid, {"helmet"})
    assert len(v) == 50
    v.retain({1, 2, 3})
    assert len(v) == 3


def test_forget_removes_one_track():
    v = voter()
    v.update(7, {"helmet"})
    v.forget(7)
    assert v.state_for(7, "helmet") is State.UNKNOWN


def test_memory_stays_bounded_under_id_churn():
    """Simulates the 24/7 failure mode: new track ids forever."""
    v = voter()
    for tid in range(5000):
        v.update(tid, {"helmet"})
        v.retain({tid})                  # only the current track is alive
    assert len(v) <= 2


# ── compliance ──

def test_unknown_is_not_a_violation():
    """One weak negative used to override a confident positive; UNKNOWN must
    never be reported as a violation in a safety system."""
    states = {"helmet": State.UNKNOWN, "vest": State.PRESENT}
    ok, violations, unknown = compliance(states, ["helmet", "vest"])
    assert violations == []
    assert unknown == ["helmet"]
    assert ok is True


def test_absent_required_class_is_a_violation():
    states = {"helmet": State.ABSENT, "vest": State.PRESENT}
    ok, violations, _ = compliance(states, ["helmet", "vest"])
    assert ok is False and violations == ["helmet"]


def test_advisory_classes_never_flip_compliance():
    """Badge and log must agree. gloves/boots are advisory because their
    measured AP does not support automatic alerting."""
    states = {"helmet": State.PRESENT, "vest": State.PRESENT, "gloves": State.ABSENT}
    ok, violations, _ = compliance(states, ["helmet", "vest", "gloves"],
                                   advisory=["gloves"])
    assert ok is True and violations == []


def test_all_present_is_compliant():
    states = {c: State.PRESENT for c in CLASSES}
    ok, violations, unknown = compliance(states, ["helmet", "vest"])
    assert ok and not violations and not unknown


def test_invalid_ratios_rejected():
    with pytest.raises(ValueError):
        TemporalVoter(CLASSES, on_ratio=0.2, off_ratio=0.8)
