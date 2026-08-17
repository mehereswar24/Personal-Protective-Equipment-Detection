"""
Temporal smoothing of per-track PPE state.

The previous implementation was broken in four ways, all of which mattered:

  1. **No warm-up.** The rule was `sum(hist) > len(hist)//2`, so on a track's
     very first frame `1 > 0` held — a single detection was instantly "stable".
     Combined with the tracker's id churn (which made *most* tracks new), the
     7-frame window debounced essentially nothing.
  2. **The fallback defeated the filter.** If nothing was stable it displayed
     the raw unfiltered detections — i.e. exactly when the voter had no
     confidence, its output was discarded.
  3. **It intersected instead of voting.** The result was always a subset of the
     current frame's detections, so an item stably present for 9 frames but
     missed on frame 10 vanished on frame 10. It therefore could not fix the
     flicker it was written to fix.
  4. **Unbounded memory.** History was keyed `(track_id, label)` in a dict that
     was never evicted; on a 24/7 stream with id churn that grows without limit.

This version votes properly, reconstructs state from the vote (not from the
current frame), applies hysteresis so state doesn't oscillate at the boundary,
and evicts history when a track dies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"       # not enough evidence yet — never treated as a violation


@dataclass
class _TrackHistory:
    windows: dict[str, deque] = field(default_factory=dict)
    state: dict[str, State] = field(default_factory=dict)
    frames_seen: int = 0


class TemporalVoter:
    """Majority vote per (track, class) with warm-up and hysteresis.

    Args:
        classes:   labels to track.
        window:    vote window length in frames.
        warmup:    frames before any state other than UNKNOWN is reported.
        on_ratio:  fraction of the window that must be positive to become PRESENT.
        off_ratio: fraction below which it becomes ABSENT. The gap between the
                   two is the hysteresis band, where the previous state persists —
                   this is what stops flapping when detections sit near threshold.
    """

    def __init__(self, classes: list[str], window: int = 9, warmup: int = 4,
                 on_ratio: float = 0.6, off_ratio: float = 0.3) -> None:
        if not 0.0 <= off_ratio <= on_ratio <= 1.0:
            raise ValueError("require 0 <= off_ratio <= on_ratio <= 1")
        self.classes = list(classes)
        self.window = int(window)
        self.warmup = int(warmup)
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self._tracks: dict[int, _TrackHistory] = {}

    def update(self, track_id: int, detected: set[str]) -> None:
        """Record one frame of observations for a track."""
        hist = self._tracks.setdefault(track_id, _TrackHistory())
        hist.frames_seen += 1
        for label in self.classes:
            dq = hist.windows.setdefault(label, deque(maxlen=self.window))
            dq.append(label in detected)

            n = len(dq)
            ratio = sum(dq) / n if n else 0.0
            prev = hist.state.get(label, State.UNKNOWN)

            if hist.frames_seen < self.warmup:
                hist.state[label] = State.UNKNOWN
            elif ratio >= self.on_ratio:
                hist.state[label] = State.PRESENT
            elif ratio <= self.off_ratio:
                hist.state[label] = State.ABSENT
            else:
                # inside the hysteresis band: hold, but never hold UNKNOWN
                # forever once we're past warm-up
                hist.state[label] = prev if prev is not State.UNKNOWN else State.ABSENT

    def state_for(self, track_id: int, label: str) -> State:
        hist = self._tracks.get(track_id)
        if hist is None or hist.frames_seen < self.warmup:
            return State.UNKNOWN
        return hist.state.get(label, State.UNKNOWN)

    def states(self, track_id: int) -> dict[str, State]:
        return {c: self.state_for(track_id, c) for c in self.classes}

    def stable_labels(self, track_id: int) -> set[str]:
        """Labels voted PRESENT — reconstructed from the vote, NOT intersected
        with the current frame, so a single missed frame no longer drops them."""
        return {c for c, s in self.states(track_id).items() if s is State.PRESENT}

    def confidence(self, track_id: int, label: str) -> float:
        hist = self._tracks.get(track_id)
        if not hist:
            return 0.0
        dq = hist.windows.get(label)
        return (sum(dq) / len(dq)) if dq else 0.0

    # ── lifecycle ──
    def retain(self, active_ids: set[int]) -> None:
        """Drop history for dead tracks. Without this the dict grows forever."""
        for tid in [t for t in self._tracks if t not in active_ids]:
            del self._tracks[tid]

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        self._tracks.clear()

    def __len__(self) -> int:
        return len(self._tracks)


def compliance(states: dict[str, State], required: list[str],
               advisory: list[str] | None = None) -> tuple[bool, list[str], list[str]]:
    """Decide compliance from smoothed state.

    Returns (is_compliant, violations, unknown).

    Two deliberate rules, both fixing real defects:

      * UNKNOWN never counts as a violation. The old code inferred "NO" from the
        mere presence of a weak negative detection, so one spurious 0.56
        `no_helmet` could override a confident 0.92 `helmet`.
      * Only `required` classes affect compliance. Advisory classes (gloves,
        boots — the ones whose measured AP does not support automatic alerting)
        are reported but never flip the badge. Previously the on-screen badge
        and the CSV disagreed about this, so a person could be drawn green SAFE
        while a violation row was written for them.
    """
    advisory = advisory or []
    violations = [c for c in required
                  if c not in advisory and states.get(c) is State.ABSENT]
    unknown = [c for c in required
               if c not in advisory and states.get(c) is State.UNKNOWN]
    return (not violations), violations, unknown
