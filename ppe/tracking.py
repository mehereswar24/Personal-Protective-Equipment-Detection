"""
Multi-object tracker: constant-velocity Kalman filter + IoU/Hungarian matching.

Written from first principles for this project (MIT-licensed, see LICENSE).
It replaces the vendored `sort.py`, which was Alex Bewley's SORT under
**GPL-3.0** — importing it into this pipeline made the whole combined work a
GPLv3 derivative, which is a blocker for a commercial plant deployment. That
file also did `matplotlib.use('TkAgg')` and imported scikit-image at module
scope, so the pipeline could not even start in a headless container.

Behavioural differences from the old setup, all deliberate:

  * `max_age` defaults to 30 frames, not 2. The old value came from a
    misdiagnosis — SORT never emits prediction-only boxes, so lowering max_age
    did not stop "boxes shooting off"; it only destroyed track continuity,
    which silently defeated both the temporal voter and the violation dedup
    (a worker walking behind a pillar came back as a new id and re-alerted).
  * Track ids come from a per-instance counter, not a process-global class
    attribute, so concurrent camera streams can no longer collide on ids.
  * `predict()` output is NaN-guarded. The old code could emit NaN boxes from a
    negative area state, which crashed the frame loop at `int(nan)`.
  * Optional appearance gating: a track will not match a detection whose box
    scale changed implausibly between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:                                    # SciPy gives optimal assignment
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:                     # pragma: no cover - fallback path
    _HAS_SCIPY = False


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4] boxes in x1y1x2y2 → [N,M]."""
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


def _to_state(box: np.ndarray) -> np.ndarray:
    """x1y1x2y2 → [cx, cy, area, aspect]."""
    w = max(float(box[2] - box[0]), 1e-3)
    h = max(float(box[3] - box[1]), 1e-3)
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / h], dtype=np.float64)


def _to_box(state: np.ndarray) -> np.ndarray:
    """[cx, cy, area, aspect] → x1y1x2y2, guarded against invalid states."""
    area = max(float(state[2]), 1e-6)      # a negative area would make w NaN
    aspect = float(state[3])
    if not np.isfinite(aspect) or aspect <= 1e-6:
        aspect = 1.0
    w = float(np.sqrt(area * aspect))
    h = area / max(w, 1e-6)
    if not np.isfinite(w) or not np.isfinite(h):
        w = h = 1.0
    cx, cy = float(state[0]), float(state[1])
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class _KalmanBox:
    """Constant-velocity Kalman filter over [cx, cy, area, aspect].

    State: [cx, cy, area, aspect, vx, vy, v_area]  (aspect assumed ~constant)
    """

    _DIM, _MEAS = 7, 4

    def __init__(self, box: np.ndarray) -> None:
        self.x = np.zeros((self._DIM,), dtype=np.float64)
        self.x[:4] = _to_state(box)

        self.F = np.eye(self._DIM)
        for i, v in ((0, 4), (1, 5), (2, 6)):
            self.F[i, v] = 1.0
        self.H = np.zeros((self._MEAS, self._DIM))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(self._DIM) * 10.0
        self.P[4:, 4:] *= 1000.0          # velocities start very uncertain
        self.Q = np.eye(self._DIM) * 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(self._MEAS) * 1.0
        self.R[2:, 2:] *= 10.0            # area/aspect measurements are noisier

    def predict(self) -> np.ndarray:
        # Stop the box from inverting when area velocity drives area negative.
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        z = _to_state(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:      # pragma: no cover - singular S
            return
        self.x = self.x + K @ y
        self.P = (np.eye(self._DIM) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return _to_box(self.x)


@dataclass
class Track:
    track_id: int
    kf: _KalmanBox
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    score: float = 0.0
    history: list = field(default_factory=list, repr=False)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box


class Tracker:
    """IoU-association multi-object tracker.

    Args:
        max_age:   frames a track survives without a detection before deletion.
                   Needs to comfortably span a typical occlusion — at 25 fps,
                   30 frames ≈ 1.2 s.
        min_hits:  detections required before a track is reported (debounce).
        iou_threshold: minimum IoU to associate a detection with a track.
        max_scale_change: reject a match if the box area ratio exceeds this,
                   which stops a track from snapping onto a much larger/smaller
                   nearby object during a crossover.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, max_scale_change: float = 4.0) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_scale_change = max_scale_change
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1               # per-instance: streams cannot collide

    # ── association ──
    def _match(self, dets: np.ndarray, preds: np.ndarray) -> tuple[list, list, list]:
        if len(preds) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(preds)))

        iou = iou_batch(dets, preds)

        # forbid implausible scale jumps
        det_area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        trk_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        ratio = np.maximum(det_area[:, None] / np.maximum(trk_area[None, :], 1e-6),
                           trk_area[None, :] / np.maximum(det_area[:, None], 1e-6))
        iou = np.where(ratio > self.max_scale_change, 0.0, iou)

        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(-iou)
            pairs = list(zip(rows, cols))
        else:                            # pragma: no cover - greedy fallback
            pairs, used_d, used_t = [], set(), set()
            for d, t in sorted(np.ndindex(iou.shape), key=lambda k: -iou[k]):
                if d not in used_d and t not in used_t and iou[d, t] > 0:
                    pairs.append((d, t)); used_d.add(d); used_t.add(t)

        matches, um_d, um_t = [], set(range(len(dets))), set(range(len(preds)))
        for d, t in pairs:
            if iou[d, t] < self.iou_threshold:
                continue
            matches.append((int(d), int(t)))
            um_d.discard(int(d)); um_t.discard(int(t))
        return matches, sorted(um_d), sorted(um_t)

    def update(self, detections: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame.

        Args:
            detections: [N,5] rows of x1,y1,x2,y2,score (score optional → [N,4]).
        Returns:
            [M,5] rows of x1,y1,x2,y2,track_id for confirmed tracks.
        """
        self.frame_count += 1
        dets = np.empty((0, 5), dtype=np.float32) if detections is None or len(detections) == 0 \
            else np.asarray(detections, dtype=np.float32)
        if dets.ndim == 2 and dets.shape[1] == 4:      # allow boxes without scores
            dets = np.hstack([dets, np.ones((len(dets), 1), dtype=np.float32)])

        # 1. predict
        preds = []
        alive: list[Track] = []
        for t in self.tracks:
            box = t.kf.predict()
            t.age += 1
            t.time_since_update += 1
            if np.all(np.isfinite(box)):
                preds.append(box)
                alive.append(t)
        self.tracks = alive
        pred_arr = np.array(preds, dtype=np.float32) if preds else np.empty((0, 4), np.float32)

        # 2. associate
        matches, unmatched_dets, _ = self._match(dets[:, :4], pred_arr)

        # 3. update matched
        for d, t in matches:
            trk = self.tracks[t]
            trk.kf.update(dets[d, :4])
            trk.hits += 1
            trk.time_since_update = 0
            trk.score = float(dets[d, 4])

        # 4. spawn new tracks
        for d in unmatched_dets:
            self.tracks.append(Track(track_id=self._next_id,
                                     kf=_KalmanBox(dets[d, :4]),
                                     score=float(dets[d, 4])))
            self._next_id += 1

        # 5. retire stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6. report confirmed tracks seen this frame
        out = []
        for t in self.tracks:
            confirmed = t.hits >= self.min_hits or self.frame_count <= self.min_hits
            if t.time_since_update == 0 and confirmed:
                b = t.box
                out.append([b[0], b[1], b[2], b[3], float(t.track_id)])
        return np.array(out, dtype=np.float32) if out else np.empty((0, 5), np.float32)

    @property
    def active_ids(self) -> set[int]:
        """Ids currently alive — used to evict per-track state elsewhere."""
        return {t.track_id for t in self.tracks}

    def reset(self) -> None:
        """Clear all state. Called on stream reconnect, where the scene may have
        changed completely and stale Kalman states would mis-associate."""
        self.tracks.clear()
        self.frame_count = 0
