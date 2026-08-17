"""Single-stage PPE compliance inference: detect → track → smooth → judge.

Replaces the two-stage crop pipeline in `scripts/pipeline/run_pipeline.py`.
That design exists because the old SSD could only find PPE inside a person
crop, so every frame meant one person pass plus one pass per person, wrapped in
hand-tuned vertical zones, box-size limits and helmet-merge heuristics to patch
up what the crops broke. FCOS detects people and PPE together in one pass over
the whole frame, so nearly all of that machinery is unnecessary rather than
merely refactored — which is why this is a new module, not an edit to that one.

What is deliberately reused rather than re-derived:

  * `ppe.config`      — per-class thresholds calibrated from the PR curve
  * `ppe.tracking`    — Kalman + IoU/Hungarian tracker (MIT, replaced GPL SORT)
  * `ppe.smoothing`   — hysteresis voter and `compliance()`
  * `ppe.dataset`     — the SAME letterbox and normalisation used in training

That last one matters more than it looks. `normalise_chw` applies ImageNet
mean/std, and torchvision's `GeneralizedRCNNTransform` then normalises again
inside the model. Training did both, so inference must do both; "fixing" the
apparent double-normalisation here would silently shift the input distribution
away from what the weights were fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import AppConfig
from .dataset import letterbox, normalise_chw
from .models import build_ppe_model
from .smoothing import State, TemporalVoter, compliance
from .taxonomy import CLASSES, PPE_ITEMS
from .tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Detected:
    """One detection in ORIGINAL frame pixels."""
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass
class PersonResult:
    track_id: int
    box: tuple[float, float, float, float]
    states: dict[str, State]
    compliant: bool
    violations: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    items: list[Detected] = field(default_factory=list)


def containment(inner: tuple, outer: tuple) -> float:
    """Fraction of `inner`'s area that lies inside `outer`.

    Association is by containment rather than IoU because the two boxes are
    wildly different sizes: a glove inside a person box has an IoU around 0.01,
    which no sane IoU threshold would accept, while its containment is ~1.0.
    """
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    if area <= 0:
        return 0.0
    return (iw * ih) / area


def items_for_person(person_box: tuple, items: list[Detected],
                     min_containment: float = 0.5) -> list[Detected]:
    """PPE items belonging to one person.

    Each item goes to the person that contains it most, so two workers standing
    close together do not both claim the same helmet — the old pipeline's crop
    padding made exactly that mistake and it inflated compliance.
    """
    return [it for it in items
            if containment(it.box, person_box) >= min_containment]


def assign_items(persons: list[tuple], items: list[Detected],
                 min_containment: float = 0.5) -> list[list[Detected]]:
    """Exclusive assignment of each item to its best-containing person."""
    out: list[list[Detected]] = [[] for _ in persons]
    for it in items:
        best, best_c = -1, min_containment
        for i, pbox in enumerate(persons):
            c = containment(it.box, pbox)
            if c >= best_c:
                best, best_c = i, c
        if best >= 0:
            out[best].append(it)
    return out


class PPEPipeline:
    """Stateful per-stream pipeline. One instance per camera.

    Stateful because tracking and smoothing are: `Tracker` holds Kalman state
    and `TemporalVoter` holds the vote window. Sharing one instance across two
    streams would let one camera's track IDs and votes leak into the other's.
    """

    def __init__(self, cfg: AppConfig | None = None, device: str | None = None,
                 ckpt_path: str | None = None, infer_size: int | None = None) -> None:
        self.cfg = cfg or AppConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        path = Path(ckpt_path or self.cfg.detection.ppe_model)
        if not path.is_absolute():
            path = ROOT / path
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.arch = ckpt.get("arch", "fcos")
        self.trained_size = int(ckpt.get("size", self.cfg.detection.input_size))
        self.classes = list(ckpt.get("classes", CLASSES))

        # Inference resolution may exceed the training resolution. FCOS is
        # anchor-free and fully convolutional, so the weights carry no fixed
        # spatial assumption and a 640-trained checkpoint runs at 1280 with no
        # state-dict change. It matters on high-resolution sources: a 3840x2160
        # frame letterboxed to 640 is scaled by 0.167, turning a 200px worker
        # into 33px. On the sample 4K clip, raising this to 1280 took a hi-vis
        # vest from undetected to 0.598.
        #
        # Note both the letterbox AND the model's own GeneralizedRCNNTransform
        # must be set: the transform re-resizes to its own min_size/max_size, so
        # feeding a bigger canvas without changing it just resamples back down.
        self.size = int(infer_size or self.trained_size)

        self.model = build_ppe_model(self.arch, pretrained=False,
                                     min_size=self.size, max_size=self.size)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        t = self.cfg.tracking
        self.tracker = Tracker(max_age=t.max_age, min_hits=t.min_hits,
                               iou_threshold=t.iou_threshold,
                               max_scale_change=t.max_scale_change)
        s = self.cfg.smoothing
        self.voter = TemporalVoter(PPE_ITEMS, window=s.window, warmup=s.warmup,
                                   on_ratio=s.on_ratio, off_ratio=s.off_ratio)
        self.checkpoint = path

    # ── detection ──

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[Detected]:
        """All detections in one pass, in ORIGINAL frame pixels."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas, _, meta = letterbox(rgb, self.size)
        tensor = torch.from_numpy(normalise_chw(canvas)).to(self.device)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=self.device.type == "cuda"
                            and self.cfg.detection.half):
            out = self.model([tensor])[0]

        scale, dx, dy = meta["scale"], meta["dx"], meta["dy"]
        thresholds = self.cfg.detection.class_scores
        h, w = frame_bgr.shape[:2]

        dets: list[Detected] = []
        for box, score, label in zip(out["boxes"].float().cpu().numpy(),
                                     out["scores"].float().cpu().numpy(),
                                     out["labels"].cpu().numpy()):
            idx = int(label) - 1              # torchvision 1-based → ours
            if not (0 <= idx < len(self.classes)):
                continue
            name = self.classes[idx]
            if float(score) < thresholds.get(name, self.cfg.detection.ppe_score):
                continue
            # undo the letterbox: boxes came back in padded-canvas coordinates
            x1 = (box[0] - dx) / scale
            y1 = (box[1] - dy) / scale
            x2 = (box[2] - dx) / scale
            y2 = (box[3] - dy) / scale
            dets.append(Detected(name, float(score), (
                float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
                float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)))))
        return dets

    # ── person plausibility ──

    def keep_person(self, box: tuple, frame_shape: tuple) -> bool:
        """Frame-relative filters, so one set of numbers works at any resolution.

        The old pipeline used absolute pixels (>=25 wide, >=40 tall), which
        silently means something different on 640x480 than on 4K.
        """
        h, w = frame_shape[:2]
        pf = self.cfg.person_filter
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw <= 0 or bh <= 0:
            return False
        if bw < pf.min_width_frac * w or bh < pf.min_height_frac * h:
            return False
        if (bw * bh) > pf.max_area_frac * (w * h):
            return False
        return (bh / bw) >= pf.min_aspect

    # ── per frame ──

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[list[PersonResult],
                                                            list[Detected]]:
        dets = self.detect(frame_bgr)
        persons = [d for d in dets if d.label == "person"
                   and self.keep_person(d.box, frame_bgr.shape)]
        persons.sort(key=lambda d: -d.score)
        persons = persons[:self.cfg.detection.max_persons]
        items = [d for d in dets if d.label in PPE_ITEMS or d.label == "head"]

        if persons:
            arr = np.array([[*p.box, p.score] for p in persons], dtype=np.float32)
        else:
            arr = np.zeros((0, 5), dtype=np.float32)
        tracks = self.tracker.update(arr)          # [M,5] x1,y1,x2,y2,id

        track_boxes = [tuple(t[:4]) for t in tracks]
        assigned = assign_items(track_boxes, items)

        results: list[PersonResult] = []
        for (t, mine) in zip(tracks, assigned):
            tid = int(t[4])
            detected = {d.label for d in mine if d.label in PPE_ITEMS}
            self.voter.update(tid, detected)
            states = self.voter.states(tid)
            ok, violations, unknown = compliance(
                states, self.cfg.events.required_ppe,
                self.cfg.events.advisory_only)
            results.append(PersonResult(tid, tuple(float(v) for v in t[:4]),
                                        states, ok, violations, unknown, mine))

        # Drop vote history for tracks the tracker has retired, otherwise the
        # voter grows without bound on a long-running stream.
        self.voter.retain(self.tracker.active_ids)
        return results, dets

    def reset(self) -> None:
        self.tracker.reset()
        self.voter.reset()
