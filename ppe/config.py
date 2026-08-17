"""
Runtime configuration.

Every operational value in the old pipeline was a module-level literal — model
paths, all thresholds, NMS, tracker params, zone bands, dedup window, and person
size filters expressed in **absolute pixels** (so behaviour silently changed
between a 640x480 webcam and a 4K camera). Per-site tuning meant editing source
and forking the file per camera.

Now: YAML + environment overrides, with per-camera sections, and the size
filters expressed as fractions of the frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DetectionConfig:
    ppe_model: str = "models/ppe_fcos_v2.pth"
    person_score: float = 0.5
    ppe_score: float = 0.35          # global floor; per-class below
    nms_iou: float = 0.5
    max_persons: int = 20
    input_size: int = 640
    half: bool = True                # fp16 inference
    # Per-class score thresholds, calibrated from the PR curve by
    # `tools/calibrate_thresholds.py` — measured, not hand-picked.
    #
    # Fitted on **val** and verified on **test** with the values frozen
    # (reports/thresholds_fcos.json), maximising F2 rather than F1: a missed
    # violation is a safety failure, a false positive costs a review click, and
    # F1 would trade the expensive error for the cheap one. On test these hold
    # recall 0.86–0.94 while cutting 47,504 raw detections to 13,009 for 11,186
    # real objects.
    #
    # Regenerate after any retrain — they are specific to this checkpoint:
    #     python tools/calibrate_thresholds.py --ckpt models/<new>.pth
    class_scores: dict[str, float] = field(default_factory=lambda: {
        "person": 0.473, "head": 0.425, "helmet": 0.461,
        "vest": 0.484, "gloves": 0.410, "boots": 0.342,
    })


@dataclass
class TrackingConfig:
    max_age: int = 30                # NOT 2 — see ppe/tracking.py for why
    min_hits: int = 3
    iou_threshold: float = 0.3
    max_scale_change: float = 4.0


@dataclass
class SmoothingConfig:
    window: int = 9
    warmup: int = 4                  # frames before a track is trusted at all
    on_ratio: float = 0.6            # vote fraction to call an item PRESENT
    off_ratio: float = 0.3           # below this → ABSENT (hysteresis band)


@dataclass
class PersonFilterConfig:
    """Frame-relative, not absolute pixels."""
    min_width_frac: float = 0.02
    min_height_frac: float = 0.05
    max_area_frac: float = 0.60
    min_aspect: float = 0.8          # height/width; people are taller than wide


@dataclass
class EventsConfig:
    enabled: bool = True
    dir: str = "output/violations"
    dedup_seconds: float = 60.0
    snapshot: bool = True
    snapshot_quality: int = 85
    retention_days: int = 30
    required_ppe: list[str] = field(default_factory=lambda: ["helmet", "vest"])
    # Classes that must never raise an automatic alert. Per the accuracy work,
    # gloves are unreliable (AP ~0.49 pre-retrain) and mask has no in-domain
    # data at all — surfacing them as alerts would train operators to ignore
    # the system.
    advisory_only: list[str] = field(default_factory=lambda: ["gloves", "boots"])


@dataclass
class StreamConfig:
    name: str = "cam"
    source: str = "0"
    enabled: bool = True
    skip_frames: int = 0
    reconnect_max_backoff: float = 30.0
    record: bool = False
    record_dir: str = "output/recordings"
    record_segment_minutes: int = 15   # segmented, so disks don't fill silently


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    person_filter: PersonFilterConfig = field(default_factory=PersonFilterConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    streams: list[StreamConfig] = field(default_factory=lambda: [StreamConfig()])
    device: str = "auto"
    log_level: str = "INFO"
    api_port: int = 8080

    # ── loading ──
    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg = cls()
        path = Path(path) if path else (ROOT / "config.yaml")
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cfg = cls._from_dict(raw)
        cfg._apply_env()
        return cfg

    @classmethod
    def _from_dict(cls, raw: dict) -> "AppConfig":
        def build(dc, data):
            if not data:
                return dc()
            known = {f.name for f in dc.__dataclass_fields__.values()}
            return dc(**{k: v for k, v in data.items() if k in known})

        streams = [build(StreamConfig, s) for s in raw.get("streams", [])] or [StreamConfig()]
        return cls(
            detection=build(DetectionConfig, raw.get("detection")),
            tracking=build(TrackingConfig, raw.get("tracking")),
            smoothing=build(SmoothingConfig, raw.get("smoothing")),
            person_filter=build(PersonFilterConfig, raw.get("person_filter")),
            events=build(EventsConfig, raw.get("events")),
            streams=streams,
            device=raw.get("device", "auto"),
            log_level=raw.get("log_level", "INFO"),
            api_port=int(raw.get("api_port", 8080)),
        )

    def _apply_env(self) -> None:
        """PPE_* env vars win over the file — needed for container deploys."""
        if v := os.getenv("PPE_MODEL"):
            self.detection.ppe_model = v
        if v := os.getenv("PPE_DEVICE"):
            self.device = v
        if v := os.getenv("PPE_LOG_LEVEL"):
            self.log_level = v
        if v := os.getenv("PPE_INPUT_SIZE"):
            self.detection.input_size = int(v)
        if v := os.getenv("PPE_EVENTS_DIR"):
            self.events.dir = v
        if v := os.getenv("PPE_API_PORT"):
            self.api_port = int(v)
        if v := os.getenv("PPE_SOURCES"):
            # comma-separated RTSP urls / camera indices
            self.streams = [StreamConfig(name=f"cam{i}", source=s.strip())
                            for i, s in enumerate(v.split(",")) if s.strip()]

    def for_stream(self, stream: StreamConfig) -> "AppConfig":
        """A per-camera view (streams differ in framing, so thresholds differ)."""
        return replace(self, streams=[stream])
