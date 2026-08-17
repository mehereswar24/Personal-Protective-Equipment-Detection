"""Run the FCOS compliance pipeline over a video or image and annotate it.

The legacy `scripts/pipeline/run_pipeline.py` still drives the OLD two-stage
SSD checkpoints and the old `no_helmet`/`no_vest` taxonomy; it cannot load the
FCOS model at all. This is the equivalent entry point for the current stack.

Usage:
    python tools/run_video.py vid.mp4 -o output/annotated.mp4
    python tools/run_video.py vid.mp4 --size 1280          # high-res source
    python tools/run_video.py frame.jpg -o out.jpg
    python tools/run_video.py vid.mp4 --raw                # no track/vote gating
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.config import AppConfig  # noqa: E402
from ppe.pipeline import PPEPipeline  # noqa: E402
from ppe.smoothing import State  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOURS = {                      # BGR
    "person": (255, 180, 60), "head": (60, 200, 255), "helmet": (60, 220, 60),
    "vest": (0, 200, 255), "gloves": (220, 120, 255), "boots": (200, 200, 60),
}
OK_COLOUR, BAD_COLOUR, UNK_COLOUR = (80, 220, 80), (60, 60, 240), (150, 150, 150)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def draw(frame, results, dets, show_raw: bool, thickness: int = 2):
    """Annotate in place. Raw detections are thin, tracked people are thick."""
    if show_raw:
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            c = COLOURS.get(d.label, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 1)
            cv2.putText(frame, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r.box)
        colour = BAD_COLOUR if r.violations else (
            UNK_COLOUR if r.unknown and not r.compliant else OK_COLOUR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        bits = [f"#{r.track_id}"]
        for label, st in r.states.items():
            mark = {State.PRESENT: "+", State.ABSENT: "-",
                    State.UNKNOWN: "?"}[st]
            bits.append(f"{mark}{label}")
        text = " ".join(bits)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1),
                      colour, -1)
        cv2.putText(frame, text, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--size", type=int, default=0,
                    help="inference resolution; default = training size. "
                         "Raise it for high-resolution sources.")
    ap.add_argument("--raw", action="store_true",
                    help="also draw every raw detection, ungated by tracking")
    ap.add_argument("--limit", type=int, default=0, help="max frames")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"no such input: {src}", file=sys.stderr)
        return 2

    cfg = AppConfig()
    pipe = PPEPipeline(cfg, ckpt_path=args.ckpt or None,
                       infer_size=args.size or None)
    print(f"model  : {pipe.checkpoint.name} (trained @{pipe.trained_size}, "
          f"running @{pipe.size})")
    print(f"thresh : {cfg.detection.class_scores}")
    print(f"required PPE: {cfg.events.required_ppe}  "
          f"advisory: {cfg.events.advisory_only}")

    # ── single image ──
    if src.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(src))
        results, dets = pipe.process_frame(frame)
        print(f"\n{len(dets)} detections: "
              f"{dict(Counter(d.label for d in dets))}")
        print(f"{len(results)} tracked people "
              f"(min_hits={cfg.tracking.min_hits} means a still image "
              f"usually yields 0)")
        out = Path(args.output or ROOT / "output" / f"{src.stem}_annotated.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw(frame, results, dets, True))
        print(f"wrote {out}")
        return 0

    # ── video ──
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"cannot open {src}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"input  : {w}x{h} {total} frames @ {fps:.1f}fps")

    out_path = Path(args.output or ROOT / "output" / f"{src.stem}_annotated.mp4")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    seen = Counter()
    people_seen: set[int] = set()
    violation_frames = 0
    n = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % args.stride:
            n += 1
            continue
        results, dets = pipe.process_frame(frame)
        seen.update(d.label for d in dets)
        people_seen.update(r.track_id for r in results)
        if any(r.violations for r in results):
            violation_frames += 1
        writer.write(draw(frame, results, dets, args.raw))
        n += 1
        if args.limit and n >= args.limit:
            break
        if n % 50 == 0:
            print(f"  {n}/{total}  {n / (time.perf_counter() - t0):.1f} fps",
                  flush=True)

    cap.release()
    writer.release()
    dt = time.perf_counter() - t0
    print(f"\nprocessed {n} frames in {dt:.0f}s ({n / max(dt, 1e-9):.1f} fps)")
    print(f"detections by class: {dict(seen)}")
    print(f"distinct tracked people: {len(people_seen)}")
    print(f"frames with a violation: {violation_frames}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
