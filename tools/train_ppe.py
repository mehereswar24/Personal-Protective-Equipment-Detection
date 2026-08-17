"""
Train the PPE detector (FCOS / RetinaNet, COCO-pretrained, no YOLO).

Fixes carried over from the audit of the old training loop:
  * selection on **AP@0.5**, not validation loss (loss was dominated by the
    localisation term, so "best" checkpoints were chosen mostly on box
    regression rather than on detection quality)
  * warmup that actually runs (the old `_set_lr(epoch-1)` made epoch 1 use the
    full LR, so there was no warmup at all)
  * per-image class-presence masking, so unannotated classes never train as
    background
  * repeat-factor sampling for the 18:1 imbalance, instead of class weights
    that corrupted the focal modulator
  * seeds set and logged; AMP; EMA; grad clipping
  * the held-out test split is never touched here

Usage:
    python tools/train_ppe.py --epochs 24 --size 800 --batch 4
    python tools/train_ppe.py --arch retinanet --epochs 24        # A/B arm
    python tools/train_ppe.py --smoke                             # 2-min wiring check
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.dataset import PPEDataset, collate, make_sampler  # noqa: E402
from ppe.metrics import Detection, GroundTruth, evaluate, format_report  # noqa: E402
from ppe.models import (  # noqa: E402
    MaskedLossWrapper, ModelEMA, build_ppe_model, freeze_backbone_bn,
)
from ppe.taxonomy import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Long runs are launched detached with stdout redirected to a file, and a
# redirected stream on Windows defaults to cp1252, which cannot encode the
# arrows and bullets in this file's output. That turned a *successful* run into
# a UnicodeEncodeError on its very last line. Encoding is a recurring hazard
# here - see the header of requirements.txt for the UTF-16 version of it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True     # fixed input size → free speedup


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float,
          floor: float = 0.01) -> float:
    """Linear warmup → cosine decay, computed PER STEP.

    The old schedule was per-epoch and off by one, so warmup silently did
    nothing. Per-step also matters because this dataset is small.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (floor + (1 - floor) * cosine)


def stratified_eval_subset(records: list[dict], n_images: int,
                           seed: int = 42) -> list[int]:
    """Pick a fixed, source-proportional subset of a split for per-epoch eval.

    Full-split eval every epoch is affordable but not free, so the loop scores a
    subset. The obvious way to take one - the first N batches with
    shuffle=False - is actively misleading here: splits_v2.json is ordered by
    source dataset, so the first 240 val images were 100% dataset1. That is 11%
    of val and not even its largest source; dataset4, at 47%, was never scored
    at all. Every val mAP the training loop printed came from a single domain,
    reading 0.6385 where the full split says 0.7910.

    Two properties matter, and taking a plain shuffle only gives the first:

      * representative - sampled proportionally across every source dataset
      * *fixed* - the same images every epoch, so an epoch-to-epoch change is
        the model changing and not the sample changing. A reshuffling subset
        adds noise directly to the signal checkpoint selection runs on.
    """
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_dataset[rec["dataset"]].append(i)

    total = max(1, len(records))
    rng = random.Random(seed)
    picked: list[int] = []
    for _name, idxs in sorted(by_dataset.items()):
        share = max(1, round(n_images * len(idxs) / total))
        picked.extend(rng.sample(idxs, min(share, len(idxs))))

    # Shuffle BEFORE truncating: `picked` is grouped by dataset, so trimming it
    # in place would drop whole sources and reintroduce exactly the bias this
    # function exists to remove.
    rng.shuffle(picked)
    return sorted(picked[:n_images])


def load_resume(path: Path, model, ema, optim, scaler, steps_per_epoch: int):
    """Restore training state. Returns (start_epoch, step, best_ap).

    Two checkpoint shapes exist, because the *_last.pth full-state file was
    added only after two 9-hour runs were lost to interruptions:

      * full state  — optimiser, scaler and EMA all restored; training
        continues exactly where it stopped.
      * best-only   — just EMA weights (what `--out` has always written). Both
        the live model and the EMA are seeded from them and the optimiser
        restarts. Not a bit-exact resume, but it keeps the learned weights,
        which is the expensive part; AdamW re-accumulates its moments quickly.

    The LR schedule is driven by `step`, so it is advanced to the resumed
    epoch either way — otherwise a resumed run would re-run warmup and hit a
    trained model with a cold-start learning rate.
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    resumed_epoch = int(ckpt.get("epoch", 0))

    if "optim" in ckpt:
        model.load_state_dict(ckpt["model_raw"])
        ema.module.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        best = float(ckpt.get("best_map50", -1.0))
        print(f"resumed full state from {path.name}: epoch {resumed_epoch} "
              f"complete, best mAP@0.5 {best:.4f}")
        return resumed_epoch + 1, int(ckpt.get("step", 0)), best

    weights = ckpt["model"]
    model.load_state_dict(weights)
    ema.module.load_state_dict(weights)
    best = float(ckpt.get("val_map50", -1.0))
    print(f"warm-started from {path.name}: epoch {resumed_epoch} weights, "
          f"mAP@0.5 {best:.4f} (optimiser state restarts)")
    return resumed_epoch + 1, resumed_epoch * steps_per_epoch, best


@torch.no_grad()
def run_eval(model, loader, max_batches: int = 0) -> dict:
    """Decode predictions on a split and score them with the AP harness."""
    model.eval()
    dets: list[Detection] = []
    gts: list[GroundTruth] = []

    for i, (images, targets) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            image_id = tgt["image_id"]
            h = w = float(images[0].shape[-1])
            area = h * w
            for box, label in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()):
                gts.append(GroundTruth(image_id, CLASSES[int(label)], tuple(box),
                                       dataset=tgt["dataset"], img_area=area))
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, lab in zip(boxes, scores, labels):
                idx = int(lab) - 1                     # torchvision reserves 0 = background
                if 0 <= idx < len(CLASSES):
                    dets.append(Detection(image_id, CLASSES[idx], float(s), tuple(b)))

    return evaluate(dets, gts, CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="fcos", choices=["fcos", "retinanet"])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--eval-batches", type=int, default=60,
                    help="val batches per epoch check (0 = full val)")
    ap.add_argument("--out", default="models/ppe_fcos.pth")
    ap.add_argument("--resume", default="",
                    help="checkpoint to continue from. A *_last.pth restores "
                         "optimiser/scaler/epoch exactly; a best-only file "
                         "warm-starts from its weights.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify wiring end to end")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.size, args.eval_batches = 1, 512, 4

    set_seed(args.seed)
    print(f"device={DEVICE}  arch={args.arch}  size={args.size}  batch={args.batch}  seed={args.seed}")

    train_ds = PPEDataset("train", size=args.size, augment=True)
    val_ds = PPEDataset("val", size=args.size, augment=False)
    if args.smoke:
        train_ds.records = train_ds.records[:40]
        val_ds.records = val_ds.records[:16]
    print(f"train={len(train_ds)} val={len(val_ds)}")
    print(f"train class counts: {train_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=make_sampler(train_ds, args.seed),
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=True)
    # Score a fixed, source-proportional subset each epoch rather than the
    # first N batches, which were one dataset. 0 means the whole split.
    if args.eval_batches:
        eval_idxs = stratified_eval_subset(
            val_ds.records, args.eval_batches * args.batch, args.seed)
        eval_ds = Subset(val_ds, eval_idxs)
        covered = Counter(val_ds.records[i]["dataset"] for i in eval_idxs)
        print(f"per-epoch eval subset: {len(eval_idxs)} images, "
              f"{dict(sorted(covered.items()))}")
    else:
        eval_ds = val_ds
        print(f"per-epoch eval: full val split ({len(val_ds)} images)")

    val_loader = DataLoader(
        eval_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_ppe_model(args.arch, pretrained=True,
                            min_size=args.size, max_size=args.size).to(DEVICE)
    freeze_backbone_bn(model)
    wrapped = MaskedLossWrapper(model)
    ema = ModelEMA(model, decay=args.ema_decay)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone") else head_params).append(p)
    optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.lr}],
        weight_decay=0.05)
    base_lrs = [args.backbone_lr, args.lr]

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, max(1, steps_per_epoch))       # ~1 epoch of warmup

    best_ap = -1.0
    history = []
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.with_name(out_path.stem + "_last.pth")
    step, start_epoch, oom_skips = 0, 1, 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            print(f"--resume: {resume_path} does not exist", file=sys.stderr)
            return 2
        start_epoch, step, best_ap = load_resume(
            resume_path, model, ema, optim, scaler, steps_per_epoch)
        if start_epoch > args.epochs:
            print(f"nothing to do: checkpoint is already at epoch "
                  f"{start_epoch - 1} of {args.epochs}")
            return 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        freeze_backbone_bn(model)          # keep BN frozen even in train mode
        t0, running = time.perf_counter(), 0.0

        for images, targets in train_loader:
            for g, base in zip(optim.param_groups, base_lrs):
                g["lr"] = lr_at(step, total_steps, warmup_steps, base)

            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]

            try:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=DEVICE.type == "cuda"):
                    losses = wrapped(images, targets)
                    loss = sum(losses.values())

                if not torch.isfinite(loss):
                    print(f"  ! non-finite loss at step {step}, skipping batch")
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    continue

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                ema.update(model)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # This GPU also drives the desktop, so a browser or Electron app
                # can take VRAM mid-run; a 9-hour run died this way at epoch 3.
                # One transient spike should cost a batch, not the whole run.
                # Best-effort: the allocator's OutOfMemoryError recovers cleanly,
                # but a driver-level "CUDA error: out of memory" can leave the
                # context unusable - hence the cap, which re-raises rather than
                # grinding on and reporting a silently degraded model.
                if "out of memory" not in str(exc).lower():
                    raise
                oom_skips += 1
                optim.zero_grad(set_to_none=True)
                del images, targets
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    # empty_cache() failing means this is a driver-level CUDA
                    # OOM, not the allocator's recoverable one: the context is
                    # dead and every later call would fail too. Say so plainly
                    # rather than raising a confusing secondary exception from
                    # inside the handler.
                    print(f"  ! CUDA context unrecoverable at step {step} - "
                          f"the GPU is out of memory at the driver level. "
                          f"Free VRAM (other GPU applications) or lower "
                          f"--batch, then --resume from "
                          f"{last_path.name}.", file=sys.stderr)
                    raise
                print(f"  ! CUDA OOM at step {step}, batch skipped "
                      f"({oom_skips} so far)")
                if oom_skips >= 50:
                    print("  ! too many OOM skips - rerun with a smaller "
                          "--batch or close other GPU applications",
                          file=sys.stderr)
                    raise
                step += 1
                continue

            running += float(loss.detach())
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps} "
                      f"loss {running / max(1, step % steps_per_epoch or steps_per_epoch):.4f} "
                      f"lr {optim.param_groups[1]['lr']:.2e}")

        train_loss = running / steps_per_epoch
        # No max_batches: val_loader is already the stratified subset, and
        # truncating it again would take the first N of it - the same
        # first-N-in-split-order bias, one layer up.
        res = run_eval(ema.module, val_loader)
        ap50 = res["map50"]
        dt = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs}  loss {train_loss:.4f}  "
              f"val mAP@0.5 {ap50:.4f}  ({dt:.0f}s)")
        for c, m in res["per_class"].items():
            if m["n_gt"]:
                print(f"    {c:8s} AP50 {m['ap50']:.4f}  (gt {m['n_gt']})")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_map50": ap50})

        if ap50 > best_ap:                 # selection on AP, never on loss
            best_ap = ap50
            torch.save({
                "model": ema.module.state_dict(),
                "arch": args.arch, "classes": CLASSES, "size": args.size,
                "epoch": epoch, "val_map50": ap50, "seed": args.seed,
            }, out_path)
            print(f"    ** saved {out_path.name} (mAP@0.5 {ap50:.4f})")

        # Full training state, every epoch regardless of AP, so an interruption
        # costs one epoch instead of the whole run. Overwritten in place.
        torch.save({
            "model": ema.module.state_dict(),
            "model_raw": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "arch": args.arch, "classes": CLASSES, "size": args.size,
            "epoch": epoch, "step": step, "best_map50": best_ap,
            "val_map50": ap50, "seed": args.seed, "oom_skips": oom_skips,
        }, last_path)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"train_{args.arch}.json").write_text(
        json.dumps({"args": vars(args), "best_map50": best_ap, "history": history},
                   indent=1), encoding="utf-8")
    print(f"\nDone. best val mAP@0.5 = {best_ap:.4f}  →  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
