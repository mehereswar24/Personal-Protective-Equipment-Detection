"""Measure host RAM used by the training dataloader before committing to a run.

This machine has ~14 GB of RAM and trains with roughly 1 GB to spare, so the
dataloader — not the GPU — is what kills long runs. On 2026-08-14 a 640px /
batch 4 / 4-worker run died of MemoryError in a worker ~90 seconds in, after the
model had already been built and moved to the GPU.

The cost is dominated by a fixed per-worker overhead, not by pixels: Windows
spawns workers rather than forking, so each one re-imports torch for ~500 MB
before it touches a sample. Measured here, 640px / batch 4:

    workers=2  ->  1.6 GB process tree
    workers=4  ->  2.7 GB process tree

So `--workers` is the first dial to turn when memory is tight, ahead of batch
size or image size — it costs throughput but not model quality, which the other
two do.

Run this before a long training job, with the same flags you intend to train
with, and check the peak against what the machine actually has free:

    python tools/check_loader_memory.py --size 640 --batch 4 --workers 4

It exits non-zero if the projected peak leaves less than --reserve GB free, so
it can gate a training run in a script.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.dataset import PPEDataset, collate, make_sampler  # noqa: E402


def tree_rss_gb(proc: psutil.Process) -> float:
    """RSS of this process plus its dataloader workers."""
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass                    # worker exited mid-measurement; skip it
    return total / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batches", type=int, default=60,
                    help="batches to draw; RSS plateaus well before 60")
    ap.add_argument("--split", default="train")
    ap.add_argument("--reserve", type=float, default=0.5,
                    help="GB that must remain free, else exit non-zero")
    args = ap.parse_args()

    vm = psutil.virtual_memory()
    print(f"host RAM: {vm.total / 1e9:.1f} GB total, {vm.available / 1e9:.1f} GB available")
    print(f"config:   size={args.size} batch={args.batch} workers={args.workers}")

    ds = PPEDataset(args.split, size=args.size, augment=args.split == "train")
    loader = DataLoader(
        ds, batch_size=args.batch,
        sampler=make_sampler(ds, 42) if args.split == "train" else None,
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=True)
    print(f"{args.split}={len(ds)} batches_available={len(loader)}")

    proc = psutil.Process(os.getpid())
    peak, floor = 0.0, vm.available / 1e9
    t0 = time.perf_counter()

    for i, (images, targets) in enumerate(loader):
        if i >= args.batches:
            break
        # Touch the batch the way training does, so the measurement includes
        # anything materialised lazily rather than just the allocation.
        assert images.shape == (args.batch, 3, args.size, args.size), images.shape
        assert torch.isfinite(images).all(), f"non-finite pixels in batch {i}"
        peak = max(peak, tree_rss_gb(proc))
        floor = min(floor, psutil.virtual_memory().available / 1e9)
        if (i + 1) % 20 == 0:
            print(f"  batch {i + 1}/{args.batches}  tree_rss {tree_rss_gb(proc):.2f} GB  "
                  f"host_free {psutil.virtual_memory().available / 1e9:.2f} GB")

    dt = time.perf_counter() - t0
    print(f"\npeak process tree: {peak:.2f} GB")
    print(f"least free RAM:    {floor:.2f} GB")
    print(f"throughput:        {args.batches / dt:.1f} batch/s "
          f"({args.batch * args.batches / dt:.0f} img/s)")

    if floor < args.reserve:
        print(f"\nTOO TIGHT: dropped to {floor:.2f} GB free, below the "
              f"{args.reserve:.2f} GB reserve.\nRetry with --workers "
              f"{max(0, args.workers - 2)}, or close other applications.")
        return 1

    print(f"\nOK: stayed above the {args.reserve:.2f} GB reserve. Safe to train "
          "at this configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
