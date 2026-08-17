"""
Build a LEAKAGE-FREE train/val/test split over the PPE datasets.

The audit found ~49% of the old val/test were augmented duplicates of training
images: Roboflow exports each source photo several times as
`<stem>_jpg.rf.<md5>.jpg`, and the old split was by file path, so copies of the
same photo landed on both sides. Every accuracy number computed on that split
was therefore partly a training score.

This script fixes that by splitting on GROUPS, not files:

  1. group key   = Roboflow stem with the `_jpg.rf.<md5>` suffix stripped
  2. union-find  = merge groups whose images are perceptual-hash duplicates
                   (catches the same photo appearing in two different datasets)
  3. split       = greedy multi-label stratification over (dataset, classes)
                   so every split keeps a similar class mix
  4. freeze      = write JSON + a content hash so the split can never silently
                   drift between experiments

It also records, per dataset, WHICH classes that dataset actually annotates.
Training uses this to mask unsupervised classes out of the loss, so an
unannotated helmet in a boots-only dataset is treated as "unknown", not as
"background" (the audit's #2 finding).

Usage:
    python tools/build_splits.py                 # build + report
    python tools/build_splits.py --report-only   # just show leakage stats
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe.taxonomy import (  # noqa: E402
    CLASSES, DATASET_DIRS, MIN_ANNS_FOR_SUPERVISED, is_known_label, map_label,
)

ROOT = Path(__file__).resolve().parents[1]
PPE_ROOT = ROOT / "data" / "ppe"
OUT_PATH = ROOT / "data" / "ppe" / "splits_v2.json"

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Roboflow augmentation suffix: `foo_jpg.rf.<32 hex>.jpg`
RF_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]{6,}$", re.IGNORECASE)


# ── grouping ────────────────────────────────────────────────────────────────

def group_key(img_path: Path) -> str:
    """Collapse Roboflow's augmented copies of one photo onto one key."""
    stem = img_path.stem
    stem = RF_SUFFIX.sub("", stem)
    # Some exports also append `_<n>` variants of the same frame.
    return stem.lower()


def phash(img_path: Path, size: int = 8) -> str | None:
    """Cheap average-hash. Used only to merge duplicates ACROSS datasets."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        bits = (small > small.mean()).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ── parsing ─────────────────────────────────────────────────────────────────

def find_image_for(xml_path: Path) -> Path | None:
    for ext in IMG_EXTS:
        cand = xml_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def parse_xml(xml_path: Path, unknown: collections.Counter) -> tuple | None:
    """VOC XML → (width, height, [(class, x1, y1, x2, y2), ...])."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        print(f"  ! malformed XML skipped: {xml_path.name} ({exc})")
        return None

    size = root.find("size")
    if size is None:
        return None
    width, height = int(float(size.findtext("width", 0))), int(float(size.findtext("height", 0)))
    if width <= 0 or height <= 0:
        return None

    boxes = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        if not is_known_label(raw):
            unknown[raw] += 1          # loud, not silently dropped
            continue
        label = map_label(raw)
        if label is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = max(0, int(float(bb.findtext("xmin"))))
            y1 = max(0, int(float(bb.findtext("ymin"))))
            x2 = min(width, int(float(bb.findtext("xmax"))))
            y2 = min(height, int(float(bb.findtext("ymax"))))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((label, x1, y1, x2, y2))
    return width, height, boxes


def collect_records() -> tuple[list[dict], collections.Counter]:
    records: list[dict] = []
    unknown: collections.Counter = collections.Counter()

    for ds in DATASET_DIRS:
        ds_dir = PPE_ROOT / ds
        if not ds_dir.exists():
            print(f"  ! missing dataset dir: {ds_dir}")
            continue
        xmls = sorted(ds_dir.rglob("*.xml"))
        kept = 0
        for xml_path in xmls:
            img_path = find_image_for(xml_path)
            if img_path is None:
                continue
            parsed = parse_xml(xml_path, unknown)
            if parsed is None:
                continue
            width, height, boxes = parsed
            records.append({
                "dataset": ds,
                # store POSIX-relative so the split file is OS-portable
                # (the old ppe_splits.json baked in Windows backslash paths)
                "img": img_path.relative_to(ROOT).as_posix(),
                "w": width,
                "h": height,
                "boxes": boxes,
                "group": f"{group_key(img_path)}",
            })
            kept += 1
        print(f"  {ds:10s} {kept:6d} images")
    return records, unknown


# ── stratified group split ──────────────────────────────────────────────────

def split_groups(groups: dict[str, list[dict]]) -> dict[str, str]:
    """Greedy multi-label stratification over (dataset, class-presence).

    Sorts groups by rarest signature first, then always assigns to whichever
    split is furthest below its quota for that signature. This keeps rare
    combinations (e.g. gloves) proportionally represented instead of landing
    entirely in one split, which a random split does badly at this scale.
    """
    import random
    rng = random.Random(SEED)

    def signature(recs: list[dict]) -> tuple:
        classes = frozenset(b[0] for r in recs for b in r["boxes"])
        return (recs[0]["dataset"], tuple(sorted(classes)))

    by_sig: dict[tuple, list[str]] = collections.defaultdict(list)
    for gid, recs in groups.items():
        by_sig[signature(recs)].append(gid)

    assignment: dict[str, str] = {}
    # rarest signatures first
    for sig in sorted(by_sig, key=lambda s: len(by_sig[s])):
        gids = by_sig[sig]
        rng.shuffle(gids)
        counts = {s: 0 for s in SPLIT_FRACTIONS}
        for gid in gids:
            # pick the split with the largest deficit vs its target share
            target = {s: SPLIT_FRACTIONS[s] * (sum(counts.values()) + 1) for s in counts}
            pick = max(counts, key=lambda s: target[s] - counts[s])
            assignment[gid] = pick
            counts[pick] += 1
    return assignment


# ── supervision map ─────────────────────────────────────────────────────────

def derive_supervision(records: list[dict]) -> dict[str, list[str]]:
    """Which classes does each dataset actually annotate?

    Anything below MIN_ANNS_FOR_SUPERVISED is treated as NOT annotated — e.g.
    dataset3 has 2 glove boxes, which is noise, not supervision. Training masks
    non-supervised classes out of the loss for images from that dataset.
    """
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        for b in r["boxes"]:
            counts[r["dataset"]][b[0]] += 1
    return {
        ds: sorted(c for c in CLASSES if counts[ds][c] >= MIN_ANNS_FOR_SUPERVISED)
        for ds in sorted(counts)
    }


# ── reporting ───────────────────────────────────────────────────────────────

def leakage_report(records: list[dict], assignment: dict[str, str]) -> None:
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_split[assignment[r["group"]]].append(r)

    print("\n── split sizes ──")
    for s in ("train", "val", "test"):
        recs = by_split[s]
        n_box = sum(len(r["boxes"]) for r in recs)
        n_grp = len({r["group"] for r in recs})
        print(f"  {s:5s} {len(recs):6d} images  {n_grp:6d} groups  {n_box:7d} boxes")

    # the number that matters: does any source photo appear on both sides?
    print("\n── leakage check (group overlap between splits) ──")
    gs = {s: {r["group"] for r in by_split[s]} for s in by_split}
    clean = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = gs.get(a, set()) & gs.get(b, set())
        flag = "OK" if not overlap else "LEAK"
        if overlap:
            clean = False
        print(f"  {a:5s} ∩ {b:5s} = {len(overlap):5d} groups   [{flag}]")
    print(f"  → {'no leakage' if clean else 'LEAKAGE PRESENT'}")

    print("\n── per-class boxes per split ──")
    print(f"  {'class':10s} {'train':>8s} {'val':>7s} {'test':>7s}")
    for c in CLASSES:
        row = [sum(1 for r in by_split[s] for b in r["boxes"] if b[0] == c)
               for s in ("train", "val", "test")]
        print(f"  {c:10s} {row[0]:8d} {row[1]:7d} {row[2]:7d}")


def old_split_leakage() -> None:
    """Quantify leakage in the EXISTING split, for the before/after record."""
    old = PPE_ROOT / "merged" / "ppe_splits.json"
    if not old.exists():
        return
    try:
        data = json.loads(old.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  (could not read old split: {exc})")
        return
    keyed = {}
    for split in ("train", "val", "test"):
        entries = data.get(split, [])
        paths = [e.get("img_path", e) if isinstance(e, dict) else e for e in entries]
        keyed[split] = {group_key(Path(str(p).replace("\\", "/"))) for p in paths}
    print("\n── OLD split leakage (for comparison) ──")
    for a, b in (("train", "val"), ("train", "test")):
        if keyed.get(a) and keyed.get(b):
            ov = keyed[a] & keyed[b]
            pct = 100.0 * len(ov) / max(1, len(keyed[b]))
            print(f"  {a} ∩ {b}: {len(ov)} groups  ({pct:.1f}% of {b} leaked from {a})")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-phash", action="store_true",
                    help="skip cross-dataset perceptual-hash dedup (faster)")
    args = ap.parse_args()

    print("Scanning datasets (dataset6 excluded: mask-only selfie data)...")
    records, unknown = collect_records()
    if not records:
        print("No records found — is data/ppe populated?")
        return 1
    print(f"  total {len(records)} images, "
          f"{sum(len(r['boxes']) for r in records)} boxes")

    if unknown:
        print("\n  ! UNMAPPED labels encountered (add to taxonomy.LABEL_MAP):")
        for lbl, n in unknown.most_common(20):
            print(f"      {n:6d}  {lbl!r}")

    old_split_leakage()

    # merge groups that are duplicates across datasets
    uf = UnionFind()
    for r in records:
        uf.find(r["group"])
    if not args.no_phash:
        print("\nPerceptual-hash pass (cross-dataset duplicate merge)...")
        seen: dict[str, str] = {}
        merged = 0
        for i, r in enumerate(records):
            if i % 2000 == 0 and i:
                print(f"  {i}/{len(records)}")
            h = phash(ROOT / r["img"])
            if h is None:
                continue
            if h in seen and uf.find(seen[h]) != uf.find(r["group"]):
                uf.union(seen[h], r["group"])
                merged += 1
            else:
                seen.setdefault(h, r["group"])
        print(f"  merged {merged} cross-dataset duplicate groups")

    for r in records:
        r["group"] = uf.find(r["group"])

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)
    print(f"\n{len(records)} images collapse to {len(groups)} unique source groups"
          f"  ({len(records)/max(1,len(groups)):.2f} copies per photo)")

    assignment = split_groups(groups)
    leakage_report(records, assignment)

    supervision = derive_supervision(records)
    print("\n── supervised classes per dataset (used to mask the loss) ──")
    for ds, classes in supervision.items():
        missing = [c for c in CLASSES if c not in classes]
        print(f"  {ds:10s} supervises {classes}")
        if missing:
            print(f"  {'':10s}   masked out: {missing}")

    if args.report_only:
        print("\n(--report-only: nothing written)")
        return 0

    payload = {
        "version": 2,
        "seed": SEED,
        "classes": CLASSES,
        "fractions": SPLIT_FRACTIONS,
        "supervised_classes_per_dataset": supervision,
        "splits": {s: [] for s in SPLIT_FRACTIONS},
    }
    for r in records:
        payload["splits"][assignment[r["group"]]].append({
            "img": r["img"], "dataset": r["dataset"],
            "w": r["w"], "h": r["h"], "group": r["group"],
            "boxes": [list(b) for b in r["boxes"]],
        })

    body = json.dumps(payload, sort_keys=True)
    payload["content_hash"] = hashlib.sha256(body.encode()).hexdigest()[:16]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}  (hash {payload['content_hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
