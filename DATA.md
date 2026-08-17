# Data and model provenance

The working tree is ~8.8 GB but only ~16 files are tracked. Excluding the
datasets and weights from git is right — they are far too large — but it also
means **nothing in the repository describes them**. If this machine's disk
fails, the code survives and the project does not.

This file is the record. Keep it current when the data changes.

## What is on disk (August 2026)

| Path | Contents | Size | Tracked |
|------|----------|------|---------|
| `INRIAPerson/Train` | 614 PNG + 614 Pascal VOC XML | ~396 MB | no |
| `INRIAPerson/Test` | 288 PNG + 288 Pascal VOC XML | ~186 MB | no |
| `data/ppe` | 19,316 JPG + 19,316 Pascal VOC XML | ~1.05 GB | no |
| `data/person` | dataset index (JSON) | ~2 MB | no |
| `models/*.pth` | 8 checkpoints, 38–51 MB each | ~350 MB | no (see `models/MANIFEST.json`, which **is** tracked) |
| `output/`, `vid.mp4` | inference output and sample footage | — | no |

## Person detection — INRIAPerson

A public dataset, so this half is reproducible. The original INRIA host has
been offline for years; the dataset is now mirrored on Kaggle and academic
mirrors. Annotations here are Pascal VOC XML rather than the original INRIA
`.txt` format, so they were converted at some point — if you re-download, you
will need to convert again before `scripts/person_detector/step1_dataset.py`
will read them.

## PPE detection — `data/ppe`

**This is the irreplaceable one.** 19,316 annotated images with no recorded
source. It is not a published dataset under that name, so it was either
assembled from several sources or annotated locally. Until its origin is
written down here, it cannot be rebuilt.

> **Action:** record where these images came from. If they were annotated by
> hand, that labour is the single most valuable artifact in the project and
> the only copy is on one disk.

## Checkpoints

`models/MANIFEST.json` records, per checkpoint: SHA-256, size, modification
time, the epoch stored inside the file, and the git revision plus torch/CUDA
versions at the time the manifest was generated.

```bash
python scripts/manifest.py            # refresh after training
python scripts/manifest.py --verify   # confirm on-disk files still match
```

`--verify` also flags checkpoints that exist on disk but are absent from the
manifest, which is how a stale manifest shows up.

## Backups

None of the above is backed up anywhere. At minimum, copy `models/` and
`data/ppe/` to external storage — together they are ~1.4 GB and they represent
essentially all of the project's unrecoverable work.

## Environment

GPU builds are not on PyPI, so install torch from the CUDA index first — see
the header of `requirements.txt`. Verified combination: torch 2.7.1+cu118,
torchvision 0.22.1+cu118, CUDA 11.8, in `venv/`.
