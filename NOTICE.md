# Third-party notices

## Removed: SORT tracker (GPL-3.0)

This project previously vendored `sort.py` — Alex Bewley's SORT implementation,
licensed **GPL-3.0-or-later**. Because the pipeline imported it directly into
the same process, the combined work was a GPLv3 derivative, which would have
obliged source disclosure for the entire system on distribution.

`sort.py` has been **removed** and replaced by `ppe/tracking.py`, an
independent constant-velocity Kalman + IoU/Hungarian tracker written for this
project and covered by the MIT LICENSE in this repository. No SORT code,
structure, or comments were copied.

If you restore `sort.py` for any reason, the GPL obligation returns.

## Dependencies

| Package | License |
|---|---|
| PyTorch, TorchVision | BSD-3-Clause |
| NumPy, SciPy | BSD-3-Clause |
| OpenCV (opencv-python) | Apache-2.0 |
| Pillow | MIT-CMU |
| Albumentations | MIT |
| FastAPI, Pydantic, Uvicorn | MIT / BSD-3-Clause |
| PyYAML | MIT |

Pretrained detector weights come from TorchVision's COCO-trained models
(BSD-3-Clause). **No YOLO / Ultralytics code or weights are used** — that is a
deliberate constraint of this project, and it also avoids Ultralytics' AGPL-3.0
licence.

## Datasets

Training data is derived from public Roboflow exports (see
`data/ppe/dataset*/README.dataset.txt` for each source and its licence) and the
INRIA Person dataset. Verify each source's terms before commercial use.
