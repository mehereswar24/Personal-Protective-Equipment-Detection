# 🦺 Personal Protective Equipment (PPE) Detection

A computer vision pipeline for **real-time detection and classification of Personal Protective Equipment** (helmets, vests, gloves, etc.) on workers in industrial environments — using a two-stage YOLOv8-based architecture.

---

## ✨ Features

- **👤 Two-Stage Pipeline**: First detects persons in the scene, then classifies PPE on each detected individual.
- **⛑️ Multi-Class PPE Detection**: Detects helmets, safety vests, gloves, goggles, and more.
- **📹 Real-Time Ready**: Designed to be run on live video feeds or static images.
- **📊 Training Logs**: Includes detailed training logs for both person detector and PPE classifier stages.
- **🔧 Modular Architecture**: Separate modules for person detection, PPE classification, and the full inference pipeline.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Detection Model** | YOLOv8 (Ultralytics) |
| **Framework** | Python |
| **Computer Vision** | OpenCV |
| **Training** | Ultralytics YOLO training pipeline |

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/mehereswar24/Personal-Protective-Equipment-Detection.git
cd Personal-Protective-Equipment-Detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run detection on an image
python sort.py
```

---

## 📂 Project Structure

```text
Personal-Protective-Equipment-Detection/
├── scripts/
│   ├── person_detector/        # Stage 1: Person detection scripts
│   ├── ppe_classifiers/        # Stage 2: PPE classification scripts
│   └── pipeline/               # End-to-end inference pipeline
├── sort.py                     # Main inference entrypoint
├── image.png                   # Sample test image
├── training_log.txt            # Full training history
└── requirements.txt            # Python dependencies
```

---

**Keeping workers safe with AI. 🦺🤖**
