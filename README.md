# Livestock Vision Analytics & MLOps Engine (Educational Project)

<p align="center">
  <img
    src="docs/assets/hero-livestock-vision.jpg"
    alt="Livestock Vision Analytics — Educational MLOps Proof-of-Concept hero banner showing pasture cattle with computer-vision bounding boxes"
    width="100%"
  />
</p>

> **Educational Proof-of-Concept / Open-Source Learning Project**  
> This repository is a self-contained portfolio demo for learning computer-vision telemetry analytics and model-failure monitoring patterns. It is **not** a production farm system, veterinary diagnostic tool, or commercial product.

---

## Architecture

Data moves from camera-frame metadata through a local DuckDB engine into SQL analytics, then through a PyTest quality harness:

```text
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│ Camera Connector    │     │ DuckDB Local Engine  │     │ SQL Analytics   │     │ PyTest Quality      │
│ (RTSP / USB / File) │────▶│ livestock_analytics  │────▶│ Lying durations │────▶│ Harness              │
│ → mock bbox metadata│     │ .duckdb              │     │ Failure metrics │     │ (confidence QA)      │
└─────────────────────┘     └──────────────────────┘     └─────────────────┘     └──────────────────────┘
```

```mermaid
sequenceDiagram
    participant Cam as Camera Connector<br/>(RTSP / USB / File)
    participant DB as DuckDB Local Engine
    participant SQL as SQL Analytics
    participant QA as PyTest Quality Harness

    Cam->>DB: Stream frame_telemetry rows<br/>(bbox labels, confidence, lighting, occlusion)
    Note over Cam,DB: Also supports offline synthetic seeding via db_setup.py
    DB->>SQL: Query daily lying durations
    DB->>SQL: Query low-confidence rates<br/>(low light vs occlusion)
    SQL->>QA: Filtered high-quality frames
    QA->>QA: Assert confidence benchmarks
```

---

## Project Overview

This educational project demonstrates a miniature **MLOps evaluation loop** for agricultural animal-welfare *telemetry* (not clinical diagnosis):

1. **Video frame bounding boxes (simulated)** — Each row in `frame_telemetry` represents a detector output for one animal in one frame: behavior label, confidence, lighting level, and occlusion flag.
2. **Model failure monitoring under low light** — Analytics compare how often low-confidence predictions occur in dim lighting versus when the camera is occluded, so learners can practice failure-mode reporting.
3. **Animal lying-time tracking** — SQL aggregates time spent with `behavior_label = 'lying'` per animal per day. In a real welfare workflow, unusually long lying periods can be a *signal worth investigating*; here the metric exists only to teach aggregation and alerting patterns.

| Module | Role (learning focus) |
|--------|------------------------|
| `db_setup.py` | Local DuckDB init + synthetic farm-condition seeding |
| `camera_connector.py` | Universal OpenCV ingest (RTSP / USB / file) → mock detect → DuckDB |
| `cattle_vision_pipeline.py` | YOLO cattle detect + standing/lying heuristics → `cattle_telemetry` |
| `analytics_engine.py` | Pure Python + SQL for lying durations & failure metrics |
| `test_eval.py` | PyTest QA that filters occluded/low-light frames and checks confidence |

---

## Project Notice

**This project is open-source software built solely for learning, skills development, and technical portfolio demonstration, with zero commercial or monetized intent.**

- It is an **Educational Proof-of-Concept / Open-Source Learning Project**.
- Synthetic data only — no real farm video, animals, or personally identifying operational data.
- Outputs are for teaching SQL analytics, DuckDB, and model-quality testing — **not** veterinary advice, welfare certification, or production monitoring.
- You may fork, study, and extend it under the spirit of open-source education; do not present it as a paid product or clinical system.

---

## Quickstart Guide

### Prerequisites

- Python 3.10+
- `pip` available on your PATH

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1) Initialize DuckDB and seed synthetic telemetry

```bash
python3 db_setup.py
```

This creates `livestock_analytics.duckdb` and inserts **1,000** synthetic `frame_telemetry` rows spanning bright light, low light, and camera occlusions.

### 2) Run the analytics engine

```bash
python3 analytics_engine.py
```

Prints:

- daily lying durations per animal (illness-flagging *demo* metric)
- model performance under low light vs occlusion

### 3) Stream camera / video telemetry into DuckDB

```bash
# Local video file (auto-generates sample_barn_footage.mp4 if missing)
python3 camera_connector.py --source sample_barn_footage.mp4 --no-display --max-frames 5

# USB webcam
python3 camera_connector.py --source 0 --target-fps 1 --no-display

# RTSP / IP camera (credentials redacted in DuckDB camera_source logs)
python3 camera_connector.py --source "rtsp://username:password@192.168.1.20:554/stream" --target-fps 1
```

Press **`q`** in the preview window (when display is enabled) or `Ctrl+C` for a graceful shutdown. Dropped IP streams automatically reconnect.

### 4) Run the YOLO cattle vision pipeline

Uses a pre-trained Ultralytics model (default `yolov8n.pt`), filters COCO **cow** detections (class id `19`), infers `standing` / `lying` from bbox aspect ratio, tags lighting (`bright` / `low_light` / `night`) and edge-occlusion, then streams rows into `cattle_telemetry`.

```bash
# Local video (press q in the preview window to quit)
python3 cattle_vision_pipeline.py --source path/to/barn_cattle.mp4

# Headless / demo cap
python3 cattle_vision_pipeline.py --source path/to/barn_cattle.mp4 --no-display --max-frames 10 --target-fps 2

# RTSP camera
python3 cattle_vision_pipeline.py --source "rtsp://username:password@192.168.1.20:554/stream" --model yolov8n.pt
```

Optional: `--include-livestock` also keeps COCO horse/sheep classes; `--model yolo11n.pt` swaps the checkpoint.

### 5) Run the PyTest quality harness

```bash
python3 -m pytest test_eval.py test_cattle_heuristics.py -v
```

The harness filters out occluded and low-light frames, then asserts that remaining “good condition” predictions meet confidence benchmarks. Heuristic unit tests cover posture / lighting / occlusion without requiring a GPU.

### One-liner (setup → short camera ingest → analytics → tests)

```bash
python3 db_setup.py && python3 camera_connector.py --source sample_barn_footage.mp4 --no-display --max-frames 5 && python3 analytics_engine.py && python3 -m pytest test_eval.py test_cattle_heuristics.py -v
```

---

## Table Schema (`frame_telemetry`)

| Column | Type | Description (educational) |
|--------|------|---------------------------|
| `frame_id` | `VARCHAR` | Frame identifier |
| `timestamp` | `TIMESTAMP` | Frame capture time |
| `camera_source` | `VARCHAR` | USB index, redacted RTSP URL, file path, or `synthetic_seed` |
| `animal_id` | `VARCHAR` | Synthetic animal ID |
| `behavior_label` | `VARCHAR` | e.g. `lying`, `standing`, `walking` |
| `confidence_score` | `DOUBLE` | Detector confidence in `[0, 1]` |
| `lighting_level` | `VARCHAR` | `bright` or `low` |
| `is_occluded` | `BOOLEAN` | Camera / subject occlusion flag |

## Table Schema (`cattle_telemetry`)

Written by `cattle_vision_pipeline.py` from live YOLO detections:

| Column | Type | Description (educational) |
|--------|------|---------------------------|
| `frame_id` | `VARCHAR` | Sampled frame identifier |
| `timestamp` | `TIMESTAMP` | Capture / process time (UTC, stored naive) |
| `animal_id` | `VARCHAR` | Tracker id (`track_0001`) or detection index |
| `behavior_label` | `VARCHAR` | `standing` or `lying` (bbox aspect-ratio heuristic) |
| `confidence_score` | `DOUBLE` | YOLO confidence |
| `bounding_box` | `VARCHAR` | JSON `[x1, y1, x2, y2]` |
| `lighting_level` | `VARCHAR` | `bright`, `low_light`, or `night` |
| `is_occluded` | `BOOLEAN` | True when the box touches extreme frame edges |

---

## Learning Outcomes

- Local analytical warehouses with **DuckDB**
- SQL-driven CV telemetry aggregations
- Separating **data-quality filters** from **model-quality metrics**
- Encoding evaluation gates as **PyTest** assertions for an MLOps-style harness
