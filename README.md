# Livestock Vision Analytics & MLOps Engine (Educational Project)

<p align="center">
  <img
    src="docs/assets/hero-livestock-vision.jpg"
    alt="Livestock Vision Analytics — Educational MLOps Proof-of-Concept hero banner showing pasture cattle with computer-vision bounding boxes"
    width="100%"
  />
</p>

**License:** [MIT](./LICENSE) · Copyright (c) 2026 **Thomas Gollogly**  
**Software cost:** Free and open source (you only pay for optional hardware/electricity)

> **Educational Proof-of-Concept / Open-Source Learning Project**  
> This repository is a self-contained portfolio demo for learning computer-vision telemetry analytics and model-failure monitoring patterns. It is **not** a production farm system, veterinary diagnostic tool, or commercial product.

---

## Architecture

Data moves from camera-frame metadata through a local DuckDB engine into SQL analytics, then through a PyTest quality harness:

<p align="center">
  <img
    src="docs/assets/architecture-sequence.png"
    alt="Architecture data flow: Camera Connector to DuckDB Local Engine to SQL Analytics to PyTest Quality Harness"
    width="100%"
  />
</p>

```text
Camera Connector (RTSP / USB / File)
        |
        v
DuckDB Local Engine (livestock_analytics.duckdb)
        |
        v
SQL Analytics (lying durations + failure metrics)
        |
        v
PyTest Quality Harness (confidence benchmarks)
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

## Key Benefits

| Benefit | What it means on a small UK farm / learning setup |
|--------|-----------------------------------------------------|
| **Free software** | MIT-licensed; no subscription for this codebase |
| **Runs locally** | Video and DuckDB stay on your machine / farm LAN by default |
| **Works with common cameras** | USB webcams, RTSP/IP cams, or a recorded video file |
| **Cattle posture cues** | Simple standing vs lying labels from bounding-box shape (teaching aid) |
| **Low-light / occlusion flags** | Helps you see when the model is guessing in poor conditions |
| **Transparent data** | SQL in DuckDB you can open, query, and audit yourself |
| **Low running cost** | Typical electricity for a mini-PC + 1–2 cameras is a few pounds per month |
| **Skills you keep** | Learn CV + MLOps patterns you can reuse or extend |

---

## Easy Farmer Setup Guide (UK)

Anyone comfortable plugging in a camera and copying a few commands can try this **free software** stack. Start small: one pen, one camera, one laptop.

### What you need

1. A Windows / Mac / Linux computer (or a small always-on mini-PC)
2. Python 3.10+ installed ([python.org](https://www.python.org/downloads/) or your package manager)
3. One camera (USB webcam **or** Wi‑Fi / PoE IP camera with RTSP)
4. Optional: Ethernet cable + PoE injector/switch for a more reliable barn camera

### 15-minute path (software only — free)

```bash
# 1) Download / clone this project, then:
pip install -r requirements.txt

# 2) Create the local database with demo data
python3 db_setup.py

# 3) Run analytics on the demo data
python3 analytics_engine.py

# 4) (Optional) Point at your camera or a barn video file
python3 cattle_vision_pipeline.py --source 0
# or
python3 cattle_vision_pipeline.py --source "rtsp://USER:PASS@CAMERA_IP:554/stream" --no-display
```

### Optimal camera & barn layout (practical)

| Choice | Recommendation | Why |
|--------|----------------|-----|
| **Resolution** | 1080p (Full HD) | Enough detail for cattle blobs without huge files |
| **Lens** | ~2.8–4 mm wide angle | Covers a pen/loafing area from one mount |
| **Night** | IR / colour night vision | Barns are dark; lighting labels need a usable image |
| **Link** | Wired PoE Ethernet if possible | More reliable than busy farm Wi‑Fi |
| **Height** | 2.5–3.5 m, angled down into the pen | Reduces people/vehicle clutter; fewer edge cut-offs |
| **Aim** | Avoid pointing into bright doorways | Cuts false “low light / night” flips and washout |
| **Count** | Start with **1 camera per priority pen** | Keep CPU load and cost low while you learn |
| **Privacy** | Keep streams on the farm LAN | Do not expose RTSP to the public internet |

**Good starter kit:** 1× outdoor/barn-rated 1080p PoE IP camera + short Ethernet run + old office PC or mini-PC. USB webcam is fine for a dry shed aisle test.

### Estimated UK costs (indicative, 2026)

Software from this repository is **£0**. Hardware is optional and bought once; prices are ballpark high-street / common online UK retail figures and will vary by brand and VAT.

#### Starter kit (1 camera — learn the system)

| Item | Typical UK price (GBP) | Notes |
|------|------------------------|-------|
| This software (MIT) | **£0** | Free forever under the license |
| USB webcam (1080p) **or** basic Wi‑Fi IP cam | £25–£70 | USB is simplest indoors |
| Existing laptop / desktop | £0 (reuse) | Most farms already have one |
| **Starter total** | **£25–£70** | If you reuse a computer |

#### Recommended always-on kit (1–2 barn cameras)

| Item | Typical UK price (GBP) | Notes |
|------|------------------------|-------|
| Mini-PC / NUC / refurbished office PC | £120–£350 | Prefer 16 GB RAM if running YOLO live |
| PoE IP camera ×1 (1080p, IR) | £45–£90 | Outdoor/barn-rated housing helps |
| Extra camera ×1 (optional) | £45–£90 | Second pen / calving area |
| PoE switch or injector | £25–£70 | Steadier than Wi‑Fi |
| Cabling, glands, mounts | £20–£60 | Keep cables out of animal reach |
| MicroSD / SSD for clips (optional) | £15–£80 | Only if you store video, not just telemetry |
| **Recommended kit total** | **≈ £225–£740** | One-off capital; software still £0 |

#### Ongoing running costs (UK, rough)

| Cost | Estimate | Notes |
|------|----------|-------|
| Electricity (mini-PC + 1–2 cams) | **≈ £3–£12 / month** | Depends on tariffs and duty cycle |
| Home/farm broadband | Usually already paid | Keep cameras on LAN; no cloud fee required |
| Cloud GPU / SaaS | **£0** | This project is designed to run offline |
| Software licence / subscription | **£0** | MIT — no fee to Thomas Gollogly for this OSS |

> Figures are **estimates for planning only**, not quotes. Shop around; refurbished PCs and sale cameras often land at the low end.

---

## Legal Disclaimer

**Copyright (c) 2026 Thomas Gollogly.** This project is released under the [MIT License](./LICENSE).

This software is provided **“as is”**, free of charge, as an **Educational Proof-of-Concept / Open-Source Learning Project** for skills development and technical portfolio demonstration, with **zero commercial or monetized intent** by the author for this repository.

- It is **not** a veterinary diagnostic device, animal-welfare certification tool, or guaranteed production monitoring system.
- Outputs (including lying-time flags, confidence scores, and occlusion/lighting tags) are **technical telemetry for learning** — they are **not** medical, husbandry, or regulatory advice.
- You are responsible for safe camera mounting, electrical work, animal safety, data protection (UK GDPR where personal data appears), and any use of the software on your farm.
- The author accepts **no liability** for loss, injury, stock outcomes, business decisions, or damages arising from use of this software or documentation.
- Third-party cameras, computers, and models (for example Ultralytics YOLO weights) remain subject to their own licences and terms.

By using this repository you agree that you understand it is free open-source learning software, not a paid farm product.

---

## Project Notice

**This project is open-source software built solely for learning, skills development, and technical portfolio demonstration, with zero commercial or monetized intent.**

- It is an **Educational Proof-of-Concept / Open-Source Learning Project**.
- Demo/seed paths use synthetic data; if you connect a live farm camera, you control that data locally.
- Outputs are for teaching SQL analytics, DuckDB, and model-quality testing — **not** veterinary advice, welfare certification, or production monitoring.
- You may fork, study, and extend it under the MIT License; do not present it as a paid clinical system.

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

---

## Attributions

This educational project stands on open tools and community work:

| Component | Attribution |
|-----------|-------------|
| **This repository** | Copyright (c) 2026 **Thomas Gollogly** — released under the [MIT License](./LICENSE) |
| **DuckDB** | Analytical database engine — [duckdb.org](https://duckdb.org/) (own licence terms) |
| **OpenCV** | Computer vision library — [opencv.org](https://opencv.org/) |
| **Ultralytics YOLO** | Pre-trained detection / tracking (`yolov8n.pt` / `yolo11n.pt`) — [ultralytics.com](https://www.ultralytics.com/) (see their licence for model weights & code) |
| **PyTorch** | Deep-learning runtime used by Ultralytics — [pytorch.org](https://pytorch.org/) |
| **PyTest** | Test harness — [pytest.org](https://pytest.org/) |
| **COCO classes** | Cattle filter uses the COCO `cow` class id (`19`) from the COCO dataset taxonomy |
| **Hero / architecture artwork** | Project documentation assets under `docs/assets/` for this portfolio demo |

Please keep the MIT copyright notice and permission notice when you copy or redistribute substantial portions of this Software, as required by the licence.
