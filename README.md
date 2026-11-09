# Cattle Vision & Welfare Telemetry Engine (Educational Project)

<p align="center">
  <img
    src="docs/assets/hero-livestock-vision.jpg"
    alt="Cattle Vision & Welfare Telemetry Engine — Educational MLOps Proof-of-Concept hero banner"
    width="100%"
  />
</p>

**License:** [MIT](./LICENSE) · Copyright (c) 2026 **Thomas Gollogly**  
**Software cost:** Free and open source (optional hardware / electricity only)

> **Educational Proof-of-Concept / Open-Source Learning Project**  
> This repository is **unmonetized open-source software** built solely for **learning**, **skills development**, and **technical portfolio demonstration**, with **zero commercial intent**.  
> It is **not** a production farm system, veterinary diagnostic tool, calving predictor, welfare certification product, or paid SaaS.  
> Framing supports UK benefit / portfolio safety: study and self-host freely; do not present this demo as clinical advice or a commercial offering.

---

## Hero Architecture Diagram

<p align="center">
  <img
    src="docs/assets/architecture-sequence.png"
    alt="Architecture: Camera Feed to YOLO Posture Inference to DuckDB Telemetry to SQL Calving Analytics to PyTest Quality Harness"
    width="100%"
  />
</p>

```mermaid
sequenceDiagram
    participant Cam as Camera Feed
    participant YOLO as YOLO Posture Inference
    participant DB as DuckDB Telemetry
    participant SQL as SQL Calving Analytics
    participant QA as PyTest Quality Harness

    Cam->>YOLO: RTSP USB or MP4 frames at about 1 FPS
    YOLO->>DB: Insert cattle_telemetry rows
    Note over YOLO,DB: standing or lying, lighting, occlusion, camera_source
    DB->>SQL: Restlessness score, lying hours, edge failures
    SQL->>QA: Filtered bright unoccluded rows
    QA->>QA: Assert schema and confidence benchmarks
```

```text
Camera Feed (RTSP / USB / .mp4)
        |
        v
YOLO Posture Inference (cattle filter + standing/lying)
        |
        v
DuckDB Telemetry (livestock_analytics.duckdb / cattle_telemetry)
        |
        v
SQL Calving Analytics (restlessness + lying + edge failures)
        |
        v
PyTest Quality Harness (filter occlusion / low light)
```

---

## Project Overview

This educational portfolio demonstrates a low-cost **computer-vision telemetry** and **model-quality monitoring** loop for cattle welfare *learning patterns* (including calving-restlessness *demo* alerts):

| Module | Role (learning focus) |
|--------|------------------------|
| `cattle_vision_pipeline.py` | YOLO cattle detect/track → posture / lighting / occlusion → DuckDB |
| `calving_analytics.py` | SQL window restlessness score, daily lying hours, edge-failure metrics |
| `test_eval.py` | PyTest: schema populated + analytics filter occluded/low-light frames |
| `camera_connector.py` | Optional mock ingest helper for frame_telemetry demos |
| `db_setup.py` / `analytics_engine.py` | Earlier synthetic frame_telemetry teaching path |

---

## Farmer Value & Impact

Teaching-oriented value themes for a UK smallholding / dairy learning setup (not clinical guarantees):

### Calving Distress Monitoring
Spikes in **standing ↔ lying** posture transitions inside a rolling **2-hour** window raise an educational **calving restlessness** alert flag. In teaching materials this models how active labour / restlessness *might* surface early from telemetry — always verify in person with a stockperson / vet.

### Early Illness & Lameness
**Daily lying duration** tracking (educational baseline band **10–14 hours/day**) highlights animals outside the band so learners can practise reviewing mastitis- or foot-related *hypotheses*. The software does **not** diagnose disease.

### 24/7 Automated Vigilance
Round-the-clock local telemetry sampling (~1 frame/sec) without standing over a pen overnight — useful as a labour-saving *learning* pattern for always-on barn cameras on a LAN.

### Camera Safeguards
**Low-light / night** intensity labels and **edge-occlusion** flags highlight dirty lenses, poor barn lighting, or cattle clipped by the FOV so model-quality failures are not mistaken for animal behaviour.

---

## Key Benefits

| Benefit | What it means |
|--------|----------------|
| **Free software** | MIT-licensed; no subscription for this codebase |
| **Runs locally** | Video + DuckDB stay on your machine / farm LAN by default |
| **Common cameras** | RTSP IP cams, USB webcams (`0`), or `.mp4` files |
| **Reconnect-aware** | Dropped IP streams retry automatically |
| **Transparent SQL** | Calving / lying / failure metrics you can read and audit |
| **Low running cost** | Typical mini-PC + 1–2 cams ≈ a few £/month electricity |

---

## Easy Farmer Setup Guide (UK)

### Optimal cameras & barn layout

| Choice | Recommendation |
|--------|----------------|
| Resolution | 1080p |
| Lens | ~2.8–4 mm wide angle |
| Night | IR / colour night vision |
| Link | Wired PoE Ethernet when possible |
| Mount | 2.5–3.5 m, angled into the pen |
| Aim | Avoid bright doorway backlight |
| Scale | Start with **1 camera per priority pen** |
| Privacy | Keep RTSP on the farm LAN only |

### Estimated UK costs (indicative, 2026)

| Kit | Approx. GBP |
|-----|-------------|
| This software (MIT) | **£0** |
| Starter (reuse PC + 1 USB/Wi‑Fi cam) | **£25–£70** |
| Always-on (mini-PC + 1–2 PoE cams + cabling) | **≈ £225–£740** |
| Electricity (mini-PC + 1–2 cams) | **≈ £3–£12 / month** |
| Cloud GPU / SaaS subscription | **£0** (offline by design) |

> Planning estimates only — not quotes. Software from Thomas Gollogly in this repo has **no fee**.

---

## Quickstart Guide

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1) Seed demo cattle telemetry (no camera required)

```bash
python3 calving_analytics.py --seed-demo
```

### 2) Run SQL calving / welfare analytics

```bash
python3 calving_analytics.py
```

Prints:

- calving restlessness scores (2-hour posture transitions + alert flag)
- daily lying hours (10–14 h teaching band)
- model quality under low light vs occlusion

### 3) Run the YOLO cattle vision pipeline

```bash
# Local video file
python3 cattle_vision_pipeline.py --source path/to/barn_cattle.mp4 --target-fps 1

# USB webcam
python3 cattle_vision_pipeline.py --source 0 --target-fps 1

# RTSP / IP camera (auto-reconnect on drop)
python3 cattle_vision_pipeline.py --source "rtsp://username:password@192.168.1.20:554/stream" --model yolov8n.pt

# Headless demo cap
python3 cattle_vision_pipeline.py --source path/to/barn_cattle.mp4 --no-display --max-frames 10
```

Press **`q`** in the preview window (or `Ctrl+C`) for a graceful shutdown.

Optional: `--model yolo11n.pt`, `--include-livestock` (also keeps COCO horse/sheep).

### 4) Run the PyTest quality harness

```bash
python3 -m pytest test_eval.py test_cattle_heuristics.py -v
```

### One-liner (seed → analytics → tests)

```bash
python3 calving_analytics.py --seed-demo && python3 calving_analytics.py && python3 -m pytest test_eval.py test_cattle_heuristics.py -v
```

---

## Table Schema (`cattle_telemetry`)

| Column | Type | Description (educational) |
|--------|------|---------------------------|
| `frame_id` | `VARCHAR` | Sampled frame identifier |
| `timestamp` | `TIMESTAMP` | Capture / process time |
| `camera_source` | `VARCHAR` | Redacted RTSP, `usb:0`, file path, or `synthetic_demo` |
| `animal_id` | `VARCHAR` | Tracker id (`track_0001`) or detection index |
| `behavior_label` | `VARCHAR` | `standing` or `lying` (bbox aspect-ratio heuristic) |
| `confidence_score` | `DOUBLE` | YOLO confidence |
| `bounding_box` | `VARCHAR` | JSON `[x1, y1, x2, y2]` |
| `lighting_level` | `VARCHAR` | `bright`, `low_light`, or `night` |
| `is_occluded` | `BOOLEAN` | True when the box touches extreme frame edges |

---

## Legal Disclaimer

**Copyright (c) 2026 Thomas Gollogly.** Released under the [MIT License](./LICENSE).

This software is provided **“as is”** as an **Educational Proof-of-Concept / Open-Source Learning Project** for skills development and technical portfolio demonstration, with **zero commercial or monetized intent**.

- Not a veterinary device, calving alarm certified for stock decisions, or welfare certification tool.
- Restlessness / lying / confidence outputs are **learning telemetry**, not medical or husbandry advice.
- You are responsible for safe mounting, electrics, animal welfare checks, and UK GDPR where personal data appears.
- No liability for loss, injury, stock outcomes, or business decisions arising from use of this software.

---

## Project Notice

**This project is open-source software built solely for learning, skills development, and technical portfolio demonstration, with zero commercial or monetized intent.**

---

## Attributions

| Component | Attribution |
|-----------|-------------|
| **This repository** | Copyright (c) 2026 **Thomas Gollogly** — [MIT License](./LICENSE) |
| **DuckDB** | [duckdb.org](https://duckdb.org/) |
| **OpenCV** | [opencv.org](https://opencv.org/) |
| **Ultralytics YOLO** | [ultralytics.com](https://www.ultralytics.com/) (see their licence for weights & code) |
| **PyTorch** | [pytorch.org](https://pytorch.org/) |
| **PyTest** | [pytest.org](https://pytest.org/) |
| **COCO taxonomy** | Cattle filter uses COCO `cow` class id `19` |
| **Docs artwork** | `docs/assets/` educational portfolio assets |
