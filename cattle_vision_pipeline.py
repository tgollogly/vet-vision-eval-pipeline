"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
cattle_vision_pipeline.py — Pre-trained YOLO cattle detection, posture
heuristics (standing vs lying), lighting / occlusion checks, RTSP reconnect,
and real-time DuckDB telemetry logging.

This module is unmonetized open-source software built solely for learning,
skills development, and technical portfolio demonstration, with zero
commercial intent. It is not a veterinary diagnostic or production
welfare-certification system.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import cv2
import duckdb
import numpy as np

from db_setup import DB_PATH, connect

# Ultralytics is imported lazily inside ``load_yolo_model`` so heuristic unit
# tests do not require a GPU stack at collection time.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cattle_vision")

# ---------------------------------------------------------------------------
# COCO cattle / animal class filters
# ---------------------------------------------------------------------------

COCO_COW_CLASS_ID = 19
COCO_LIVESTOCK_CLASS_IDS = {
    17: "horse",
    18: "sheep",
    19: "cow",
}
DEFAULT_CATTLE_CLASS_IDS = {COCO_COW_CLASS_ID}

CATTLE_TELEMETRY_TABLE = "cattle_telemetry"
DEFAULT_MODEL = "yolov8n.pt"

# Aspect-ratio threshold: width/height above this → lying heuristic.
LYING_ASPECT_RATIO = 1.35

# Mean grayscale intensity thresholds for field lighting labels.
NIGHT_LUMA_MAX = 35.0
LOW_LIGHT_LUMA_MAX = 85.0

# Pixels from the frame border that count as "touching the edge".
EDGE_MARGIN_PX = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Runtime configuration for the educational cattle vision pipeline."""

    source: str
    db_path: Path = DB_PATH
    model_path: str = DEFAULT_MODEL
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    target_fps: float = 1.0  # process ~1 frame/sec by default
    display: bool = True
    max_frames: int | None = None
    include_livestock: bool = False
    lying_aspect_ratio: float = LYING_ASPECT_RATIO
    imgsz: int = 640
    reconnect_delay_sec: float = 3.0
    max_reconnect_attempts: int = 0  # 0 = retry forever for live streams
    camera_label: str | None = None

    @property
    def class_ids(self) -> set[int]:
        if self.include_livestock:
            return set(COCO_LIVESTOCK_CLASS_IDS.keys())
        return set(DEFAULT_CATTLE_CLASS_IDS)

    @property
    def camera_source(self) -> str:
        """Human-readable camera_source value written to DuckDB."""
        if self.camera_label:
            return self.camera_label
        text = str(self.source)
        if text.isdigit():
            return f"usb:{text}"
        parsed = urlparse(text)
        if parsed.scheme in {"rtsp", "rtsps", "http", "https"}:
            host = parsed.hostname or "unknown-host"
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{host}{port}{parsed.path}"
        return text


def classify_source(source: str) -> str:
    """Return ``rtsp``, ``usb``, or ``file`` for reconnect policy."""
    text = str(source)
    if text.isdigit():
        return "usb"
    scheme = urlparse(text).scheme.lower()
    if scheme in {"rtsp", "rtsps", "http", "https"}:
        return "rtsp"
    return "file"


# ---------------------------------------------------------------------------
# DuckDB: cattle_telemetry schema + writer
# ---------------------------------------------------------------------------


def ensure_cattle_telemetry_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure ``cattle_telemetry`` exists with the educational portfolio schema.

    Columns:
      frame_id, timestamp, camera_source, animal_id, behavior_label,
      confidence_score, bounding_box, lighting_level, is_occluded
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CATTLE_TELEMETRY_TABLE} (
            frame_id         VARCHAR NOT NULL,
            timestamp        TIMESTAMP NOT NULL,
            camera_source    VARCHAR NOT NULL,
            animal_id        VARCHAR NOT NULL,
            behavior_label   VARCHAR NOT NULL,
            confidence_score DOUBLE NOT NULL,
            bounding_box     VARCHAR NOT NULL,
            lighting_level   VARCHAR NOT NULL,
            is_occluded      BOOLEAN NOT NULL
        )
        """
    )
    # Forward-compatible for DBs created before camera_source existed.
    con.execute(
        f"""
        ALTER TABLE {CATTLE_TELEMETRY_TABLE}
        ADD COLUMN IF NOT EXISTS camera_source VARCHAR DEFAULT 'unknown'
        """
    )


class CattleTelemetryWriter:
    """Insert cattle detection telemetry into DuckDB in real time."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._con: duckdb.DuckDBPyConnection | None = None
        self.inserts = 0

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = connect(self.db_path)
        ensure_cattle_telemetry_table(self._con)
        log.info("DuckDB ready | db=%s | table=%s", self.db_path, CATTLE_TELEMETRY_TABLE)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
            log.info("DuckDB closed | session_inserts=%d", self.inserts)

    def insert_records(self, records: Sequence[dict[str, Any]]) -> int:
        if self._con is None:
            raise RuntimeError("CattleTelemetryWriter is not open")
        if not records:
            return 0

        rows = [
            (
                r["frame_id"],
                r["timestamp"],
                r["camera_source"],
                r["animal_id"],
                r["behavior_label"],
                float(r["confidence_score"]),
                r["bounding_box"],
                r["lighting_level"],
                bool(r["is_occluded"]),
            )
            for r in records
        ]
        self._con.executemany(
            f"""
            INSERT INTO {CATTLE_TELEMETRY_TABLE} (
                frame_id,
                timestamp,
                camera_source,
                animal_id,
                behavior_label,
                confidence_score,
                bounding_box,
                lighting_level,
                is_occluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.inserts += len(rows)
        return len(rows)


# ---------------------------------------------------------------------------
# Field environment + posture heuristics
# ---------------------------------------------------------------------------


def classify_lighting_level(frame: np.ndarray) -> str:
    """
    Classify frame illumination from mean pixel intensity.

    Returns one of: ``bright``, ``low_light``, ``night``.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    mean_luma = float(np.mean(gray))
    if mean_luma <= NIGHT_LUMA_MAX:
        return "night"
    if mean_luma <= LOW_LIGHT_LUMA_MAX:
        return "low_light"
    return "bright"


def bbox_aspect_ratio(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return width / height for a bounding box (guards against zero height)."""
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    return width / height


def infer_posture_label(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    lying_aspect_ratio: float = LYING_ASPECT_RATIO,
) -> str:
    """
    Infer standing vs lying from bounding-box geometry.

    Educational heuristic only: if width is significantly greater than height
    (aspect ratio >= threshold), label as ``lying``; otherwise ``standing``.
    """
    ratio = bbox_aspect_ratio(x1, y1, x2, y2)
    return "lying" if ratio >= lying_aspect_ratio else "standing"


def is_bbox_occluded_by_edge(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    *,
    margin_px: int = EDGE_MARGIN_PX,
) -> bool:
    """
    Flag potential occlusion / truncation when a box touches extreme frame edges.

    Cattle clipped by the camera FOV are often partially visible; this
    educational flag marks those detections for downstream QA filters.
    """
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= (frame_width - 1 - margin_px)
        or y2 >= (frame_height - 1 - margin_px)
    )


def serialize_bbox(x1: float, y1: float, x2: float, y2: float) -> str:
    """Store ``[x1, y1, x2, y2]`` as JSON text inside DuckDB VARCHAR."""
    return json.dumps([round(float(v), 2) for v in (x1, y1, x2, y2)])


# ---------------------------------------------------------------------------
# YOLO detection helpers
# ---------------------------------------------------------------------------


def load_yolo_model(model_path: str = DEFAULT_MODEL) -> Any:
    """Load a pre-trained Ultralytics YOLO checkpoint (downloads on first use)."""
    from ultralytics import YOLO  # lazy: heavy dependency

    log.info("Loading YOLO model: %s", model_path)
    model = YOLO(model_path)
    log.info("YOLO model ready")
    return model


def open_video_source(source: str) -> cv2.VideoCapture:
    """
    Open RTSP/IP streams, USB webcam indices (``0`` / ``1``), or local video files.
    """
    kind = classify_source(source)
    if kind == "usb":
        cap = cv2.VideoCapture(int(source))
    elif kind == "rtsp":
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Video source not found: {path}")
        cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source!r}")
    return cap


@dataclass
class CattleDetection:
    """One filtered cattle / livestock detection for a frame."""

    animal_id: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    behavior_label: str
    lighting_level: str
    is_occluded: bool

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


def extract_cattle_detections(
    result: Any,
    *,
    frame_shape: tuple[int, ...],
    lighting_level: str,
    allowed_class_ids: set[int],
    lying_aspect_ratio: float,
    conf_threshold: float,
) -> list[CattleDetection]:
    """Parse an Ultralytics result, keep cattle/animal classes, attach heuristics."""
    frame_h, frame_w = frame_shape[:2]
    detections: list[CattleDetection] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    track_ids = None
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        if int(cls_id) not in allowed_class_ids:
            continue
        if float(conf) < conf_threshold:
            continue

        x1, y1, x2, y2 = map(float, box)
        behavior = infer_posture_label(
            x1, y1, x2, y2, lying_aspect_ratio=lying_aspect_ratio
        )
        occluded = is_bbox_occluded_by_edge(x1, y1, x2, y2, frame_w, frame_h)
        if track_ids is not None:
            animal_id = f"track_{int(track_ids[i]):04d}"
        else:
            animal_id = f"det_{i:02d}"

        class_name = COCO_LIVESTOCK_CLASS_IDS.get(int(cls_id), f"class_{int(cls_id)}")
        detections.append(
            CattleDetection(
                animal_id=animal_id,
                class_id=int(cls_id),
                class_name=class_name,
                confidence=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                behavior_label=behavior,
                lighting_level=lighting_level,
                is_occluded=occluded,
            )
        )
    return detections


def draw_detections(frame: np.ndarray, detections: Iterable[CattleDetection]) -> np.ndarray:
    """Overlay boxes + behavior labels for local visualization."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        color = (60, 180, 75) if det.behavior_label == "standing" else (40, 160, 230)
        if det.is_occluded:
            color = (50, 50, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{det.animal_id} {det.class_name} | {det.behavior_label} "
            f"{det.confidence:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    lighting = (
        next(iter(detections)).lighting_level
        if detections
        else classify_lighting_level(frame)
    )
    cv2.putText(
        annotated,
        f"lighting={lighting} | EDU PoC (q=quit)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return annotated


# ---------------------------------------------------------------------------
# Main pipeline (throttle + reconnect + graceful shutdown)
# ---------------------------------------------------------------------------


class CattleVisionPipeline:
    """
    End-to-end educational loop:

    Camera/file → sample @ ~1 FPS → YOLO cattle filter → posture / lighting /
    occlusion heuristics → DuckDB ``cattle_telemetry`` → optional preview.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.writer = CattleTelemetryWriter(config.db_path)
        self.model = load_yolo_model(config.model_path)
        self._stop = False
        self._processed = 0
        self._source_kind = classify_source(config.source)

    def request_stop(self, *_args: Any) -> None:
        if not self._stop:
            log.info("Shutdown requested — finishing current frame")
        self._stop = True

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def _should_reconnect(self) -> bool:
        return self._source_kind in {"rtsp", "usb"}

    def _open_with_retries(self) -> cv2.VideoCapture | None:
        attempts = 0
        while not self._stop:
            try:
                cap = open_video_source(self.config.source)
                log.info(
                    "Camera connected | kind=%s | source=%s",
                    self._source_kind,
                    self.config.camera_source,
                )
                return cap
            except Exception as exc:  # noqa: BLE001 — field demos need resilient logs
                attempts += 1
                log.warning("Connection failed (attempt %d): %s", attempts, exc)
                if (
                    self.config.max_reconnect_attempts > 0
                    and attempts >= self.config.max_reconnect_attempts
                ):
                    log.error("Max reconnect attempts reached — giving up")
                    return None
                if self._source_kind == "file":
                    return None
                time.sleep(self.config.reconnect_delay_sec)
        return None

    def _detections_to_records(
        self,
        *,
        frame_id: str,
        timestamp: datetime,
        detections: Sequence[CattleDetection],
    ) -> list[dict[str, Any]]:
        camera_source = self.config.camera_source
        return [
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "camera_source": camera_source,
                "animal_id": det.animal_id,
                "behavior_label": det.behavior_label,
                "confidence_score": det.confidence,
                "bounding_box": serialize_bbox(*det.bbox_xyxy),
                "lighting_level": det.lighting_level,
                "is_occluded": det.is_occluded,
            }
            for det in detections
        ]

    def process_frame(self, frame: np.ndarray, frame_index: int) -> list[CattleDetection]:
        """Run YOLO + heuristics on a single BGR frame."""
        lighting = classify_lighting_level(frame)
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            classes=sorted(self.config.class_ids),
            imgsz=self.config.imgsz,
            verbose=False,
        )
        return extract_cattle_detections(
            results[0],
            frame_shape=frame.shape,
            lighting_level=lighting,
            allowed_class_ids=self.config.class_ids,
            lying_aspect_ratio=self.config.lying_aspect_ratio,
            conf_threshold=self.config.conf_threshold,
        )

    def _file_frame_stride(self, cap: cv2.VideoCapture) -> int:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        stride = max(1, int(round(source_fps / max(self.config.target_fps, 0.05))))
        log.info(
            "File throttle | source_fps=%.2f | target_fps=%.2f | stride=%d",
            source_fps,
            self.config.target_fps,
            stride,
        )
        return stride

    def run(self) -> int:
        """Main ingest loop with RTSP/USB reconnect. Returns sampled frame count."""
        self._install_signal_handlers()
        self.writer.open()
        window = "Cattle Vision Pipeline (Educational PoC) — press q to quit"
        interval = 1.0 / max(self.config.target_fps, 0.05)

        log.info(
            "Ingest started | source=%s | classes=%s | model=%s | target_fps=%.2f",
            self.config.camera_source,
            sorted(self.config.class_ids),
            self.config.model_path,
            self.config.target_fps,
        )

        try:
            while not self._stop:
                cap = self._open_with_retries()
                if cap is None:
                    break

                next_sample_at = time.monotonic()
                raw_index = 0
                file_stride = (
                    self._file_frame_stride(cap) if self._source_kind == "file" else None
                )

                try:
                    while not self._stop:
                        ok, frame = cap.read()
                        raw_index += 1
                        if not ok or frame is None:
                            if self._source_kind == "file":
                                log.info(
                                    "Reached end of video file after %d reads",
                                    raw_index - 1,
                                )
                            else:
                                log.warning(
                                    "Dropped / failed frame read — stream may have stalled"
                                )
                            break

                        if file_stride is not None:
                            if (raw_index - 1) % file_stride != 0:
                                continue
                        else:
                            now = time.monotonic()
                            if now < next_sample_at:
                                continue
                            next_sample_at = now + interval

                        detections = self.process_frame(frame, raw_index)
                        frame_id = (
                            f"cattle_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
                            f"_{raw_index:06d}"
                        )
                        ts = datetime.now(timezone.utc).replace(tzinfo=None)
                        inserted = self.writer.insert_records(
                            self._detections_to_records(
                                frame_id=frame_id,
                                timestamp=ts,
                                detections=detections,
                            )
                        )
                        self._processed += 1
                        log.info(
                            "Frame %d | detections=%d | inserts=%d | lighting=%s",
                            self._processed,
                            len(detections),
                            inserted,
                            classify_lighting_level(frame),
                        )

                        if self.config.display:
                            preview = draw_detections(frame, detections)
                            try:
                                cv2.imshow(window, preview)
                                key = cv2.waitKey(1) & 0xFF
                                if key in (ord("q"), ord("Q")):
                                    log.info("Quit key pressed")
                                    self.request_stop()
                            except cv2.error:
                                log.warning("Display unavailable — continuing headless")
                                self.config.display = False

                        if (
                            self.config.max_frames is not None
                            and self._processed >= self.config.max_frames
                        ):
                            log.info("Reached max_frames=%d", self.config.max_frames)
                            self.request_stop()
                finally:
                    cap.release()

                if self._stop:
                    break
                if not self._should_reconnect():
                    log.info("End of video file — ingestion complete")
                    break

                log.warning(
                    "Stream disconnected — reconnecting in %.1fs",
                    self.config.reconnect_delay_sec,
                )
                time.sleep(self.config.reconnect_delay_sec)
        finally:
            self.writer.close()
            if self.config.display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            log.info(
                "Graceful shutdown | sampled_frames=%d | db_inserts=%d",
                self._processed,
                self.writer.inserts,
            )
        return self._processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Educational cattle vision pipeline: YOLO cattle detection, "
            "standing/lying heuristics, lighting/occlusion flags → DuckDB"
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="RTSP URL, USB index (0/1), or local video path (.mp4)",
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help="Path to livestock_analytics.duckdb",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ultralytics checkpoint (e.g. yolov8n.pt or yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        help="Sample / process approximately this many frames per second",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on sampled frames (CI / demos)",
    )
    parser.add_argument(
        "--lying-aspect-ratio",
        type=float,
        default=LYING_ASPECT_RATIO,
        help="width/height threshold for lying posture heuristic",
    )
    parser.add_argument(
        "--include-livestock",
        action="store_true",
        help="Also keep COCO horse/sheep classes (still educational)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting a dropped IP/USB stream",
    )
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=0,
        help="0 = retry forever for live cameras",
    )
    parser.add_argument(
        "--camera-label",
        default=None,
        help="Override camera_source string stored in DuckDB",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview (servers / headless)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PipelineConfig(
        source=args.source,
        db_path=Path(args.db_path),
        model_path=args.model,
        conf_threshold=args.conf,
        target_fps=args.target_fps,
        display=not args.no_display,
        max_frames=args.max_frames,
        include_livestock=args.include_livestock,
        lying_aspect_ratio=args.lying_aspect_ratio,
        imgsz=args.imgsz,
        reconnect_delay_sec=args.reconnect_delay,
        max_reconnect_attempts=args.max_reconnect_attempts,
        camera_label=args.camera_label,
    )
    CattleVisionPipeline(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
