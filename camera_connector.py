"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
camera_connector.py — Universal camera ingestion for the livestock vision
analytics portfolio demo.

Connects to RTSP/IP cameras, USB webcams, or local video files via OpenCV,
throttles frame sampling, runs a *mock* animal-detection pipeline, and
streams detection metadata into DuckDB (`livestock_analytics.duckdb`).

This module is open-source learning / skills-development software with zero
commercial or monetized intent. It is not a production farm security system
or veterinary diagnostic product.
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import cv2
import duckdb
import numpy as np

from db_setup import (
    ANIMAL_IDS,
    BEHAVIOR_LABELS,
    DB_PATH,
    TABLE_NAME,
    connect,
    ensure_frame_telemetry_schema,
)

# ---------------------------------------------------------------------------
# Logging (clean console status for field / lab demos)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("camera_connector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CameraConfig:
    """
    Flexible source configuration for the educational ingestion loop.

    ``source`` may be:
      - RTSP / HTTP URL string  → IP / security camera
      - integer / digit string  → USB webcam device index (0, 1, ...)
      - filesystem path         → local video file (e.g. sample_barn_footage.mp4)
    """

    source: str | int
    db_path: Path = DB_PATH
    target_fps: float = 1.0  # process ~1 frame/sec by default (CPU-friendly)
    reconnect_delay_sec: float = 3.0
    max_reconnect_attempts: int = 0  # 0 = retry forever (IP cameras in the field)
    display: bool = True
    max_frames: int | None = None  # optional cap for headless / CI demos
    camera_label: str | None = None  # override stored camera_source value

    @property
    def source_label(self) -> str:
        """Human-readable camera_source value written to DuckDB."""
        if self.camera_label:
            return self.camera_label
        if isinstance(self.source, int):
            return f"usb:{self.source}"
        text = str(self.source)
        parsed = urlparse(text)
        if parsed.scheme in {"rtsp", "rtsps", "http", "https"}:
            # Redact credentials if present in the URL.
            host = parsed.hostname or "unknown-host"
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{host}{port}{parsed.path}"
        return text


def config_from_dict(data: dict[str, Any]) -> CameraConfig:
    """Build a CameraConfig from a plain dictionary (notebook / service friendly)."""
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return CameraConfig(
        source=source,
        db_path=Path(data.get("db_path", DB_PATH)),
        target_fps=float(data.get("target_fps", 1.0)),
        reconnect_delay_sec=float(data.get("reconnect_delay_sec", 3.0)),
        max_reconnect_attempts=int(data.get("max_reconnect_attempts", 0)),
        display=bool(data.get("display", True)),
        max_frames=data.get("max_frames"),
        camera_label=data.get("camera_label"),
    )


# ---------------------------------------------------------------------------
# Capture open / classify
# ---------------------------------------------------------------------------


def classify_source(source: str | int) -> str:
    """Return ``rtsp``, ``usb``, or ``file`` for logging and reconnect policy."""
    if isinstance(source, int):
        return "usb"
    text = str(source)
    if text.isdigit():
        return "usb"
    scheme = urlparse(text).scheme.lower()
    if scheme in {"rtsp", "rtsps", "http", "https"}:
        return "rtsp"
    return "file"


def open_capture(source: str | int) -> cv2.VideoCapture:
    """
    Open any supported camera / file source with OpenCV.

    RTSP streams use FFMPEG backend when available for broader IP-cam support.
    """
    kind = classify_source(source)
    if kind == "usb":
        index = int(source)
        cap = cv2.VideoCapture(index)
    elif kind == "rtsp":
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        # Prefer low-latency buffering when the backend honors these hints.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        path = Path(str(source))
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source!r}")
    return cap


# ---------------------------------------------------------------------------
# Mock lightweight detection pipeline (educational stand-in for a real model)
# ---------------------------------------------------------------------------


def estimate_lighting_level(frame: np.ndarray) -> str:
    """Heuristic lighting label from mean luma — teaches condition tagging."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    return "low" if mean_luma < 70.0 else "bright"


def mock_detect_animals(frame: np.ndarray) -> list[dict[str, Any]]:
    """
    Simulate bounding-box inference for livestock monitoring demos.

    Returns zero or more detections with animal_id, behavior, confidence,
    lighting, and occlusion — no real neural net weights are loaded.
    """
    h, w = frame.shape[:2]
    lighting = estimate_lighting_level(frame)
    # Mild random occlusion flag; also treat very dark frames as occluded-ish.
    is_occluded = random.random() < 0.12 or lighting == "low" and random.random() < 0.25

    n_detections = random.choices([0, 1, 2, 3], weights=[0.15, 0.45, 0.30, 0.10], k=1)[0]
    detections: list[dict[str, Any]] = []

    for _ in range(n_detections):
        box_w = random.randint(max(20, w // 12), max(40, w // 4))
        box_h = random.randint(max(20, h // 12), max(40, h // 4))
        x1 = random.randint(0, max(0, w - box_w))
        y1 = random.randint(0, max(0, h - box_h))

        if is_occluded:
            confidence = round(random.uniform(0.15, 0.55), 4)
        elif lighting == "low":
            confidence = round(random.uniform(0.35, 0.75), 4)
        else:
            confidence = round(random.uniform(0.72, 0.99), 4)

        detections.append(
            {
                "animal_id": random.choice(ANIMAL_IDS),
                "behavior_label": random.choice(BEHAVIOR_LABELS),
                "confidence_score": confidence,
                "lighting_level": lighting,
                "is_occluded": is_occluded,
                "bbox": (x1, y1, x1 + box_w, y1 + box_h),
            }
        )
    return detections


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Overlay mock boxes for optional local preview windows."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (40, 180, 80) if det["confidence_score"] >= 0.7 else (40, 160, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['animal_id']} {det['confidence_score']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# DuckDB streaming writer
# ---------------------------------------------------------------------------


class TelemetryWriter:
    """Insert live frame telemetry rows into ``frame_telemetry`` in real time."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._con: duckdb.DuckDBPyConnection | None = None
        self.inserts = 0

    def open(self) -> None:
        self._con = connect(self.db_path)
        ensure_frame_telemetry_schema(self._con)
        log.info("Connected to DuckDB at %s", self.db_path)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
            log.info("DuckDB connection closed (%s inserts this session)", self.inserts)

    def insert_detections(
        self,
        *,
        frame_id: str,
        timestamp: datetime,
        camera_source: str,
        detections: list[dict[str, Any]],
    ) -> int:
        if self._con is None:
            raise RuntimeError("TelemetryWriter is not open")
        if not detections:
            return 0

        rows = [
            (
                frame_id,
                timestamp,
                camera_source,
                det["animal_id"],
                det["behavior_label"],
                float(det["confidence_score"]),
                det["lighting_level"],
                bool(det["is_occluded"]),
            )
            for det in detections
        ]
        self._con.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                frame_id,
                timestamp,
                camera_source,
                animal_id,
                behavior_label,
                confidence_score,
                lighting_level,
                is_occluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.inserts += len(rows)
        log.info(
            "DB insert OK | frame=%s | detections=%d | total_inserts=%d",
            frame_id,
            len(rows),
            self.inserts,
        )
        return len(rows)


# ---------------------------------------------------------------------------
# Ingestion pipeline (throttle + reconnect + graceful shutdown)
# ---------------------------------------------------------------------------


class CameraIngestionPipeline:
    """
    End-to-end educational ingest loop:

    Camera / file → sample @ target_fps → mock detect → DuckDB insert → optional preview
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.writer = TelemetryWriter(config.db_path)
        self._stop = False
        self._processed = 0
        self._dropped_read_failures = 0
        self._source_kind = classify_source(config.source)

    def request_stop(self, *_args: Any) -> None:
        """SIGINT / SIGTERM / 'q' → cooperative shutdown."""
        if not self._stop:
            log.info("Shutdown requested — draining current frame then exiting")
        self._stop = True

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def _should_reconnect(self) -> bool:
        # Files are finite; do not spin forever when the clip ends.
        return self._source_kind in {"rtsp", "usb"}

    def _open_with_retries(self) -> cv2.VideoCapture | None:
        attempts = 0
        while not self._stop:
            try:
                cap = open_capture(self.config.source)
                log.info(
                    "Camera connected | kind=%s | source=%s",
                    self._source_kind,
                    self.config.source_label,
                )
                return cap
            except Exception as exc:  # noqa: BLE001 — field demos need resilient logs
                attempts += 1
                log.warning(
                    "Connection failed (attempt %d): %s",
                    attempts,
                    exc,
                )
                if (
                    self.config.max_reconnect_attempts > 0
                    and attempts >= self.config.max_reconnect_attempts
                ):
                    log.error("Max reconnect attempts reached — giving up")
                    return None
                if not self._should_reconnect() and self._source_kind == "file":
                    return None
                time.sleep(self.config.reconnect_delay_sec)
        return None

    def _frame_interval(self) -> float:
        fps = max(self.config.target_fps, 0.05)
        return 1.0 / fps

    def _file_frame_stride(self, cap: cv2.VideoCapture) -> int:
        """
        For video files, convert target_fps into an integer frame stride so
        short clips still yield multiple samples under fast sequential decode.
        """
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        stride = max(1, int(round(source_fps / max(self.config.target_fps, 0.05))))
        log.info(
            "File throttle | source_fps=%.2f | target_fps=%.2f | stride=%d",
            source_fps,
            self.config.target_fps,
            stride,
        )
        return stride

    def iter_sampled_frames(
        self, cap: cv2.VideoCapture
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yield (raw_index, frame) at approximately ``target_fps``.

        Live RTSP/USB sources use wall-clock throttling and discard intermediate
        frames so camera buffers do not stall. Local files use a frame stride
        derived from the container FPS so educational clips still produce
        multiple telemetry rows when decoded faster than real time.
        """
        interval = self._frame_interval()
        next_sample_at = time.monotonic()
        raw_index = 0
        file_stride = (
            self._file_frame_stride(cap) if self._source_kind == "file" else None
        )

        while not self._stop:
            ok, frame = cap.read()
            raw_index += 1
            if not ok or frame is None:
                if self._source_kind == "file":
                    log.info("Reached end of video file after %d reads", raw_index - 1)
                else:
                    self._dropped_read_failures += 1
                    log.warning(
                        "Dropped / failed frame read (#%d) — stream may have stalled",
                        self._dropped_read_failures,
                    )
                break

            if file_stride is not None:
                # Process frames 1, 1+stride, 1+2*stride, ...
                if (raw_index - 1) % file_stride != 0:
                    continue
            else:
                now = time.monotonic()
                if now < next_sample_at:
                    # Live throttle: skip processing this frame to save CPU/GPU.
                    continue
                next_sample_at = now + interval

            yield raw_index, frame

            if self.config.max_frames is not None and self._processed >= self.config.max_frames:
                break

    def run(self) -> int:
        """
        Main loop. Returns the number of sampled frames that produced DB writes
        (including frames with zero detections, which still count as processed).
        """
        self._install_signal_handlers()
        self.writer.open()
        window_name = "Livestock Vision Ingest (Educational PoC) — press q to quit"

        try:
            while not self._stop:
                cap = self._open_with_retries()
                if cap is None:
                    break

                try:
                    for raw_index, frame in self.iter_sampled_frames(cap):
                        if self._stop:
                            break

                        detections = mock_detect_animals(frame)
                        frame_id = (
                            f"live_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
                            f"_{raw_index:06d}"
                        )
                        ts = datetime.now(timezone.utc).replace(tzinfo=None)
                        self.writer.insert_detections(
                            frame_id=frame_id,
                            timestamp=ts,
                            camera_source=self.config.source_label,
                            detections=detections,
                        )
                        self._processed += 1

                        if self.config.display:
                            preview = draw_detections(frame, detections)
                            try:
                                cv2.imshow(window_name, preview)
                                key = cv2.waitKey(1) & 0xFF
                                if key in (ord("q"), ord("Q")):
                                    log.info("Quit key pressed")
                                    self.request_stop()
                                    break
                            except cv2.error:
                                # Headless environments: disable further display attempts.
                                log.warning(
                                    "Display unavailable (headless?) — continuing without preview"
                                )
                                self.config.display = False

                        if (
                            self.config.max_frames is not None
                            and self._processed >= self.config.max_frames
                        ):
                            log.info("Reached max_frames=%d", self.config.max_frames)
                            self.request_stop()
                            break
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
                "Graceful shutdown complete | sampled_frames=%d | db_inserts=%d | "
                "failed_reads=%d",
                self._processed,
                self.writer.inserts,
                self._dropped_read_failures,
            )

        return self._processed


# ---------------------------------------------------------------------------
# Sample asset helper (optional demo footage when no barn camera is attached)
# ---------------------------------------------------------------------------


def write_sample_barn_footage(
    path: Path | str = "sample_barn_footage.mp4",
    *,
    frames: int = 60,
    fps: int = 10,
    width: int = 640,
    height: int = 360,
) -> Path:
    """
    Synthesize a short clip so learners can exercise the file-source path
    without a physical camera. Educational placeholder only.
    """
    out = Path(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create sample video at {out}")

    for i in range(frames):
        # Soft barn-aisle plate + moving blob. Alternate brighter / dimmer
        # segments so mock lighting_level has educational variety.
        dim = (i // 15) % 2 == 1
        base = 45 if dim else 130
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (base + (i % 20), base + 10, base)
        cx = int((i / max(frames - 1, 1)) * (width - 80)) + 40
        cy = height // 2
        blob = (70, 120, 90) if dim else (110, 170, 130)
        cv2.ellipse(frame, (cx, cy), (50, 30), 0, 0, 360, blob, -1)
        cv2.putText(
            frame,
            "EDU SAMPLE BARN FOOTAGE",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    log.info("Wrote educational sample footage → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Educational PoC camera connector: RTSP / USB / video file → "
            "mock detections → DuckDB livestock_analytics.duckdb"
        )
    )
    parser.add_argument(
        "--source",
        default="sample_barn_footage.mp4",
        help=(
            "RTSP URL, USB index (0/1), or video file path "
            "(default: sample_barn_footage.mp4)"
        ),
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help="Path to livestock_analytics.duckdb",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        help="Sample / process approximately this many frames per second",
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
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on sampled frames (handy for CI / headless demos)",
    )
    parser.add_argument(
        "--camera-label",
        default=None,
        help="Override camera_source string stored in DuckDB",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window (recommended on servers)",
    )
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help="Generate sample_barn_footage.mp4 before connecting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.make_sample:
        write_sample_barn_footage("sample_barn_footage.mp4")

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Auto-create educational sample clip when a file source is missing.
    if (
        isinstance(source, str)
        and classify_source(source) == "file"
        and not Path(source).exists()
    ):
        log.warning("Source file missing (%s) — generating educational sample clip", source)
        write_sample_barn_footage(source)

    # Ensure DB exists for first-time learners who skip db_setup.py.
    db_path = Path(args.db_path)
    if not db_path.exists():
        log.warning("Database missing — creating empty schema at %s", db_path)
        con = connect(db_path)
        try:
            ensure_frame_telemetry_schema(con)
        finally:
            con.close()

    config = CameraConfig(
        source=source,
        db_path=db_path,
        target_fps=args.target_fps,
        reconnect_delay_sec=args.reconnect_delay,
        max_reconnect_attempts=args.max_reconnect_attempts,
        display=not args.no_display,
        max_frames=args.max_frames,
        camera_label=args.camera_label,
    )

    # Dictionary-style usage is also supported programmatically:
    #   CameraIngestionPipeline(config_from_dict({...})).run()
    pipeline = CameraIngestionPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
