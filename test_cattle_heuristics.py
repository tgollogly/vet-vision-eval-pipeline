"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
Unit tests for cattle posture / lighting / occlusion heuristics and the
``cattle_telemetry`` DuckDB schema. These tests intentionally avoid loading
Ultralytics YOLO weights so CI stays CPU-light.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cattle_vision_pipeline import (
    COCO_COW_CLASS_ID,
    CattleTelemetryWriter,
    classify_lighting_level,
    ensure_cattle_telemetry_table,
    extract_cattle_detections,
    infer_posture_label,
    is_bbox_occluded_by_edge,
    serialize_bbox,
)
from db_setup import connect


class _FakeTensor:
    """Minimal tensor stub mimicking Ultralytics ``Boxes`` CPU tensors."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls, track_ids=None):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)
        self.id = None if track_ids is None else _FakeTensor(track_ids)

    def __len__(self):
        return len(self.conf.numpy())


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_infer_posture_lying_when_wide() -> None:
    # Wide box → lying
    assert infer_posture_label(0, 0, 200, 80) == "lying"


def test_infer_posture_standing_when_tall() -> None:
    # Tall box → standing
    assert infer_posture_label(0, 0, 80, 200) == "standing"


def test_classify_lighting_levels() -> None:
    night = np.full((40, 40, 3), 10, dtype=np.uint8)
    low = np.full((40, 40, 3), 60, dtype=np.uint8)
    bright = np.full((40, 40, 3), 180, dtype=np.uint8)
    assert classify_lighting_level(night) == "night"
    assert classify_lighting_level(low) == "low_light"
    assert classify_lighting_level(bright) == "bright"


def test_edge_occlusion_flag() -> None:
    assert is_bbox_occluded_by_edge(0, 10, 50, 80, 320, 240) is True
    assert is_bbox_occluded_by_edge(40, 40, 120, 160, 320, 240) is False


def test_serialize_bbox_json() -> None:
    assert serialize_bbox(1.234, 2, 3.5, 4) == "[1.23, 2.0, 3.5, 4.0]"


def test_extract_cattle_detections_filters_and_labels() -> None:
    # Include a person (class 0) that must be filtered out, plus a wide cow box.
    boxes = _FakeBoxes(
        xyxy=[[5, 5, 60, 180], [10, 40, 220, 100]],
        conf=[0.9, 0.88],
        cls=[0, COCO_COW_CLASS_ID],
        track_ids=[7, 3],
    )
    dets = extract_cattle_detections(
        _FakeResult(boxes),
        frame_shape=(240, 320, 3),
        lighting_level="bright",
        allowed_class_ids={COCO_COW_CLASS_ID},
        lying_aspect_ratio=1.35,
        conf_threshold=0.25,
    )
    assert len(dets) == 1
    assert dets[0].animal_id == "track_0003"
    assert dets[0].behavior_label == "lying"
    assert dets[0].class_name == "cow"
    assert dets[0].is_occluded is False


def test_cattle_telemetry_insert(tmp_path: Path) -> None:
    db_path = tmp_path / "livestock_analytics.duckdb"
    writer = CattleTelemetryWriter(db_path)
    writer.open()
    try:
        n = writer.insert_records(
            [
                {
                    "frame_id": "cattle_test_001",
                    "timestamp": datetime(2026, 1, 1, 12, 0, 0),
                    "camera_source": "usb:0",
                    "animal_id": "track_0001",
                    "behavior_label": "lying",
                    "confidence_score": 0.91,
                    "bounding_box": serialize_bbox(10, 20, 200, 90),
                    "lighting_level": "bright",
                    "is_occluded": False,
                }
            ]
        )
        assert n == 1
        con = connect(db_path)
        try:
            ensure_cattle_telemetry_table(con)
            row = con.execute(
                "SELECT animal_id, behavior_label, lighting_level, camera_source "
                "FROM cattle_telemetry"
            ).fetchone()
            assert row == ("track_0001", "lying", "bright", "usb:0")
        finally:
            con.close()
    finally:
        writer.close()
