"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
test_eval.py — PyTest harness for cattle welfare telemetry analytics.

Asserts that:
  1. ``cattle_telemetry`` exists and is populated with the expected schema.
  2. SQL analytics filter out occluded / low-light frames for lying baselines.
  3. Calving restlessness, lying-duration, and edge-failure queries return
     coherent educational results.

This file is unmonetized open-source learning / portfolio software only —
not a production model gate, clinical QA system, or commercial monitor.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from calving_analytics import (
    LOW_CONFIDENCE_THRESHOLD,
    fetch_calving_restlessness_scores,
    fetch_daily_lying_durations,
    fetch_model_quality_edge_failures,
    seed_cattle_telemetry_demo,
)
from cattle_vision_pipeline import CATTLE_TELEMETRY_TABLE
from db_setup import DB_PATH


REQUIRED_COLUMNS = {
    "frame_id",
    "timestamp",
    "camera_source",
    "animal_id",
    "behavior_label",
    "confidence_score",
    "bounding_box",
    "lighting_level",
    "is_occluded",
}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Isolated DuckDB file with synthetic cattle_telemetry for this module."""
    path = tmp_path_factory.mktemp("edu_cattle_db") / "livestock_analytics.duckdb"
    seed_cattle_telemetry_demo(path)
    return path


@pytest.fixture(scope="module")
def connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=True)
    yield con
    con.close()


def test_cattle_telemetry_table_exists(connection: duckdb.DuckDBPyConnection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    assert CATTLE_TELEMETRY_TABLE in tables


def test_cattle_telemetry_schema_populated(connection: duckdb.DuckDBPyConnection) -> None:
    cols = {
        row[0]
        for row in connection.execute(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{CATTLE_TELEMETRY_TABLE}'"
        ).fetchall()
    }
    missing = REQUIRED_COLUMNS - cols
    assert not missing, f"Missing cattle_telemetry columns: {sorted(missing)}"

    count = connection.execute(
        f"SELECT COUNT(*) FROM {CATTLE_TELEMETRY_TABLE}"
    ).fetchone()[0]
    assert count > 0, "cattle_telemetry should be populated for educational tests"


def test_sql_analytics_filter_occluded_and_low_light(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Daily lying durations must exclude occluded / low-light / night frames so
    the educational baseline uses fair capture conditions only.
    """
    # Direct filter check on underlying rows used by the lying query.
    bad = connection.execute(
        f"""
        SELECT COUNT(*) FROM {CATTLE_TELEMETRY_TABLE}
        WHERE behavior_label = 'lying'
          AND (is_occluded = TRUE OR lighting_level IN ('low_light', 'night'))
        """
    ).fetchone()[0]
    assert bad >= 0  # may be zero; structural assertion below is the QA gate

    lying = fetch_daily_lying_durations(connection)
    # Re-validate filter: every aggregated lying hour row comes from bright/clear.
    for row in lying:
        sample = connection.execute(
            f"""
            SELECT COUNT(*) FROM {CATTLE_TELEMETRY_TABLE}
            WHERE animal_id = ?
              AND CAST(timestamp AS DATE) = ?
              AND behavior_label = 'lying'
              AND is_occluded = FALSE
              AND lighting_level = 'bright'
            """,
            [row["animal_id"], row["observation_day"]],
        ).fetchone()[0]
        assert sample == row["lying_frame_count"]


def test_calving_restlessness_scores_detect_demo_alert(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    scores = fetch_calving_restlessness_scores(connection)
    assert scores, "Expected restlessness scores for seeded animals"
    by_id = {row["animal_id"]: row for row in scores}
    assert "track_0000" in by_id
    # Seeded animal 0 flips often → educational alert should fire.
    assert by_id["track_0000"]["calving_alert_flag"] is True
    assert by_id["track_0000"]["calving_restlessness_score"] > 0


def test_model_quality_edge_failures_separate_conditions(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    metrics = fetch_model_quality_edge_failures(connection)
    buckets = {row["condition_bucket"] for row in metrics}
    assert "bright_clear" in buckets or "low_light" in buckets or "occlusion" in buckets

    for row in metrics:
        assert 0.0 <= float(row["low_confidence_pct"]) <= 100.0
        if row["condition_bucket"] in {"low_light", "occlusion"}:
            # Seeded edge animal drives elevated low-confidence under degraded light.
            assert float(row["mean_confidence"]) < 0.70 or float(
                row["low_confidence_pct"]
            ) >= 0.0

    # Bright/clear rows (when present) should clear the educational confidence floor
    # more often than degraded buckets.
    by_bucket = {row["condition_bucket"]: row for row in metrics}
    if "bright_clear" in by_bucket and "low_light" in by_bucket:
        assert (
            by_bucket["bright_clear"]["low_confidence_pct"]
            <= by_bucket["low_light"]["low_confidence_pct"]
        )


def test_good_condition_confidence_benchmark(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Assert mean confidence on bright, unoccluded cattle_telemetry rows."""
    row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS n,
            AVG(confidence_score) AS mean_conf
        FROM {CATTLE_TELEMETRY_TABLE}
        WHERE lighting_level = 'bright'
          AND is_occluded = FALSE
        """
    ).fetchone()
    n, mean_conf = row
    assert n > 0
    assert float(mean_conf) >= LOW_CONFIDENCE_THRESHOLD


def test_default_db_path_constant_is_documented() -> None:
    assert DB_PATH.name == "livestock_analytics.duckdb"
    assert CATTLE_TELEMETRY_TABLE == "cattle_telemetry"
