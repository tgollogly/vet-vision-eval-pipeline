"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
test_eval.py — PyTest quality harness for the livestock vision analytics demo.

The harness:
  1. Ensures the local DuckDB store is initialized with synthetic telemetry.
  2. Filters out occluded and low-light frames (degraded capture conditions).
  3. Asserts confidence benchmarks on the remaining "good condition" frames.

This file is open-source learning / portfolio software only — not a
production model gate, clinical QA system, or commercial monitor.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db_setup import DB_PATH, TABLE_NAME, initialize_database

# ---------------------------------------------------------------------------
# Educational QA benchmarks (chosen to pass the seeded synthetic generator)
# ---------------------------------------------------------------------------

# Minimum mean confidence expected on bright, unoccluded frames.
MIN_MEAN_CONFIDENCE_GOOD_FRAMES = 0.80

# Share of good-condition frames that must meet a "usable" confidence floor.
MIN_USABLE_CONFIDENCE = 0.70
MIN_USABLE_FRAME_RATIO = 0.90

# Absolute floor: even the weakest good-condition frame should clear this.
MIN_FLOOR_CONFIDENCE_GOOD_FRAMES = 0.65


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Build an isolated DuckDB file for the module so tests do not depend on
    a pre-existing workspace database from a manual quickstart run.
    """
    path = tmp_path_factory.mktemp("edu_poc_db") / "livestock_analytics.duckdb"
    initialize_database(db_path=path)
    return path


@pytest.fixture(scope="module")
def connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Read-only-style connection to the educational test database."""
    con = duckdb.connect(str(db_path), read_only=True)
    yield con
    con.close()


def _good_condition_stats(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, float | int]:
    """
    Filter out occluded / low-light frames and summarize confidence quality.

    Educational intent: QA should evaluate model skill on frames where the
    capture conditions are fair, then report degraded conditions separately
    (see analytics_engine.py).
    """
    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS good_frame_count,
            AVG(confidence_score) AS mean_confidence,
            MIN(confidence_score) AS min_confidence,
            SUM(
                CASE WHEN confidence_score >= {MIN_USABLE_CONFIDENCE} THEN 1 ELSE 0 END
            ) AS usable_frame_count
        FROM {TABLE_NAME}
        WHERE lighting_level = 'bright'
          AND is_occluded = FALSE
        """
    ).fetchone()

    good_frame_count, mean_confidence, min_confidence, usable_frame_count = row
    usable_ratio = (
        float(usable_frame_count) / float(good_frame_count)
        if good_frame_count
        else 0.0
    )
    return {
        "good_frame_count": int(good_frame_count),
        "mean_confidence": float(mean_confidence or 0.0),
        "min_confidence": float(min_confidence or 0.0),
        "usable_frame_count": int(usable_frame_count),
        "usable_ratio": usable_ratio,
    }


def test_database_contains_seeded_telemetry(connection: duckdb.DuckDBPyConnection) -> None:
    """Sanity check: educational seed produced the expected volume of rows."""
    count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert count == 1000, "Educational PoC expects 1,000 synthetic telemetry rows"


def test_good_condition_frames_exist_after_filters(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """After dropping occluded/low-light frames, some bright frames remain."""
    stats = _good_condition_stats(connection)
    assert stats["good_frame_count"] > 0, (
        "No bright/unoccluded frames left after filters — check synthetic seed"
    )


def test_mean_confidence_benchmark_on_good_frames(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Assert mean detector confidence on filtered (good-condition) frames.

    Low-light and occluded frames are excluded so the QA gate measures model
    quality under fair capture conditions, not environmental failure modes.
    """
    stats = _good_condition_stats(connection)
    assert stats["mean_confidence"] >= MIN_MEAN_CONFIDENCE_GOOD_FRAMES, (
        f"Mean confidence {stats['mean_confidence']:.4f} below educational "
        f"benchmark {MIN_MEAN_CONFIDENCE_GOOD_FRAMES}"
    )


def test_usable_confidence_ratio_on_good_frames(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Most good-condition frames should clear the usable confidence floor."""
    stats = _good_condition_stats(connection)
    assert stats["usable_ratio"] >= MIN_USABLE_FRAME_RATIO, (
        f"Usable ratio {stats['usable_ratio']:.3f} below educational "
        f"benchmark {MIN_USABLE_FRAME_RATIO}"
    )


def test_confidence_floor_on_good_frames(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Even the weakest bright/unoccluded frame should clear a soft floor."""
    stats = _good_condition_stats(connection)
    assert stats["min_confidence"] >= MIN_FLOOR_CONFIDENCE_GOOD_FRAMES, (
        f"Min confidence {stats['min_confidence']:.4f} below educational "
        f"floor {MIN_FLOOR_CONFIDENCE_GOOD_FRAMES}"
    )


def test_default_db_path_constant_is_documented() -> None:
    """
    Lightweight documentation guard: quickstart DB filename stays discoverable.

    Does not require the on-disk file to exist; that is created by db_setup.py.
    """
    assert DB_PATH.name == "livestock_analytics.duckdb"
