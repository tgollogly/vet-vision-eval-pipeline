"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
db_setup.py — Initialize a local DuckDB database and seed synthetic
frame telemetry for a livestock vision analytics portfolio demo.

This module is intended solely for learning, skills development, and
technical portfolio demonstration. It has zero commercial or monetized
intent and does not process real farm or veterinary data.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Educational project constants
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "livestock_analytics.duckdb"
TABLE_NAME = "frame_telemetry"
SEED_ROW_COUNT = 1000
RANDOM_SEED = 42  # Reproducible synthetic data for demos and tests

# Simulated farm / model vocabulary (synthetic only)
ANIMAL_IDS = [f"animal_{i:03d}" for i in range(1, 21)]
BEHAVIOR_LABELS = ("lying", "standing", "walking", "feeding")
LIGHTING_LEVELS = ("bright", "low")


def connect(db_path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Open (or create) the local DuckDB file used by this learning project.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file. Defaults to livestock_analytics.duckdb
        next to this script.
    """
    return duckdb.connect(str(db_path))


def create_frame_telemetry_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the frame_telemetry table that stores simulated CV detector outputs.

    Fields mirror a typical educational telemetry schema:
    frame identity, time, animal id, behavior, confidence, lighting, occlusion.
    """
    # Drop + recreate so repeated local runs stay deterministic for learners.
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            frame_id         VARCHAR NOT NULL,
            timestamp        TIMESTAMP NOT NULL,
            camera_source    VARCHAR NOT NULL,
            animal_id        VARCHAR NOT NULL,
            behavior_label   VARCHAR NOT NULL,
            confidence_score DOUBLE NOT NULL,
            lighting_level   VARCHAR NOT NULL,
            is_occluded      BOOLEAN NOT NULL
        )
        """
    )


def ensure_frame_telemetry_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create ``frame_telemetry`` if missing, and add ``camera_source`` when an
    older educational DB was seeded before live camera ingestion existed.

    Safe for the camera connector to call on every startup (idempotent).
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            frame_id         VARCHAR NOT NULL,
            timestamp        TIMESTAMP NOT NULL,
            camera_source    VARCHAR NOT NULL,
            animal_id        VARCHAR NOT NULL,
            behavior_label   VARCHAR NOT NULL,
            confidence_score DOUBLE NOT NULL,
            lighting_level   VARCHAR NOT NULL,
            is_occluded      BOOLEAN NOT NULL
        )
        """
    )
    # DuckDB supports IF NOT EXISTS on ADD COLUMN for forward-compatible demos.
    con.execute(
        f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS camera_source VARCHAR DEFAULT 'synthetic_seed'
        """
    )


def _synthetic_confidence(lighting_level: str, is_occluded: bool) -> float:
    """
    Generate a plausible confidence score shaped by lighting and occlusion.

    Educational intent: low light and occlusions should tend to produce
    weaker model confidence so downstream failure metrics have signal.
    """
    if is_occluded:
        # Occlusion is a harsh failure mode in this synthetic generator.
        return round(random.uniform(0.15, 0.55), 4)
    if lighting_level == "low":
        # Low-light frames: mixed quality, often below a "good" threshold.
        return round(random.uniform(0.35, 0.75), 4)
    # Bright, unoccluded: generally strong detections for the QA harness.
    return round(random.uniform(0.72, 0.99), 4)


def seed_synthetic_rows(
    con: duckdb.DuckDBPyConnection,
    n_rows: int = SEED_ROW_COUNT,
    *,
    seed: int = RANDOM_SEED,
) -> int:
    """
    Insert synthetic telemetry rows representing real-world farm conditions
    (bright scenes, low light, and camera occlusions).

    Returns
    -------
    int
        Number of rows inserted (always ``n_rows`` on success).
    """
    random.seed(seed)

    # Spread frames over a few calendar days so lying-duration analytics
    # can group by day in the educational engine.
    base_time = datetime(2026, 1, 1, 6, 0, 0)
    rows: list[tuple] = []

    # Force a small set of "review candidate" animals with dense lying frames
    # so the illness-flag *demo* metric fires during the educational walkthrough.
    prolonged_lying_animals = {"animal_003", "animal_011"}

    for i in range(n_rows):
        lighting_level = random.choices(
            LIGHTING_LEVELS,
            weights=(0.65, 0.35),  # Majority bright; meaningful low-light share
            k=1,
        )[0]
        is_occluded = random.random() < 0.18  # ~18% occlusion rate

        # Bias animal selection toward the demo review candidates on day 1.
        if i < 400 and random.random() < 0.35:
            animal_id = random.choice(sorted(prolonged_lying_animals))
        else:
            animal_id = random.choice(ANIMAL_IDS)

        if animal_id in prolonged_lying_animals and i < 400:
            behavior = "lying"
        else:
            behavior = random.choices(
                BEHAVIOR_LABELS,
                weights=(0.30, 0.30, 0.25, 0.15),
                k=1,
            )[0]

        rows.append(
            (
                f"frame_{i:05d}",
                base_time + timedelta(minutes=i * 3),  # ~3-minute cadence
                "synthetic_seed",  # camera_source for offline educational seeds
                animal_id,
                behavior,
                _synthetic_confidence(lighting_level, is_occluded),
                lighting_level,
                is_occluded,
            )
        )

    con.executemany(
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
    return n_rows


def initialize_database(
    db_path: Path | str = DB_PATH,
    n_rows: int = SEED_ROW_COUNT,
) -> Path:
    """
    Full educational bootstrap: connect, create schema, seed synthetic rows.

    Returns the path to the DuckDB file for convenience in scripts/tests.
    """
    path = Path(db_path)
    con = connect(path)
    try:
        create_frame_telemetry_table(con)
        inserted = seed_synthetic_rows(con, n_rows=n_rows)
        count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        print(
            "[educational-poc] Initialized DuckDB at "
            f"{path} with {count} frame_telemetry rows "
            f"(requested seed={inserted})."
        )
    finally:
        con.close()
    return path


def main() -> None:
    """CLI entrypoint for the Quickstart Guide in README.md."""
    initialize_database()


if __name__ == "__main__":
    main()
