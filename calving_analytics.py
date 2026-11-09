"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
calving_analytics.py — SQL analytics over ``cattle_telemetry`` for:

  (a) Calving Restlessness Score — posture transitions in a rolling window
  (b) Daily Lying Durations — hours lying per animal ID
  (c) Model Quality & Edge Failures — low-confidence rates under low light
      or occlusion

This module is unmonetized open-source software built solely for learning,
skills development, and technical portfolio demonstration, with zero
commercial intent. Metrics are teaching aids — not veterinary diagnosis,
calving prediction certification, or commercial farm monitoring.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from cattle_vision_pipeline import CATTLE_TELEMETRY_TABLE, ensure_cattle_telemetry_table
from db_setup import DB_PATH, connect

# ---------------------------------------------------------------------------
# Educational thresholds (portfolio / learning defaults — not clinical rules)
# ---------------------------------------------------------------------------

# Rolling window for posture-shift counting (calving restlessness demo).
RESTLESSNESS_WINDOW_HOURS = 2.0

# Alert when standing <-> lying transitions in the window exceed this count.
# Tuned for ~1 Hz educational sampling: calm animals stay below; restless demo exceeds.
RESTLESSNESS_TRANSITION_THRESHOLD = 200

# Assumed sampling interval when converting lying-frame counts → hours.
# Matches the pipeline default of ~1 processed frame per second.
ASSUMED_SAMPLE_INTERVAL_SECONDS = 1.0

# Confidence below this is treated as a low-confidence / edge failure.
LOW_CONFIDENCE_THRESHOLD = 0.60


def fetch_calving_restlessness_scores(
    con: duckdb.DuckDBPyConnection,
    *,
    window_hours: float = RESTLESSNESS_WINDOW_HOURS,
    transition_threshold: int = RESTLESSNESS_TRANSITION_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    (a) Count posture shifts (standing <-> lying) per animal in a rolling window.

    Educational framing: spikes in transitions are used in teaching materials as
    a *proxy signal* for restlessness / active labour patterns. This is **not**
    a clinical calving detector.
    """
    # DuckDB interval from hours as seconds for a portable window bound.
    window_seconds = int(window_hours * 3600)
    query = f"""
        WITH ordered AS (
            SELECT
                animal_id,
                timestamp,
                behavior_label,
                LAG(behavior_label) OVER (
                    PARTITION BY animal_id
                    ORDER BY timestamp
                ) AS prev_behavior
            FROM {CATTLE_TELEMETRY_TABLE}
            WHERE behavior_label IN ('standing', 'lying')
        ),
        transitions AS (
            SELECT
                animal_id,
                timestamp,
                CASE
                    WHEN prev_behavior IS NOT NULL
                     AND prev_behavior <> behavior_label
                    THEN 1
                    ELSE 0
                END AS is_transition
            FROM ordered
        ),
        windowed AS (
            SELECT
                animal_id,
                timestamp AS window_end,
                SUM(is_transition) OVER (
                    PARTITION BY animal_id
                    ORDER BY timestamp
                    RANGE BETWEEN INTERVAL '{window_seconds} seconds' PRECEDING
                              AND CURRENT ROW
                ) AS posture_transitions_in_window
            FROM transitions
        ),
        latest AS (
            SELECT
                animal_id,
                MAX(window_end) AS latest_timestamp
            FROM windowed
            GROUP BY animal_id
        )
        SELECT
            w.animal_id,
            w.window_end AS as_of_timestamp,
            w.posture_transitions_in_window AS calving_restlessness_score,
            CASE
                WHEN w.posture_transitions_in_window >= {transition_threshold}
                THEN TRUE
                ELSE FALSE
            END AS calving_alert_flag
        FROM windowed w
        INNER JOIN latest l
            ON w.animal_id = l.animal_id
           AND w.window_end = l.latest_timestamp
        ORDER BY w.animal_id
    """
    return con.execute(query).fetchdf().to_dict(orient="records")


def fetch_daily_lying_durations(
    con: duckdb.DuckDBPyConnection,
    *,
    sample_interval_seconds: float = ASSUMED_SAMPLE_INTERVAL_SECONDS,
) -> list[dict[str, Any]]:
    """
    (b) Aggregate approximate daily lying hours per animal ID.

    Educational baseline often cited in teaching notes is ~10–14 hours/day lying
    for dairy cattle; deviations are discussed as *signals worth reviewing*,
    not diagnoses of mastitis or lameness.
    """
    hours_per_frame = sample_interval_seconds / 3600.0
    query = f"""
        SELECT
            animal_id,
            CAST(timestamp AS DATE) AS observation_day,
            COUNT(*) AS lying_frame_count,
            ROUND(COUNT(*) * {hours_per_frame}, 3) AS lying_hours,
            CASE
                WHEN COUNT(*) * {hours_per_frame} < 10.0
                  OR COUNT(*) * {hours_per_frame} > 14.0
                THEN TRUE
                ELSE FALSE
            END AS outside_baseline_band
        FROM {CATTLE_TELEMETRY_TABLE}
        WHERE behavior_label = 'lying'
          AND is_occluded = FALSE
          AND lighting_level = 'bright'
        GROUP BY animal_id, CAST(timestamp AS DATE)
        ORDER BY observation_day, animal_id
    """
    return con.execute(query).fetchdf().to_dict(orient="records")


def fetch_model_quality_edge_failures(
    con: duckdb.DuckDBPyConnection,
    *,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    (c) Percentage of low-confidence detections under low light or occlusion.

    Also reports the complementary "good condition" slice (bright + unoccluded)
    so learners can compare edge failures vs fair capture conditions.
    """
    query = f"""
        WITH labeled AS (
            SELECT
                CASE
                    WHEN is_occluded THEN 'occlusion'
                    WHEN lighting_level IN ('low_light', 'night') THEN 'low_light'
                    ELSE 'bright_clear'
                END AS condition_bucket,
                confidence_score,
                CASE
                    WHEN confidence_score < {low_confidence_threshold} THEN 1
                    ELSE 0
                END AS is_low_confidence
            FROM {CATTLE_TELEMETRY_TABLE}
        )
        SELECT
            condition_bucket,
            COUNT(*) AS detection_count,
            SUM(is_low_confidence) AS low_confidence_count,
            ROUND(
                100.0 * SUM(is_low_confidence) / NULLIF(COUNT(*), 0),
                2
            ) AS low_confidence_pct,
            ROUND(AVG(confidence_score), 4) AS mean_confidence
        FROM labeled
        GROUP BY condition_bucket
        ORDER BY condition_bucket
    """
    return con.execute(query).fetchdf().to_dict(orient="records")


def run_calving_analytics(
    db_path: Path | str = DB_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Execute all three educational analytics queries against DuckDB."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run cattle_vision_pipeline.py "
            "or seed_cattle_telemetry_demo() first (Educational PoC)."
        )

    con = connect(path)
    try:
        ensure_cattle_telemetry_table(con)
        count = con.execute(
            f"SELECT COUNT(*) FROM {CATTLE_TELEMETRY_TABLE}"
        ).fetchone()[0]
        if count == 0:
            raise RuntimeError(
                f"{CATTLE_TELEMETRY_TABLE} is empty. Ingest camera telemetry "
                "or run with --seed-demo first."
            )
        return {
            "calving_restlessness": fetch_calving_restlessness_scores(con),
            "daily_lying_durations": fetch_daily_lying_durations(con),
            "model_quality_edge_failures": fetch_model_quality_edge_failures(con),
        }
    finally:
        con.close()


def seed_cattle_telemetry_demo(
    db_path: Path | str = DB_PATH,
    *,
    n_animals: int = 3,
    hours: int = 3,
) -> Path:
    """
    Insert synthetic cattle_telemetry rows so calving analytics and tests
    run without a live barn camera (Educational PoC only).

    Rows are spaced at ``ASSUMED_SAMPLE_INTERVAL_SECONDS`` so lying-hour
    math matches the ~1 FPS cattle vision pipeline default.
    """
    from cattle_vision_pipeline import CattleTelemetryWriter

    path = Path(db_path)
    writer = CattleTelemetryWriter(path)
    writer.open()
    try:
        base = datetime(2026, 3, 1, 6, 0, 0)
        records: list[dict[str, Any]] = []
        step = max(int(ASSUMED_SAMPLE_INTERVAL_SECONDS), 1)
        n_steps = int((hours * 3600) / step)
        # Animal 0 is "restless" (many posture flips) for alert-demo teaching.
        for step_i in range(n_steps):
            ts = base + timedelta(seconds=step_i * step)
            for animal_idx in range(n_animals):
                animal_id = f"track_{animal_idx:04d}"
                if animal_idx == 0:
                    # Flip often → high restlessness score in a 2h window.
                    behavior = "lying" if (step_i % 4) < 2 else "standing"
                    confidence = 0.85
                    lighting = "bright"
                    occluded = False
                elif animal_idx == 1:
                    # Mostly lying / calm — few posture flips (below alert threshold).
                    behavior = "lying" if step_i % 600 < 540 else "standing"
                    confidence = 0.82
                    lighting = "bright"
                    occluded = False
                else:
                    # Edge-failure demo animal: dim / occluded / lower conf; calm posture.
                    behavior = "standing" if step_i % 900 < 800 else "lying"
                    confidence = 0.45 if step_i % 3 == 0 else 0.55
                    lighting = "low_light" if step_i % 2 == 0 else "night"
                    occluded = step_i % 4 == 0

                records.append(
                    {
                        "frame_id": f"demo_{animal_idx}_{step_i:05d}",
                        "timestamp": ts,
                        "camera_source": "synthetic_demo",
                        "animal_id": animal_id,
                        "behavior_label": behavior,
                        "confidence_score": confidence,
                        "bounding_box": "[10.0, 20.0, 200.0, 90.0]"
                        if behavior == "lying"
                        else "[10.0, 20.0, 80.0, 220.0]",
                        "lighting_level": lighting,
                        "is_occluded": occluded,
                    }
                )
        chunk = 1000
        for i in range(0, len(records), chunk):
            writer.insert_records(records[i : i + chunk])
        print(
            f"[educational-poc] Seeded {len(records)} cattle_telemetry rows at {path}"
        )
    finally:
        writer.close()
    return path


def _print_report(payload: dict[str, list[dict[str, Any]]]) -> None:
    print("\n=== Educational PoC: Calving restlessness (2h window) ===")
    print(
        "(calving_alert_flag is a synthetic teaching alert, "
        "not a veterinary labour diagnosis)\n"
    )
    for row in payload["calving_restlessness"]:
        flag = "ALERT" if row["calving_alert_flag"] else "ok"
        print(
            f"  {row['animal_id']}: score={row['calving_restlessness_score']} "
            f"as_of={row['as_of_timestamp']} [{flag}]"
        )

    print("\n=== Educational PoC: Daily lying durations ===")
    print("(outside_baseline_band uses a 10–14 h teaching band)\n")
    for row in payload["daily_lying_durations"]:
        flag = "REVIEW" if row["outside_baseline_band"] else "ok"
        print(
            f"  {row['observation_day']} | {row['animal_id']}: "
            f"{row['lying_hours']} h ({row['lying_frame_count']} frames) [{flag}]"
        )

    print("\n=== Educational PoC: Model quality & edge failures ===")
    print(f"(low confidence = score < {LOW_CONFIDENCE_THRESHOLD})\n")
    for row in payload["model_quality_edge_failures"]:
        print(
            f"  {row['condition_bucket']}: "
            f"{row['low_confidence_pct']}% low-confidence "
            f"({row['low_confidence_count']}/{row['detection_count']}), "
            f"mean={row['mean_confidence']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Educational calving / welfare SQL analytics over cattle_telemetry "
            "in livestock_analytics.duckdb"
        )
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help="Path to livestock_analytics.duckdb",
    )
    parser.add_argument(
        "--seed-demo",
        action="store_true",
        help="Seed synthetic cattle_telemetry rows before running analytics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    db_path = Path(args.db_path)
    if args.seed_demo or not db_path.exists():
        seed_cattle_telemetry_demo(db_path)
    payload = run_calving_analytics(db_path)
    _print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
