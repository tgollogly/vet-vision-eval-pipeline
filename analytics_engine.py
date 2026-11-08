"""
Educational Proof-of-Concept / Open-Source Learning Project
------------------------------------------------------------
analytics_engine.py — Pure Python + DuckDB SQL analytics for:
  (a) Daily animal lying durations (illness-flagging *demo* pattern)
  (b) Model performance under low light vs camera occlusions

This module exists only for learning, skills development, and portfolio
demonstration. Metrics are synthetic teaching aids — not veterinary
diagnosis, welfare certification, or commercial monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from db_setup import DB_PATH, TABLE_NAME, connect

# Confidence below this value is treated as a "low-confidence" prediction
# in the educational failure-mode report.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Synthetic demo threshold: lying longer than this many minutes/day is
# printed as a "review candidate" so learners see how alerts might look.
# Not a clinical rule.
LYING_FLAG_MINUTES = 180.0

# Assumed frame sampling interval used when converting lying-frame counts
# into approximate duration (matches the 3-minute cadence in db_setup.py).
FRAME_INTERVAL_MINUTES = 3.0


def fetch_daily_lying_durations(
    con: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """
    (a) Aggregate approximate daily lying time per animal.

    Educational framing: prolonged lying can be a signal worth reviewing
    in real welfare workflows. Here we only practice SQL aggregation and
    a simple threshold flag for portfolio demonstration.
    """
    query = f"""
        SELECT
            animal_id,
            CAST(timestamp AS DATE) AS observation_day,
            COUNT(*) AS lying_frame_count,
            COUNT(*) * {FRAME_INTERVAL_MINUTES} AS lying_duration_minutes,
            CASE
                WHEN COUNT(*) * {FRAME_INTERVAL_MINUTES} >= {LYING_FLAG_MINUTES}
                THEN TRUE
                ELSE FALSE
            END AS potential_illness_flag
        FROM {TABLE_NAME}
        WHERE behavior_label = 'lying'
        GROUP BY animal_id, CAST(timestamp AS DATE)
        ORDER BY observation_day, animal_id
    """
    result = con.execute(query).fetchdf()
    return result.to_dict(orient="records")


def fetch_model_performance_metrics(
    con: duckdb.DuckDBPyConnection,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    (b) Compare share of low-confidence predictions under low light vs occlusion.

    Failure-mode monitoring pattern for learning MLOps evaluation reports:
    isolate environmental conditions and measure how often the model is unsure.
    """
    query = f"""
        WITH labeled AS (
            SELECT
                CASE
                    WHEN is_occluded THEN 'occlusion'
                    WHEN lighting_level = 'low' THEN 'low_light'
                    ELSE 'bright_clear'
                END AS condition_bucket,
                confidence_score,
                CASE
                    WHEN confidence_score < {low_confidence_threshold} THEN 1
                    ELSE 0
                END AS is_low_confidence
            FROM {TABLE_NAME}
        )
        SELECT
            condition_bucket,
            COUNT(*) AS frame_count,
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
    result = con.execute(query).fetchdf()
    return result.to_dict(orient="records")


def run_analytics(db_path: Path | str = DB_PATH) -> dict[str, list[dict[str, Any]]]:
    """
    Execute both educational analytics queries against the local DuckDB file.

    Returns a dict with ``lying_durations`` and ``model_performance`` lists.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run `python3 db_setup.py` first "
            "(Educational PoC quickstart)."
        )

    con = connect(path)
    try:
        lying = fetch_daily_lying_durations(con)
        performance = fetch_model_performance_metrics(con)
    finally:
        con.close()

    return {
        "lying_durations": lying,
        "model_performance": performance,
    }


def _print_report(payload: dict[str, list[dict[str, Any]]]) -> None:
    """Pretty-print analytics for the educational CLI demo."""
    print("\n=== Educational PoC: Daily lying durations ===")
    print(
        "(potential_illness_flag is a synthetic teaching alert, "
        "not a veterinary diagnosis)\n"
    )
    lying_rows = payload["lying_durations"]
    if not lying_rows:
        print("No lying frames found in synthetic telemetry.")
    else:
        flagged = 0
        for row in lying_rows:
            flag = "FLAG" if row["potential_illness_flag"] else "ok"
            if row["potential_illness_flag"]:
                flagged += 1
            print(
                f"  {row['observation_day']} | {row['animal_id']}: "
                f"{row['lying_duration_minutes']:.0f} min "
                f"({row['lying_frame_count']} frames) [{flag}]"
            )
        print(f"\n  Review candidates (synthetic): {flagged}/{len(lying_rows)}")

    print("\n=== Educational PoC: Model failure metrics ===")
    print(
        f"(low confidence = score < {LOW_CONFIDENCE_THRESHOLD}; "
        "compares low light vs occlusions)\n"
    )
    for row in payload["model_performance"]:
        print(
            f"  {row['condition_bucket']}: "
            f"{row['low_confidence_pct']}% low-confidence "
            f"({row['low_confidence_count']}/{row['frame_count']} frames), "
            f"mean confidence={row['mean_confidence']}"
        )


def main() -> None:
    """CLI entrypoint for the Quickstart Guide in README.md."""
    payload = run_analytics()
    _print_report(payload)


if __name__ == "__main__":
    main()
