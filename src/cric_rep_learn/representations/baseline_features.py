"""Generate leakage-safe rolling baseline features for neural residual learning."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from cric_rep_learn.baselines.historical import (
    BASELINE_LEVELS,
    HistoricalBaseline,
    MatchContext,
    SmoothingConfig,
)

BASELINE_PROBABILITY_COLUMNS = (
    [f"baseline_runs_{index}" for index in range(8)]
    + [f"baseline_extras_{index}" for index in range(8)]
    + [f"baseline_legality_{index}" for index in range(3)]
    + ["baseline_batter_dismissal", "baseline_bowler_wicket"]
)
EVIDENCE_COLUMNS = [
    "batter_evidence_log1p",
    "bowler_evidence_log1p",
    "venue_evidence_log1p",
    "matchup_evidence_log1p",
]


def _escape_sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _feature_schema() -> pa.Schema:
    return pa.schema(
        [
            ("match_id", pa.string()),
            ("innings", pa.int16()),
            ("attempt_index_in_innings", pa.int32()),
            *[(column, pa.float32()) for column in BASELINE_PROBABILITY_COLUMNS],
            *[(column, pa.float32()) for column in EVIDENCE_COLUMNS],
        ]
    )


def _ordered_reader(canonical_dir: Path, batch_size: int):
    deliveries = _escape_sql_path(canonical_dir / "deliveries.parquet")
    matches = _escape_sql_path(canonical_dir / "matches.parquet")
    query = f"""
        SELECT
            d.match_date,
            d.match_id,
            d.innings,
            d.attempt_index_in_innings,
            d.is_super_over,
            d.phase,
            d.wickets_before,
            d.batter_id,
            d.bowler_id,
            d.runs_batter,
            d.runs_extras,
            d.extras_wides,
            d.extras_noballs,
            d.batter_dismissed,
            d.bowler_wicket_count,
            m.gender,
            m.team_type,
            m.venue
        FROM read_parquet('{deliveries}') d
        JOIN read_parquet('{matches}') m USING (match_id)
        ORDER BY
            d.match_date,
            d.match_id,
            d.innings,
            d.attempt_index_in_innings
    """
    connection = duckdb.connect()
    return connection, connection.execute(query).to_arrow_reader(batch_size)


def _feature_row(
    row: dict[str, Any],
    model: HistoricalBaseline,
    level: str,
) -> dict[str, Any]:
    context = MatchContext(
        gender=row["gender"],
        team_type=row["team_type"],
        venue=row["venue"],
    )
    prediction = model.predict_all(row, context)[level]
    evidence = prediction.evidence
    values = {
        "match_id": row["match_id"],
        "innings": row["innings"],
        "attempt_index_in_innings": row["attempt_index_in_innings"],
        **{
            f"baseline_runs_{index}": float(probability)
            for index, probability in enumerate(prediction.batter_runs)
        },
        **{
            f"baseline_extras_{index}": float(probability)
            for index, probability in enumerate(prediction.extras_runs)
        },
        **{
            f"baseline_legality_{index}": float(probability)
            for index, probability in enumerate(prediction.legality)
        },
        "baseline_batter_dismissal": float(prediction.batter_dismissal),
        "baseline_bowler_wicket": float(prediction.bowler_wicket),
    }
    for key in ("batter", "bowler", "venue", "matchup"):
        values[f"{key}_evidence_log1p"] = math.log1p(evidence[key])
    return values


def generate_baseline_features(
    canonical_dir: Path,
    output_path: Path,
    *,
    level: str = "context",
    smoothing: SmoothingConfig | None = None,
    batch_size: int = 100_000,
    player_attributes_path: Path | None = None,
) -> dict[str, Any]:
    if level not in BASELINE_LEVELS:
        raise ValueError(f"Unknown baseline level {level!r}; expected one of {BASELINE_LEVELS}")

    from cric_rep_learn.data.player_attributes import load_attributes_index

    attributes = None
    if player_attributes_path is not None and player_attributes_path.exists():
        attributes = load_attributes_index(player_attributes_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = HistoricalBaseline(smoothing, player_attributes=attributes)
    writer = pq.ParquetWriter(output_path, _feature_schema(), compression="zstd")
    connection, reader = _ordered_reader(canonical_dir, batch_size)
    current_date: date | None = None
    day_rows: list[dict[str, Any]] = []
    feature_buffer: list[dict[str, Any]] = []
    row_count = 0
    next_report = 250_000

    def process_day() -> None:
        nonlocal next_report, row_count
        for row in day_rows:
            feature_buffer.append(_feature_row(row, model, level))
        for row in day_rows:
            model.update(
                row,
                MatchContext(
                    gender=row["gender"],
                    team_type=row["team_type"],
                    venue=row["venue"],
                ),
            )
        row_count += len(day_rows)
        if row_count >= next_report:
            print(f"baseline_features rows={row_count:,}", flush=True)
            next_report += 250_000
        if len(feature_buffer) >= batch_size:
            writer.write_table(pa.Table.from_pylist(feature_buffer, schema=_feature_schema()))
            feature_buffer.clear()

    try:
        for batch in reader:
            for row in batch.to_pylist():
                match_date = row["match_date"]
                if current_date is None:
                    current_date = match_date
                if match_date != current_date:
                    process_day()
                    day_rows.clear()
                    current_date = match_date
                day_rows.append(row)
        if day_rows:
            process_day()
        if feature_buffer:
            writer.write_table(pa.Table.from_pylist(feature_buffer, schema=_feature_schema()))
    finally:
        writer.close()
        connection.close()

    return {
        "rows": row_count,
        "level": level,
        "same_date_policy": "predict_all_then_update_all",
        "history_policy": "all_prior_dates_without_windowing",
    }
