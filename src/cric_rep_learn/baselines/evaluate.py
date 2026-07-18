"""Evaluate hierarchical baselines with day-by-day chronological updates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .historical import (
    BASELINE_LEVELS,
    N_EXTRAS_CLASSES,
    N_LEGALITY_CLASSES,
    N_RUN_CLASSES,
    HistoricalBaseline,
    MatchContext,
    SmoothingConfig,
)
from .metrics import BinaryMetrics, MulticlassMetrics


@dataclass(slots=True)
class MetricBundle:
    batter_runs: MulticlassMetrics = field(default_factory=lambda: MulticlassMetrics(N_RUN_CLASSES))
    extras_runs: MulticlassMetrics = field(
        default_factory=lambda: MulticlassMetrics(N_EXTRAS_CLASSES)
    )
    legality: MulticlassMetrics = field(
        default_factory=lambda: MulticlassMetrics(N_LEGALITY_CLASSES)
    )
    batter_dismissal: BinaryMetrics = field(default_factory=BinaryMetrics)
    bowler_wicket: BinaryMetrics = field(default_factory=BinaryMetrics)

    def update(self, prediction: Any, row: dict[str, Any]) -> None:
        self.batter_runs.update(
            prediction.batter_runs,
            min(int(row["runs_batter"]), N_RUN_CLASSES - 1),
        )
        self.extras_runs.update(
            prediction.extras_runs,
            min(int(row["runs_extras"]), N_EXTRAS_CLASSES - 1),
        )
        legality_target = (
            1 if int(row["extras_wides"]) > 0 else 2 if int(row["extras_noballs"]) > 0 else 0
        )
        self.legality.update(prediction.legality, legality_target)
        self.batter_dismissal.update(prediction.batter_dismissal, row["batter_dismissed"])
        self.bowler_wicket.update(prediction.bowler_wicket, int(row["bowler_wicket_count"]) > 0)

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            "batter_runs": self.batter_runs.as_dict(),
            "extras_runs": self.extras_runs.as_dict(),
            "legality": self.legality.as_dict(),
            "batter_dismissal": self.batter_dismissal.as_dict(),
            "bowler_wicket": self.bowler_wicket.as_dict(),
        }


@dataclass(slots=True)
class EvidenceMetrics:
    n: int = 0
    batter_seen: int = 0
    bowler_seen: int = 0
    venue_seen: int = 0
    vs_pace_seen: int = 0
    vs_arm_pace_seen: int = 0
    vs_nation_arm_pace_seen: int = 0
    matchup_seen: int = 0
    batter_evidence_sum: int = 0
    bowler_evidence_sum: int = 0
    venue_evidence_sum: int = 0
    vs_pace_evidence_sum: int = 0
    vs_arm_pace_evidence_sum: int = 0
    vs_nation_arm_pace_evidence_sum: int = 0
    matchup_evidence_sum: int = 0

    _KEYS = (
        "batter",
        "bowler",
        "venue",
        "vs_pace",
        "vs_arm_pace",
        "vs_nation_arm_pace",
        "matchup",
    )

    def update(self, evidence: dict[str, int]) -> None:
        self.n += 1
        for key in self._KEYS:
            value = evidence.get(key, 0)
            setattr(self, f"{key}_seen", getattr(self, f"{key}_seen") + int(value > 0))
            setattr(
                self,
                f"{key}_evidence_sum",
                getattr(self, f"{key}_evidence_sum") + value,
            )

    def as_dict(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        result: dict[str, float | int] = {"n": self.n}
        for key in self._KEYS:
            result[f"{key}_coverage"] = getattr(self, f"{key}_seen") / self.n
            result[f"{key}_mean_prior_deliveries"] = getattr(self, f"{key}_evidence_sum") / self.n
        return result


def _escape_sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _ordered_delivery_reader(dataset_dir: Path, batch_size: int):
    deliveries = _escape_sql_path(dataset_dir / "deliveries.parquet")
    matches = _escape_sql_path(dataset_dir / "matches.parquet")
    split_manifest = _escape_sql_path(dataset_dir / "split_manifest.parquet")
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
            m.venue,
            s.split
        FROM read_parquet('{deliveries}') AS d
        JOIN read_parquet('{matches}') AS m USING (match_id)
        JOIN read_parquet('{split_manifest}') AS s USING (match_id)
        ORDER BY
            d.match_date,
            d.match_id,
            d.innings,
            d.attempt_index_in_innings
    """
    connection = duckdb.connect()
    return connection, connection.execute(query).fetch_record_batch(batch_size)


def _process_day(
    rows: list[dict[str, Any]],
    model: HistoricalBaseline,
    metrics: dict[str, dict[str, MetricBundle]],
    evidence_metrics: dict[str, EvidenceMetrics],
) -> None:
    # Predict the complete date first. Start times are unavailable, so no
    # match on this date is allowed to become history for another.
    for row in rows:
        split = row["split"]
        if split not in metrics:
            continue
        context = MatchContext(
            gender=row["gender"],
            team_type=row["team_type"],
            venue=row["venue"],
        )
        predictions = model.predict_all(row, context)
        evidence_metrics[split].update(predictions["matchup"].evidence)
        for level, prediction in predictions.items():
            metrics[split][level].update(prediction, row)

    for row in rows:
        model.update(
            row,
            MatchContext(
                gender=row["gender"],
                team_type=row["team_type"],
                venue=row["venue"],
            ),
        )


def evaluate_baselines(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    smoothing: SmoothingConfig | None = None,
    batch_size: int = 100_000,
    player_attributes_path: Path | None = None,
) -> dict[str, Any]:
    from cric_rep_learn.data.player_attributes import load_attributes_index

    smoothing = smoothing or SmoothingConfig()
    attributes = None
    if player_attributes_path is not None and player_attributes_path.exists():
        attributes = load_attributes_index(player_attributes_path)
    model = HistoricalBaseline(smoothing, player_attributes=attributes)
    metrics = {
        split: {level: MetricBundle() for level in BASELINE_LEVELS}
        for split in ("validation", "test")
    }
    evidence_metrics = {split: EvidenceMetrics() for split in ("validation", "test")}

    current_date: date | None = None
    day_rows: list[dict[str, Any]] = []
    first_date: date | None = None
    last_date: date | None = None

    connection, reader = _ordered_delivery_reader(dataset_dir, batch_size)
    try:
        for batch in reader:
            for row in batch.to_pylist():
                match_date = row["match_date"]
                if first_date is None:
                    first_date = match_date
                last_date = match_date
                if current_date is None:
                    current_date = match_date
                if match_date != current_date:
                    _process_day(day_rows, model, metrics, evidence_metrics)
                    day_rows.clear()
                    current_date = match_date
                day_rows.append(row)
        if day_rows:
            _process_day(day_rows, model, metrics, evidence_metrics)
    finally:
        connection.close()

    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "rolling_historical_empirical_bayes",
        "same_date_policy": "predict_all_then_update_all",
        "history_policy": "all_prior_dates_without_windowing",
        "first_date": first_date.isoformat() if first_date else None,
        "last_date": last_date.isoformat() if last_date else None,
        "smoothing": asdict(smoothing),
        "player_attributes": (
            str(player_attributes_path.resolve()) if player_attributes_path else None
        ),
        "player_attributes_loaded": attributes is not None,
        "metrics": {
            split: {level: bundle.as_dict() for level, bundle in split_metrics.items()}
            for split, split_metrics in metrics.items()
        },
        "evidence": {split: values.as_dict() for split, values in evidence_metrics.items()},
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baselines/metrics.json"),
    )
    parser.add_argument(
        "--player-attributes",
        type=Path,
        default=Path("artifacts/player-attributes/player_attributes.parquet"),
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    attributes_path = args.player_attributes if args.player_attributes.exists() else None
    result = evaluate_baselines(
        args.dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        player_attributes_path=attributes_path,
    )
    summary = {
        split: {
            level: {
                "runs_log_loss": values["batter_runs"].get("log_loss"),
                "dismissal_brier": values["batter_dismissal"].get("brier_score"),
                "wicket_brier": values["bowler_wicket"].get("brier_score"),
            }
            for level, values in split_metrics.items()
        }
        for split, split_metrics in result["metrics"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
