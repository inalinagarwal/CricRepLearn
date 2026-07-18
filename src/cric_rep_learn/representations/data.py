"""Build and load leakage-safe encoded datasets for representation learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cric_rep_learn.baselines.historical import BASELINE_LEVELS

from .baseline_features import (
    BASELINE_PROBABILITY_COLUMNS,
    EVIDENCE_COLUMNS,
    generate_baseline_features,
)

CATEGORICAL_COLUMNS = [
    "batter_idx",
    "bowler_idx",
    "venue_idx",
    "phase_idx",
    "gender_idx",
    "team_type_idx",
    "innings_group_idx",
    "wickets_bucket_idx",
]

NUMERIC_COLUMNS = [
    "score_before_z",
    "wickets_before_z",
    "innings_progress_z",
    "current_run_rate_z",
    "target_runs_remaining_z",
    "required_run_rate_z",
    "has_target",
    *EVIDENCE_COLUMNS,
]

CATEGORICAL_TARGET_COLUMNS = [
    "runs_target",
    "extras_target",
    "legality_target",
]

BINARY_TARGET_COLUMNS = [
    "batter_dismissal_target",
    "bowler_wicket_target",
]

PHASE_TO_ID = {"unknown": 0, "powerplay": 1, "middle": 2, "death": 3}
GENDER_TO_ID = {"unknown": 0, "male": 1, "female": 2}
TEAM_TYPE_TO_ID = {"unknown": 0, "club": 1, "international": 2}
INNINGS_GROUP_TO_ID = {"unknown": 0, "first_innings": 1, "chase": 2, "super_over": 3}
WICKETS_BUCKET_TO_ID = {"unknown": 0, "0-2": 1, "3-5": 2, "6-9": 3}


def _escape_sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name_query(aliases_path: str) -> str:
    return f"""
        SELECT player_id, player_name
        FROM read_parquet('{aliases_path}')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY match_count DESC, last_seen DESC, player_name
        ) = 1
    """


def build_model_dataset(
    canonical_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    baseline_level: str = "context",
    player_attributes_path: Path | None = None,
) -> dict[str, Any]:
    if baseline_level not in BASELINE_LEVELS:
        raise ValueError(
            f"Unknown baseline level {baseline_level!r}; expected one of {BASELINE_LEVELS}"
        )
    canonical_dir = canonical_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    deliveries = _escape_sql_path(canonical_dir / "deliveries.parquet")
    matches = _escape_sql_path(canonical_dir / "matches.parquet")
    aliases = _escape_sql_path(canonical_dir / "player_aliases.parquet")
    split_manifest = _escape_sql_path(canonical_dir / "split_manifest.parquet")
    baseline_features_path = output_dir / "baseline_features.parquet"
    if player_attributes_path is None:
        default_attrs = Path("artifacts/player-attributes/player_attributes.parquet")
        player_attributes_path = default_attrs if default_attrs.exists() else None
    baseline_feature_metadata = generate_baseline_features(
        canonical_dir,
        baseline_features_path,
        level=baseline_level,
        player_attributes_path=player_attributes_path,
    )
    baseline_features = _escape_sql_path(baseline_features_path)

    connection = duckdb.connect()
    try:
        role_evidence_sql = {
            "batting": f"""
                SELECT
                    d.batter_id AS player_id,
                    COUNT(*)::BIGINT AS deliveries,
                    COUNT(DISTINCT d.match_id)::BIGINT AS matches,
                    MIN(d.match_date) AS first_date,
                    MAX(d.match_date) AS last_date
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split_manifest}') s USING (match_id)
                WHERE s.split = 'train'
                GROUP BY d.batter_id
            """,
            "bowling": f"""
                SELECT
                    d.bowler_id AS player_id,
                    COUNT(*)::BIGINT AS deliveries,
                    COUNT(DISTINCT d.match_id)::BIGINT AS matches,
                    MIN(d.match_date) AS first_date,
                    MAX(d.match_date) AS last_date
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split_manifest}') s USING (match_id)
                WHERE s.split = 'train'
                GROUP BY d.bowler_id
            """,
        }

        role_vocabularies = {}
        for role, evidence_sql in role_evidence_sql.items():
            evidence = connection.execute(
                f"""
                WITH evidence AS ({evidence_sql}),
                names AS ({_canonical_name_query(aliases)})
                SELECT evidence.*, names.player_name
                FROM evidence
                LEFT JOIN names USING (player_id)
                ORDER BY player_id
                """
            ).to_arrow_table()
            role_rows = []
            for index, row in enumerate(evidence.to_pylist(), start=1):
                role_rows.append(
                    {
                        "index": index,
                        **row,
                        "first_date": (
                            row["first_date"].isoformat()
                            if hasattr(row["first_date"], "isoformat")
                            else str(row["first_date"])
                        ),
                        "last_date": (
                            row["last_date"].isoformat()
                            if hasattr(row["last_date"], "isoformat")
                            else str(row["last_date"])
                        ),
                    }
                )
            role_vocabularies[role] = role_rows

        batters = role_vocabularies["batting"]
        bowlers = role_vocabularies["bowling"]
        batter_mapping = pa.table(
            {
                "player_id": [row["player_id"] for row in batters],
                "player_idx": pa.array(
                    [row["index"] for row in batters], type=pa.int32()
                ),
            }
        )
        bowler_mapping = pa.table(
            {
                "player_id": [row["player_id"] for row in bowlers],
                "player_idx": pa.array(
                    [row["index"] for row in bowlers], type=pa.int32()
                ),
            }
        )
        connection.register("batter_mapping", batter_mapping)
        connection.register("bowler_mapping", bowler_mapping)

        venue_evidence = connection.execute(
            f"""
            SELECT m.venue, COUNT(*)::BIGINT AS match_count
            FROM read_parquet('{matches}') m
            JOIN read_parquet('{split_manifest}') s USING (match_id)
            WHERE s.split = 'train'
            GROUP BY m.venue
            ORDER BY m.venue
            """
        ).to_arrow_table()
        venues = [
            {"index": index, **row} for index, row in enumerate(venue_evidence.to_pylist(), start=1)
        ]
        venue_mapping = pa.table(
            {
                "venue": [row["venue"] for row in venues],
                "venue_idx": pa.array([row["index"] for row in venues], type=pa.int16()),
            }
        )
        connection.register("venue_mapping", venue_mapping)

        connection.execute(
            f"""
            CREATE TEMP VIEW model_rows AS
            SELECT
                d.*,
                m.gender,
                m.team_type,
                m.venue,
                s.split,
                {", ".join(f"bf.{column}" for column in BASELINE_PROBABILITY_COLUMNS + EVIDENCE_COLUMNS)},
                CASE
                    WHEN d.is_super_over THEN 'super_over'
                    WHEN d.innings = 1 THEN 'first_innings'
                    ELSE 'chase'
                END AS innings_group,
                CASE
                    WHEN d.wickets_before <= 2 THEN '0-2'
                    WHEN d.wickets_before <= 5 THEN '3-5'
                    ELSE '6-9'
                END AS wickets_bucket,
                CAST(d.score_before AS DOUBLE) AS f_score_before,
                CAST(d.wickets_before AS DOUBLE) AS f_wickets_before,
                CAST(
                    CASE
                        WHEN d.scheduled_balls > 0
                        THEN d.legal_balls_before::DOUBLE / d.scheduled_balls
                        ELSE 0
                    END AS DOUBLE
                ) AS f_innings_progress,
                CAST(
                    CASE
                        WHEN d.legal_balls_before > 0
                        THEN 6.0 * d.score_before / d.legal_balls_before
                        ELSE 0
                    END AS DOUBLE
                ) AS f_current_run_rate,
                CAST(
                    CASE
                        WHEN d.target_runs IS NOT NULL
                        THEN GREATEST(d.target_runs - d.score_before, 0)
                        ELSE 0
                    END AS DOUBLE
                ) AS f_target_runs_remaining,
                CAST(
                    CASE
                        WHEN d.target_runs IS NOT NULL
                        THEN 6.0 * GREATEST(d.target_runs - d.score_before, 0)
                            / GREATEST(COALESCE(d.target_balls, d.scheduled_balls)
                                - d.legal_balls_before, 1)
                        ELSE 0
                    END AS DOUBLE
                ) AS f_required_run_rate,
                CASE WHEN d.target_runs IS NOT NULL THEN 1.0 ELSE 0.0 END AS f_has_target
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{matches}') m USING (match_id)
            JOIN read_parquet('{split_manifest}') s USING (match_id)
            JOIN read_parquet('{baseline_features}') bf
                USING (match_id, innings, attempt_index_in_innings)
            """
        )

        raw_numeric_columns = [
            "f_score_before",
            "f_wickets_before",
            "f_innings_progress",
            "f_current_run_rate",
            "f_target_runs_remaining",
            "f_required_run_rate",
        ]
        aggregate_sql = ", ".join(
            f"AVG({column}) AS {column}_mean, STDDEV_POP({column}) AS {column}_std"
            for column in raw_numeric_columns
        )
        numeric_row = connection.execute(
            f"SELECT {aggregate_sql} FROM model_rows WHERE split = 'train'"
        ).fetchone()
        numeric_stats = {}
        for position, column in enumerate(raw_numeric_columns):
            mean = float(numeric_row[position * 2])
            raw_std = numeric_row[position * 2 + 1]
            std = float(raw_std) if raw_std is not None else 1.0
            numeric_stats[column] = {"mean": mean, "std": std if std > 1e-8 else 1.0}

        normalized_sql = []
        output_names = NUMERIC_COLUMNS[:6]
        for column, output_name in zip(raw_numeric_columns, output_names, strict=True):
            stats = numeric_stats[column]
            normalized_sql.append(
                f"CAST(({column} - {stats['mean']}) / {stats['std']} AS FLOAT) AS {output_name}"
            )

        phase_cases = " ".join(f"WHEN '{key}' THEN {value}" for key, value in PHASE_TO_ID.items())
        gender_cases = " ".join(f"WHEN '{key}' THEN {value}" for key, value in GENDER_TO_ID.items())
        team_type_cases = " ".join(
            f"WHEN '{key}' THEN {value}" for key, value in TEAM_TYPE_TO_ID.items()
        )
        innings_cases = " ".join(
            f"WHEN '{key}' THEN {value}" for key, value in INNINGS_GROUP_TO_ID.items()
        )
        wickets_cases = " ".join(
            f"WHEN '{key}' THEN {value}" for key, value in WICKETS_BUCKET_TO_ID.items()
        )

        row_counts = {}
        unknown_rates = {}
        for split in ("train", "validation", "test"):
            output_path = _escape_sql_path(output_dir / f"{split}.parquet")
            select_sql = f"""
                SELECT
                    r.match_id,
                    r.match_date,
                    COALESCE(bp.player_idx, 0)::INTEGER AS batter_idx,
                    COALESCE(bw.player_idx, 0)::INTEGER AS bowler_idx,
                    COALESCE(v.venue_idx, 0)::SMALLINT AS venue_idx,
                    CASE r.phase {phase_cases} ELSE 0 END::TINYINT AS phase_idx,
                    CASE r.gender {gender_cases} ELSE 0 END::TINYINT AS gender_idx,
                    CASE r.team_type {team_type_cases} ELSE 0 END::TINYINT AS team_type_idx,
                    CASE r.innings_group {innings_cases} ELSE 0 END::TINYINT AS innings_group_idx,
                    CASE r.wickets_bucket {wickets_cases} ELSE 0 END::TINYINT AS wickets_bucket_idx,
                    {", ".join(normalized_sql)},
                    CAST(r.f_has_target AS FLOAT) AS has_target,
                    {", ".join(f"CAST(r.{column} AS FLOAT) AS {column}" for column in EVIDENCE_COLUMNS)},
                    {", ".join(f"CAST(r.{column} AS FLOAT) AS {column}" for column in BASELINE_PROBABILITY_COLUMNS)},
                    LEAST(r.runs_batter, 7)::TINYINT AS runs_target,
                    LEAST(r.runs_extras, 7)::TINYINT AS extras_target,
                    CASE
                        WHEN r.extras_wides > 0 THEN 1
                        WHEN r.extras_noballs > 0 THEN 2
                        ELSE 0
                    END::TINYINT AS legality_target,
                    CAST(r.batter_dismissed AS FLOAT) AS batter_dismissal_target,
                    CAST(r.bowler_wicket_count > 0 AS FLOAT) AS bowler_wicket_target
                FROM model_rows r
                LEFT JOIN batter_mapping bp ON r.batter_id = bp.player_id
                LEFT JOIN bowler_mapping bw ON r.bowler_id = bw.player_id
                LEFT JOIN venue_mapping v ON r.venue = v.venue
                WHERE r.split = '{split}'
            """
            connection.execute(
                f"""
                COPY ({select_sql}) TO '{output_path}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
            summary = connection.execute(
                f"""
                SELECT
                    COUNT(*)::BIGINT,
                    AVG(CASE WHEN batter_idx = 0 THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN bowler_idx = 0 THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN venue_idx = 0 THEN 1.0 ELSE 0.0 END)
                FROM ({select_sql})
                """
            ).fetchone()
            row_counts[split] = int(summary[0])
            unknown_rates[split] = {
                "batter": float(summary[1]),
                "bowler": float(summary[2]),
                "venue": float(summary[3]),
            }

        vocab = {
            "unknown_index": 0,
            "batters": batters,
            "bowlers": bowlers,
            "venues": venues,
            "phase_to_id": PHASE_TO_ID,
            "gender_to_id": GENDER_TO_ID,
            "team_type_to_id": TEAM_TYPE_TO_ID,
            "innings_group_to_id": INNINGS_GROUP_TO_ID,
            "wickets_bucket_to_id": WICKETS_BUCKET_TO_ID,
        }
        (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "vocab_policy": "train-only role-specific vocabularies with learned unknown index 0",
            "numeric_stats_policy": "training_split_only",
            "baseline_features": {
                **baseline_feature_metadata,
                "sha256": _sha256(baseline_features_path),
            },
            "canonical_metadata_sha256": _sha256(canonical_dir / "metadata.json"),
            "split_manifest_sha256": _sha256(canonical_dir / "split_manifest.parquet"),
            "batters": len(batters),
            "bowlers": len(bowlers),
            "venues": len(venues),
            "rows": row_counts,
            "unknown_rates": unknown_rates,
            "numeric_stats": numeric_stats,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "baseline_probability_columns": BASELINE_PROBABILITY_COLUMNS,
            "categorical_target_columns": CATEGORICAL_TARGET_COLUMNS,
            "binary_target_columns": BINARY_TARGET_COLUMNS,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        connection.close()


class EncodedDeliveryDataset(Dataset):
    """In-memory tensor view of one encoded chronological split."""

    def __init__(self, path: Path):
        columns = (
            CATEGORICAL_COLUMNS
            + NUMERIC_COLUMNS
            + BASELINE_PROBABILITY_COLUMNS
            + CATEGORICAL_TARGET_COLUMNS
            + BINARY_TARGET_COLUMNS
        )
        table = pq.read_table(path, columns=columns)
        self.categorical = torch.from_numpy(
            np.column_stack(
                [table[column].to_numpy(zero_copy_only=False) for column in CATEGORICAL_COLUMNS]
            ).astype(np.int64, copy=False)
        )
        self.numeric = torch.from_numpy(
            np.column_stack(
                [table[column].to_numpy(zero_copy_only=False) for column in NUMERIC_COLUMNS]
            ).astype(np.float32, copy=False)
        )
        self.baseline_probabilities = torch.from_numpy(
            np.column_stack(
                [
                    table[column].to_numpy(zero_copy_only=False)
                    for column in BASELINE_PROBABILITY_COLUMNS
                ]
            ).astype(np.float32, copy=False)
        )
        self.categorical_targets = torch.from_numpy(
            np.column_stack(
                [
                    table[column].to_numpy(zero_copy_only=False)
                    for column in CATEGORICAL_TARGET_COLUMNS
                ]
            ).astype(np.int64, copy=False)
        )
        self.binary_targets = torch.from_numpy(
            np.column_stack(
                [table[column].to_numpy(zero_copy_only=False) for column in BINARY_TARGET_COLUMNS]
            ).astype(np.float32, copy=False)
        )

    def __len__(self) -> int:
        return self.categorical.shape[0]

    def __getitem__(self, index: int):
        return (
            self.categorical[index],
            self.numeric[index],
            self.baseline_probabilities[index],
            self.categorical_targets[index],
            self.binary_targets[index],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-data"))
    parser.add_argument(
        "--baseline-level",
        choices=BASELINE_LEVELS,
        default="context",
        help="Empirical-Bayes hierarchy level used as the residual prior",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_model_dataset(
        args.canonical,
        args.output,
        overwrite=args.overwrite,
        baseline_level=args.baseline_level,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
