"""Build leakage-safe bowling-conditioned batting contribution datasets."""

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

TOP_BOWLERS = 4

CATEGORICAL_COLUMNS = [
    "batter_idx",
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
    "balls_faced_log1p",
]

BASELINE_COLUMNS = [
    "baseline_runs",
]

TARGET_COLUMNS = [
    "runs",
    "dismissed",
]

BOWLER_INDEX_COLUMNS = [f"bowler_idx_{index}" for index in range(TOP_BOWLERS)]
BOWLER_WEIGHT_COLUMNS = [f"bowler_weight_{index}" for index in range(TOP_BOWLERS)]

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


def build_stints_table(connection: duckdb.DuckDBPyConnection, canonical_dir: Path) -> None:
    deliveries = _escape_sql_path(canonical_dir / "deliveries.parquet")
    matches = _escape_sql_path(canonical_dir / "matches.parquet")
    split_manifest = _escape_sql_path(canonical_dir / "split_manifest.parquet")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE faced AS
        SELECT
            d.match_id,
            d.match_date,
            d.innings,
            d.batter_id,
            d.bowler_id,
            d.attempt_index_in_innings,
            d.phase,
            d.score_before,
            d.wickets_before,
            d.runs_batter,
            d.batter_dismissed,
            CASE
                WHEN d.is_legal OR d.extras_noballs > 0 THEN 1
                ELSE 0
            END AS ball_faced
        FROM read_parquet('{deliveries}') d
        WHERE NOT d.is_super_over
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE ranked AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY match_id, innings, batter_id
                ORDER BY attempt_index_in_innings
            ) AS stint_ball_rank
        FROM faced
        WHERE ball_faced = 1
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stints AS
        WITH aggregated AS (
            SELECT
                match_id,
                match_date,
                innings,
                batter_id,
                COUNT(*)::INTEGER AS balls_faced,
                SUM(runs_batter)::INTEGER AS runs,
                MAX(CASE WHEN batter_dismissed THEN 1 ELSE 0 END)::INTEGER AS dismissed
            FROM ranked
            GROUP BY 1, 2, 3, 4
        ),
        entry AS (
            SELECT
                a.*,
                f.phase AS entry_phase,
                f.score_before AS entry_score_before,
                f.wickets_before AS entry_wickets_before
            FROM aggregated a
            JOIN ranked f
              ON a.match_id = f.match_id
             AND a.innings = f.innings
             AND a.batter_id = f.batter_id
             AND f.stint_ball_rank = 1
        )
        SELECT
            e.*,
            m.gender,
            m.team_type,
            m.venue,
            s.split,
            CASE WHEN e.innings = 1 THEN 'first_innings' ELSE 'chase' END AS innings_group,
            CASE
                WHEN e.entry_wickets_before <= 2 THEN '0-2'
                WHEN e.entry_wickets_before <= 5 THEN '3-5'
                ELSE '6-9'
            END AS wickets_bucket
        FROM entry e
        JOIN read_parquet('{matches}') m USING (match_id)
        JOIN read_parquet('{split_manifest}') s USING (match_id)
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE bowler_exposure AS
        SELECT
            match_id,
            innings,
            batter_id,
            bowler_id,
            COUNT(*)::INTEGER AS balls
        FROM ranked
        GROUP BY 1, 2, 3, 4
        """
    )


def _top_bowlers_for_stint(
    exposure_rows: list[dict[str, Any]],
    bowler_to_idx: dict[str, int],
    *,
    top_k: int = TOP_BOWLERS,
) -> tuple[list[int], list[float]]:
    if not exposure_rows:
        return [0] * top_k, [1.0] + [0.0] * (top_k - 1)
    total = float(sum(int(row["balls"]) for row in exposure_rows))
    ranked = sorted(exposure_rows, key=lambda row: (-int(row["balls"]), str(row["bowler_id"])))
    idxs: list[int] = []
    weights: list[float] = []
    used = 0.0
    for row in ranked[:top_k]:
        idxs.append(int(bowler_to_idx.get(str(row["bowler_id"]), 0)))
        weight = float(row["balls"]) / total if total else 0.0
        weights.append(weight)
        used += weight
    while len(idxs) < top_k:
        idxs.append(0)
        weights.append(0.0)
    # Fold leftover mass into UNK if more than K bowlers.
    leftover = max(0.0, 1.0 - used)
    if leftover > 0:
        if idxs[0] == 0:
            weights[0] += leftover
        else:
            # push mass onto last slot as UNK if free, else first weight
            if idxs[-1] == 0 and weights[-1] == 0:
                weights[-1] = leftover
            else:
                weights[0] += leftover
    weight_sum = sum(weights) or 1.0
    weights = [weight / weight_sum for weight in weights]
    return idxs, weights


def build_contribution_dataset(
    canonical_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    min_balls_eval: int = 3,
) -> dict[str, Any]:
    canonical_dir = canonical_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    aliases = _escape_sql_path(canonical_dir / "player_aliases.parquet")
    connection = duckdb.connect()
    try:
        build_stints_table(connection, canonical_dir)
        names = connection.execute(_canonical_name_query(aliases)).fetchdf()
        name_map = dict(zip(names["player_id"], names["player_name"], strict=False))

        train_batters = connection.execute(
            """
            SELECT
                batter_id AS player_id,
                COUNT(*)::BIGINT AS stints,
                SUM(balls_faced)::BIGINT AS balls_faced,
                SUM(runs)::BIGINT AS runs,
                MIN(match_date)::VARCHAR AS first_date,
                MAX(match_date)::VARCHAR AS last_date
            FROM stints
            WHERE split = 'train'
            GROUP BY 1
            ORDER BY stints DESC, balls_faced DESC, player_id
            """
        ).fetchdf()
        batter_vocab = [{"index": 0, "player_id": "UNK_BATTER", "player_name": "UNK_BATTER"}]
        batter_to_idx = {"UNK_BATTER": 0}
        for row in train_batters.to_dict(orient="records"):
            index = len(batter_vocab)
            batter_to_idx[row["player_id"]] = index
            batter_vocab.append(
                {
                    "index": index,
                    "player_id": row["player_id"],
                    "player_name": name_map.get(row["player_id"], row["player_id"]),
                    "stints": int(row["stints"]),
                    "balls_faced": int(row["balls_faced"]),
                    "runs": int(row["runs"]),
                    "first_date": row["first_date"],
                    "last_date": row["last_date"],
                }
            )

        train_bowlers = connection.execute(
            """
            SELECT
                e.bowler_id AS player_id,
                COUNT(*)::BIGINT AS stint_exposures,
                SUM(e.balls)::BIGINT AS balls_bowled,
                MIN(s.match_date)::VARCHAR AS first_date,
                MAX(s.match_date)::VARCHAR AS last_date
            FROM bowler_exposure e
            JOIN stints s
              ON e.match_id = s.match_id
             AND e.innings = s.innings
             AND e.batter_id = s.batter_id
            WHERE s.split = 'train'
            GROUP BY 1
            ORDER BY balls_bowled DESC, stint_exposures DESC, player_id
            """
        ).fetchdf()
        bowler_vocab = [{"index": 0, "player_id": "UNK_BOWLER", "player_name": "UNK_BOWLER"}]
        bowler_to_idx = {"UNK_BOWLER": 0}
        for row in train_bowlers.to_dict(orient="records"):
            index = len(bowler_vocab)
            bowler_to_idx[row["player_id"]] = index
            bowler_vocab.append(
                {
                    "index": index,
                    "player_id": row["player_id"],
                    "player_name": name_map.get(row["player_id"], row["player_id"]),
                    "stint_exposures": int(row["stint_exposures"]),
                    "balls_bowled": int(row["balls_bowled"]),
                    "first_date": row["first_date"],
                    "last_date": row["last_date"],
                }
            )

        venues = connection.execute(
            """
            SELECT venue, COUNT(*)::BIGINT AS stints
            FROM stints
            WHERE split = 'train'
            GROUP BY 1
            ORDER BY stints DESC, venue
            """
        ).fetchdf()
        venue_vocab = [{"index": 0, "venue": "UNK_VENUE"}]
        venue_to_idx = {"UNK_VENUE": 0}
        for row in venues.to_dict(orient="records"):
            index = len(venue_vocab)
            venue_to_idx[row["venue"]] = index
            venue_vocab.append({"index": index, "venue": row["venue"], "stints": int(row["stints"])})

        train_stats = connection.execute(
            """
            SELECT
                AVG(entry_score_before)::DOUBLE AS score_mean,
                STDDEV_SAMP(entry_score_before)::DOUBLE AS score_std,
                AVG(entry_wickets_before)::DOUBLE AS wickets_mean,
                STDDEV_SAMP(entry_wickets_before)::DOUBLE AS wickets_std,
                SUM(runs)::DOUBLE / NULLIF(SUM(balls_faced), 0) AS global_strike_rate
            FROM stints
            WHERE split = 'train'
            """
        ).fetchone()
        score_mean, score_std, wickets_mean, wickets_std, global_sr = train_stats
        score_std = score_std or 1.0
        wickets_std = wickets_std or 1.0
        global_sr = float(global_sr or 1.0)

        context_rates = connection.execute(
            """
            SELECT
                gender, team_type, innings_group, entry_phase, wickets_bucket,
                SUM(runs)::DOUBLE / NULLIF(SUM(balls_faced), 0) AS strike_rate
            FROM stints
            WHERE split = 'train'
            GROUP BY 1, 2, 3, 4, 5
            """
        ).fetchdf()
        context_sr: dict[tuple[Any, ...], float] = {}
        for row in context_rates.to_dict(orient="records"):
            if row["strike_rate"] is None:
                continue
            key = (
                str(row["gender"]),
                str(row["team_type"]),
                str(row["innings_group"]),
                str(row["entry_phase"]),
                str(row["wickets_bucket"]),
            )
            context_sr[key] = float(row["strike_rate"])

        stints = connection.execute(
            "SELECT * FROM stints ORDER BY match_date, match_id, innings"
        ).fetchdf()
        exposures = connection.execute(
            "SELECT match_id, innings, batter_id, bowler_id, balls FROM bowler_exposure"
        ).fetchdf()
    finally:
        connection.close()

    exposure_map: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in exposures.to_dict(orient="records"):
        key = (str(row["match_id"]), int(row["innings"]), str(row["batter_id"]))
        exposure_map.setdefault(key, []).append(row)

    split_counts: dict[str, int] = {}
    baseline_mae_sums: dict[str, float] = {}
    baseline_mae_ns: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = stints[stints["split"] == split_name].copy()
        rows: list[dict[str, Any]] = []
        abs_err_sum = 0.0
        abs_err_n = 0
        for record in split_frame.to_dict(orient="records"):
            balls = int(record["balls_faced"])
            key = (
                str(record["gender"]),
                str(record["team_type"]),
                str(record["innings_group"]),
                str(record["entry_phase"]),
                str(record["wickets_bucket"]),
            )
            strike_rate = context_sr.get(key, global_sr)
            baseline_runs = float(strike_rate * balls)
            runs = int(record["runs"])
            eligible = balls >= min_balls_eval
            if eligible:
                abs_err_sum += abs(baseline_runs - runs)
                abs_err_n += 1
            stint_key = (str(record["match_id"]), int(record["innings"]), str(record["batter_id"]))
            bowler_idxs, bowler_weights = _top_bowlers_for_stint(
                exposure_map.get(stint_key, []), bowler_to_idx
            )
            row = {
                "match_id": record["match_id"],
                "match_date": str(record["match_date"]),
                "innings": int(record["innings"]),
                "batter_id": record["batter_id"],
                "batter_idx": int(batter_to_idx.get(record["batter_id"], 0)),
                "venue_idx": int(venue_to_idx.get(record["venue"], 0)),
                "phase_idx": int(PHASE_TO_ID.get(str(record["entry_phase"]), 0)),
                "gender_idx": int(GENDER_TO_ID.get(str(record["gender"]), 0)),
                "team_type_idx": int(TEAM_TYPE_TO_ID.get(str(record["team_type"]), 0)),
                "innings_group_idx": int(
                    INNINGS_GROUP_TO_ID.get(str(record["innings_group"]), 0)
                ),
                "wickets_bucket_idx": int(
                    WICKETS_BUCKET_TO_ID.get(str(record["wickets_bucket"]), 0)
                ),
                "score_before_z": float(
                    (float(record["entry_score_before"]) - score_mean) / score_std
                ),
                "wickets_before_z": float(
                    (float(record["entry_wickets_before"]) - wickets_mean) / wickets_std
                ),
                "balls_faced": balls,
                "balls_faced_log1p": float(np.log1p(balls)),
                "baseline_runs": baseline_runs,
                "runs": runs,
                "dismissed": float(record["dismissed"]),
                "eval_eligible": eligible,
            }
            for index, (bowler_idx, weight) in enumerate(zip(bowler_idxs, bowler_weights, strict=True)):
                row[f"bowler_idx_{index}"] = int(bowler_idx)
                row[f"bowler_weight_{index}"] = float(weight)
            rows.append(row)
        pq.write_table(
            pa.Table.from_pylist(rows),
            output_dir / f"{split_name}.parquet",
            compression="zstd",
        )
        split_counts[split_name] = len(rows)
        baseline_mae_sums[split_name] = abs_err_sum
        baseline_mae_ns[split_name] = abs_err_n

    vocab = {
        "batters": batter_vocab,
        "bowlers": bowler_vocab,
        "venues": venue_vocab,
        "top_bowlers": TOP_BOWLERS,
        "normalization": {
            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "wickets_mean": float(wickets_mean),
            "wickets_std": float(wickets_std),
            "global_strike_rate": global_sr,
        },
        "context_strike_rates": {
            "|".join(key): value for key, value in sorted(context_sr.items())
        },
        "min_balls_eval": min_balls_eval,
    }
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir),
        "objective": "bowling_conditioned_residual_batting_contribution",
        "grain": "match_id,innings,batter_id",
        "bowling_attack": f"top-{TOP_BOWLERS} bowlers by balls faced, weight = ball share",
        "baseline": "train-only context strike rate × balls_faced",
        "excludes_super_overs": True,
        "min_balls_eval": min_balls_eval,
        "split_counts": split_counts,
        "baseline_runs_mae_min3": {
            split: (
                baseline_mae_sums[split] / baseline_mae_ns[split]
                if baseline_mae_ns[split]
                else None
            )
            for split in ("train", "validation", "test")
        },
        "n_batters": len(batter_vocab) - 1,
        "n_bowlers": len(bowler_vocab) - 1,
        "n_venues": len(venue_vocab) - 1,
        "canonical_hashes": {
            "deliveries": _sha256(canonical_dir / "deliveries.parquet"),
            "matches": _sha256(canonical_dir / "matches.parquet"),
            "split_manifest": _sha256(canonical_dir / "split_manifest.parquet"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class EncodedStintDataset(Dataset):
    def __init__(self, path: Path, *, eval_only: bool = False):
        frame = pq.read_table(path).to_pandas()
        if eval_only:
            frame = frame[frame["eval_eligible"]].reset_index(drop=True)
        self.categorical = torch.tensor(
            frame[CATEGORICAL_COLUMNS].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.numeric = torch.tensor(
            frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.baseline = torch.tensor(
            frame[BASELINE_COLUMNS].to_numpy(dtype=np.float32).reshape(-1),
            dtype=torch.float32,
        )
        self.bowler_idxs = torch.tensor(
            frame[BOWLER_INDEX_COLUMNS].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.bowler_weights = torch.tensor(
            frame[BOWLER_WEIGHT_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.targets = torch.tensor(
            frame[TARGET_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.balls_faced = torch.tensor(
            frame["balls_faced"].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.eval_eligible = torch.tensor(
            frame["eval_eligible"].to_numpy(dtype=np.bool_), dtype=torch.bool
        )

    def __len__(self) -> int:
        return int(self.categorical.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.categorical[index],
            self.numeric[index],
            self.baseline[index],
            self.bowler_idxs[index],
            self.bowler_weights[index],
            self.targets[index],
            self.balls_faced[index],
            self.eval_eligible[index],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/contribution-data-bowling")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-balls-eval", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            build_contribution_dataset(
                args.canonical,
                args.output,
                overwrite=args.overwrite,
                min_balls_eval=args.min_balls_eval,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
