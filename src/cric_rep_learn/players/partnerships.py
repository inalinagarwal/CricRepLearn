"""Partnership / co-batter graph from non-striker deliveries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_co_batters(
    canonical_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    canonical_dir = canonical_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    connection = duckdb.connect()
    try:
        # Count legal/faced balls where A is striker and B is non-striker, then
        # symmetrize so "batted with" is undirected.
        pairs = connection.execute(
            f"""
            WITH faced AS (
                SELECT
                    d.batter_id,
                    d.non_striker_id,
                    d.match_id
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND d.non_striker_id IS NOT NULL
                  AND d.batter_id <> d.non_striker_id
            ),
            directed AS (
                SELECT
                    batter_id AS player_id,
                    non_striker_id AS partner_id,
                    COUNT(*)::BIGINT AS balls_together,
                    COUNT(DISTINCT match_id)::BIGINT AS matches_together
                FROM faced
                GROUP BY 1, 2
            ),
            undirected AS (
                SELECT
                    LEAST(player_id, partner_id) AS player_a,
                    GREATEST(player_id, partner_id) AS player_b,
                    SUM(balls_together)::BIGINT AS balls_together,
                    SUM(matches_together)::BIGINT AS matches_together
                FROM directed
                GROUP BY 1, 2
            )
            SELECT * FROM undirected
            ORDER BY balls_together DESC
            """
        ).fetchdf()
    finally:
        connection.close()

    path = output_dir / "co_batters.parquet"
    pq.write_table(pa.Table.from_pandas(pairs), path, compression="zstd")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir),
        "split": "train",
        "definition": "undirected non-striker partnership balls on faced deliveries",
        "pairs": int(len(pairs)),
        "output": str(path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def top_partners(
    co_batters_path: Path,
    player_id: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    frame = pq.read_table(co_batters_path).to_pandas()
    left = frame[frame["player_a"] == player_id][
        ["player_b", "balls_together", "matches_together"]
    ].rename(columns={"player_b": "partner_id"})
    right = frame[frame["player_b"] == player_id][
        ["player_a", "balls_together", "matches_together"]
    ].rename(columns={"player_a": "partner_id"})
    partners = (
        pd.concat([left, right], ignore_index=True)
        .groupby("partner_id", as_index=False)
        .sum()
        .sort_values(["balls_together", "matches_together"], ascending=False)
        .head(limit)
    )
    return partners.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/co-batters"))
    args = parser.parse_args()
    print(json.dumps(build_co_batters(args.canonical, args.output), indent=2))


if __name__ == "__main__":
    main()
