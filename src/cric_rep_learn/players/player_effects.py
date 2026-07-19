"""Hierarchical empirical-Bayes batting effect vectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _posterior_rate(successes: float, n: float, prior: float, strength: float) -> float:
    if n <= 0:
        return float(prior)
    return float((successes + strength * prior) / (n + strength))


def build_player_effects(
    canonical_dir: Path,
    attributes_path: Path,
    output_dir: Path,
    *,
    player_strength: float = 400.0,
    archetype_strength: float = 120.0,
    matchup_strength: float = 40.0,
) -> dict[str, Any]:
    """
    Fit train-only hierarchical batting rates and export one vector per batter.

    Hierarchy for expected runs / ball:
      global → player → player×pace → player×arm_pace
    plus direct player×bowler rates when available.
    """
    canonical_dir = canonical_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    deliveries = _escape(canonical_dir / "deliveries.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    aliases = _escape(canonical_dir / "player_aliases.parquet")
    attributes = _escape(attributes_path)

    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE train_balls AS
            SELECT
                d.batter_id,
                d.bowler_id,
                d.runs_batter,
                d.batter_dismissed,
                CASE WHEN d.is_legal OR d.extras_noballs > 0 THEN 1 ELSE 0 END AS ball_faced,
                COALESCE(a.bowling_arm, 'unknown') AS bowling_arm,
                COALESCE(a.pace_group, 'unknown') AS pace_group,
                COALESCE(a.country, 'unknown') AS bowler_country
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{matches}') m USING (match_id)
            JOIN read_parquet('{split}') s USING (match_id)
            LEFT JOIN read_parquet('{attributes}') a
              ON d.bowler_id = a.player_id
            WHERE s.split = 'train'
              AND NOT d.is_super_over
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE faced AS
            SELECT * FROM train_balls WHERE ball_faced = 1
            """
        )

        global_row = connection.execute(
            """
            SELECT
                SUM(runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(CASE WHEN batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
            FROM faced
            """
        ).fetchone()
        global_runs, global_balls, global_dismissals = global_row
        global_sr = float(global_runs / global_balls) if global_balls else 1.0
        global_dismiss = float(global_dismissals / global_balls) if global_balls else 0.05

        player_stats = connection.execute(
            """
            SELECT
                batter_id,
                SUM(runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(CASE WHEN batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
            FROM faced
            GROUP BY 1
            """
        ).fetchdf()

        pace_stats = connection.execute(
            """
            SELECT
                batter_id,
                pace_group,
                SUM(runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls
            FROM faced
            GROUP BY 1, 2
            """
        ).fetchdf()

        arm_pace_stats = connection.execute(
            """
            SELECT
                batter_id,
                bowling_arm,
                pace_group,
                SUM(runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls
            FROM faced
            GROUP BY 1, 2, 3
            """
        ).fetchdf()

        matchup_stats = connection.execute(
            """
            SELECT
                batter_id,
                bowler_id,
                SUM(runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(CASE WHEN batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
            FROM faced
            GROUP BY 1, 2
            """
        ).fetchdf()

        names = connection.execute(
            f"""
            SELECT player_id, player_name
            FROM read_parquet('{aliases}')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY player_id
                ORDER BY match_count DESC, last_seen DESC, player_name
            ) = 1
            """
        ).fetchdf()
    finally:
        connection.close()

    name_map = dict(zip(names["player_id"], names["player_name"], strict=False))

    pace_map: dict[tuple[str, str], tuple[float, float]] = {}
    for row in pace_stats.to_dict(orient="records"):
        pace_map[(row["batter_id"], row["pace_group"])] = (
            float(row["runs"]),
            float(row["balls"]),
        )

    arm_pace_map: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in arm_pace_stats.to_dict(orient="records"):
        arm_pace_map[(row["batter_id"], row["bowling_arm"], row["pace_group"])] = (
            float(row["runs"]),
            float(row["balls"]),
        )

    matchup_map: dict[tuple[str, str], dict[str, float]] = {}
    for row in matchup_stats.to_dict(orient="records"):
        matchup_map[(row["batter_id"], row["bowler_id"])] = {
            "runs": float(row["runs"]),
            "balls": float(row["balls"]),
            "dismissals": float(row["dismissals"]),
        }

    effect_rows: list[dict[str, Any]] = []
    for row in player_stats.to_dict(orient="records"):
        batter_id = row["batter_id"]
        balls = float(row["balls"])
        runs = float(row["runs"])
        dismissals = float(row["dismissals"])
        player_sr = _posterior_rate(runs, balls, global_sr, player_strength)
        player_dismiss = _posterior_rate(
            dismissals, balls, global_dismiss, player_strength
        )

        def pace_rate(pace: str) -> float:
            r, n = pace_map.get((batter_id, pace), (0.0, 0.0))
            return _posterior_rate(r, n, player_sr, archetype_strength)

        def arm_pace_rate(arm: str, pace: str) -> float:
            parent = pace_rate(pace)
            r, n = arm_pace_map.get((batter_id, arm, pace), (0.0, 0.0))
            return _posterior_rate(r, n, parent, archetype_strength)

        vector = [
            player_sr - global_sr,
            pace_rate("pace") - global_sr,
            pace_rate("spin") - global_sr,
            arm_pace_rate("left", "pace") - global_sr,
            arm_pace_rate("right", "pace") - global_sr,
            arm_pace_rate("left", "spin") - global_sr,
            arm_pace_rate("right", "spin") - global_sr,
            player_dismiss,
            float(np.log1p(balls)),
        ]
        effect_rows.append(
            {
                "player_id": batter_id,
                "player_name": name_map.get(batter_id, batter_id),
                "balls": int(balls),
                "runs": int(runs),
                "dismissals": int(dismissals),
                "global_sr": global_sr,
                "player_sr": player_sr,
                "player_sr_effect": player_sr - global_sr,
                "sr_vs_pace": pace_rate("pace"),
                "sr_vs_spin": pace_rate("spin"),
                "sr_vs_left_pace": arm_pace_rate("left", "pace"),
                "sr_vs_right_pace": arm_pace_rate("right", "pace"),
                "sr_vs_left_spin": arm_pace_rate("left", "spin"),
                "sr_vs_right_spin": arm_pace_rate("right", "spin"),
                "dismissal_rate": player_dismiss,
                "effect_vector": vector,
            }
        )

    schema = pa.schema(
        [
            ("player_id", pa.string()),
            ("player_name", pa.string()),
            ("balls", pa.int64()),
            ("runs", pa.int64()),
            ("dismissals", pa.int64()),
            ("global_sr", pa.float64()),
            ("player_sr", pa.float64()),
            ("player_sr_effect", pa.float64()),
            ("sr_vs_pace", pa.float64()),
            ("sr_vs_spin", pa.float64()),
            ("sr_vs_left_pace", pa.float64()),
            ("sr_vs_right_pace", pa.float64()),
            ("sr_vs_left_spin", pa.float64()),
            ("sr_vs_right_spin", pa.float64()),
            ("dismissal_rate", pa.float64()),
            ("effect_vector", pa.list_(pa.float64(), 9)),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(effect_rows, schema=schema),
        output_dir / "player_effects.parquet",
        compression="zstd",
    )

    # Compact matchup table for ranking lookups.
    matchup_rows = [
        {
            "batter_id": batter_id,
            "bowler_id": bowler_id,
            "balls": int(stats["balls"]),
            "runs": int(stats["runs"]),
            "dismissals": int(stats["dismissals"]),
            "raw_sr": float(stats["runs"] / stats["balls"]) if stats["balls"] else None,
        }
        for (batter_id, bowler_id), stats in matchup_map.items()
    ]
    pq.write_table(
        pa.Table.from_pylist(matchup_rows),
        output_dir / "batter_bowler_matchups.parquet",
        compression="zstd",
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir),
        "attributes": str(attributes_path.resolve()),
        "split": "train",
        "player_strength": player_strength,
        "archetype_strength": archetype_strength,
        "matchup_strength": matchup_strength,
        "global_sr": global_sr,
        "global_dismissal_rate": global_dismiss,
        "players": len(effect_rows),
        "matchups": len(matchup_rows),
        "effect_vector_fields": [
            "player_sr_effect",
            "pace_effect",
            "spin_effect",
            "left_pace_effect",
            "right_pace_effect",
            "left_spin_effect",
            "right_spin_effect",
            "dismissal_rate",
            "log1p_balls",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    # Persist strengths for ranking CLI.
    (output_dir / "smoothing.json").write_text(
        json.dumps(
            {
                "player_strength": player_strength,
                "archetype_strength": archetype_strength,
                "matchup_strength": matchup_strength,
                "global_sr": global_sr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def expected_runs_vs_bowler(
    *,
    batter_id: str,
    bowler_id: str,
    balls: float,
    effects: dict[str, dict[str, Any]],
    matchups: dict[tuple[str, str], dict[str, float]],
    bowler_attrs: dict[str, Any],
    global_sr: float,
    matchup_strength: float,
    archetype_strength: float,
) -> dict[str, Any]:
    """Shrink direct matchup → arm/pace → pace → player → global."""
    effect = effects.get(batter_id)
    if effect is None:
        sr = global_sr
        level = "global"
        evidence = 0
        parent = global_sr
    else:
        arm = str(bowler_attrs.get("bowling_arm") or "unknown")
        pace = str(bowler_attrs.get("pace_group") or "unknown")
        if arm == "left" and pace == "pace":
            parent = float(effect["sr_vs_left_pace"])
            level = "vs_left_pace"
        elif arm == "right" and pace == "pace":
            parent = float(effect["sr_vs_right_pace"])
            level = "vs_right_pace"
        elif arm == "left" and pace == "spin":
            parent = float(effect["sr_vs_left_spin"])
            level = "vs_left_spin"
        elif arm == "right" and pace == "spin":
            parent = float(effect["sr_vs_right_spin"])
            level = "vs_right_spin"
        elif pace == "pace":
            parent = float(effect["sr_vs_pace"])
            level = "vs_pace"
        elif pace == "spin":
            parent = float(effect["sr_vs_spin"])
            level = "vs_spin"
        else:
            parent = float(effect["player_sr"])
            level = "player"

        matchup = matchups.get((batter_id, bowler_id))
        if matchup and matchup["balls"] > 0:
            sr = _posterior_rate(
                matchup["runs"], matchup["balls"], parent, matchup_strength
            )
            evidence = int(matchup["balls"])
            level = "matchup→" + level
        else:
            sr = parent
            evidence = 0

    return {
        "expected_runs": float(sr * balls),
        "expected_sr": float(sr),
        "level": level,
        "matchup_balls": evidence,
        "parent_sr": float(parent) if effect is not None else global_sr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("artifacts/player-attributes/player_attributes.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/player-effects")
    )
    parser.add_argument("--player-strength", type=float, default=400.0)
    parser.add_argument("--archetype-strength", type=float, default=120.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_player_effects(
                args.canonical,
                args.attributes,
                args.output,
                player_strength=args.player_strength,
                archetype_strength=args.archetype_strength,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
