"""Validate canonical tables and fail on identity, state, or leakage errors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REGISTRY_ID = re.compile(r"^[0-9a-f]{8}$")


def validate_canonical_dataset(dataset_dir: Path) -> dict[str, Any]:
    matches = pq.read_table(
        dataset_dir / "matches.parquet", columns=["match_id", "match_date"]
    ).to_pylist()
    match_players = pq.read_table(
        dataset_dir / "match_players.parquet",
        columns=["match_id", "player_id"],
    ).to_pylist()
    innings_rows = pq.read_table(
        dataset_dir / "innings.parquet",
        columns=["match_id", "innings", "penalty_runs_pre"],
    ).to_pylist()

    match_ids = {row["match_id"] for row in matches}
    if len(match_ids) != len(matches):
        raise ValueError("matches.parquet contains duplicate match_id values")

    registered_by_match: dict[str, set[str]] = {}
    invalid_registry_ids: set[str] = set()
    for row in match_players:
        registered_by_match.setdefault(row["match_id"], set()).add(row["player_id"])
        if not REGISTRY_ID.fullmatch(row["player_id"]):
            invalid_registry_ids.add(row["player_id"])
    if invalid_registry_ids:
        examples = sorted(invalid_registry_ids)[:5]
        raise ValueError(f"Non-canonical player IDs found: {examples}")

    columns = [
        "match_id",
        "innings",
        "over_number",
        "attempt_index_in_innings",
        "batter_id",
        "bowler_id",
        "non_striker_id",
        "is_legal",
        "legal_balls_in_over_before",
        "legal_balls_before",
        "score_before",
        "wickets_before",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "extras_noballs",
        "extras_wides",
        "wicket_count",
        "bowler_wicket_count",
    ]
    parquet_file = pq.ParquetFile(dataset_dir / "deliveries.parquet")
    state: dict[tuple[str, int], tuple[int, int, int]] = {}
    over_state: dict[tuple[str, int, int], int] = {}
    attempts: dict[tuple[str, int], int] = {}
    innings_initial_score = {
        (row["match_id"], row["innings"]): row["penalty_runs_pre"] for row in innings_rows
    }
    delivery_count = 0

    for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            delivery_count += 1
            match_id = row["match_id"]
            if match_id not in match_ids:
                raise ValueError(f"Delivery references unknown match {match_id}")

            registered = registered_by_match.get(match_id, set())
            for role in ("batter_id", "bowler_id", "non_striker_id"):
                if row[role] not in registered:
                    raise ValueError(
                        f"{match_id}: delivery {role}={row[role]} is absent from match_players"
                    )

            key = (match_id, row["innings"])
            expected_score, expected_wickets, expected_legal_balls = state.get(
                key, (innings_initial_score.get(key, 0), 0, 0)
            )
            expected_attempt = attempts.get(key, 0) + 1
            if row["attempt_index_in_innings"] != expected_attempt:
                raise ValueError(f"{match_id}: non-sequential innings attempt index")
            attempts[key] = expected_attempt

            over_key = (match_id, row["innings"], row["over_number"])
            expected_over_balls = over_state.get(over_key, 0)
            if row["legal_balls_in_over_before"] != expected_over_balls:
                raise ValueError(f"{match_id}: invalid legal-ball count within over")
            actual_state = (
                row["score_before"],
                row["wickets_before"],
                row["legal_balls_before"],
            )
            if actual_state != (
                expected_score,
                expected_wickets,
                expected_legal_balls,
            ):
                raise ValueError(
                    f"{match_id} innings {row['innings']}: state mismatch; "
                    f"expected {(expected_score, expected_wickets, expected_legal_balls)}, "
                    f"got {actual_state}"
                )

            if row["runs_batter"] + row["runs_extras"] != row["runs_total"]:
                raise ValueError(f"{match_id}: batter + extra runs != total runs")
            expected_legal = not row["extras_wides"] and not row["extras_noballs"]
            if row["is_legal"] != expected_legal:
                raise ValueError(f"{match_id}: invalid legal-delivery flag")
            if row["bowler_wicket_count"] > row["wicket_count"]:
                raise ValueError(f"{match_id}: bowler wickets exceed team wickets")

            state[key] = (
                expected_score + row["runs_total"],
                expected_wickets + row["wicket_count"],
                expected_legal_balls + int(row["is_legal"]),
            )
            over_state[over_key] = expected_over_balls + int(row["is_legal"])

    result = {
        "matches": len(matches),
        "match_players": len(match_players),
        "deliveries": delivery_count,
        "innings": len(state),
        "canonical_registry_ids": True,
        "foreign_keys_valid": True,
        "pre_delivery_state_valid": True,
        "run_totals_valid": True,
        "legal_ball_flags_valid": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/canonical"))
    args = parser.parse_args()
    print(json.dumps(validate_canonical_dataset(args.dataset), indent=2))


if __name__ == "__main__":
    main()
