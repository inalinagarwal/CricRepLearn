"""Tests for fantasy scoring and XI constraints."""

from __future__ import annotations

import pytest

from cric_rep_learn.fantasy.optimize import is_legal, optimize_xi
from cric_rep_learn.fantasy.scoring import batting_points, bowling_points, merge_player_points


def test_batting_and_bowling_points() -> None:
    bat = batting_points({"expected_runs": 45.0, "expected_balls": 30.0})
    assert bat["batting_points"] > 45  # milestone 30
    bowl = bowling_points(
        {"expected_wickets": 2.2, "expected_overs": 4.0, "expected_economy": 6.5}
    )
    assert bowl["bowling_points"] > 2.2 * 30


def test_optimize_respects_constraints() -> None:
    pool = []
    # 2 WK, 6 BAT, 4 AR, 6 BOWL across two teams
    for i in range(2):
        pool.append(
            merge_player_points(
                player_id=f"wk{i}",
                player_name=f"WK{i}",
                team="A" if i == 0 else "B",
                role="WK",
                batting={"expected_runs": 30 + i, "expected_balls": 20},
            )
        )
    for i in range(6):
        pool.append(
            merge_player_points(
                player_id=f"bat{i}",
                player_name=f"BAT{i}",
                team="A" if i < 3 else "B",
                role="BAT",
                batting={"expected_runs": 40 - i, "expected_balls": 25},
            )
        )
    for i in range(4):
        pool.append(
            merge_player_points(
                player_id=f"ar{i}",
                player_name=f"AR{i}",
                team="A" if i < 2 else "B",
                role="AR",
                batting={"expected_runs": 15, "expected_balls": 12},
                bowling={
                    "expected_wickets": 1.0,
                    "expected_overs": 3.0,
                    "expected_economy": 7.0,
                },
            )
        )
    for i in range(6):
        pool.append(
            merge_player_points(
                player_id=f"bowl{i}",
                player_name=f"BOWL{i}",
                team="A" if i < 3 else "B",
                role="BOWL",
                bowling={
                    "expected_wickets": 1.5 - 0.05 * i,
                    "expected_overs": 4.0,
                    "expected_economy": 7.5,
                },
            )
        )
    assert len(pool) == 18
    result = optimize_xi(pool, top_k=3)
    xi = result["best_xi"]
    assert is_legal(
        [
            {**p, "fantasy_points": p["fantasy_points"]}
            for p in xi["players"]
        ]
    )
    assert xi["roles"]["WK"] >= 1
    assert xi["roles"]["BAT"] >= 3
    assert xi["roles"]["BOWL"] >= 3
    assert xi["roles"]["BAT"] <= 5
    assert xi["roles"]["BOWL"] <= 5
    assert xi["roles"]["AR"] >= 1
    assert max(xi["teams"].values()) <= 7
    assert "objective_score" in xi
    assert xi["captain"]["multiplier"] == 2.0
    assert xi["vice_captain"]["multiplier"] == 1.5


def test_max_from_team_blocks_stacked_xi() -> None:
    # Build a pool where stacking 8 from A would be best unconstrained.
    pool = []
    for i in range(8):
        pool.append(
            merge_player_points(
                player_id=f"a{i}",
                player_name=f"A{i}",
                team="A",
                role="BAT" if i < 4 else ("WK" if i == 4 else ("AR" if i == 5 else "BOWL")),
                batting={"expected_runs": 50, "expected_balls": 30},
                bowling={
                    "expected_wickets": 2.0,
                    "expected_overs": 4.0,
                    "expected_economy": 6.0,
                }
                if i >= 5
                else None,
            )
        )
    for i in range(8):
        pool.append(
            merge_player_points(
                player_id=f"b{i}",
                player_name=f"B{i}",
                team="B",
                role="BAT" if i < 3 else ("WK" if i == 3 else ("AR" if i == 4 else "BOWL")),
                batting={"expected_runs": 5, "expected_balls": 8},
                bowling={
                    "expected_wickets": 0.3,
                    "expected_overs": 2.0,
                    "expected_economy": 9.0,
                }
                if i >= 5
                else None,
            )
        )
    # Need legal role coverage — adjust: A has WK/AR/BOWLs, B has WK/AR/BOWLs
    result = optimize_xi(pool, constraints={"max_from_team": 7}, top_k=1)
    assert max(result["best_xi"]["teams"].values()) <= 7
