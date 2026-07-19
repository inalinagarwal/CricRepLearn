"""Tests for hierarchical Bayes player effects and co-batter ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cric_rep_learn.players.partnerships import top_partners
from cric_rep_learn.players.player_effects import expected_runs_vs_bowler
from cric_rep_learn.players.rank_vs_bowler import _combine_representation


def test_expected_runs_shrinks_matchup_toward_arm_pace() -> None:
    effects = {
        "rohit": {
            "player_sr": 1.4,
            "sr_vs_pace": 1.3,
            "sr_vs_spin": 1.5,
            "sr_vs_left_pace": 1.1,
            "sr_vs_right_pace": 1.35,
            "sr_vs_left_spin": 1.45,
            "sr_vs_right_spin": 1.55,
        }
    }
    # Sparse hot matchup should pull toward left-pace parent, not stay at raw 4.0.
    matchups = {("rohit", "starc"): {"runs": 20.0, "balls": 5.0, "dismissals": 0.0}}
    forecast = expected_runs_vs_bowler(
        batter_id="rohit",
        bowler_id="starc",
        balls=12.0,
        effects=effects,
        matchups=matchups,
        bowler_attrs={"bowling_arm": "left", "pace_group": "pace"},
        global_sr=1.2,
        matchup_strength=40.0,
        archetype_strength=120.0,
    )
    assert forecast["level"].startswith("matchup→")
    assert forecast["matchup_balls"] == 5
    # (20 + 40*1.1) / (5+40) = 64/45 ≈ 1.422
    assert abs(forecast["expected_sr"] - 64.0 / 45.0) < 1e-9
    assert abs(forecast["expected_runs"] - 12.0 * 64.0 / 45.0) < 1e-9


def test_expected_runs_falls_back_to_arm_pace_without_matchup() -> None:
    effects = {
        "rohit": {
            "player_sr": 1.4,
            "sr_vs_pace": 1.3,
            "sr_vs_spin": 1.5,
            "sr_vs_left_pace": 1.1,
            "sr_vs_right_pace": 1.35,
            "sr_vs_left_spin": 1.45,
            "sr_vs_right_spin": 1.55,
        }
    }
    forecast = expected_runs_vs_bowler(
        batter_id="rohit",
        bowler_id="starc",
        balls=10.0,
        effects=effects,
        matchups={},
        bowler_attrs={"bowling_arm": "left", "pace_group": "pace"},
        global_sr=1.2,
        matchup_strength=40.0,
        archetype_strength=120.0,
    )
    assert forecast["level"] == "vs_left_pace"
    assert forecast["expected_sr"] == 1.1
    assert forecast["expected_runs"] == 11.0


def test_top_partners_undirected(tmp_path: Path) -> None:
    table = pa.table(
        {
            "player_a": ["rohit", "rohit", "virat"],
            "player_b": ["virat", "surya", "surya"],
            "balls_together": [500, 300, 50],
            "matches_together": [40, 25, 5],
        }
    )
    path = tmp_path / "co_batters.parquet"
    pq.write_table(table, path)
    partners = top_partners(path, "virat", limit=5)
    assert [row["partner_id"] for row in partners] == ["rohit", "surya"]
    assert partners[0]["balls_together"] == 500


def test_combine_representation_concatenates_hb_and_embedding() -> None:
    combined = _combine_representation([1.0, 0.0, 0.0], np.array([0.0, 2.0]))
    assert combined is not None
    assert combined.shape == (5,)
    # HB weight 1.0, embedding weight 0.25 after unit-norm.
    assert abs(combined[0] - 1.0) < 1e-8
    assert abs(combined[4] - 0.25) < 1e-8


def test_country_and_venue_helpers() -> None:
    from cric_rep_learn.players.rank_bowlers_vs_batter import _country_matches
    from cric_rep_learn.players.venue_similarity import (
        cluster_for_query,
        normalize_place,
    )

    assert normalize_place("Rawalpindi Cricket Stadium") == "rawalpindi cricket stadium"
    assert _country_matches("Pakistan", "pakistan")
    assert _country_matches("India", "India")
    assert not _country_matches("Australia", "Pakistan")
    assert cluster_for_query("Islamabad") == "pakistan_plains"
    assert cluster_for_query("Rawalpindi") == "pakistan_plains"


def test_survival_endogenous_balls() -> None:
    from cric_rep_learn.players.forecast_vs_attack import simulate_expected_innings

    # High dismissal rate → few balls / runs; low rate → more opportunity.
    dangerous = simulate_expected_innings(
        rates=[{"expected_sr": 1.0, "dismissal_rate": 0.2}],
        max_balls=120,
    )
    safe = simulate_expected_innings(
        rates=[{"expected_sr": 1.0, "dismissal_rate": 0.02}],
        max_balls=120,
    )
    assert dangerous["expected_balls"] < safe["expected_balls"]
    assert dangerous["expected_runs"] < safe["expected_runs"]
    # Alone vs a killer bowler is shorter than vs a soft one even in a mix.
    mixed = simulate_expected_innings(
        rates=[
            {"expected_sr": 1.0, "dismissal_rate": 0.15},
            {"expected_sr": 1.0, "dismissal_rate": 0.03},
        ],
        max_balls=120,
    )
    assert mixed["expected_balls"] < safe["expected_balls"]
    assert mixed["expected_balls"] > dangerous["expected_balls"]
