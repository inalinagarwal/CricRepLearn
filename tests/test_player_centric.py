"""Tests for bowling-style parsing and hierarchical matchup fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cric_rep_learn.baselines.historical import HistoricalBaseline, MatchContext
from cric_rep_learn.data.bowling_style import parse_batting_hand, parse_bowling_style
from cric_rep_learn.players.card import hierarchical_matchup, resolve_player
from tests.test_baselines import delivery


def test_parse_starc_style() -> None:
    parsed = parse_bowling_style("Left arm Fast")
    assert parsed.bowling_arm == "left"
    assert parsed.pace_group == "pace"
    assert parsed.arm_pace_key == "left_pace"
    assert parse_batting_hand("Left hand Bat") == "left"


def test_parse_rohit_offbreak() -> None:
    parsed = parse_bowling_style("Right arm Offbreak")
    assert parsed.bowling_arm == "right"
    assert parsed.pace_group == "spin"
    assert parsed.bowling_family == "offbreak"


def test_archetype_level_uses_shared_bowler_style_evidence() -> None:
    attrs = {
        "starc": {
            "country": "Australia",
            "bowling_arm": "left",
            "pace_group": "pace",
        },
        "other-left-pacer": {
            "country": "Australia",
            "bowling_arm": "left",
            "pace_group": "pace",
        },
        "right-spinner": {
            "country": "India",
            "bowling_arm": "right",
            "pace_group": "spin",
        },
    }
    model = HistoricalBaseline(player_attributes=attrs)
    context = MatchContext(gender="male", team_type="international", venue="Ground")

    for _ in range(40):
        model.update(delivery("rohit", "other-left-pacer", runs=6), context)
    for _ in range(40):
        model.update(delivery("rohit", "right-spinner", runs=0), context)

    predictions = model.predict_all(delivery("rohit", "starc"), context)

    assert predictions["vs_arm_pace"].evidence["vs_arm_pace"] == 40
    assert predictions["vs_nation_arm_pace"].evidence["vs_nation_arm_pace"] == 40
    assert predictions["matchup"].evidence["matchup"] == 0
    assert predictions["vs_arm_pace"].batter_runs[6] > predictions["venue"].batter_runs[6]


def test_resolve_player_prefers_full_name_alias() -> None:
    aliases = pd.DataFrame(
        [
            {"player_id": "rare", "player_name": "Rohit Sharma", "match_count": 5},
            {"player_id": "rg", "player_name": "RG Sharma", "match_count": 400},
        ]
    )
    attributes = {
        "rg": {"full_name": "Rohit Gurunath Sharma"},
        "rare": {"full_name": "Rohit Sharma"},
    }
    resolved = resolve_player("Rohit Sharma", aliases, attributes=attributes)
    assert resolved["player_id"] == "rg"


def test_hierarchical_matchup_chain_counts() -> None:
    deliveries = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "batter_id": "rohit",
                "bowler_id": "starc",
                "runs_batter": 4,
                "batter_dismissed": False,
                "bowler_wicket_count": 0,
            },
            {
                "match_id": "m2",
                "batter_id": "rohit",
                "bowler_id": "hazlewood",
                "runs_batter": 1,
                "batter_dismissed": False,
                "bowler_wicket_count": 0,
            },
            {
                "match_id": "m3",
                "batter_id": "rohit",
                "bowler_id": "ashwin",
                "runs_batter": 0,
                "batter_dismissed": True,
                "bowler_wicket_count": 1,
            },
        ]
    )
    attributes = {
        "starc": {
            "country": "Australia",
            "bowling_arm": "left",
            "pace_group": "pace",
            "bowling_style_raw": "Left arm Fast",
        },
        "hazlewood": {
            "country": "Australia",
            "bowling_arm": "right",
            "pace_group": "pace",
            "bowling_style_raw": "Right arm Fast",
        },
        "ashwin": {
            "country": "India",
            "bowling_arm": "right",
            "pace_group": "spin",
            "bowling_style_raw": "Right arm Offbreak",
        },
    }
    chain = {row["level"]: row for row in hierarchical_matchup(deliveries, attributes, "rohit", "starc")}
    assert chain["matchup"]["deliveries"] == 1
    assert chain["vs_arm_pace"]["deliveries"] == 1  # only starc is left pace
    assert chain["vs_pace"]["deliveries"] == 2  # starc + hazlewood
    assert chain["batter"]["deliveries"] == 3
    assert np.isclose(chain["matchup"]["mean_batter_runs"], 4.0)
