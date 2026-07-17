from __future__ import annotations

import pytest

from cric_rep_learn.data.parser import CricsheetParser, ParseError


def match_fixture() -> dict:
    people = {
        "Batter One": "aaaaaaaa",
        "Batter Two": "bbbbbbbb",
        "Bowler One": "cccccccc",
        "Fielder One": "dddddddd",
    }
    return {
        "meta": {"data_version": "1.1.0", "created": "2026-01-02", "revision": 1},
        "info": {
            "balls_per_over": 6,
            "dates": ["2026-01-01"],
            "event": {"name": "Example T20", "match_number": 1},
            "gender": "male",
            "match_type": "T20",
            "overs": 20,
            "players": {
                "Team A": ["Batter One", "Batter Two"],
                "Team B": ["Bowler One", "Fielder One"],
            },
            "registry": {"people": people},
            "season": "2026",
            "team_type": "international",
            "teams": ["Team A", "Team B"],
            "toss": {"winner": "Team A", "decision": "bat"},
            "venue": "Example Ground",
        },
        "innings": [
            {
                "team": "Team A",
                "powerplays": [{"from": 0.1, "to": 5.6, "type": "mandatory"}],
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter One",
                                "bowler": "Bowler One",
                                "non_striker": "Batter Two",
                                "extras": {"wides": 1},
                                "runs": {"batter": 0, "extras": 1, "total": 1},
                            },
                            {
                                "batter": "Batter One",
                                "bowler": "Bowler One",
                                "non_striker": "Batter Two",
                                "runs": {"batter": 4, "extras": 0, "total": 4},
                            },
                            {
                                "batter": "Batter One",
                                "bowler": "Bowler One",
                                "non_striker": "Batter Two",
                                "extras": {"noballs": 1},
                                "runs": {"batter": 1, "extras": 1, "total": 2},
                                "wickets": [
                                    {
                                        "kind": "run out",
                                        "player_out": "Batter Two",
                                        "fielders": [{"name": "Fielder One"}],
                                    }
                                ],
                            },
                            {
                                "batter": "Batter One",
                                "bowler": "Bowler One",
                                "non_striker": "Batter Two",
                                "runs": {"batter": 0, "extras": 0, "total": 0},
                                "wickets": [{"kind": "bowled", "player_out": "Batter One"}],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_parser_preserves_pre_delivery_state_and_illegal_balls() -> None:
    parsed = CricsheetParser().parse_dict(match_fixture(), match_id="123")
    rows = parsed.deliveries

    assert [row["score_before"] for row in rows] == [0, 1, 5, 7]
    assert [row["wickets_before"] for row in rows] == [0, 0, 0, 1]
    assert [row["legal_balls_before"] for row in rows] == [0, 0, 1, 1]
    assert [row["is_legal"] for row in rows] == [False, True, False, True]
    assert [row["source_ball_label"] for row in rows] == ["0.1", "0.1", "0.2", "0.2"]
    assert [row["legal_balls_in_over_before"] for row in rows] == [0, 0, 1, 1]
    assert all(row["phase"] == "powerplay" for row in rows)
    assert rows[1]["is_boundary"] is True


def test_parser_separates_team_and_bowler_wickets() -> None:
    parsed = CricsheetParser().parse_dict(match_fixture(), match_id="123")

    assert parsed.deliveries[2]["wicket_count"] == 1
    assert parsed.deliveries[2]["bowler_wicket_count"] == 0
    assert parsed.deliveries[2]["batter_dismissed"] is False
    assert parsed.deliveries[3]["bowler_wicket_count"] == 1
    assert parsed.deliveries[3]["batter_dismissed"] is True
    assert parsed.wickets[0]["fielder_ids"] == ["dddddddd"]


def test_parser_uses_stable_registry_ids_for_both_roles() -> None:
    parsed = CricsheetParser().parse_dict(match_fixture(), match_id="123")

    assert parsed.deliveries[0]["batter_id"] == "aaaaaaaa"
    assert parsed.deliveries[0]["bowler_id"] == "cccccccc"
    assert {row["player_id"] for row in parsed.match_players} == {
        "aaaaaaaa",
        "bbbbbbbb",
        "cccccccc",
        "dddddddd",
    }


def test_parser_rejects_unregistered_participants() -> None:
    fixture = match_fixture()
    del fixture["info"]["registry"]["people"]["Bowler One"]

    with pytest.raises(ParseError, match="missing from people registry"):
        CricsheetParser(require_registry=True).parse_dict(fixture, match_id="123")


def test_parser_preserves_fractional_target_and_pre_innings_penalty() -> None:
    fixture = match_fixture()
    fixture["innings"][0]["target"] = {"runs": 42, "overs": 5.3}
    fixture["innings"][0]["penalty_runs"] = {"pre": 5, "post": 2}

    parsed = CricsheetParser().parse_dict(fixture, match_id="123")

    assert parsed.innings[0]["target_overs_raw"] == "5.3"
    assert parsed.innings[0]["target_balls"] == 33
    assert parsed.innings[0]["penalty_runs_pre"] == 5
    assert parsed.innings[0]["penalty_runs_post"] == 2
    assert parsed.deliveries[0]["score_before"] == 5


def test_parser_preserves_replacements_reviews_and_unknown_fielders() -> None:
    fixture = match_fixture()
    fixture["info"]["registry"]["people"].update(
        {"Bowler Two": "eeeeeeee", "Umpire One": "ffffffff"}
    )
    first_delivery = fixture["innings"][0]["overs"][0]["deliveries"][0]
    first_delivery["replacements"] = {
        "role": [
            {
                "in": "Bowler Two",
                "out": "Bowler One",
                "reason": "injury",
                "role": "bowler",
            }
        ]
    }
    first_delivery["review"] = {
        "by": "Team B",
        "batter": "Batter One",
        "umpire": "Umpire One",
        "decision": "struck down",
        "umpires_call": True,
    }
    fixture["innings"][0]["overs"][0]["deliveries"][2]["wickets"][0]["fielders"] = [
        {"substitute": True}
    ]

    parsed = CricsheetParser().parse_dict(fixture, match_id="123")

    assert parsed.replacements[0]["player_in_id"] == "eeeeeeee"
    assert parsed.replacements[0]["team"] == "Team B"
    assert parsed.reviews[0]["umpire_id"] == "ffffffff"
    assert parsed.reviews[0]["umpires_call"] is True
    assert parsed.wickets[0]["unknown_fielder_count"] == 1
