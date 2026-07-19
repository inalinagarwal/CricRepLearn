"""Tests for fantasy scoring and XI constraints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cric_rep_learn.fantasy.calibration import _spearman, tune_bowl_wicket_weight
from cric_rep_learn.fantasy.optimize import assign_captain_vice, is_legal, optimize_xi
from cric_rep_learn.fantasy.roles import credit_proxy, infer_role_from_attributes, map_playing_role
from cric_rep_learn.fantasy.scoring import (
    DEFAULT_WEIGHTS,
    batting_points,
    bowling_points,
    load_scoring_weights,
    merge_player_points,
    save_scoring_weights,
)
from cric_rep_learn.simulation.run_sampler import sample_runs


def test_batting_and_bowling_points() -> None:
    bat = batting_points({"expected_runs": 45.0, "expected_balls": 30.0})
    assert bat["batting_points"] > 45  # milestone 30
    bowl = bowling_points(
        {"expected_wickets": 2.2, "expected_overs": 4.0, "expected_economy": 6.5}
    )
    assert bowl["bowling_points"] > 2.2 * 30


def test_boundary_fantasy_components() -> None:
    bat = batting_points(
        {
            "expected_runs": 40.0,
            "expected_balls": 25.0,
            "expected_fours": 4.0,
            "expected_sixes": 2.0,
        }
    )
    assert bat["boundary_component"] == pytest.approx(4.0 * 1.0 + 2.0 * 2.0)


def test_role_mapper() -> None:
    assert map_playing_role("Wicketkeeper Batter") == "WK"
    assert map_playing_role("Middle order Batter") == "BAT"
    assert map_playing_role("Bowling Allrounder") == "AR"
    assert map_playing_role("Bowler") == "BOWL"
    assert (
        infer_role_from_attributes(
            {"bowling_style_raw": "Right-arm fast"},
            batting_order=9,
            bowls_in_attack=True,
        )
        == "BOWL"
    )
    assert (
        infer_role_from_attributes(
            {"bowling_style_raw": "Right-arm offbreak"},
            batting_order=6,
            bowls_in_attack=True,
        )
        == "AR"
    )
    assert credit_proxy("WK") == 8.5
    assert credit_proxy("BAT", batting_sr=1.5) == 10.0


def test_cvc_search_picks_top_scorers_over_naive_fixed_slots() -> None:
    """Search C/VC among top-N; skipping #2 for VC is strictly worse."""
    xi = [
        merge_player_points(
            player_id="p1",
            player_name="P1",
            team="A",
            role="BAT",
            batting={"expected_runs": 100, "expected_balls": 60},
        ),
        merge_player_points(
            player_id="p2",
            player_name="P2",
            team="A",
            role="BAT",
            batting={"expected_runs": 80, "expected_balls": 50},
        ),
        merge_player_points(
            player_id="p3",
            player_name="P3",
            team="B",
            role="BOWL",
            bowling={
                "expected_wickets": 2.0,
                "expected_overs": 4.0,
                "expected_economy": 7.0,
            },
        ),
        merge_player_points(
            player_id="p4",
            player_name="P4",
            team="B",
            role="AR",
            batting={"expected_runs": 10, "expected_balls": 10},
        ),
    ]
    ranked = sorted(xi, key=lambda r: -r["fantasy_points"])
    assert ranked[0]["player_id"] == "p1"
    cv = assign_captain_vice(xi, captain_candidates=3)
    assert cv["captain"]["player_id"] == ranked[0]["player_id"]
    assert cv["vice_captain"]["player_id"] == ranked[1]["player_id"]
    c_pts = ranked[0]["fantasy_points"]
    v2 = ranked[1]["fantasy_points"]
    v3 = ranked[2]["fantasy_points"]
    naive_skip = (
        c_pts * 2.0
        + v3 * 1.5
        + sum(p["fantasy_points"] for p in ranked[1:2] + ranked[3:])
    )
    assert cv["xi_points_with_cv"] > naive_skip
    assert cv["xi_points_with_cv"] == pytest.approx(
        c_pts * 2.0 + v2 * 1.5 + sum(p["fantasy_points"] for p in ranked[2:])
    )


def test_optimize_respects_constraints() -> None:
    pool = []
    for i in range(2):
        pool.append(
            merge_player_points(
                player_id=f"wk{i}",
                player_name=f"WK{i}",
                team="A" if i == 0 else "B",
                role="WK",
                batting={"expected_runs": 30 + i, "expected_balls": 20},
                credits=8.5,
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
                credits=9.0,
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
                credits=8.5,
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
                credits=8.5,
            )
        )
    assert len(pool) == 18
    result = optimize_xi(pool, top_k=3, captain_candidates=5)
    xi = result["best_xi"]
    assert is_legal([{**p, "fantasy_points": p["fantasy_points"]} for p in xi["players"]])
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
    assert xi["captain_candidates"] == 5


def test_max_credits_constraint() -> None:
    pool = []
    for i in range(2):
        pool.append(
            merge_player_points(
                player_id=f"wk{i}",
                player_name=f"WK{i}",
                team="A" if i == 0 else "B",
                role="WK",
                batting={"expected_runs": 30, "expected_balls": 20},
                credits=8.0,
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
                credits=9.0 if i < 3 else 8.0,
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
                credits=8.5,
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
                    "expected_wickets": 1.5,
                    "expected_overs": 4.0,
                    "expected_economy": 7.5,
                },
                credits=9.5 if i < 3 else 7.0,
            )
        )
    # Expensive A-heavy XI exceeds 90; mixed cheap XI fits.
    result = optimize_xi(pool, constraints={"max_credits": 90}, top_k=1)
    assert result["best_xi"]["credits_used"] <= 90 + 1e-6
    with pytest.raises(RuntimeError):
        optimize_xi(pool, constraints={"max_credits": 50}, top_k=1)


def test_max_from_team_blocks_stacked_xi() -> None:
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
    result = optimize_xi(pool, constraints={"max_from_team": 7}, top_k=1)
    assert max(result["best_xi"]["teams"].values()) <= 7


def test_run_sampler_mean_near_sr() -> None:
    rng = np.random.default_rng(0)
    table = {
        "outcomes": [0, 1, 2, 4, 6],
        "buckets": {
            "sr_1.1_1.4": [0.36, 0.34, 0.12, 0.12, 0.06],
        },
    }
    draws = [sample_runs(rng, 1.25, table=table) for _ in range(4000)]
    mean = float(np.mean(draws))
    assert abs(mean - 1.25) < 0.25


def test_calibration_tune_smoke(tmp_path: Path) -> None:
    import pandas as pd

    rows = []
    pred_rows = []
    for m in range(5):
        for p in range(12):
            rows.append(
                {
                    "match_id": f"m{m}",
                    "player_id": f"p{p}",
                    "player_name": f"P{p}",
                    "team": "A" if p < 6 else "B",
                    "runs": 10 + p,
                    "balls": 8 + p // 2,
                    "fours": p % 3,
                    "sixes": p % 2,
                    "dismissals": 0,
                    "wickets": 1 if p >= 8 else 0,
                    "overs": 4.0 if p >= 8 else 0.0,
                    "runs_conceded": 28.0 if p >= 8 else 0.0,
                    "economy": 7.0 if p >= 8 else float("nan"),
                    "fantasy_points": 0.0,
                }
            )
            # Noisy MC-like predictions (not shrink-of-actual).
            pred_rows.append(
                {
                    "match_id": f"m{m}",
                    "player_id": f"p{p}",
                    "player_name": f"P{p}",
                    "team": "A" if p < 6 else "B",
                    "expected_runs": 8 + 0.8 * p + (m % 3),
                    "expected_balls": 7 + 0.5 * p,
                    "expected_fours": max(0, (p % 3) - 0.2),
                    "expected_sixes": max(0, (p % 2) - 0.1),
                    "expected_wickets": 0.7 if p >= 8 else 0.05,
                    "expected_overs": 3.5 if p >= 8 else 0.0,
                    "expected_economy": 7.2 if p >= 8 else None,
                }
            )
    frame = pd.DataFrame(rows)
    pred = pd.DataFrame(pred_rows)
    result = tune_bowl_wicket_weight(frame, pred_frame=pred, max_matches=5)
    assert result["best"]["BOWL_WICKET"] in {25.0, 30.0, 35.0}
    assert result["method"] == "hb_mc_holdout"
    assert "spearman" in result["best"]
    weights_path = tmp_path / "scoring_weights.json"
    save_scoring_weights(
        {**DEFAULT_WEIGHTS, "BOWL_WICKET": result["best"]["BOWL_WICKET"]},
        weights_path,
    )
    loaded = load_scoring_weights(weights_path)
    assert loaded["BOWL_WICKET"] == result["best"]["BOWL_WICKET"]
    assert _spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.5])) > 0.9


def test_reconstruct_holdout_smoke() -> None:
    from pathlib import Path

    from cric_rep_learn.fantasy.holdout_mc import reconstruct_holdout_matches

    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical artifacts not present")
    setups = reconstruct_holdout_matches(
        canonical, splits=("validation",), max_matches=3, seed=0
    )
    assert len(setups) >= 1
    s0 = setups[0]
    assert len(s0.first_lineup) >= 2
    assert len(s0.first_attack) >= 1
    assert s0.first_team and s0.chase_team
