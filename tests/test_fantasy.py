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
    from cric_rep_learn.fantasy.scoring import DEFAULT_WEIGHTS, save_scoring_weights

    trial = Path("/tmp/fantasy_default_weights_test.json")
    save_scoring_weights(dict(DEFAULT_WEIGHTS), trial)
    load_scoring_weights(trial)
    bat = batting_points({"expected_runs": 45.0, "expected_balls": 30.0})
    assert bat["batting_points"] > 45  # milestone 30
    bowl = bowling_points(
        {"expected_wickets": 2.2, "expected_overs": 4.0, "expected_economy": 6.5}
    )
    assert bowl["bowling_points"] > 2.2 * DEFAULT_WEIGHTS["BOWL_WICKET"]


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


def test_expected_haul_from_mc_tail_probs() -> None:
    """Haul bonuses are nonlinear — use MC P(W>=k), not threshold on E[W]."""
    from cric_rep_learn.fantasy.scoring import _expected_tier_points

    mean_only = bowling_points(
        {"expected_wickets": 1.4, "expected_overs": 4.0, "expected_economy": 7.5}
    )
    assert mean_only["haul_component"] == 0.0

    with_tails = bowling_points(
        {
            "expected_wickets": 1.4,
            "expected_overs": 4.0,
            "expected_economy": 7.5,
            "p_wickets_ge3": 0.20,
            "p_wickets_ge4": 0.06,
            "p_wickets_ge5": 0.01,
        }
    )
    expected = _expected_tier_points(
        p_ge_low=0.20,
        p_ge_mid=0.06,
        p_ge_high=0.01,
        pts_low=4.0,
        pts_mid=8.0,
        pts_high=16.0,
    )
    assert with_tails["haul_component"] == pytest.approx(expected)
    assert with_tails["haul_component"] > 0.0
    assert with_tails["bowling_points"] > mean_only["bowling_points"]


def test_expected_milestone_from_mc_tail_probs() -> None:
    """Milestone bonuses use MC P(R>=k) when present."""
    from cric_rep_learn.fantasy.scoring import _expected_tier_points

    mean_only = batting_points({"expected_runs": 22.0, "expected_balls": 18.0})
    assert mean_only["milestone_component"] == 0.0

    with_tails = batting_points(
        {
            "expected_runs": 22.0,
            "expected_balls": 18.0,
            "p_runs_ge30": 0.35,
            "p_runs_ge50": 0.10,
            "p_runs_ge100": 0.01,
        }
    )
    expected = _expected_tier_points(
        p_ge_low=0.35,
        p_ge_mid=0.10,
        p_ge_high=0.01,
        pts_low=4.0,
        pts_mid=8.0,
        pts_high=16.0,
    )
    assert with_tails["milestone_component"] == pytest.approx(expected)
    assert with_tails["batting_points"] > mean_only["batting_points"]


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


def test_box_scores_preserve_allrounder_bowling() -> None:
    """Batting appearance must not wipe overs/wickets for the same player."""
    from pathlib import Path

    from cric_rep_learn.fantasy.calibration import build_match_box_scores

    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical artifacts not present")
    table = build_match_box_scores(canonical, splits=("validation",))
    frame = table.to_pandas()
    # Players who both faced balls and bowled legal deliveries.
    both = frame[(frame["balls"] > 0) & (frame["overs"] > 0)]
    assert len(both) >= 1
    assert float(both["wickets"].sum()) >= 0.0
    # Spot-check: at least one all-rounder with a real bowling spell.
    assert float(both["overs"].max()) >= 1.0


def test_reconstruct_holdout_smoke() -> None:
    from pathlib import Path

    from cric_rep_learn.fantasy.holdout_mc import reconstruct_holdout_matches

    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical artifacts not present")
    setups = reconstruct_holdout_matches(
        canonical,
        splits=("validation",),
        max_matches=3,
        seed=0,
        opportunity="scheduled",
    )
    assert len(setups) >= 1
    s0 = setups[0]
    assert len(s0.first_lineup) >= 2
    assert len(s0.first_attack) == 5
    assert s0.first_team and s0.chase_team
    assert sum(b.max_overs for b in s0.first_attack) == 20
    setups_actual = reconstruct_holdout_matches(
        canonical,
        splits=("validation",),
        max_matches=3,
        seed=0,
        opportunity="actual",
    )
    assert len(setups_actual[0].first_attack) >= 1
    assert sum(b.max_overs for b in setups_actual[0].first_attack) >= 1


def test_holdout_min_batters_avoids_high_wicket_bias() -> None:
    """
    Requiring many *faced* batters selects collapse-heavy matches and makes
    calibrated hazards look under-rate. Default min_batters=2 keeps the pool
    near validation mean wickets (~12).
    """
    import duckdb
    import numpy as np

    from cric_rep_learn.fantasy.holdout_mc import reconstruct_holdout_matches

    canonical = Path("artifacts/canonical")
    deliveries = canonical / "deliveries.parquet"
    if not deliveries.exists():
        pytest.skip("canonical artifacts not present")

    loose = reconstruct_holdout_matches(
        canonical, splits=("validation",), max_matches=80, seed=7, min_batters=2
    )
    tight = reconstruct_holdout_matches(
        canonical, splits=("validation",), max_matches=80, seed=7, min_batters=8
    )
    assert len(loose) == 80 and len(tight) == 80

    def _mean_wickets(match_ids: list[str]) -> float:
        con = duckdb.connect()
        try:
            row = con.execute(
                """
                SELECT AVG(w) FROM (
                  SELECT match_id, SUM(bowler_wicket_count)::DOUBLE AS w
                  FROM read_parquet(?)
                  WHERE match_id IN (SELECT * FROM UNNEST(?))
                    AND NOT is_super_over
                    AND (is_legal OR extras_noballs > 0)
                  GROUP BY 1
                )
                """,
                [str(deliveries.resolve()), match_ids],
            ).fetchone()
        finally:
            con.close()
        return float(row[0])

    loose_w = _mean_wickets([s.match_id for s in loose])
    tight_w = _mean_wickets([s.match_id for s in tight])
    # High facer threshold inflates wickets; loose pool should sit nearer ~12.
    assert tight_w > loose_w + 1.5
    assert loose_w < 13.5
    assert np.isfinite(loose_w)


def test_innings_expected_runs_reads_nested_team() -> None:
    """Chase targets must use team.expected_runs from simulate_innings."""
    from cric_rep_learn.fantasy.holdout_mc import _innings_expected_runs

    nested = {"team": {"expected_runs": 148.5}, "batters": [], "bowlers": []}
    assert _innings_expected_runs(nested) == pytest.approx(148.5)
    # Legacy/slim payload with top-level key still works.
    assert _innings_expected_runs({"expected_runs": 120.0}) == pytest.approx(120.0)
    # Missing key must not silently become a 1-run chase target via `or 0`.
    assert _innings_expected_runs({"batters": [], "bowlers": []}) == 0.0
    assert _innings_expected_runs(nested) + 1.0 == pytest.approx(149.5)


def test_holdout_mc_chase_contributes_both_innings(tmp_path: Path) -> None:
    """
    Holdout MC must not collapse chase to target≈1 (half match scale).

    Smoke on one reconstructed match: both teams bat and ~40 overs are scheduled.
    """
    from cric_rep_learn.fantasy.holdout_mc import (
        predict_holdout_via_mc,
        reconstruct_holdout_matches,
    )

    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical artifacts not present")
    attrs = Path("artifacts/player-attributes/player_attributes.parquet")
    effects = Path("artifacts/player-effects/player_effects.parquet")
    matchups = Path("artifacts/player-effects/batter_bowler_matchups.parquet")
    if not attrs.exists() or not effects.exists() or not matchups.exists():
        pytest.skip("player-effect artifacts not present")

    setups = reconstruct_holdout_matches(
        canonical,
        splits=("validation",),
        max_matches=1,
        seed=7,
        opportunity="scheduled",
    )
    assert setups
    pred = predict_holdout_via_mc(
        setups,
        canonical_dir=canonical,
        attributes_path=attrs,
        effects_path=effects,
        matchups_path=matchups,
        chase_impacts_path=Path("artifacts/baselines/chase_impacts.json"),
        co_batters_path=Path("artifacts/co-batters/co_batters.parquet"),
        n_sims=8,
        seed=7,
    )
    assert len(pred) >= 20
    totals = pred.groupby("match_id").agg(
        runs=("expected_runs", "sum"),
        overs=("expected_overs", "sum"),
        balls=("expected_balls", "sum"),
    )
    row = totals.iloc[0]
    # Full match: ~2×20 overs and batting from both XIs (not one innings only).
    assert float(row["overs"]) >= 30.0
    assert float(row["runs"]) >= 180.0
    assert float(row["balls"]) >= 180.0
    by_team = pred.groupby("team")["expected_runs"].sum()
    assert (by_team > 40.0).all()


def test_baseline_strategies_smoke() -> None:
    from cric_rep_learn.fantasy.baselines import (
        pick_dream_xi,
        pick_greedy_points_xi,
        pick_naive_top11,
        pick_random_legal_xi,
        score_xi_actual,
    )
    from cric_rep_learn.fantasy.scoring import merge_player_points

    pool = []
    roles = ("WK", "BAT", "BAT", "BAT", "AR", "AR", "BOWL", "BOWL", "BOWL", "BAT", "BAT", "BOWL")
    for i, role in enumerate(roles):
        pool.append(
            merge_player_points(
                player_id=f"p{i}",
                player_name=f"P{i}",
                team="A" if i < 6 else "B",
                role=role,
                batting={"expected_runs": 30 - i, "expected_balls": 20},
                bowling={
                    "expected_wickets": max(0.0, 2.5 - i * 0.2),
                    "expected_overs": 3.0,
                    "expected_economy": 7.0,
                },
                credits=8.5,
            )
        )
    actual = {p["player_id"]: float(20 + i) for i, p in enumerate(pool)}
    random_pick = pick_random_legal_xi(pool, seed=3)
    naive = pick_naive_top11(pool)
    greedy = pick_greedy_points_xi(pool)
    dream = pick_dream_xi(pool)
    assert random_pick["legal"]
    assert len(naive["players"]) == 11
    assert greedy["legal"] and dream["legal"]
    pts = score_xi_actual(
        dream["players"],
        actual_points=actual,
        captain_id=dream["captain_id"],
        vice_id=dream["vice_id"],
    )
    assert pts > 0


def test_bowler_wicket_prior_multiplier() -> None:
    from cric_rep_learn.simulation.priors import load_bowler_wicket_priors

    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical data not built")
    priors = load_bowler_wicket_priors(canonical)
    assert priors
    vals = list(priors.values())
    assert all(0.75 <= v <= 1.35 for v in vals)
    assert max(vals) > 1.0 and min(vals) < 1.0


def test_optimize_uses_role_composition_for_large_pools() -> None:
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
    for i in range(8):
        pool.append(
            merge_player_points(
                player_id=f"bat{i}",
                player_name=f"BAT{i}",
                team="A" if i < 4 else "B",
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
    for i in range(8):
        pool.append(
            merge_player_points(
                player_id=f"bowl{i}",
                player_name=f"BOWL{i}",
                team="A" if i < 4 else "B",
                role="BOWL",
                bowling={
                    "expected_wickets": 1.5 - 0.05 * i,
                    "expected_overs": 4.0,
                    "expected_economy": 7.5,
                },
                credits=8.5,
            )
        )
    assert len(pool) == 22
    result = optimize_xi(pool, top_k=1, captain_candidates=3)
    assert result["search"] == "role_composition"
    # C(22,11) = 705432; pruned role search should be far smaller.
    assert result["combinations_checked"] < 150_000
    assert result["cv_candidates"] <= 250
    assert is_legal(result["best_xi"]["players"])


def test_fantasy_uncertainty_quantiles() -> None:
    row = merge_player_points(
        player_id="x",
        player_name="X",
        team="A",
        role="BAT",
        batting={
            "expected_runs": 40,
            "expected_balls": 28,
            "runs_p10": 10,
            "runs_p50": 35,
            "runs_p90": 70,
            "p_runs_ge30": 0.55,
            "p_runs_ge50": 0.25,
            "p_runs_ge100": 0.02,
        },
        bowling={
            "expected_wickets": 1.2,
            "expected_overs": 4.0,
            "expected_economy": 7.0,
            "wickets_p10": 0.0,
            "wickets_p50": 1.0,
            "wickets_p90": 3.0,
            "p_wickets_ge3": 0.2,
            "p_wickets_ge4": 0.05,
            "p_wickets_ge5": 0.01,
        },
    )
    assert row["fantasy_points_p10"] is not None
    assert row["fantasy_points_p90"] is not None
    assert row["fantasy_points_p10"] < row["fantasy_points"] < row["fantasy_points_p90"]


def test_batting_order_opportunity_preserves_totals() -> None:
    from cric_rep_learn.simulation.batting_opportunity import (
        apply_batting_order_opportunity,
    )

    batters = [
        {"player_id": f"b{i}", "expected_balls": 10.0, "expected_runs": 12.0,
         "expected_fours": 1.0, "expected_sixes": 0.5}
        for i in range(11)
    ]
    # Front-load shares like T20 openers.
    shares = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    out = apply_batting_order_opportunity(batters, shares=shares, blend=0.5)
    assert abs(sum(b["expected_balls"] for b in out) - 110.0) < 1e-6
    assert abs(sum(b["expected_runs"] for b in out) - 132.0) < 1e-6
    assert out[0]["expected_balls"] > out[-1]["expected_balls"]
