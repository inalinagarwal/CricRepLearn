"""Loadable fantasy scoring weights + point calculators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Defaults (Dream11-inspired, not identical). Overridden by scoring_weights.json.
DEFAULT_WEIGHTS: dict[str, float] = {
    "BAT_RUN": 1.0,
    "BAT_MILESTONE_30": 4.0,
    "BAT_MILESTONE_50": 8.0,
    "BAT_MILESTONE_100": 16.0,
    "BAT_FOUR": 1.0,
    "BAT_SIX": 2.0,
    "BAT_SR_BONUS_THRESHOLD": 140.0,
    "BAT_SR_BONUS": 2.0,
    "BAT_SR_PENALTY_THRESHOLD": 100.0,
    "BAT_SR_PENALTY": -2.0,
    "BAT_MIN_BALLS_FOR_SR": 10.0,
    "BOWL_WICKET": 30.0,
    "BOWL_HAUL_3": 4.0,
    "BOWL_HAUL_4": 8.0,
    "BOWL_HAUL_5": 16.0,
    "BOWL_ECON_BASELINE": 7.5,
    "BOWL_ECON_PER_RUN": 1.0,
    "BOWL_MIN_OVERS_FOR_ECON": 2.0,
    "CAPTAIN_MULT": 2.0,
    "VICE_MULT": 1.5,
    "BALANCE_PENALTY_PER_SLOT": 8.0,
}

_WEIGHTS: dict[str, float] = dict(DEFAULT_WEIGHTS)


def load_scoring_weights(path: Path | None = None) -> dict[str, float]:
    """Merge JSON overrides into module weights; return active weights."""
    global _WEIGHTS
    _WEIGHTS = dict(DEFAULT_WEIGHTS)
    path = path or Path("artifacts/fantasy/scoring_weights.json")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        overrides = payload.get("weights") if isinstance(payload, dict) else payload
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if key in _WEIGHTS:
                    _WEIGHTS[key] = float(value)
    _refresh_exports()
    return dict(_WEIGHTS)


def save_scoring_weights(
    weights: dict[str, float],
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_WEIGHTS, **weights}
    payload = {"weights": merged, **(metadata or {})}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_scoring_weights(path)


def W(key: str) -> float:
    return float(_WEIGHTS.get(key, DEFAULT_WEIGHTS[key]))


def _prob_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _expected_tier_points(
    *,
    p_ge_low: float,
    p_ge_mid: float,
    p_ge_high: float,
    pts_low: float,
    pts_mid: float,
    pts_high: float,
) -> float:
    """
    Expected points for mutually exclusive escalating tiers.

    Tier pays ``pts_low`` on [low, mid), ``pts_mid`` on [mid, high),
    ``pts_high`` on [high, ∞). With survival probs p_ge_*:

        E = pts_low*(p_low-p_mid) + pts_mid*(p_mid-p_high) + pts_high*p_high
          = (pts_low)*p_low + (pts_mid-pts_low)*p_mid + (pts_high-pts_mid)*p_high
    """
    p_low = float(min(max(p_ge_low, 0.0), 1.0))
    p_mid = float(min(max(p_ge_mid, 0.0), p_low))
    p_high = float(min(max(p_ge_high, 0.0), p_mid))
    return float(
        pts_low * p_low
        + (pts_mid - pts_low) * p_mid
        + (pts_high - pts_mid) * p_high
    )


# Back-compat module-level names used elsewhere.
def _refresh_exports() -> None:
    global BAT_RUN, BAT_MILESTONE_30, BAT_MILESTONE_50, BAT_MILESTONE_100
    global BAT_FOUR, BAT_SIX
    global BAT_SR_BONUS_THRESHOLD, BAT_SR_BONUS, BAT_SR_PENALTY_THRESHOLD, BAT_SR_PENALTY
    global BAT_MIN_BALLS_FOR_SR
    global BOWL_WICKET, BOWL_HAUL_3, BOWL_HAUL_4, BOWL_HAUL_5
    global BOWL_ECON_BASELINE, BOWL_ECON_PER_RUN, BOWL_MIN_OVERS_FOR_ECON
    global CAPTAIN_MULT, VICE_MULT, BALANCE_PENALTY_PER_SLOT
    BAT_RUN = W("BAT_RUN")
    BAT_MILESTONE_30 = W("BAT_MILESTONE_30")
    BAT_MILESTONE_50 = W("BAT_MILESTONE_50")
    BAT_MILESTONE_100 = W("BAT_MILESTONE_100")
    BAT_FOUR = W("BAT_FOUR")
    BAT_SIX = W("BAT_SIX")
    BAT_SR_BONUS_THRESHOLD = W("BAT_SR_BONUS_THRESHOLD")
    BAT_SR_BONUS = W("BAT_SR_BONUS")
    BAT_SR_PENALTY_THRESHOLD = W("BAT_SR_PENALTY_THRESHOLD")
    BAT_SR_PENALTY = W("BAT_SR_PENALTY")
    BAT_MIN_BALLS_FOR_SR = W("BAT_MIN_BALLS_FOR_SR")
    BOWL_WICKET = W("BOWL_WICKET")
    BOWL_HAUL_3 = W("BOWL_HAUL_3")
    BOWL_HAUL_4 = W("BOWL_HAUL_4")
    BOWL_HAUL_5 = W("BOWL_HAUL_5")
    BOWL_ECON_BASELINE = W("BOWL_ECON_BASELINE")
    BOWL_ECON_PER_RUN = W("BOWL_ECON_PER_RUN")
    BOWL_MIN_OVERS_FOR_ECON = W("BOWL_MIN_OVERS_FOR_ECON")
    CAPTAIN_MULT = W("CAPTAIN_MULT")
    VICE_MULT = W("VICE_MULT")
    BALANCE_PENALTY_PER_SLOT = W("BALANCE_PENALTY_PER_SLOT")


load_scoring_weights()
_refresh_exports()


def batting_points(batter: dict[str, Any]) -> dict[str, float]:
    runs = float(
        batter.get("expected_runs")
        if batter.get("expected_runs") is not None
        else batter.get("runs")
        or 0.0
    )
    balls = float(
        batter.get("expected_balls")
        if batter.get("expected_balls") is not None
        else batter.get("balls")
        or 0.0
    )
    fours = float(
        batter.get("expected_fours")
        if batter.get("expected_fours") is not None
        else batter.get("fours")
        or 0.0
    )
    sixes = float(
        batter.get("expected_sixes")
        if batter.get("expected_sixes") is not None
        else batter.get("sixes")
        or 0.0
    )
    pts = runs * W("BAT_RUN")
    boundary = fours * W("BAT_FOUR") + sixes * W("BAT_SIX")
    pts += boundary
    milestone = 0.0
    p30 = _prob_or_none(batter.get("p_runs_ge30"))
    p50 = _prob_or_none(batter.get("p_runs_ge50"))
    p100 = _prob_or_none(batter.get("p_runs_ge100"))
    if p30 is not None or p50 is not None or p100 is not None:
        # MC survival probs → expected milestone (nonlinear in runs).
        milestone = _expected_tier_points(
            p_ge_low=float(p30 or 0.0),
            p_ge_mid=float(p50 or 0.0),
            p_ge_high=float(p100 or 0.0),
            pts_low=W("BAT_MILESTONE_30"),
            pts_mid=W("BAT_MILESTONE_50"),
            pts_high=W("BAT_MILESTONE_100"),
        )
    elif runs >= 100:
        milestone = W("BAT_MILESTONE_100")
    elif runs >= 50:
        milestone = W("BAT_MILESTONE_50")
    elif runs >= 30:
        milestone = W("BAT_MILESTONE_30")
    pts += milestone
    sr_bonus = 0.0
    if balls >= W("BAT_MIN_BALLS_FOR_SR"):
        sr = 100.0 * runs / balls
        if sr >= W("BAT_SR_BONUS_THRESHOLD"):
            sr_bonus = W("BAT_SR_BONUS")
        elif sr < W("BAT_SR_PENALTY_THRESHOLD"):
            sr_bonus = W("BAT_SR_PENALTY")
        pts += sr_bonus
    return {
        "batting_points": float(pts),
        "runs_component": float(runs * W("BAT_RUN")),
        "boundary_component": float(boundary),
        "milestone_component": float(milestone),
        "sr_component": float(sr_bonus),
    }


def bowling_points(bowler: dict[str, Any]) -> dict[str, float]:
    wickets = float(
        bowler.get("expected_wickets")
        if bowler.get("expected_wickets") is not None
        else bowler.get("wickets")
        or 0.0
    )
    overs = float(
        bowler.get("expected_overs")
        if bowler.get("expected_overs") is not None
        else bowler.get("overs")
        or 0.0
    )
    econ = bowler.get("expected_economy")
    if econ is None:
        econ = bowler.get("economy")
    pts = wickets * W("BOWL_WICKET")
    haul = 0.0
    p3 = _prob_or_none(bowler.get("p_wickets_ge3"))
    p4 = _prob_or_none(bowler.get("p_wickets_ge4"))
    p5 = _prob_or_none(bowler.get("p_wickets_ge5"))
    if p3 is not None or p4 is not None or p5 is not None:
        # MC survival probs → expected haul bonus (nonlinear in wickets).
        haul = _expected_tier_points(
            p_ge_low=float(p3 or 0.0),
            p_ge_mid=float(p4 or 0.0),
            p_ge_high=float(p5 or 0.0),
            pts_low=W("BOWL_HAUL_3"),
            pts_mid=W("BOWL_HAUL_4"),
            pts_high=W("BOWL_HAUL_5"),
        )
    elif wickets >= 5:
        haul = W("BOWL_HAUL_5")
    elif wickets >= 4:
        haul = W("BOWL_HAUL_4")
    elif wickets >= 3:
        haul = W("BOWL_HAUL_3")
    pts += haul
    econ_pts = 0.0
    if econ is not None and overs >= W("BOWL_MIN_OVERS_FOR_ECON"):
        econ_pts = (W("BOWL_ECON_BASELINE") - float(econ)) * overs * W("BOWL_ECON_PER_RUN")
        pts += econ_pts
    return {
        "bowling_points": float(pts),
        "wicket_component": float(wickets * W("BOWL_WICKET")),
        "haul_component": float(haul),
        "economy_component": float(econ_pts),
    }


def _fantasy_quantile(
    batting: dict[str, Any],
    bowling: dict[str, Any],
    *,
    runs_key: str,
    wickets_key: str,
) -> float | None:
    """Approximate fantasy P10/P50/P90 from MC run/wicket quantiles."""
    runs = batting.get(runs_key)
    wickets = bowling.get(wickets_key)
    if runs is None and wickets is None:
        return None
    bat_q = dict(batting)
    bowl_q = dict(bowling)
    if runs is not None:
        bat_q["expected_runs"] = float(runs)
        # Scale balls with runs so SR stays sane at the quantile.
        base_runs = float(batting.get("expected_runs") or 0.0)
        base_balls = float(batting.get("expected_balls") or 0.0)
        if base_runs > 1e-9:
            bat_q["expected_balls"] = base_balls * (float(runs) / base_runs)
    if wickets is not None:
        bowl_q["expected_wickets"] = float(wickets)
    # Quantile paths should not double-count survival EV tiers.
    for key in (
        "p_runs_ge30",
        "p_runs_ge50",
        "p_runs_ge100",
        "p_wickets_ge3",
        "p_wickets_ge4",
        "p_wickets_ge5",
    ):
        bat_q.pop(key, None)
        bowl_q.pop(key, None)
    return float(
        batting_points(bat_q)["batting_points"] + bowling_points(bowl_q)["bowling_points"]
    )


def merge_player_points(
    *,
    player_id: str,
    player_name: str,
    team: str,
    role: str,
    batting: dict[str, Any] | None = None,
    bowling: dict[str, Any] | None = None,
    credits: float | None = None,
) -> dict[str, Any]:
    bat = batting_points(batting or {})
    bowl = bowling_points(bowling or {})
    total = bat["batting_points"] + bowl["bowling_points"]
    batting = batting or {}
    bowling = bowling or {}
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "role": role,
        "credits": float(credits) if credits is not None else None,
        "expected_runs": float(
            batting.get("expected_runs")
            if batting.get("expected_runs") is not None
            else batting.get("runs")
            or 0.0
        ),
        "expected_balls": float(
            batting.get("expected_balls")
            if batting.get("expected_balls") is not None
            else batting.get("balls")
            or 0.0
        ),
        "expected_fours": float(
            batting.get("expected_fours")
            if batting.get("expected_fours") is not None
            else batting.get("fours")
            or 0.0
        ),
        "expected_sixes": float(
            batting.get("expected_sixes")
            if batting.get("expected_sixes") is not None
            else batting.get("sixes")
            or 0.0
        ),
        "runs_p10": batting.get("runs_p10"),
        "runs_p50": batting.get("runs_p50"),
        "runs_p90": batting.get("runs_p90"),
        "balls_p10": batting.get("balls_p10"),
        "balls_p50": batting.get("balls_p50"),
        "balls_p90": batting.get("balls_p90"),
        "p_runs_ge30": batting.get("p_runs_ge30"),
        "p_runs_ge50": batting.get("p_runs_ge50"),
        "p_runs_ge100": batting.get("p_runs_ge100"),
        "expected_wickets": float(
            bowling.get("expected_wickets")
            if bowling.get("expected_wickets") is not None
            else bowling.get("wickets")
            or 0.0
        ),
        "expected_overs": float(
            bowling.get("expected_overs")
            if bowling.get("expected_overs") is not None
            else bowling.get("overs")
            or 0.0
        ),
        "expected_economy": bowling.get("expected_economy")
        if bowling.get("expected_economy") is not None
        else bowling.get("economy"),
        "wickets_p10": bowling.get("wickets_p10"),
        "wickets_p50": bowling.get("wickets_p50"),
        "wickets_p90": bowling.get("wickets_p90"),
        "p_wickets_ge3": bowling.get("p_wickets_ge3"),
        "p_wickets_ge4": bowling.get("p_wickets_ge4"),
        "p_wickets_ge5": bowling.get("p_wickets_ge5"),
        **bat,
        **bowl,
        "fantasy_points": float(total),
        "fantasy_points_p10": _fantasy_quantile(
            batting, bowling, runs_key="runs_p10", wickets_key="wickets_p10"
        ),
        "fantasy_points_p50": _fantasy_quantile(
            batting, bowling, runs_key="runs_p50", wickets_key="wickets_p50"
        ),
        "fantasy_points_p90": _fantasy_quantile(
            batting, bowling, runs_key="runs_p90", wickets_key="wickets_p90"
        ),
    }


def average_player_pools(
    pools: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average fantasy points across toss scenarios (same player_id keys)."""
    if not pools:
        return []
    by_id: dict[str, list[dict[str, Any]]] = {}
    for pool in pools:
        for row in pool:
            by_id.setdefault(row["player_id"], []).append(row)
    averaged: list[dict[str, Any]] = []
    for _pid, rows in by_id.items():
        n = float(len(rows))
        base = dict(rows[0])
        numeric = (
            "expected_runs",
            "expected_balls",
            "expected_fours",
            "expected_sixes",
            "runs_p10",
            "runs_p50",
            "runs_p90",
            "balls_p10",
            "balls_p50",
            "balls_p90",
            "p_runs_ge30",
            "p_runs_ge50",
            "p_runs_ge100",
            "expected_wickets",
            "expected_overs",
            "wickets_p10",
            "wickets_p50",
            "wickets_p90",
            "p_wickets_ge3",
            "p_wickets_ge4",
            "p_wickets_ge5",
            "batting_points",
            "bowling_points",
            "fantasy_points",
            "fantasy_points_p10",
            "fantasy_points_p50",
            "fantasy_points_p90",
            "runs_component",
            "boundary_component",
            "milestone_component",
            "sr_component",
            "wicket_component",
            "haul_component",
            "economy_component",
            "credits",
        )
        for key in numeric:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if not vals:
                continue
            base[key] = float(sum(float(v) for v in vals) / len(vals))
        econs = [
            r.get("expected_economy")
            for r in rows
            if r.get("expected_economy") is not None
        ]
        base["expected_economy"] = float(sum(econs) / len(econs)) if econs else None
        base["n_scenarios"] = int(n)
        averaged.append(base)
    averaged.sort(key=lambda r: -r["fantasy_points"])
    return averaged


# Ensure exports exist after load.
BAT_RUN = W("BAT_RUN")
BAT_MILESTONE_30 = W("BAT_MILESTONE_30")
BAT_MILESTONE_50 = W("BAT_MILESTONE_50")
BAT_MILESTONE_100 = W("BAT_MILESTONE_100")
BAT_FOUR = W("BAT_FOUR")
BAT_SIX = W("BAT_SIX")
BAT_SR_BONUS_THRESHOLD = W("BAT_SR_BONUS_THRESHOLD")
BAT_SR_BONUS = W("BAT_SR_BONUS")
BAT_SR_PENALTY_THRESHOLD = W("BAT_SR_PENALTY_THRESHOLD")
BAT_SR_PENALTY = W("BAT_SR_PENALTY")
BAT_MIN_BALLS_FOR_SR = W("BAT_MIN_BALLS_FOR_SR")
BOWL_WICKET = W("BOWL_WICKET")
BOWL_HAUL_3 = W("BOWL_HAUL_3")
BOWL_HAUL_4 = W("BOWL_HAUL_4")
BOWL_HAUL_5 = W("BOWL_HAUL_5")
BOWL_ECON_BASELINE = W("BOWL_ECON_BASELINE")
BOWL_ECON_PER_RUN = W("BOWL_ECON_PER_RUN")
BOWL_MIN_OVERS_FOR_ECON = W("BOWL_MIN_OVERS_FOR_ECON")
CAPTAIN_MULT = W("CAPTAIN_MULT")
VICE_MULT = W("VICE_MULT")
BALANCE_PENALTY_PER_SLOT = W("BALANCE_PENALTY_PER_SLOT")
