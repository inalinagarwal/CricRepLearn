"""Custom fantasy points from simulated batting and bowling contributions."""

from __future__ import annotations

from typing import Any


# Performance-based scoring (Dream11-inspired, not identical).
BAT_RUN = 1.0
BAT_MILESTONE_30 = 4.0
BAT_MILESTONE_50 = 8.0
BAT_MILESTONE_100 = 16.0
# Soft strike-rate bonus when balls are meaningful.
BAT_SR_BONUS_THRESHOLD = 140.0  # runs per 100 balls
BAT_SR_BONUS = 2.0
BAT_SR_PENALTY_THRESHOLD = 100.0
BAT_SR_PENALTY = -2.0
BAT_MIN_BALLS_FOR_SR = 10.0

BOWL_WICKET = 30.0
BOWL_HAUL_3 = 4.0
BOWL_HAUL_4 = 8.0
BOWL_HAUL_5 = 16.0
# Economy: points per over relative to 7.5 baseline (lower econ → more points).
BOWL_ECON_BASELINE = 7.5
BOWL_ECON_PER_RUN = 1.0  # +1 pt per run/over under baseline; -1 over
BOWL_MIN_OVERS_FOR_ECON = 2.0

CAPTAIN_MULT = 2.0
VICE_MULT = 1.5

# Soft penalty per role-slot away from venue target mix (keeps sides balanced).
BALANCE_PENALTY_PER_SLOT = 8.0


def batting_points(batter: dict[str, Any]) -> dict[str, float]:
    runs = float(batter.get("expected_runs") or 0.0)
    balls = float(batter.get("expected_balls") or 0.0)
    pts = runs * BAT_RUN
    if runs >= 100:
        pts += BAT_MILESTONE_100
    elif runs >= 50:
        pts += BAT_MILESTONE_50
    elif runs >= 30:
        pts += BAT_MILESTONE_30
    sr_bonus = 0.0
    if balls >= BAT_MIN_BALLS_FOR_SR:
        sr = 100.0 * runs / balls
        if sr >= BAT_SR_BONUS_THRESHOLD:
            sr_bonus = BAT_SR_BONUS
        elif sr < BAT_SR_PENALTY_THRESHOLD:
            sr_bonus = BAT_SR_PENALTY
        pts += sr_bonus
    return {
        "batting_points": float(pts),
        "runs_component": float(runs * BAT_RUN),
        "milestone_component": float(pts - runs * BAT_RUN - sr_bonus),
        "sr_component": float(sr_bonus),
    }


def bowling_points(bowler: dict[str, Any]) -> dict[str, float]:
    wickets = float(bowler.get("expected_wickets") or 0.0)
    overs = float(bowler.get("expected_overs") or 0.0)
    econ = bowler.get("expected_economy")
    pts = wickets * BOWL_WICKET
    if wickets >= 5:
        pts += BOWL_HAUL_5
    elif wickets >= 4:
        pts += BOWL_HAUL_4
    elif wickets >= 3:
        pts += BOWL_HAUL_3
    econ_pts = 0.0
    if econ is not None and overs >= BOWL_MIN_OVERS_FOR_ECON:
        econ_pts = (BOWL_ECON_BASELINE - float(econ)) * overs * BOWL_ECON_PER_RUN
        pts += econ_pts
    return {
        "bowling_points": float(pts),
        "wicket_component": float(wickets * BOWL_WICKET),
        "haul_component": float(
            BOWL_HAUL_5
            if wickets >= 5
            else BOWL_HAUL_4
            if wickets >= 4
            else BOWL_HAUL_3
            if wickets >= 3
            else 0.0
        ),
        "economy_component": float(econ_pts),
    }


def merge_player_points(
    *,
    player_id: str,
    player_name: str,
    team: str,
    role: str,
    batting: dict[str, Any] | None = None,
    bowling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bat = batting_points(batting or {})
    bowl = bowling_points(bowling or {})
    total = bat["batting_points"] + bowl["bowling_points"]
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "role": role,
        "expected_runs": float((batting or {}).get("expected_runs") or 0.0),
        "expected_balls": float((batting or {}).get("expected_balls") or 0.0),
        "expected_wickets": float((bowling or {}).get("expected_wickets") or 0.0),
        "expected_overs": float((bowling or {}).get("expected_overs") or 0.0),
        "expected_economy": (bowling or {}).get("expected_economy"),
        **bat,
        **bowl,
        "fantasy_points": float(total),
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
    for pid, rows in by_id.items():
        n = float(len(rows))
        base = dict(rows[0])
        numeric = (
            "expected_runs",
            "expected_balls",
            "expected_wickets",
            "expected_overs",
            "batting_points",
            "bowling_points",
            "fantasy_points",
            "runs_component",
            "milestone_component",
            "sr_component",
            "wicket_component",
            "haul_component",
            "economy_component",
        )
        for key in numeric:
            base[key] = float(sum(float(r.get(key) or 0.0) for r in rows) / n)
        econs = [r.get("expected_economy") for r in rows if r.get("expected_economy") is not None]
        base["expected_economy"] = float(sum(econs) / len(econs)) if econs else None
        base["n_scenarios"] = int(n)
        averaged.append(base)
    averaged.sort(key=lambda r: -r["fantasy_points"])
    return averaged
