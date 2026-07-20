"""Holdout baseline strategies for constrained fantasy XI selection."""

from __future__ import annotations

import random
from typing import Any, Callable

from cric_rep_learn.fantasy.optimize import (
    DEFAULT_CONSTRAINTS,
    assign_captain_vice,
    is_legal,
    optimize_xi,
)
from cric_rep_learn.fantasy.scoring import W, load_scoring_weights


StrategyFn = Callable[..., dict[str, Any]]


def _xi_player_rows(xi: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "team": p["team"],
            "role": p["role"],
            "fantasy_points": float(p["fantasy_points"]),
            "credits": p.get("credits"),
        }
        for p in xi
    ]


def score_xi_actual(
    xi: list[dict[str, Any]],
    *,
    actual_points: dict[str, float],
    captain_id: str,
    vice_id: str,
) -> float:
    """Score a XI using realized fantasy points and C/VC multipliers."""
    load_scoring_weights()
    c_mult = W("CAPTAIN_MULT")
    v_mult = W("VICE_MULT")
    total = 0.0
    for player in xi:
        pts = float(actual_points.get(player["player_id"], 0.0))
        if player["player_id"] == captain_id:
            total += pts * c_mult
        elif player["player_id"] == vice_id:
            total += pts * v_mult
        else:
            total += pts
    return float(total)


def _result_from_xi(
    xi: list[dict[str, Any]],
    *,
    method: str,
    constraints: dict[str, Any] | None,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    cv = assign_captain_vice(xi, captain_candidates=captain_candidates)
    return {
        "method": method,
        "players": _xi_player_rows(xi),
        "captain_id": cv["captain"]["player_id"],
        "vice_id": cv["vice_captain"]["player_id"],
        "predicted_xi_points": float(cv["xi_points_with_cv"]),
        "legal": is_legal(xi, constraints=constraints),
    }


def pick_random_legal_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    seed: int = 0,
    max_tries: int = 5000,
) -> dict[str, Any]:
    """Uniform random legal XI (rejection sampling)."""
    rng = random.Random(seed)
    n = len(pool)
    xi_size = int((constraints or {}).get("xi_size", 11))
    if n < xi_size:
        raise ValueError(f"pool too small: {n} < {xi_size}")
    indices = list(range(n))
    for _ in range(max_tries):
        combo = rng.sample(indices, xi_size)
        xi = [pool[i] for i in combo]
        if is_legal(xi, constraints=constraints):
            return _result_from_xi(xi, method="random", constraints=constraints)
    # Rare: role inference leaves no easy rejection hit — shuffle + greedy fill.
    order = {p["player_id"]: rng.random() for p in pool}
    return pick_greedy_legal_xi(
        pool,
        constraints=constraints,
        score_key=lambda p, order=order: order[p["player_id"]],
        method="random",
    )


def pick_naive_top11(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Top-11 by predicted points, ignoring role/team constraints."""
    xi = sorted(pool, key=lambda p: -float(p["fantasy_points"]))[:11]
    return _result_from_xi(
        xi,
        method="naive_top11",
        constraints=constraints,
        captain_candidates=captain_candidates,
    )


def _can_complete_legal(
    xi: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    picked: set[str],
    constraints: dict[str, Any],
) -> bool:
    from cric_rep_learn.fantasy.optimize import _counts as opt_counts

    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    xi_size = int(c["xi_size"])
    slots_left = xi_size - len(xi)
    if slots_left < 0:
        return False
    remaining = [p for p in pool if p["player_id"] not in picked]
    if len(remaining) < slots_left:
        return False
    roles = opt_counts(xi)["roles"]
    teams = opt_counts(xi)["teams"]
    for role, need in (
        ("WK", c["min_wk"]),
        ("BAT", c["min_bat"]),
        ("BOWL", c["min_bowl"]),
        ("AR", c["min_ar"]),
    ):
        deficit = max(0, int(need) - int(roles.get(role, 0)))
        avail = sum(1 for p in remaining if p["role"] == role)
        if avail < deficit:
            return False
    max_team = int(c["max_from_team"])
    for team, count in teams.items():
        if count > max_team:
            return False
        team_left = sum(1 for p in remaining if p["team"] == team)
        if count + team_left < slots_left and count == max_team:
            # still ok if other teams can fill — loose check only on hard cap
            pass
        if count > max_team:
            return False
    max_credits = c.get("max_credits")
    if max_credits is not None:
        used = opt_counts(xi)["credits_used"]
        cheapest_fill = sorted(
            float(p.get("credits") or 8.5) for p in remaining
        )[:slots_left]
        if used + sum(cheapest_fill) > float(max_credits) + 1e-6:
            return False
    return True


def pick_greedy_legal_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    score_key: Callable[[dict[str, Any]], float] | None = None,
    method: str = "greedy_legal",
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Fast constraint-aware greedy XI (points or value/credit)."""
    score_key = score_key or (lambda p: float(p["fantasy_points"]))
    ranked = sorted(pool, key=score_key, reverse=True)
    xi: list[dict[str, Any]] = []
    picked: set[str] = set()
    for player in ranked:
        if len(xi) >= 11:
            break
        if player["player_id"] in picked:
            continue
        trial = xi + [player]
        if _can_complete_legal(trial, pool, picked=picked | {player["player_id"]}, constraints=constraints):
            xi.append(player)
            picked.add(player["player_id"])
    if not is_legal(xi, constraints=constraints):
        opt = optimize_xi(
            pool,
            constraints=constraints,
            balance_penalty_per_slot=0.0,
            captain_candidates=captain_candidates,
            top_k=1,
        )
        xi = [
            {**p, "fantasy_points": float(p["fantasy_points"])}
            for p in opt["best_xi"]["players"]
        ]
        return {
            "method": method,
            "players": _xi_player_rows(xi),
            "captain_id": opt["best_xi"]["captain"]["player_id"],
            "vice_id": opt["best_xi"]["vice_captain"]["player_id"],
            "predicted_xi_points": float(opt["best_xi"]["xi_points_with_cv"]),
            "legal": True,
        }
    return _result_from_xi(
        xi,
        method=method,
        constraints=constraints,
        captain_candidates=captain_candidates,
    )


def pick_greedy_points_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Constrained XI maximizing predicted points (fast greedy)."""
    return pick_greedy_legal_xi(
        pool,
        constraints=constraints,
        score_key=lambda p: float(p["fantasy_points"]),
        method="greedy_points",
        captain_candidates=captain_candidates,
    )


def _pick_credits_value_optimize(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    max_credits: float = 100.0,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Constrained XI under a credit budget, ranking pool by points / credit."""
    merged_constraints = {**(constraints or {}), "max_credits": max_credits}

    def value_score(p: dict[str, Any]) -> float:
        credits = float(p.get("credits") or 8.5)
        return float(p["fantasy_points"]) / max(credits, 0.5)

    fast = pick_greedy_legal_xi(
        pool,
        constraints=merged_constraints,
        score_key=value_score,
        method="credits_value",
        captain_candidates=captain_candidates,
    )
    if fast["method"] == "credits_value":
        fast["max_credits"] = max_credits
        return fast
    value_pool = []
    for row in pool:
        credits = float(row.get("credits") or 8.5)
        value_pool.append(
            {
                **row,
                "fantasy_points": float(row["fantasy_points"]) / max(credits, 0.5),
                "_pred_points": float(row["fantasy_points"]),
            }
        )
    opt = optimize_xi(
        value_pool,
        constraints={**(constraints or {}), "max_credits": max_credits},
        balance_penalty_per_slot=0.0,
        captain_candidates=captain_candidates,
        top_k=1,
    )
    by_id = {p["player_id"]: p for p in pool}
    xi = []
    for p in opt["best_xi"]["players"]:
        base = by_id[p["player_id"]]
        xi.append({**base, "fantasy_points": float(base["fantasy_points"])})
    cv = assign_captain_vice(xi, captain_candidates=captain_candidates)
    return {
        "method": "credits_value",
        "players": _xi_player_rows(xi),
        "captain_id": cv["captain"]["player_id"],
        "vice_id": cv["vice_captain"]["player_id"],
        "predicted_xi_points": float(cv["xi_points_with_cv"]),
        "legal": True,
        "max_credits": max_credits,
    }


def pick_credits_value_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    max_credits: float = 100.0,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    return _pick_credits_value_optimize(
        pool,
        constraints=constraints,
        max_credits=max_credits,
        captain_candidates=captain_candidates,
    )


def pick_dream_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    target_roles: dict[str, int] | None = None,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Production Dream XI optimizer (C/VC search + balance penalty)."""
    opt = optimize_xi(
        pool,
        constraints=constraints,
        target_roles=target_roles,
        captain_candidates=captain_candidates,
        top_k=1,
    )
    xi = [
        {
            **p,
            "fantasy_points": float(p["fantasy_points"]),
        }
        for p in opt["best_xi"]["players"]
    ]
    return {
        "method": "dream_xi",
        "players": _xi_player_rows(xi),
        "captain_id": opt["best_xi"]["captain"]["player_id"],
        "vice_id": opt["best_xi"]["vice_captain"]["player_id"],
        "predicted_xi_points": float(opt["best_xi"]["xi_points_with_cv"]),
        "legal": True,
    }


def pick_oracle_xi(
    pool: list[dict[str, Any]],
    *,
    actual_points: dict[str, float],
    constraints: dict[str, Any] | None = None,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Best legal XI using realized points (cheating upper bound)."""
    oracle_pool = [
        {**p, "fantasy_points": float(actual_points.get(p["player_id"], 0.0))}
        for p in pool
    ]
    opt = optimize_xi(
        oracle_pool,
        constraints=constraints,
        balance_penalty_per_slot=0.0,
        captain_candidates=captain_candidates,
        top_k=1,
    )
    xi = []
    for p in opt["best_xi"]["players"]:
        base = next(row for row in pool if row["player_id"] == p["player_id"])
        xi.append({**base, "fantasy_points": float(base["fantasy_points"])})
    return {
        "method": "oracle_actual",
        "players": _xi_player_rows(xi),
        "captain_id": opt["best_xi"]["captain"]["player_id"],
        "vice_id": opt["best_xi"]["vice_captain"]["player_id"],
        "predicted_xi_points": float(opt["best_xi"]["xi_points_with_cv"]),
        "legal": True,
    }


BASELINE_STRATEGIES: dict[str, StrategyFn] = {
    "random": pick_random_legal_xi,
    "naive_top11": pick_naive_top11,
    "greedy_points": pick_greedy_points_xi,
    "credits_value": pick_credits_value_xi,
    "dream_xi": pick_dream_xi,
}


def top11_overlap(
    selected_ids: set[str],
    actual_top_ids: set[str],
    *,
    k: int = 11,
) -> float:
    if not actual_top_ids:
        return 0.0
    return len(selected_ids & actual_top_ids) / float(k)
