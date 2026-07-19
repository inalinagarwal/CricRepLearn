"""Constrained fantasy XI optimizer with venue role tilt."""

from __future__ import annotations

import itertools
from typing import Any

from cric_rep_learn.fantasy.scoring import (
    BALANCE_PENALTY_PER_SLOT,
    CAPTAIN_MULT,
    VICE_MULT,
)


DEFAULT_CONSTRAINTS = {
    "xi_size": 11,
    "max_from_team": 7,
    "min_wk": 1,
    "min_bat": 3,
    "min_bowl": 3,
    "min_ar": 1,
    "max_bat": 5,
    "max_bowl": 5,
}

DEFAULT_TARGET_ROLES = {"WK": 1, "BAT": 4, "AR": 2, "BOWL": 4}


def _counts(xi: list[dict[str, Any]]) -> dict[str, Any]:
    roles = {"WK": 0, "BAT": 0, "AR": 0, "BOWL": 0}
    teams: dict[str, int] = {}
    for row in xi:
        role = str(row["role"]).upper()
        roles[role] = roles.get(role, 0) + 1
        teams[row["team"]] = teams.get(row["team"], 0) + 1
    return {"roles": roles, "teams": teams}


def is_legal(
    xi: list[dict[str, Any]],
    *,
    constraints: dict[str, int] | None = None,
) -> bool:
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    if len(xi) != c["xi_size"]:
        return False
    counts = _counts(xi)
    if any(v > c["max_from_team"] for v in counts["teams"].values()):
        return False
    roles = counts["roles"]
    if roles.get("WK", 0) < c["min_wk"]:
        return False
    if roles.get("BAT", 0) < c["min_bat"]:
        return False
    if roles.get("BOWL", 0) < c["min_bowl"]:
        return False
    if roles.get("AR", 0) < c["min_ar"]:
        return False
    if roles.get("BAT", 0) > c.get("max_bat", 99):
        return False
    if roles.get("BOWL", 0) > c.get("max_bowl", 99):
        return False
    return True


def xi_base_points(xi: list[dict[str, Any]]) -> float:
    return float(sum(float(p["fantasy_points"]) for p in xi))


def balance_penalty(
    xi: list[dict[str, Any]],
    *,
    target_roles: dict[str, int] | None = None,
    penalty_per_slot: float = BALANCE_PENALTY_PER_SLOT,
) -> float:
    """L1 distance from target WK-BAT-AR-BOWL mix."""
    target = {**DEFAULT_TARGET_ROLES, **(target_roles or {})}
    roles = _counts(xi)["roles"]
    dist = 0
    for role in ("WK", "BAT", "AR", "BOWL"):
        dist += abs(int(roles.get(role, 0)) - int(target.get(role, 0)))
    # Each mis-slot is counted twice in a pure L1 over a fixed sum — use half.
    return float((dist / 2.0) * penalty_per_slot)


def assign_captain_vice(
    xi: list[dict[str, Any]],
) -> dict[str, Any]:
    """C = highest points, VC = second; team score with multipliers."""
    ranked = sorted(xi, key=lambda p: -float(p["fantasy_points"]))
    captain = ranked[0]
    vice = ranked[1]
    total = 0.0
    for player in xi:
        pts = float(player["fantasy_points"])
        if player["player_id"] == captain["player_id"]:
            total += pts * CAPTAIN_MULT
        elif player["player_id"] == vice["player_id"]:
            total += pts * VICE_MULT
        else:
            total += pts
    return {
        "captain": {
            "player_id": captain["player_id"],
            "player_name": captain["player_name"],
            "fantasy_points": captain["fantasy_points"],
            "multiplier": CAPTAIN_MULT,
        },
        "vice_captain": {
            "player_id": vice["player_id"],
            "player_name": vice["player_name"],
            "fantasy_points": vice["fantasy_points"],
            "multiplier": VICE_MULT,
        },
        "xi_points_raw": xi_base_points(xi),
        "xi_points_with_cv": float(total),
    }


def optimize_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, int] | None = None,
    target_roles: dict[str, int] | None = None,
    top_k: int = 5,
    balance_penalty_per_slot: float = BALANCE_PENALTY_PER_SLOT,
) -> dict[str, Any]:
    """
    Enumerate legal XIs and rank by C/VC score minus venue-balance penalty.
    """
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    target = {**DEFAULT_TARGET_ROLES, **(target_roles or {})}
    n = len(pool)
    if n < c["xi_size"]:
        raise ValueError(f"need at least {c['xi_size']} players, got {n}")

    role_pool = {
        role: [p for p in pool if p["role"] == role] for role in ("WK", "BAT", "AR", "BOWL")
    }
    for role, need in (
        ("WK", c["min_wk"]),
        ("BAT", c["min_bat"]),
        ("BOWL", c["min_bowl"]),
        ("AR", c["min_ar"]),
    ):
        if len(role_pool[role]) < need:
            raise ValueError(
                f"pool has {len(role_pool[role])} {role} players; need ≥{need}"
            )

    best: list[dict[str, Any]] = []
    checked = 0
    legal = 0
    for combo in itertools.combinations(range(n), c["xi_size"]):
        checked += 1
        xi = [pool[i] for i in combo]
        if not is_legal(xi, constraints=c):
            continue
        legal += 1
        cv = assign_captain_vice(xi)
        penalty = balance_penalty(
            xi, target_roles=target, penalty_per_slot=balance_penalty_per_slot
        )
        score = float(cv["xi_points_with_cv"] - penalty)
        row = {
            "players": [
                {
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "team": p["team"],
                    "role": p["role"],
                    "fantasy_points": p["fantasy_points"],
                }
                for p in sorted(xi, key=lambda x: -x["fantasy_points"])
            ],
            **cv,
            **_counts(xi),
            "balance_penalty": penalty,
            "objective_score": score,
            "target_roles": target,
        }
        best.append(row)
        best.sort(key=lambda r: -r["objective_score"])
        if len(best) > top_k:
            best = best[:top_k]

    if not best:
        raise RuntimeError("no legal XI found under constraints")

    return {
        "constraints": c,
        "target_roles": target,
        "pool_size": n,
        "combinations_checked": checked,
        "legal_xis": legal,
        "best_xi": best[0],
        "top_xis": best,
        "method": (
            "Enumerate C(n,11) under max-from-team + role min/max; "
            "rank by C/VC points − venue balance penalty; "
            "C=top scorer ×2, VC=2nd ×1.5"
        ),
    }
