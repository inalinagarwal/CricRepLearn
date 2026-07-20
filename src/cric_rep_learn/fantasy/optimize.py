"""Constrained fantasy XI optimizer with venue tilt and C/VC search."""

from __future__ import annotations

import itertools
from typing import Any

from cric_rep_learn.fantasy.scoring import W, load_scoring_weights


DEFAULT_CONSTRAINTS = {
    "xi_size": 11,
    "max_from_team": 7,
    "min_wk": 1,
    "min_bat": 3,
    "min_bowl": 3,
    "min_ar": 1,
    "max_bat": 5,
    "max_bowl": 5,
    "max_credits": None,
}

DEFAULT_TARGET_ROLES = {"WK": 1, "BAT": 4, "AR": 2, "BOWL": 4}

# Exact enum is cheap below this; above it we use role-composition search.
_EXACT_COMBO_LIMIT = 50_000
# Keep only top candidates for full C/VC search after a cheap base-points pass.
_CV_CANDIDATE_LIMIT = 250
# Cap players considered per role in large pools.
_ROLE_POOL_CAP = 8


def _counts(xi: list[dict[str, Any]]) -> dict[str, Any]:
    roles = {"WK": 0, "BAT": 0, "AR": 0, "BOWL": 0}
    teams: dict[str, int] = {}
    credits = 0.0
    for row in xi:
        role = str(row["role"]).upper()
        roles[role] = roles.get(role, 0) + 1
        teams[row["team"]] = teams.get(row["team"], 0) + 1
        if row.get("credits") is not None:
            credits += float(row["credits"])
    return {"roles": roles, "teams": teams, "credits_used": credits}


def is_legal(
    xi: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
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
    max_credits = c.get("max_credits")
    if max_credits is not None and counts["credits_used"] > float(max_credits) + 1e-6:
        return False
    return True


def xi_base_points(xi: list[dict[str, Any]]) -> float:
    return float(sum(float(p["fantasy_points"]) for p in xi))


def balance_penalty(
    xi: list[dict[str, Any]],
    *,
    target_roles: dict[str, int] | None = None,
    penalty_per_slot: float | None = None,
) -> float:
    if penalty_per_slot is None:
        penalty_per_slot = W("BALANCE_PENALTY_PER_SLOT")
    target = {**DEFAULT_TARGET_ROLES, **(target_roles or {})}
    roles = _counts(xi)["roles"]
    dist = 0
    for role in ("WK", "BAT", "AR", "BOWL"):
        dist += abs(int(roles.get(role, 0)) - int(target.get(role, 0)))
    return float((dist / 2.0) * penalty_per_slot)


def assign_captain_vice(
    xi: list[dict[str, Any]],
    *,
    captain_candidates: int = 5,
) -> dict[str, Any]:
    """Search C/VC among top-N scorers in the XI."""
    load_scoring_weights()
    c_mult = W("CAPTAIN_MULT")
    v_mult = W("VICE_MULT")
    ranked = sorted(xi, key=lambda p: -float(p["fantasy_points"]))
    n = min(max(int(captain_candidates), 2), len(ranked))
    candidates = ranked[:n]
    best: dict[str, Any] | None = None
    for i, captain in enumerate(candidates):
        for j, vice in enumerate(candidates):
            if i == j:
                continue
            total = 0.0
            for player in xi:
                pts = float(player["fantasy_points"])
                if player["player_id"] == captain["player_id"]:
                    total += pts * c_mult
                elif player["player_id"] == vice["player_id"]:
                    total += pts * v_mult
                else:
                    total += pts
            row = {
                "captain": {
                    "player_id": captain["player_id"],
                    "player_name": captain["player_name"],
                    "fantasy_points": captain["fantasy_points"],
                    "multiplier": c_mult,
                },
                "vice_captain": {
                    "player_id": vice["player_id"],
                    "player_name": vice["player_name"],
                    "fantasy_points": vice["fantasy_points"],
                    "multiplier": v_mult,
                },
                "xi_points_raw": xi_base_points(xi),
                "xi_points_with_cv": float(total),
                "captain_candidates": n,
            }
            if best is None or row["xi_points_with_cv"] > best["xi_points_with_cv"]:
                best = row
    assert best is not None
    return best


def _xi_row(
    xi: list[dict[str, Any]],
    *,
    target: dict[str, int],
    balance_penalty_per_slot: float,
    captain_candidates: int,
    cv: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cv is None:
        cv = assign_captain_vice(xi, captain_candidates=captain_candidates)
    penalty = balance_penalty(
        xi, target_roles=target, penalty_per_slot=balance_penalty_per_slot
    )
    score = float(cv["xi_points_with_cv"] - penalty)
    return {
        "players": [
            {
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "team": p["team"],
                "role": p["role"],
                "fantasy_points": p["fantasy_points"],
                "credits": p.get("credits"),
                "fantasy_points_p10": p.get("fantasy_points_p10"),
                "fantasy_points_p50": p.get("fantasy_points_p50"),
                "fantasy_points_p90": p.get("fantasy_points_p90"),
            }
            for p in sorted(xi, key=lambda x: -float(x["fantasy_points"]))
        ],
        **cv,
        **_counts(xi),
        "balance_penalty": penalty,
        "objective_score": score,
        "target_roles": target,
    }


def _role_compositions(
    constraints: dict[str, Any],
    *,
    target: dict[str, int],
    max_role_dist: int = 2,
) -> list[dict[str, int]]:
    """Legal role counts near the venue target (keeps search small)."""
    xi_size = int(constraints["xi_size"])
    comps: list[dict[str, int]] = []
    for wk in range(int(constraints["min_wk"]), xi_size + 1):
        for bat in range(int(constraints["min_bat"]), int(constraints["max_bat"]) + 1):
            for bowl in range(
                int(constraints["min_bowl"]), int(constraints["max_bowl"]) + 1
            ):
                for ar in range(int(constraints["min_ar"]), xi_size + 1):
                    if wk + bat + bowl + ar != xi_size:
                        continue
                    dist = (
                        abs(wk - int(target.get("WK", 1)))
                        + abs(bat - int(target.get("BAT", 4)))
                        + abs(ar - int(target.get("AR", 2)))
                        + abs(bowl - int(target.get("BOWL", 4)))
                    )
                    # dist is L1; /2 because each move transfers one slot.
                    if dist // 2 > max_role_dist:
                        continue
                    comps.append({"WK": wk, "BAT": bat, "AR": ar, "BOWL": bowl})
    if not comps:
        # Fallback: any legal composition.
        for wk in range(int(constraints["min_wk"]), xi_size + 1):
            for bat in range(
                int(constraints["min_bat"]), int(constraints["max_bat"]) + 1
            ):
                for bowl in range(
                    int(constraints["min_bowl"]), int(constraints["max_bowl"]) + 1
                ):
                    ar = xi_size - wk - bat - bowl
                    if ar < int(constraints["min_ar"]):
                        continue
                    comps.append({"WK": wk, "BAT": bat, "AR": ar, "BOWL": bowl})
    return comps


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k in {0, n}:
        return 1
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def _prune_role_pool(
    players: list[dict[str, Any]],
    *,
    need: int,
    cap: int = _ROLE_POOL_CAP,
) -> list[dict[str, Any]]:
    ranked = sorted(players, key=lambda p: -float(p["fantasy_points"]))
    keep = max(need, min(cap, len(ranked)))
    return ranked[:keep]


def optimize_xi(
    pool: list[dict[str, Any]],
    *,
    constraints: dict[str, Any] | None = None,
    target_roles: dict[str, int] | None = None,
    top_k: int = 5,
    balance_penalty_per_slot: float | None = None,
    captain_candidates: int = 5,
    prune_roles: bool = True,
) -> dict[str, Any]:
    """
    Constrained XI search.

    Large pools use role-composition enumeration with:
      1) role-pool pruning (top scorers per role)
      2) compositions near the venue target
      3) cheap base-points pass, then C/VC only on top candidates
    """
    load_scoring_weights()
    if balance_penalty_per_slot is None:
        balance_penalty_per_slot = W("BALANCE_PENALTY_PER_SLOT")
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    target = {**DEFAULT_TARGET_ROLES, **(target_roles or {})}
    n = len(pool)
    if n < c["xi_size"]:
        raise ValueError(f"need at least {c['xi_size']} players, got {n}")

    role_pool_full = {
        role: [p for p in pool if str(p["role"]).upper() == role]
        for role in ("WK", "BAT", "AR", "BOWL")
    }
    for role, need in (
        ("WK", c["min_wk"]),
        ("BAT", c["min_bat"]),
        ("BOWL", c["min_bowl"]),
        ("AR", c["min_ar"]),
    ):
        if len(role_pool_full[role]) < need:
            raise ValueError(
                f"pool has {len(role_pool_full[role])} {role} players; need ≥{need}"
            )

    exact_combos = _n_choose_k(n, int(c["xi_size"]))
    use_role_search = exact_combos > _EXACT_COMBO_LIMIT
    role_pool = role_pool_full
    if use_role_search and prune_roles:
        role_pool = {
            "WK": _prune_role_pool(role_pool_full["WK"], need=int(c["min_wk"])),
            "BAT": _prune_role_pool(
                role_pool_full["BAT"], need=int(c["max_bat"]), cap=_ROLE_POOL_CAP
            ),
            "AR": _prune_role_pool(role_pool_full["AR"], need=int(c["min_ar"])),
            "BOWL": _prune_role_pool(
                role_pool_full["BOWL"], need=int(c["max_bowl"]), cap=_ROLE_POOL_CAP
            ),
        }

    checked = 0
    legal = 0
    import heapq

    cheap_heap: list[tuple[float, int, list[dict[str, Any]]]] = []
    counter = 0

    def consider_cheap(xi: list[dict[str, Any]]) -> None:
        nonlocal checked, legal, counter
        checked += 1
        if not is_legal(xi, constraints=c):
            return
        legal += 1
        penalty = balance_penalty(
            xi,
            target_roles=target,
            penalty_per_slot=float(balance_penalty_per_slot),
        )
        score = float(xi_base_points(xi) - penalty)
        counter += 1
        item = (score, counter, xi)
        if len(cheap_heap) < _CV_CANDIDATE_LIMIT:
            heapq.heappush(cheap_heap, item)
        elif score > cheap_heap[0][0]:
            heapq.heapreplace(cheap_heap, item)

    if use_role_search:
        method = (
            "Role-composition search (pruned pools, target-near comps) with "
            f"deferred C/VC over top-{_CV_CANDIDATE_LIMIT}; "
            f"C/VC among top-{captain_candidates}; "
            "rank by C/VC points − venue balance penalty"
        )
        for comp in _role_compositions(c, target=target):
            if any(len(role_pool[r]) < comp[r] for r in comp):
                continue
            for wk_ids in itertools.combinations(role_pool["WK"], comp["WK"]):
                for bat_ids in itertools.combinations(role_pool["BAT"], comp["BAT"]):
                    for ar_ids in itertools.combinations(role_pool["AR"], comp["AR"]):
                        for bowl_ids in itertools.combinations(
                            role_pool["BOWL"], comp["BOWL"]
                        ):
                            xi = (
                                list(wk_ids)
                                + list(bat_ids)
                                + list(ar_ids)
                                + list(bowl_ids)
                            )
                            consider_cheap(xi)
        # Pruning can miss legal XIs (team caps); retry unpruned once.
        if not cheap_heap and prune_roles:
            return optimize_xi(
                pool,
                constraints=constraints,
                target_roles=target_roles,
                top_k=top_k,
                balance_penalty_per_slot=balance_penalty_per_slot,
                captain_candidates=captain_candidates,
                prune_roles=False,
            )
    else:
        method = (
            "Enumerate C(n,11) under max-from-team + role min/max + optional credits; "
            f"C/VC search over top-{captain_candidates}; "
            "rank by C/VC points − venue balance penalty"
        )
        for combo in itertools.combinations(range(n), c["xi_size"]):
            consider_cheap([pool[i] for i in combo])

    if not cheap_heap:
        raise RuntimeError("no legal XI found under constraints")

    cheap_best = sorted(cheap_heap, key=lambda r: -r[0])
    best: list[dict[str, Any]] = []
    for _, _, xi in cheap_best:
        row = _xi_row(
            xi,
            target=target,
            balance_penalty_per_slot=float(balance_penalty_per_slot),
            captain_candidates=captain_candidates,
        )
        best.append(row)
    best.sort(key=lambda r: -r["objective_score"])
    best = best[:top_k]

    return {
        "constraints": c,
        "target_roles": target,
        "pool_size": n,
        "combinations_checked": checked,
        "legal_xis": legal,
        "cv_candidates": len(cheap_best),
        "best_xi": best[0],
        "top_xis": best,
        "search": "role_composition" if use_role_search else "exact",
        "method": method,
    }
