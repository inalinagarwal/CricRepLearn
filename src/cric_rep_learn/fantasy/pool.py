"""Build fantasy player pools from joint match simulation outputs."""

from __future__ import annotations

from typing import Any

from cric_rep_learn.fantasy.scoring import average_player_pools, merge_player_points


def pool_from_match(
    match_result: dict[str, Any],
    *,
    first_team: str,
    chase_team: str,
    roles: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Map one toss scenario to a 22-player fantasy pool.

    ``roles`` keyed by player_id → WK|BAT|AR|BOWL.
    First-innings batters + chase bowlers are ``first_team`` contributors;
    chase batters + first-innings bowlers are ``chase_team``.
    """
    first = match_result["first_innings"]
    chase = match_result["chase_innings"]
    bat_first = {b["player_id"]: b for b in first["batters"]}
    bat_chase = {b["player_id"]: b for b in chase["batters"]}
    bowl_first = {b["player_id"]: b for b in first["bowlers"]}
    bowl_chase = {b["player_id"]: b for b in chase["bowlers"]}

    # All unique players across both lineups.
    players: dict[str, dict[str, Any]] = {}
    for row in match_result.get("context", {}).get("first_batters") or []:
        players[row["player_id"]] = {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "team": first_team,
        }
    for row in match_result.get("context", {}).get("chase_batters") or []:
        players[row["player_id"]] = {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "team": chase_team,
        }
    # Ensure bowlers appear even if not in bat list (shouldn't happen for XI).
    for src, team in ((bowl_first, chase_team), (bowl_chase, first_team)):
        for pid, b in src.items():
            if pid not in players:
                players[pid] = {
                    "player_id": pid,
                    "player_name": b["player_name"],
                    "team": team,
                }

    pool: list[dict[str, Any]] = []
    for pid, meta in players.items():
        role = roles.get(pid)
        if role is None:
            raise ValueError(f"missing fantasy role for {meta['player_name']} ({pid})")
        batting = bat_first.get(pid) or bat_chase.get(pid)
        bowling = bowl_first.get(pid) or bowl_chase.get(pid)
        pool.append(
            merge_player_points(
                player_id=pid,
                player_name=meta["player_name"],
                team=meta["team"],
                role=role.upper(),
                batting=batting,
                bowling=bowling,
            )
        )
    pool.sort(key=lambda r: -r["fantasy_points"])
    return pool


def pool_average_tosses(
    scenarios: list[tuple[dict[str, Any], str, str]],
    *,
    roles: dict[str, str],
) -> list[dict[str, Any]]:
    """
    ``scenarios`` = list of (match_result, first_team, chase_team).
    Average fantasy points across toss outcomes.
    """
    pools = [
        pool_from_match(result, first_team=ft, chase_team=ct, roles=roles)
        for result, ft, ct in scenarios
    ]
    return average_player_pools(pools)
