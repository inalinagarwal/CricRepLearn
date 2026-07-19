"""Shared service layer for CLI modes and the public web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.fantasy.optimize import optimize_xi
from cric_rep_learn.fantasy.pool import pool_average_tosses
from cric_rep_learn.fantasy.roles import resolve_squad_roles
from cric_rep_learn.fantasy.scoring import load_scoring_weights
from cric_rep_learn.fantasy.venue_tilt import venue_scoring_profile
from cric_rep_learn.players.card import resolve_player
from cric_rep_learn.players.forecast_vs_attack import forecast_vs_attack
from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    attach_phase_profiles,
    load_bowler_phase_profiles,
)
from cric_rep_learn.simulation.chase import load_chase_impacts
from cric_rep_learn.simulation.match import simulate_match
from cric_rep_learn.simulation.partnership import load_partnership_index
from cric_rep_learn.simulation.priors import InningsRateModel

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "canonical": ROOT / "artifacts" / "canonical",
    "attributes": ROOT / "artifacts" / "player-attributes" / "player_attributes.parquet",
    "effects": ROOT / "artifacts" / "player-effects" / "player_effects.parquet",
    "matchups": ROOT / "artifacts" / "player-effects" / "batter_bowler_matchups.parquet",
    "weather": ROOT / "artifacts" / "weather",
    "chase_impacts": ROOT / "artifacts" / "baselines" / "chase_impacts.json",
    "co_batters": ROOT / "artifacts" / "co-batters" / "co_batters.parquet",
    "scoring_weights": ROOT / "artifacts" / "fantasy" / "scoring_weights.json",
}


def _parse_names(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]


def _resolve_lineup(names: list[str], aliases, attributes) -> list[dict[str, Any]]:
    lineup = []
    for query in names:
        resolved = resolve_player(query, aliases, attributes=attributes)
        attrs = attributes.get(resolved["player_id"], {})
        lineup.append(
            {
                "player_id": resolved["player_id"],
                "player_name": resolved["player_name"],
                "batting_hand": str(attrs.get("batting_hand") or "unknown"),
                "query": query,
            }
        )
    return lineup


def _resolve_attack(
    names: list[str], aliases, attributes, *, canonical_dir: Path
) -> list[BowlerSpell]:
    attack: list[BowlerSpell] = []
    for q in names:
        resolved = resolve_player(q, aliases, attributes=attributes)
        attack.append(
            BowlerSpell(
                player_id=resolved["player_id"],
                player_name=resolved["player_name"],
                max_overs=4,
            )
        )
    profiles = load_bowler_phase_profiles(canonical_dir, [b.player_id for b in attack])
    return attach_phase_profiles(attack, profiles)


def run_dream_xi(
    *,
    team_a_batters: str | list[str],
    team_b_batters: str | list[str],
    team_a_bowlers: str | list[str],
    team_b_bowlers: str | list[str],
    team_a_name: str = "IND",
    team_b_name: str = "ENG",
    venue: str | None = "Lord's",
    date: str | None = None,
    sims: int = 80,
    seed: int = 7,
    captain_candidates: int = 5,
    max_credits: float | None = 100.0,
    max_from_team: int = 7,
    top_k: int = 3,
) -> dict[str, Any]:
    """Mode 1: toss-averaged fantasy MC → constrained Dream XI."""
    canonical = DEFAULTS["canonical"]
    load_scoring_weights(
        DEFAULTS["scoring_weights"] if DEFAULTS["scoring_weights"].exists() else None
    )
    aliases = pq.read_table(canonical / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(DEFAULTS["attributes"])

    a_bat = _resolve_lineup(_parse_names(team_a_batters), aliases, attributes)
    b_bat = _resolve_lineup(_parse_names(team_b_batters), aliases, attributes)
    a_bowl = _resolve_attack(
        _parse_names(team_a_bowlers), aliases, attributes, canonical_dir=canonical
    )
    b_bowl = _resolve_attack(
        _parse_names(team_b_bowlers), aliases, attributes, canonical_dir=canonical
    )
    for i, row in enumerate(a_bat):
        row["batting_order"] = i + 1
    for i, row in enumerate(b_bat):
        row["batting_order"] = i + 1

    attack_ids = {b.player_id for b in a_bowl + b_bowl}
    role_info = resolve_squad_roles(
        a_bat + b_bat, attributes=attributes, attack_ids=attack_ids
    )
    roles = {pid: info["role"] for pid, info in role_info.items()}
    credits = {pid: info["credits"] for pid, info in role_info.items()}

    def rates(group: str) -> InningsRateModel:
        return InningsRateModel(
            canonical_dir=canonical,
            effects_path=DEFAULTS["effects"],
            matchups_path=DEFAULTS["matchups"],
            attributes=attributes,
            venue=venue,
            innings_group=group,
            match_date=date,
            weather_dir=DEFAULTS["weather"] if date else None,
        )

    chase_impacts = load_chase_impacts(DEFAULTS["chase_impacts"], canonical_dir=canonical)
    partnership_index = load_partnership_index(DEFAULTS["co_batters"])

    toss_a = simulate_match(
        first_lineup=a_bat,
        first_attack=b_bowl,
        chase_lineup=b_bat,
        chase_attack=a_bowl,
        first_rates=rates("first_innings"),
        chase_rates=rates("chase"),
        n_sims=sims,
        seed=seed,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )
    toss_a["context"] = {
        "first_batters": a_bat,
        "chase_batters": b_bat,
        "first_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in b_bowl],
        "chase_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in a_bowl],
    }
    toss_b = simulate_match(
        first_lineup=b_bat,
        first_attack=a_bowl,
        chase_lineup=a_bat,
        chase_attack=b_bowl,
        first_rates=rates("first_innings"),
        chase_rates=rates("chase"),
        n_sims=sims,
        seed=seed + 17,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )
    toss_b["context"] = {
        "first_batters": b_bat,
        "chase_batters": a_bat,
        "first_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in a_bowl],
        "chase_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in b_bowl],
    }

    pool = pool_average_tosses(
        [(toss_a, team_a_name, team_b_name), (toss_b, team_b_name, team_a_name)],
        roles=roles,
        credits=credits,
    )
    venue_profile = venue_scoring_profile(canonical, venue)
    constraints = {"max_from_team": max_from_team, **venue_profile.get("constraints", {})}
    if max_credits is not None:
        constraints["max_credits"] = max_credits
    opt = optimize_xi(
        pool,
        constraints=constraints,
        target_roles=venue_profile.get("target_roles"),
        top_k=top_k,
        captain_candidates=captain_candidates,
    )
    xi = opt["best_xi"]
    return {
        "mode": "dream_xi",
        "venue": venue,
        "match_date": date,
        "n_sims": sims,
        "toss_a": toss_a["match"],
        "toss_b": toss_b["match"],
        "best_xi": {
            "captain": xi["captain"],
            "vice_captain": xi["vice_captain"],
            "objective_score": xi["objective_score"],
            "credits_used": xi.get("credits_used"),
            "roles": xi["roles"],
            "teams": xi["teams"],
            "players": xi["players"],
        },
        "player_pool": [
            {
                "player_name": p["player_name"],
                "team": p["team"],
                "role": p["role"],
                "fantasy_points": p["fantasy_points"],
                "batting_points": p.get("batting_points"),
                "bowling_points": p.get("bowling_points"),
            }
            for p in pool
        ],
    }


def run_match_sim(
    *,
    first_batters: str | list[str],
    first_bowlers: str | list[str],
    chase_batters: str | list[str],
    chase_bowlers: str | list[str],
    venue: str | None = "Lord's",
    date: str | None = None,
    sims: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """Mode 2: full match MC card with player + over summaries."""
    canonical = DEFAULTS["canonical"]
    aliases = pq.read_table(canonical / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(DEFAULTS["attributes"])
    first_lineup = _resolve_lineup(_parse_names(first_batters), aliases, attributes)
    chase_lineup = _resolve_lineup(_parse_names(chase_batters), aliases, attributes)
    first_attack = _resolve_attack(
        _parse_names(first_bowlers), aliases, attributes, canonical_dir=canonical
    )
    chase_attack = _resolve_attack(
        _parse_names(chase_bowlers), aliases, attributes, canonical_dir=canonical
    )

    def rates(group: str) -> InningsRateModel:
        return InningsRateModel(
            canonical_dir=canonical,
            effects_path=DEFAULTS["effects"],
            matchups_path=DEFAULTS["matchups"],
            attributes=attributes,
            venue=venue,
            innings_group=group,
            match_date=date,
            weather_dir=DEFAULTS["weather"] if date else None,
        )

    result = simulate_match(
        first_lineup=first_lineup,
        first_attack=first_attack,
        chase_lineup=chase_lineup,
        chase_attack=chase_attack,
        first_rates=rates("first_innings"),
        chase_rates=rates("chase"),
        n_sims=sims,
        seed=seed,
        chase_impacts=load_chase_impacts(DEFAULTS["chase_impacts"], canonical_dir=canonical),
        partnership_index=load_partnership_index(DEFAULTS["co_batters"]),
    )

    def slim_innings(block: dict[str, Any]) -> dict[str, Any]:
        return {
            "expected_runs": block.get("expected_runs"),
            "runs_p50": block.get("runs_p50"),
            "expected_wickets": block.get("expected_wickets"),
            "p_chase_win": block.get("p_chase_win"),
            "batters": [
                {
                    "player_name": b["player_name"],
                    "expected_runs": b.get("expected_runs"),
                    "expected_balls": b.get("expected_balls"),
                    "expected_fours": b.get("expected_fours"),
                    "expected_sixes": b.get("expected_sixes"),
                }
                for b in block.get("batters") or []
            ],
            "bowlers": [
                {
                    "player_name": b["player_name"],
                    "expected_wickets": b.get("expected_wickets"),
                    "expected_overs": b.get("expected_overs"),
                    "expected_economy": b.get("expected_economy"),
                }
                for b in block.get("bowlers") or []
            ],
            "overs": [
                {
                    "over": o.get("over"),
                    "expected_runs": o.get("expected_runs"),
                    "expected_wickets": o.get("expected_wickets"),
                }
                for o in (block.get("overs") or [])[:20]
            ],
        }

    return {
        "mode": "match_sim",
        "venue": venue,
        "match_date": date,
        "n_sims": sims,
        "match": result["match"],
        "first_innings": slim_innings(result["first_innings"]),
        "chase_innings": slim_innings(result["chase_innings"]),
    }


def run_player_dive(
    *,
    batter: str,
    bowlers: str | list[str],
    venue: str | None = None,
    max_balls: int = 120,
) -> dict[str, Any]:
    """Mode 3: batter vs named attack at a venue (Gayle-style deep dive)."""
    result = forecast_vs_attack(
        batter_query=batter,
        bowler_queries=_parse_names(bowlers),
        canonical_dir=DEFAULTS["canonical"],
        attributes_path=DEFAULTS["attributes"],
        effects_path=DEFAULTS["effects"],
        matchups_path=DEFAULTS["matchups"],
        venue=venue,
        max_balls=max_balls,
    )
    result["mode"] = "player_dive"
    return result
