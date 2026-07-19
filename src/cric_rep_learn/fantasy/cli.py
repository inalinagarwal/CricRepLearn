"""CLI: simulate both tosses, score fantasy points, optimize constrained XI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.fantasy.optimize import optimize_xi
from cric_rep_learn.fantasy.pool import pool_average_tosses
from cric_rep_learn.fantasy.venue_tilt import venue_scoring_profile
from cric_rep_learn.players.card import resolve_player
from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    attach_phase_profiles,
    load_bowler_phase_profiles,
)
from cric_rep_learn.simulation.chase import load_chase_impacts
from cric_rep_learn.simulation.match import simulate_match
from cric_rep_learn.simulation.partnership import load_partnership_index
from cric_rep_learn.simulation.priors import InningsRateModel


def _parse_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def _parse_roles(raw: str) -> dict[str, str]:
    """
    Format: 'Rohit Sharma:BAT,Jos Buttler:WK,...'
    Roles: WK, BAT, AR, BOWL.
    """
    out: dict[str, str] = {}
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"role entry must be Name:ROLE, got {part!r}")
        name, role = part.rsplit(":", 1)
        role = role.strip().upper()
        if role not in {"WK", "BAT", "AR", "BOWL"}:
            raise ValueError(f"invalid role {role!r} for {name.strip()!r}")
        out[name.strip()] = role
    return out


def _resolve_lineup(names, aliases, attributes):
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


def _resolve_attack(names, aliases, attributes, *, canonical_dir: Path):
    attack: list[BowlerSpell] = []
    for query in names:
        resolved = resolve_player(query, aliases, attributes=attributes)
        attack.append(
            BowlerSpell(
                player_id=resolved["player_id"],
                player_name=resolved["player_name"],
                max_overs=4,
            )
        )
    profiles = load_bowler_phase_profiles(
        canonical_dir, [b.player_id for b in attack]
    )
    return attach_phase_profiles(attack, profiles)


def _role_ids(
    role_by_name: dict[str, str],
    aliases,
    attributes,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, role in role_by_name.items():
        resolved = resolve_player(name, aliases, attributes=attributes)
        out[resolved["player_id"]] = role
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-a-name", default="IND")
    parser.add_argument("--team-b-name", default="ENG")
    parser.add_argument("--team-a-batters", required=True, help="Full XI team A (11)")
    parser.add_argument("--team-b-batters", required=True, help="Full XI team B (11)")
    parser.add_argument("--team-a-bowlers", required=True, help="5 bowlers from team A")
    parser.add_argument("--team-b-bowlers", required=True, help="5 bowlers from team B")
    parser.add_argument(
        "--roles",
        required=True,
        help="Comma list Name:ROLE for all 22 (WK/BAT/AR/BOWL)",
    )
    parser.add_argument("--venue", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-from-team", type=int, default=7)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("artifacts/player-attributes/player_attributes.parquet"),
    )
    parser.add_argument(
        "--effects",
        type=Path,
        default=Path("artifacts/player-effects/player_effects.parquet"),
    )
    parser.add_argument(
        "--matchups",
        type=Path,
        default=Path("artifacts/player-effects/batter_bowler_matchups.parquet"),
    )
    parser.add_argument("--weather", type=Path, default=Path("artifacts/weather"))
    parser.add_argument(
        "--chase-impacts",
        type=Path,
        default=Path("artifacts/baselines/chase_impacts.json"),
    )
    parser.add_argument(
        "--co-batters",
        type=Path,
        default=Path("artifacts/co-batters/co_batters.parquet"),
    )
    args = parser.parse_args()

    aliases = pq.read_table(args.canonical / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(args.attributes)
    role_by_name = _parse_roles(args.roles)
    roles = _role_ids(role_by_name, aliases, attributes)

    a_bat = _resolve_lineup(_parse_names(args.team_a_batters), aliases, attributes)
    b_bat = _resolve_lineup(_parse_names(args.team_b_batters), aliases, attributes)
    a_bowl = _resolve_attack(
        _parse_names(args.team_a_bowlers), aliases, attributes, canonical_dir=args.canonical
    )
    b_bowl = _resolve_attack(
        _parse_names(args.team_b_bowlers), aliases, attributes, canonical_dir=args.canonical
    )

    missing = []
    for row in a_bat + b_bat:
        if row["player_id"] not in roles:
            missing.append(row["query"])
    if missing:
        parser.error(f"--roles missing entries for: {missing}")

    def rates(group: str) -> InningsRateModel:
        return InningsRateModel(
            canonical_dir=args.canonical,
            effects_path=args.effects,
            matchups_path=args.matchups,
            attributes=attributes,
            venue=args.venue,
            innings_group=group,
            match_date=args.date,
            weather_dir=args.weather if args.date else None,
        )

    chase_impacts = load_chase_impacts(args.chase_impacts, canonical_dir=args.canonical)
    partnership_index = load_partnership_index(args.co_batters)

    # Toss A: team A bats first
    print("simulating toss A (team A bat first)...", file=sys.stderr)
    toss_a = simulate_match(
        first_lineup=a_bat,
        first_attack=b_bowl,
        chase_lineup=b_bat,
        chase_attack=a_bowl,
        first_rates=rates("first_innings"),
        chase_rates=rates("chase"),
        n_sims=args.sims,
        seed=args.seed,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )
    toss_a["context"] = {
        "first_batters": a_bat,
        "chase_batters": b_bat,
        "first_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in b_bowl],
        "chase_bowlers": [{"player_id": b.player_id, "player_name": b.player_name} for b in a_bowl],
    }

    # Toss B: team B bats first
    print("simulating toss B (team B bat first)...", file=sys.stderr)
    toss_b = simulate_match(
        first_lineup=b_bat,
        first_attack=a_bowl,
        chase_lineup=a_bat,
        chase_attack=b_bowl,
        first_rates=rates("first_innings"),
        chase_rates=rates("chase"),
        n_sims=args.sims,
        seed=args.seed + 17,
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
        [
            (toss_a, args.team_a_name, args.team_b_name),
            (toss_b, args.team_b_name, args.team_a_name),
        ],
        roles=roles,
    )
    venue_profile = venue_scoring_profile(args.canonical, args.venue)
    constraints = {
        "max_from_team": args.max_from_team,
        **venue_profile.get("constraints", {}),
    }
    opt = optimize_xi(
        pool,
        constraints=constraints,
        target_roles=venue_profile.get("target_roles"),
        top_k=args.top_k,
    )
    result: dict[str, Any] = {
        "venue": args.venue,
        "match_date": args.date,
        "venue_profile": venue_profile,
        "toss_average": True,
        "toss_a_match": toss_a["match"],
        "toss_b_match": toss_b["match"],
        "player_pool": pool,
        "optimized": opt,
        "scoring": {
            "bat": "1/run + milestones 30/50/100 + SR tilt",
            "bowl": "30/wicket + haul bonuses + economy vs 7.5",
            "captain": "×2 top scorer, ×1.5 second",
            "balance": "soft penalty vs venue target WK-BAT-AR-BOWL mix",
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
