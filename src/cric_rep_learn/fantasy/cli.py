"""CLI: simulate both tosses, score fantasy points, optimize constrained XI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.fantasy.embeddings_tiebreak import apply_embedding_tiebreak
from cric_rep_learn.fantasy.optimize import optimize_xi
from cric_rep_learn.fantasy.pool import pool_average_tosses
from cric_rep_learn.fantasy.roles import resolve_squad_roles
from cric_rep_learn.fantasy.scoring import load_scoring_weights
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
        default=None,
        help="Optional overrides Name:ROLE (WK/BAT/AR/BOWL). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--captain-candidates",
        type=int,
        default=5,
        help="Search C/VC among top-N fantasy scorers in the XI (default 5)",
    )
    parser.add_argument(
        "--max-credits",
        type=float,
        default=None,
        help="Optional credit cap (e.g. 100); uses role-based credit proxies",
    )
    parser.add_argument(
        "--embedding-tiebreak",
        action="store_true",
        help="Optional HB⊕embedding garnish when ranking near-tied pool points",
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
    parser.add_argument(
        "--scoring-weights",
        type=Path,
        default=Path("artifacts/fantasy/scoring_weights.json"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/embeddings-residual-mps-user/player_embeddings.parquet"),
    )
    args = parser.parse_args()

    load_scoring_weights(args.scoring_weights if args.scoring_weights.exists() else None)

    aliases = pq.read_table(args.canonical / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(args.attributes)
    overrides = _parse_roles(args.roles) if args.roles else {}

    a_bat = _resolve_lineup(_parse_names(args.team_a_batters), aliases, attributes)
    b_bat = _resolve_lineup(_parse_names(args.team_b_batters), aliases, attributes)
    a_bowl = _resolve_attack(
        _parse_names(args.team_a_bowlers), aliases, attributes, canonical_dir=args.canonical
    )
    b_bowl = _resolve_attack(
        _parse_names(args.team_b_bowlers), aliases, attributes, canonical_dir=args.canonical
    )

    attack_ids = {b.player_id for b in a_bowl + b_bowl}
    squad = a_bat + b_bat
    # Attach batting order within each team for role inference.
    for i, row in enumerate(a_bat):
        row["batting_order"] = i + 1
    for i, row in enumerate(b_bat):
        row["batting_order"] = i + 1
    role_info = resolve_squad_roles(
        squad,
        attributes=attributes,
        attack_ids=attack_ids,
        overrides=overrides,
    )
    roles = {pid: info["role"] for pid, info in role_info.items()}
    credits = {pid: info["credits"] for pid, info in role_info.items()}

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
        credits=credits,
    )
    if args.embedding_tiebreak:
        pool = apply_embedding_tiebreak(
            pool,
            effects_path=args.effects,
            embeddings_path=args.embeddings,
        )

    venue_profile = venue_scoring_profile(args.canonical, args.venue)
    constraints = {
        "max_from_team": args.max_from_team,
        **venue_profile.get("constraints", {}),
    }
    if args.max_credits is not None:
        constraints["max_credits"] = args.max_credits
    opt = optimize_xi(
        pool,
        constraints=constraints,
        target_roles=venue_profile.get("target_roles"),
        top_k=args.top_k,
        captain_candidates=args.captain_candidates,
    )
    result: dict[str, Any] = {
        "venue": args.venue,
        "match_date": args.date,
        "venue_profile": venue_profile,
        "role_resolution": {
            pid: {
                "player_name": next(
                    (p["player_name"] for p in squad if p["player_id"] == pid), pid
                ),
                **info,
            }
            for pid, info in role_info.items()
        },
        "toss_average": True,
        "toss_a_match": toss_a["match"],
        "toss_b_match": toss_b["match"],
        "player_pool": pool,
        "optimized": opt,
        "scoring": {
            "weights": str(args.scoring_weights),
            "bat": "1/run + boundary 4/6 + milestones + SR tilt",
            "bowl": f"{load_scoring_weights().get('BOWL_WICKET', 30)}/wicket + hauls + economy",
            "captain": f"C/VC search over top-{args.captain_candidates}",
            "balance": "soft penalty vs venue target WK-BAT-AR-BOWL mix",
            "credits": args.max_credits,
            "embedding_tiebreak": bool(args.embedding_tiebreak),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
