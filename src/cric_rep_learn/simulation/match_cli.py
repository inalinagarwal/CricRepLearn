"""CLI: simulate a full T20 match (first innings → chase with sampled target)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cric_rep_learn.data.player_attributes import load_attributes_index
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


def _resolve_lineup(
    names: list[str],
    aliases,
    attributes: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
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
    names: list[str],
    aliases,
    attributes: dict[str, dict[str, Any]],
    *,
    canonical_dir: Path,
    max_overs: int = 4,
) -> list[BowlerSpell]:
    attack: list[BowlerSpell] = []
    for query in names:
        resolved = resolve_player(query, aliases, attributes=attributes)
        attack.append(
            BowlerSpell(
                player_id=resolved["player_id"],
                player_name=resolved["player_name"],
                max_overs=max_overs,
            )
        )
    profiles = load_bowler_phase_profiles(
        canonical_dir, [b.player_id for b in attack]
    )
    return attach_phase_profiles(attack, profiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-batters", required=True, help="Full XI batting first (11)")
    parser.add_argument("--first-bowlers", required=True, help="Bowling attack in first innings (5)")
    parser.add_argument("--chase-batters", required=True, help="Full XI chasing (11)")
    parser.add_argument("--chase-bowlers", required=True, help="Bowling attack in chase (5)")
    parser.add_argument("--venue", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("artifacts/weather"),
    )
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

    first_lineup = _resolve_lineup(_parse_names(args.first_batters), aliases, attributes)
    chase_lineup = _resolve_lineup(_parse_names(args.chase_batters), aliases, attributes)
    first_attack = _resolve_attack(
        _parse_names(args.first_bowlers), aliases, attributes, canonical_dir=args.canonical
    )
    chase_attack = _resolve_attack(
        _parse_names(args.chase_bowlers), aliases, attributes, canonical_dir=args.canonical
    )

    for label, lineup, attack in (
        ("first", first_lineup, first_attack),
        ("chase", chase_lineup, chase_attack),
    ):
        if len(lineup) < 11:
            print(
                f"note: {label} has {len(lineup)} batters; prefer full XI (11)",
                file=sys.stderr,
            )
        if len(attack) != 5:
            print(
                f"note: {label} has {len(attack)} bowlers; typical attack is 5",
                file=sys.stderr,
            )
        if sum(b.max_overs for b in attack) < 20:
            parser.error(f"{label} attack overs < 20")

    first_rates = InningsRateModel(
        canonical_dir=args.canonical,
        effects_path=args.effects,
        matchups_path=args.matchups,
        attributes=attributes,
        venue=args.venue,
        innings_group="first_innings",
        match_date=args.date,
        weather_dir=args.weather if args.date else None,
    )
    chase_rates = InningsRateModel(
        canonical_dir=args.canonical,
        effects_path=args.effects,
        matchups_path=args.matchups,
        attributes=attributes,
        venue=args.venue,
        innings_group="chase",
        match_date=args.date,
        weather_dir=args.weather if args.date else None,
    )
    chase_impacts = load_chase_impacts(args.chase_impacts, canonical_dir=args.canonical)
    partnership_index = load_partnership_index(args.co_batters)

    result = simulate_match(
        first_lineup=first_lineup,
        first_attack=first_attack,
        chase_lineup=chase_lineup,
        chase_attack=chase_attack,
        first_rates=first_rates,
        chase_rates=chase_rates,
        n_sims=args.sims,
        seed=args.seed,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )
    result["context"] = {
        "venue": args.venue,
        "match_date": args.date,
        "weather": first_rates.weather_features,
        "first_batters": first_lineup,
        "chase_batters": chase_lineup,
        "first_bowlers": [
            {"player_id": b.player_id, "player_name": b.player_name} for b in first_attack
        ],
        "chase_bowlers": [
            {"player_id": b.player_id, "player_name": b.player_name} for b in chase_attack
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
