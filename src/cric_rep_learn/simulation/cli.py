"""CLI: simulate a T20 innings for a batting order vs a bowling attack."""

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
from cric_rep_learn.simulation.innings import simulate_innings
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
    parser.add_argument(
        "--batters",
        required=True,
        help=(
            "Full batting order (typically 11). Lower-order players contribute "
            "fewer expected runs but still add to the team total."
        ),
    )
    parser.add_argument(
        "--bowlers",
        required=True,
        help=(
            "Bowling attack (typically 5 × max 4 overs = 20). "
            'e.g. "JJ Bumrah,B Kumar,R Ashwin,YS Chahal,HH Pandya"'
        ),
    )
    parser.add_argument(
        "--venue",
        default=None,
        help="Venue/city for priors (sparse grounds expand to similar-condition cluster)",
    )
    parser.add_argument(
        "--innings",
        choices=["first", "chase"],
        default="first",
        help="Batting first or chasing (default first)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help=(
            "Chase target (runs to win). Enables RRR×wickets pressure and "
            "empirical chase win-confidence; innings stops when reached."
        ),
    )
    parser.add_argument(
        "--chase-impacts",
        type=Path,
        default=Path("artifacts/baselines/chase_impacts.json"),
        help="Chase pressure table (built from train if missing)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Match date YYYY-MM-DD for daily weather priors (Open-Meteo archive)",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("artifacts/weather"),
        help="Weather artifact directory from cric-build-weather",
    )
    parser.add_argument("--sims", type=int, default=400)
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
        "--co-batters",
        type=Path,
        default=Path("artifacts/co-batters/co_batters.parquet"),
        help="Co-batter partnership graph for familiarity tilts",
    )
    args = parser.parse_args()

    aliases = pq.read_table(args.canonical / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(args.attributes)
    lineup = _resolve_lineup(_parse_names(args.batters), aliases, attributes)
    attack = _resolve_attack(
        _parse_names(args.bowlers), aliases, attributes, canonical_dir=args.canonical
    )
    if len(lineup) < 11:
        print(
            f"note: {len(lineup)} batters supplied; prefer the full XI (11) "
            "so lower-order contributions enter the team total",
            file=sys.stderr,
        )
    if len(attack) != 5:
        print(
            f"note: {len(attack)} bowlers supplied; typical T20 attack is 5 "
            "(4 overs each)",
            file=sys.stderr,
        )
    if sum(b.max_overs for b in attack) < 20:
        parser.error(
            f"attack max overs sum to {sum(b.max_overs for b in attack)}; need ≥20"
        )
    if args.target is not None and args.innings != "chase":
        parser.error("--target requires --innings chase")
    innings_group = "first_innings" if args.innings == "first" else "chase"
    rates = InningsRateModel(
        canonical_dir=args.canonical,
        effects_path=args.effects,
        matchups_path=args.matchups,
        attributes=attributes,
        venue=args.venue,
        innings_group=innings_group,
        match_date=args.date,
        weather_dir=args.weather if args.date else None,
    )
    chase_impacts = None
    if args.target is not None:
        chase_impacts = load_chase_impacts(
            args.chase_impacts, canonical_dir=args.canonical
        )
    partnership_index = load_partnership_index(args.co_batters)
    result = simulate_innings(
        lineup=lineup,
        attack=attack,
        rates=rates,
        n_sims=args.sims,
        seed=args.seed,
        target=args.target,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )
    result["lineup"] = lineup
    result["context"] = {
        "venue": args.venue,
        "venue_scope": rates.venue_scope,
        "innings": args.innings,
        "innings_group": innings_group,
        "target": args.target,
        "match_date": args.date,
        "weather": rates.weather_features,
        "weather_notes": rates.weather_notes,
    }
    if rates.venue_resolution:
        result["context"]["venue_cluster"] = rates.venue_resolution.get("cluster")
        result["context"]["venue_note"] = rates.venue_resolution.get("note")
    result["attack"] = [
        {
            "player_id": b.player_id,
            "player_name": b.player_name,
            "max_overs": b.max_overs,
            "phase_scores": b.phase_scores,
            "phase_evidence": {
                phase: {
                    "balls": stats.get("balls"),
                    "sr": stats.get("sr"),
                    "wicket_rate": stats.get("wicket_rate"),
                    "score": stats.get("score"),
                }
                for phase, stats in b.phase_evidence.items()
            },
        }
        for b in attack
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
