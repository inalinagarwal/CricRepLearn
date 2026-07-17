#!/usr/bin/env python3
"""CLI: score any T20 match using exported embeddings."""

import argparse
import json
from pathlib import Path

from match_predictor import MatchContext, MatchPredictor, PlayerRole, TeamSquad, load_squad_from_json


def print_results(results: dict):
    print(f"\n{'=' * 60}")
    print(f"{results['team_a']} vs {results['team_b']}")
    print(f"Venue: {results['venue']}  |  League: {results['league']}")
    print(f"{'=' * 60}")

    print("\n--- Batting (expected runs / ball vs opposition bowling) ---")
    for row in results["batting_scores"]:
        print(
            f"  {row['team']:12} {row['player']:22} "
            f"runs/ball={row['exp_runs_per_ball']:.3f}  "
            f"wicket%={row['wicket_prob_per_ball']*100:.1f}  "
            f"boundary%={row['boundary_prob_per_ball']*100:.1f}"
        )

    print("\n--- Bowling (vs opposition batting) ---")
    for row in results["bowling_scores"]:
        print(
            f"  {row['team']:12} {row['player']:22} "
            f"runs conceded/ball={row['exp_runs_conceded_per_ball']:.3f}  "
            f"wicket%={row['wicket_prob_per_ball']*100:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict T20 player matchups from learned embeddings"
    )
    parser.add_argument(
        "--match",
        type=Path,
        help="JSON file describing the fixture (see examples/ind_vs_eng.json)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write JSON results",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in India vs England example",
    )
    args = parser.parse_args()

    if args.demo:
        ctx = MatchContext(
            team_a=TeamSquad(
                name="India",
                players=[
                    PlayerRole("V Kohli", bats=True),
                    PlayerRole("RG Sharma", bats=True),
                    PlayerRole("HH Pandya", bats=True, bowls=True),
                    PlayerRole("RR Pant", bats=True),
                    PlayerRole("SA Yadav", bats=True),
                    PlayerRole("AR Patel", bats=True, bowls=True),
                    PlayerRole("JJ Bumrah", bats=False, bowls=True),
                    PlayerRole("Kuldeep Yadav", bats=False, bowls=True),
                    PlayerRole("YS Chahal", bats=False, bowls=True),
                ],
            ),
            team_b=TeamSquad(
                name="England",
                players=[
                    PlayerRole("JE Root", bats=True, bowls=True),
                    PlayerRole("BA Stokes", bats=True, bowls=True),
                    PlayerRole("JC Buttler", bats=True),
                    PlayerRole("PD Salt", bats=True),
                    PlayerRole("HC Brook", bats=True),
                    PlayerRole("LS Livingstone", bats=True, bowls=True),
                    PlayerRole("AU Rashid", bats=False, bowls=True),
                    PlayerRole("MA Wood", bats=False, bowls=True),
                    PlayerRole("CR Woakes", bats=False, bowls=True),
                ],
            ),
            venue="Eden Gardens",
            league="t20i",
        )
    elif args.match:
        ctx = load_squad_from_json(args.match)
    else:
        parser.error("Provide --match <file.json> or --demo")

    predictor = MatchPredictor(checkpoint_path=Path(args.checkpoint))
    results = predictor.predict_match(ctx)
    print_results(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
