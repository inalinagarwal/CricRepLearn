"""Three-mode entrypoint: Dream XI, match sim, player deep-dive."""

from __future__ import annotations

import argparse
import json
import sys


LORDS_IND_BAT = (
    "Rohit Sharma,Shubman Gill,Virat Kohli,Ishan Kishan,Shreyas Iyer,"
    "Washington Sundar,Shivam Dube,Axar Patel,Gurnoor Brar,Jasprit Bumrah,Prasidh Krishna"
)
LORDS_ENG_BAT = (
    "Jacob Bethell,Ben Duckett,Joe Root,Harry Brook,Jos Buttler,"
    "Sam Curran,Will Jacks,Gus Atkinson,Jofra Archer,Adil Rashid,Saqib Mahmood"
)
LORDS_IND_BOWL = "Jasprit Bumrah,Prasidh Krishna,Gurnoor Brar,Axar Patel,Washington Sundar"
LORDS_ENG_BOWL = "Jofra Archer,Gus Atkinson,Saqib Mahmood,Adil Rashid,Sam Curran"


def _print_dream(result: dict) -> None:
    xi = result["best_xi"]
    print(f"Venue: {result.get('venue')}  Date: {result.get('match_date')}  sims={result['n_sims']}")
    print(
        f"Toss A: {result['toss_a'].get('first_expected_runs', 0):.1f} → "
        f"chase {result['toss_a'].get('chase_expected_runs', 0):.1f}  "
        f"P(chase)={result['toss_a'].get('p_chase_win', 0):.2f}"
    )
    print(
        f"Toss B: {result['toss_b'].get('first_expected_runs', 0):.1f} → "
        f"chase {result['toss_b'].get('chase_expected_runs', 0):.1f}  "
        f"P(chase)={result['toss_b'].get('p_chase_win', 0):.2f}"
    )
    print(
        f"\nBEST XI  obj={xi['objective_score']:.1f}  "
        f"credits={xi.get('credits_used')}  {xi['roles']}  {xi['teams']}"
    )
    print(f"C:  {xi['captain']['player_name']}")
    print(f"VC: {xi['vice_captain']['player_name']}")
    print(f"{'Role':4} {'Tm':3} {'Player':22} {'Pts':>7}")
    for p in xi["players"]:
        print(f"{p['role']:4} {p['team']:3} {p['player_name']:22} {p['fantasy_points']:7.1f}")


def _print_match(result: dict) -> None:
    m = result["match"]
    print(f"Venue: {result.get('venue')}  Date: {result.get('match_date')}  sims={result['n_sims']}")
    print(
        f"First {m.get('first_expected_runs', 0):.1f}  "
        f"Chase {m.get('chase_expected_runs', 0):.1f}  "
        f"P(chase win)={m.get('p_chase_win', 0):.2f}"
    )
    for label, block in (
        ("FIRST INNINGS", result["first_innings"]),
        ("CHASE", result["chase_innings"]),
    ):
        print(f"\n=== {label} ===")
        print(f"{'Batter':22} {'Runs':>7} {'Balls':>7} {'4s':>5} {'6s':>5}")
        for b in block.get("batters") or []:
            print(
                f"{b['player_name']:22} {float(b.get('expected_runs') or 0):7.1f} "
                f"{float(b.get('expected_balls') or 0):7.1f} "
                f"{float(b.get('expected_fours') or 0):5.1f} "
                f"{float(b.get('expected_sixes') or 0):5.1f}"
            )
        print(f"{'Bowler':22} {'Wkts':>7} {'Overs':>7} {'Econ':>7}")
        for b in block.get("bowlers") or []:
            econ = b.get("expected_economy")
            econ_s = f"{float(econ):7.2f}" if econ is not None else f"{'—':>7}"
            print(
                f"{b['player_name']:22} {float(b.get('expected_wickets') or 0):7.2f} "
                f"{float(b.get('expected_overs') or 0):7.2f} {econ_s}"
            )
        overs = block.get("overs") or []
        if overs:
            print("Overs (runs / wickets):")
            chunks = []
            for o in overs:
                chunks.append(
                    f"{int(o.get('over', 0))+1}:"
                    f"{float(o.get('expected_runs') or 0):.1f}/"
                    f"{float(o.get('expected_wickets') or 0):.2f}"
                )
            print("  " + "  ".join(chunks))


def _print_dive(result: dict) -> None:
    print(f"Batter: {result.get('batter', {}).get('player_name') or result.get('batter')}")
    print(f"Venue: {result.get('venue')}")
    if result.get("warning"):
        print(f"Warning: {result['warning']}")
    print(f"Expected runs: {result.get('expected_runs')}")
    print(f"Expected balls: {result.get('expected_balls')}")
    lines = result.get("attack") or []
    if isinstance(lines, list) and lines:
        print(f"{'Bowler':22} {'Runs':>7} {'Balls':>7} {'Level':14}")
        for row in lines:
            name = row.get("bowler_name") or row.get("player_name") or "?"
            print(
                f"{name:22} {float(row.get('expected_runs') or 0):7.2f} "
                f"{float(row.get('expected_balls_faced') or 0):7.2f} "
                f"{str(row.get('level') or '')[:14]:14}"
            )


def _cmd_dream(args: argparse.Namespace) -> None:
    from cric_rep_learn.app.services import run_dream_xi

    result = run_dream_xi(
        team_a_batters=args.team_a_batters,
        team_b_batters=args.team_b_batters,
        team_a_bowlers=args.team_a_bowlers,
        team_b_bowlers=args.team_b_bowlers,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
        venue=args.venue,
        date=args.date,
        sims=args.sims,
        max_credits=args.max_credits,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_dream(result)


def _cmd_match(args: argparse.Namespace) -> None:
    from cric_rep_learn.app.services import run_match_sim

    result = run_match_sim(
        first_batters=args.first_batters,
        first_bowlers=args.first_bowlers,
        chase_batters=args.chase_batters,
        chase_bowlers=args.chase_bowlers,
        venue=args.venue,
        date=args.date,
        sims=args.sims,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_match(result)


def _cmd_dive(args: argparse.Namespace) -> None:
    from cric_rep_learn.app.services import run_player_dive

    result = run_player_dive(
        batter=args.batter,
        bowlers=args.bowlers,
        venue=args.venue,
        max_balls=args.max_balls,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_dive(result)


def _interactive() -> None:
    print(
        """
CricRepLearn — three modes
  1) Dream XI     — fantasy points MC → constrained XI
  2) Match sim    — expected scores, wickets, over-by-over
  3) Player dive  — one batter vs an attack at a venue
  q) Quit
""".strip()
    )
    choice = input("Mode [1/2/3/q]: ").strip().lower()
    if choice in {"q", "quit", ""}:
        return
    if choice == "1":
        sys.argv = [
            "cric",
            "dream-xi",
            "--team-a-batters",
            LORDS_IND_BAT,
            "--team-b-batters",
            LORDS_ENG_BAT,
            "--team-a-bowlers",
            LORDS_IND_BOWL,
            "--team-b-bowlers",
            LORDS_ENG_BOWL,
            "--venue",
            "Lord's",
            "--sims",
            "40",
        ]
        main()
        return
    if choice == "2":
        sys.argv = [
            "cric",
            "match-sim",
            "--first-batters",
            LORDS_IND_BAT,
            "--first-bowlers",
            LORDS_ENG_BOWL,
            "--chase-batters",
            LORDS_ENG_BAT,
            "--chase-bowlers",
            LORDS_IND_BOWL,
            "--venue",
            "Lord's",
            "--sims",
            "60",
        ]
        main()
        return
    if choice == "3":
        batter = input("Batter [Chris Gayle]: ").strip() or "Chris Gayle"
        bowlers = (
            input(
                "Bowlers comma-list [Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf]: "
            ).strip()
            or "Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf"
        )
        venue = input("Venue [Rawalpindi]: ").strip() or "Rawalpindi"
        sys.argv = [
            "cric",
            "player-dive",
            "--batter",
            batter,
            "--bowlers",
            bowlers,
            "--venue",
            venue,
        ]
        main()
        return
    print("Unknown choice.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cric",
        description="CricRepLearn product entry: Dream XI, match sim, player deep-dive",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    sub = parser.add_subparsers(dest="mode")

    dream = sub.add_parser("dream-xi", help="Optimize a constrained fantasy XI")
    dream.add_argument("--team-a-name", default="IND")
    dream.add_argument("--team-b-name", default="ENG")
    dream.add_argument("--team-a-batters", default=LORDS_IND_BAT)
    dream.add_argument("--team-b-batters", default=LORDS_ENG_BAT)
    dream.add_argument("--team-a-bowlers", default=LORDS_IND_BOWL)
    dream.add_argument("--team-b-bowlers", default=LORDS_ENG_BOWL)
    dream.add_argument("--venue", default="Lord's")
    dream.add_argument("--date", default=None)
    dream.add_argument("--sims", type=int, default=80)
    dream.add_argument("--max-credits", type=float, default=100.0)
    dream.set_defaults(func=_cmd_dream)

    match = sub.add_parser("match-sim", help="Simulate a full match card")
    match.add_argument("--first-batters", default=LORDS_IND_BAT)
    match.add_argument("--first-bowlers", default=LORDS_ENG_BOWL)
    match.add_argument("--chase-batters", default=LORDS_ENG_BAT)
    match.add_argument("--chase-bowlers", default=LORDS_IND_BOWL)
    match.add_argument("--venue", default="Lord's")
    match.add_argument("--date", default=None)
    match.add_argument("--sims", type=int, default=100)
    match.set_defaults(func=_cmd_match)

    dive = sub.add_parser("player-dive", help="Batter vs attack deep-dive")
    dive.add_argument("--batter", required=True)
    dive.add_argument("--bowlers", required=True)
    dive.add_argument("--venue", default=None)
    dive.add_argument("--max-balls", type=int, default=120)
    dive.set_defaults(func=_cmd_dive)

    ui = sub.add_parser("ui", help="Launch the local web UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.set_defaults(func=None)

    args = parser.parse_args()
    if args.mode is None:
        if sys.stdin.isatty():
            _interactive()
        else:
            parser.print_help()
        return
    if args.mode == "ui":
        from cric_rep_learn.app.web import serve

        serve(host=args.host, port=args.port)
        return
    args.func(args)


if __name__ == "__main__":
    main()
