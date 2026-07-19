"""Forecast expected runs for a batter vs a named bowling attack at a venue.

Balls faced are not guessed up-front. They come from the batter's dismissal
hazard vs each bowler: if he tends to get out early to someone, fewer balls
(and runs) accrue against the rest of the attack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from cric_rep_learn.data.bowling_style import nation_arm_pace_label, parse_bowling_style
from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.players.card import resolve_player
from cric_rep_learn.players.player_effects import _posterior_rate, expected_runs_vs_bowler
from cric_rep_learn.players.venue_similarity import resolve_venues, venue_sql_clause


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _load_effects(path: Path) -> dict[str, dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    return {row["player_id"]: row for row in frame.to_dict(orient="records")}


def _load_overall_matchups(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    frame = pq.read_table(path).to_pandas()
    return {
        (row["batter_id"], row["bowler_id"]): {
            "runs": float(row["runs"]),
            "balls": float(row["balls"]),
            "dismissals": float(row["dismissals"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _parse_bowlers(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _parse_floats(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _matchups_at_venues(
    *,
    canonical_dir: Path,
    batter_id: str,
    bowler_ids: list[str],
    venue_clause: str,
) -> dict[str, dict[str, float]]:
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    if not bowler_ids:
        return {}
    id_list = ", ".join(f"'{bid}'" for bid in bowler_ids)
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            SELECT
                d.bowler_id,
                SUM(d.runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{matches}') m USING (match_id)
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND d.batter_id = '{batter_id}'
              AND d.bowler_id IN ({id_list})
              AND {venue_clause}
            GROUP BY 1
            """
        ).fetchdf()
    finally:
        connection.close()
    return {
        row["bowler_id"]: {
            "runs": float(row["runs"]),
            "balls": float(row["balls"]),
            "dismissals": float(row["dismissals"]),
        }
        for row in frame.to_dict(orient="records")
    }


def expected_dismissal_rate_vs_bowler(
    *,
    batter_id: str,
    bowler_id: str,
    effects: dict[str, dict[str, Any]],
    matchups: dict[tuple[str, str], dict[str, float]],
    global_dismiss: float,
    matchup_strength: float,
) -> dict[str, Any]:
    """Shrink matchup dismissal rate toward the batter's overall dismiss prior."""
    effect = effects.get(batter_id)
    parent = float(effect["dismissal_rate"]) if effect is not None else global_dismiss
    matchup = matchups.get((batter_id, bowler_id))
    if matchup and matchup["balls"] > 0:
        rate = _posterior_rate(
            matchup["dismissals"], matchup["balls"], parent, matchup_strength
        )
        level = "matchup→player_dismiss"
        evidence = int(matchup["balls"])
    else:
        rate = parent
        level = "player_dismiss" if effect is not None else "global_dismiss"
        evidence = 0
    # Keep hazards in a sane band for geometric survival.
    rate = float(min(max(rate, 1e-4), 0.35))
    return {
        "dismissal_rate": rate,
        "parent_dismissal_rate": float(parent),
        "matchup_balls": evidence,
        "level": level,
    }


def simulate_expected_innings(
    *,
    rates: list[dict[str, float]],
    max_balls: int,
    bowl_weights: list[float] | None = None,
) -> dict[str, Any]:
    """
    Expected balls/runs until dismissal (or max_balls), rotating through bowlers.

    ``bowl_weights`` are relative shares of the attack (default equal). The
    bowler for ball t is drawn by cycling a deterministic weighted schedule:
    expand weights into a short repeating block of indices.
    """
    n = len(rates)
    if n == 0:
        return {
            "expected_balls": 0.0,
            "expected_runs": 0.0,
            "p_still_in_at_max": 1.0,
            "per_bowler_balls": [],
            "per_bowler_runs": [],
            "schedule": [],
        }

    if bowl_weights is None:
        weights = [1.0] * n
    else:
        if len(bowl_weights) != n:
            raise ValueError("bowl_weights length must match bowlers")
        weights = [max(float(w), 0.0) for w in bowl_weights]
        if sum(weights) <= 0:
            weights = [1.0] * n

    # Build a repeating schedule proportional to weights (approx overs share).
    scale = 6  # one "over" unit
    schedule: list[int] = []
    for index, weight in enumerate(weights):
        slots = max(1, int(round(weight / sum(weights) * n * scale)))
        schedule.extend([index] * slots)
    if not schedule:
        schedule = list(range(n))

    survival = 1.0
    exp_balls = 0.0
    exp_runs = 0.0
    per_balls = [0.0] * n
    per_runs = [0.0] * n

    for t in range(max_balls):
        bowler_index = schedule[t % len(schedule)]
        sr = float(rates[bowler_index]["expected_sr"])
        p_out = float(rates[bowler_index]["dismissal_rate"])
        per_balls[bowler_index] += survival
        per_runs[bowler_index] += survival * sr
        exp_balls += survival
        exp_runs += survival * sr
        survival *= 1.0 - p_out
        if survival < 1e-12:
            break

    return {
        "expected_balls": float(exp_balls),
        "expected_runs": float(exp_runs),
        "p_still_in_at_max": float(survival),
        "per_bowler_balls": per_balls,
        "per_bowler_runs": per_runs,
        "schedule_block": schedule,
    }


def forecast_vs_attack(
    *,
    batter_query: str,
    bowler_queries: list[str],
    canonical_dir: Path,
    attributes_path: Path,
    effects_path: Path,
    matchups_path: Path,
    venue: str | None = None,
    max_balls: int = 120,
    bowl_weights: list[float] | None = None,
    sparse_balls: float = 24.0,
    venue_matchup_strength: float = 30.0,
) -> dict[str, Any]:
    """
    Expected runs scored by the batter vs a named attack.

    Opportunity is endogenous: balls faced follow survival under each bowler's
    dismissal hazard (matchup-shrunk), capped at ``max_balls``.
    """
    aliases = pq.read_table(canonical_dir / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(attributes_path)
    effects = _load_effects(effects_path)
    overall_matchups = _load_overall_matchups(matchups_path)
    smoothing = json.loads(
        (effects_path.parent / "smoothing.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (effects_path.parent / "metadata.json").read_text(encoding="utf-8")
    )
    global_dismiss = float(metadata.get("global_dismissal_rate", 0.05))

    batter = resolve_player(batter_query, aliases, attributes=attributes)
    bowlers = [
        resolve_player(query, aliases, attributes=attributes) for query in bowler_queries
    ]

    venue_resolution = None
    venue_matchups: dict[str, dict[str, float]] = {}
    venue_scope = "none"
    if venue:
        venue_resolution = resolve_venues(canonical_dir, venue, include_similar=False)
        primary = venue_resolution["primary"]
        if not primary:
            return {
                "batter": {
                    "player_id": batter["player_id"],
                    "player_name": batter["player_name"],
                    "query": batter_query,
                },
                "venue": venue,
                "warning": (
                    f"no venue/city matched {venue!r}; "
                    f"suggestions={venue_resolution.get('suggestions')}"
                ),
                "expected_runs": None,
                "venue_resolution": venue_resolution,
            }

        bowler_ids = [row["player_id"] for row in bowlers]
        venue_matchups = _matchups_at_venues(
            canonical_dir=canonical_dir,
            batter_id=batter["player_id"],
            bowler_ids=bowler_ids,
            venue_clause=venue_sql_clause(primary),
        )
        primary_balls = sum(item["balls"] for item in venue_matchups.values())
        venue_scope = "primary"
        if primary_balls < sparse_balls:
            expanded = resolve_venues(canonical_dir, venue, include_similar=True)
            venue_matchups = _matchups_at_venues(
                canonical_dir=canonical_dir,
                batter_id=batter["player_id"],
                bowler_ids=bowler_ids,
                venue_clause=venue_sql_clause(expanded["accepted"]),
            )
            venue_scope = "primary+similar_conditions"
            venue_resolution = expanded

    rates: list[dict[str, float]] = []
    lines: list[dict[str, Any]] = []
    for bowler, query in zip(bowlers, bowler_queries, strict=True):
        ba = dict(attributes.get(bowler["player_id"], {}))
        parsed = parse_bowling_style(ba.get("bowling_style_raw"))
        if not ba.get("bowling_arm"):
            ba["bowling_arm"] = parsed.bowling_arm
        if not ba.get("pace_group"):
            ba["pace_group"] = parsed.pace_group

        run_prior = expected_runs_vs_bowler(
            batter_id=batter["player_id"],
            bowler_id=bowler["player_id"],
            balls=1.0,
            effects=effects,
            matchups=overall_matchups,
            bowler_attrs=ba,
            global_sr=float(smoothing["global_sr"]),
            matchup_strength=float(smoothing["matchup_strength"]),
            archetype_strength=float(smoothing["archetype_strength"]),
        )
        dismiss_prior = expected_dismissal_rate_vs_bowler(
            batter_id=batter["player_id"],
            bowler_id=bowler["player_id"],
            effects=effects,
            matchups=overall_matchups,
            global_dismiss=global_dismiss,
            matchup_strength=float(smoothing["matchup_strength"]),
        )

        sr = float(run_prior["expected_sr"])
        p_out = float(dismiss_prior["dismissal_rate"])
        level = f"{run_prior['level']}|{dismiss_prior['level']}"
        venue_balls = 0
        venue_raw_sr = None
        venue_raw_dismiss = None
        if venue and bowler["player_id"] in venue_matchups:
            vm = venue_matchups[bowler["player_id"]]
            venue_balls = int(vm["balls"])
            if vm["balls"] > 0:
                venue_raw_sr = float(vm["runs"] / vm["balls"])
                venue_raw_dismiss = float(vm["dismissals"] / vm["balls"])
                sr = _posterior_rate(
                    vm["runs"], vm["balls"], sr, venue_matchup_strength
                )
                p_out = _posterior_rate(
                    vm["dismissals"], vm["balls"], p_out, venue_matchup_strength
                )
                p_out = float(min(max(p_out, 1e-4), 0.35))
                level = f"venue({venue_scope})→{level}"

        rates.append({"expected_sr": sr, "dismissal_rate": p_out})
        lines.append(
            {
                "bowler_id": bowler["player_id"],
                "bowler_name": bowler["player_name"],
                "query": query,
                "country": ba.get("country"),
                "bowling_style_raw": ba.get("bowling_style_raw"),
                "label": nation_arm_pace_label(ba.get("country"), parsed),
                "overall_matchup_balls": run_prior["matchup_balls"],
                "expected_sr": sr,
                "dismissal_rate": p_out,
                "balls_per_dismissal": (1.0 / p_out) if p_out > 0 else None,
                "venue_matchup_balls": venue_balls,
                "venue_raw_sr": venue_raw_sr,
                "venue_raw_dismissal_rate": venue_raw_dismiss,
                "level": level,
            }
        )

    sim = simulate_expected_innings(
        rates=rates, max_balls=max_balls, bowl_weights=bowl_weights
    )
    for index, line in enumerate(lines):
        line["expected_balls_faced"] = sim["per_bowler_balls"][index]
        line["expected_runs"] = sim["per_bowler_runs"][index]

    return {
        "batter": {
            "player_id": batter["player_id"],
            "player_name": batter["player_name"],
            "query": batter_query,
        },
        "venue": venue,
        "venue_scope": venue_scope,
        "venue_resolution": {
            "cluster": (venue_resolution or {}).get("cluster"),
            "primary": (venue_resolution or {}).get("primary", [])[:5],
            "similar_used": [
                row
                for row in (venue_resolution or {}).get("accepted", [])
                if row.get("scope") == "similar_conditions"
            ][:8],
            "note": (venue_resolution or {}).get("note"),
        }
        if venue
        else None,
        "max_balls": max_balls,
        "bowl_weights": bowl_weights or [1.0] * len(bowlers),
        "expected_balls": sim["expected_balls"],
        "expected_runs": sim["expected_runs"],
        "expected_sr": (
            sim["expected_runs"] / sim["expected_balls"]
            if sim["expected_balls"] > 0
            else None
        ),
        "p_still_batting_at_max": sim["p_still_in_at_max"],
        "attack": lines,
        "method": (
            "expected runs scored by the batter; balls faced are endogenous from "
            "per-bowler dismissal hazards under a rotating attack schedule "
            f"(cap={max_balls}); venue(+similar) updates SR and dismiss rates when available"
        ),
        "sparse_balls_threshold": sparse_balls if venue else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batter", required=True)
    parser.add_argument(
        "--bowlers",
        required=True,
        help='Comma-separated bowlers, e.g. "Hafeez,Wahab Riaz,Shaheen Afridi,Shadab,Haris Rauf"',
    )
    parser.add_argument("--venue", default=None)
    parser.add_argument(
        "--max-balls",
        type=int,
        default=120,
        help="Ceiling on balls faced if not dismissed (default 120)",
    )
    parser.add_argument(
        "--bowl-weights",
        default=None,
        help="Relative overs/share per bowler, e.g. 4,4,4,2,2 (default equal)",
    )
    parser.add_argument(
        "--sparse-balls",
        type=float,
        default=24.0,
        help="If venue matchup balls vs attack below this, expand to similar venues",
    )
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
    args = parser.parse_args()
    result = forecast_vs_attack(
        batter_query=args.batter,
        bowler_queries=_parse_bowlers(args.bowlers),
        canonical_dir=args.canonical,
        attributes_path=args.attributes,
        effects_path=args.effects,
        matchups_path=args.matchups,
        venue=args.venue,
        max_balls=args.max_balls,
        bowl_weights=_parse_floats(args.bowl_weights),
        sparse_balls=args.sparse_balls,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
