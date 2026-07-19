"""Rank bowlers against a batter, with nation / venue filters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from cric_rep_learn.data.bowling_style import nation_arm_pace_label, parse_bowling_style
from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.players.card import resolve_player
from cric_rep_learn.players.player_effects import expected_runs_vs_bowler


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _clean_place(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        import math

        if math.isnan(value):
            return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


# Cities with no Cricsheet rows → nearby grounds that do exist in-corpus.
_VENUE_NEARBY: dict[str, list[str]] = {
    "islamabad": ["Rawalpindi", "Lahore", "Karachi"],
}


def resolve_venues(
    canonical_dir: Path,
    query: str,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Fuzzy-match a venue/city query against match metadata."""
    matches = pq.read_table(canonical_dir / "matches.parquet").to_pandas()
    needle = _normalize(query)
    tokens = [token for token in needle.split() if token]
    scored: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for row in matches[["venue", "city"]].drop_duplicates().to_dict(orient="records"):
        venue = _clean_place(row.get("venue"))
        city = _clean_place(row.get("city"))
        key = (venue, city)
        if key in seen or (venue is None and city is None):
            continue
        seen.add(key)
        hay = _normalize(f"{venue or ''} {city or ''}")
        if not hay:
            continue
        if needle == hay:
            score = 100
        elif needle and needle in hay:
            score = 80
        elif tokens and all(token in hay for token in tokens):
            score = 60
        elif tokens and any(token in hay for token in tokens):
            score = 30
        else:
            continue
        scored.append({"venue": venue, "city": city, "score": score})
    scored.sort(
        key=lambda item: (-item["score"], item["venue"] or "", item["city"] or "")
    )
    suggestions: list[str] = []
    if not scored:
        for key, nearby in _VENUE_NEARBY.items():
            if key in needle or needle in key:
                suggestions = nearby
                break
    return {
        "query": query,
        "matches": scored[:limit],
        "exact_or_substring": [row for row in scored if row["score"] >= 60][:limit],
        "suggestions": suggestions,
    }


def _load_effects(path: Path) -> dict[str, dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    return {row["player_id"]: row for row in frame.to_dict(orient="records")}


def _bowler_country(attrs: dict[str, Any] | None) -> str | None:
    if not attrs:
        return None
    country = attrs.get("country")
    if country is None:
        return None
    text = str(country).strip()
    return text or None


def _country_matches(country: str | None, query: str) -> bool:
    if not country:
        return False
    return _normalize(country) == _normalize(query) or _normalize(query) in _normalize(
        country
    )


def _venue_sql_clause(accepted: list[dict[str, Any]]) -> str:
    predicates = []
    for row in accepted:
        parts = []
        if row["venue"]:
            esc = str(row["venue"]).replace("'", "''")
            parts.append(f"m.venue = '{esc}'")
        if row["city"]:
            esc = str(row["city"]).replace("'", "''")
            parts.append(f"m.city = '{esc}'")
        if parts:
            predicates.append("(" + " AND ".join(parts) + ")")
    return "(" + " OR ".join(predicates) + ")" if predicates else "FALSE"


def build_filtered_matchups(
    *,
    canonical_dir: Path,
    batter_id: str,
    against_country: str | None = None,
    bowling_team: str | None = None,
    venue_query: str | None = None,
    venue_mode: str = "bowlers",
    attributes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Aggregate train faced balls for batter vs each bowler, optionally filtered.

    Filters:
      - against_country: keep bowlers whose attribute country matches
      - bowling_team: keep deliveries where bowling_team matches (substring ok)
      - venue_query + venue_mode:
          * bowlers (default): only keep bowlers who have bowled at that venue
            anywhere; score with the batter's overall matchup vs them
          * deliveries: only count balls the batter faced at that venue
    """
    attributes = attributes or {}
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")

    if venue_mode not in {"bowlers", "deliveries"}:
        raise ValueError("venue_mode must be 'bowlers' or 'deliveries'")

    venue_resolution = None
    venue_clause = "TRUE"
    accepted: list[dict[str, Any]] = []
    if venue_query:
        venue_resolution = resolve_venues(canonical_dir, venue_query)
        accepted = venue_resolution["exact_or_substring"] or venue_resolution["matches"][:3]
        if not accepted:
            suggestions = venue_resolution.get("suggestions") or []
            hint = (
                f"; try nearby grounds in this corpus: {', '.join(suggestions)}"
                if suggestions
                else ""
            )
            return {
                "matchups": {},
                "venue_resolution": venue_resolution,
                "balls_total": 0,
                "bowlers": 0,
                "venue_mode": venue_mode,
                "warning": f"no venue/city matched {venue_query!r}{hint}",
            }
        venue_clause = _venue_sql_clause(accepted)

    team_clause = "TRUE"
    if bowling_team:
        team = bowling_team.replace("'", "''")
        team_clause = f"lower(d.bowling_team) LIKE '%{team.lower()}%'"

    connection = duckdb.connect()
    try:
        eligible_bowlers: set[str] | None = None
        venues_for_pool: list[str] = []
        if venue_query and venue_mode == "bowlers":
            pool = connection.execute(
                f"""
                SELECT DISTINCT
                    d.bowler_id,
                    m.venue,
                    m.city
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{matches}') m USING (match_id)
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND {venue_clause}
                """
            ).fetchdf()
            eligible_bowlers = set(pool["bowler_id"].tolist())
            venues_for_pool = sorted(
                {
                    " / ".join(
                        part
                        for part in [_clean_place(r.get("venue")), _clean_place(r.get("city"))]
                        if part
                    )
                    for r in pool.to_dict(orient="records")
                }
                - {""}
            )

        # For delivery mode, restrict Gayle balls to the venue; for bowler mode,
        # use overall Gayle matchups then intersect the pool.
        delivery_venue_clause = venue_clause if venue_mode == "deliveries" else "TRUE"
        frame = connection.execute(
            f"""
            SELECT
                d.bowler_id,
                d.bowling_team,
                m.venue,
                m.city,
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
              AND {delivery_venue_clause}
              AND {team_clause}
            GROUP BY 1, 2, 3, 4
            """
        ).fetchdf()
    finally:
        connection.close()

    aggregated: dict[str, dict[str, float]] = {}
    teams_seen: dict[str, set[str]] = {}
    venues_seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        bowler_id = row["bowler_id"]
        if eligible_bowlers is not None and bowler_id not in eligible_bowlers:
            continue
        if against_country is not None:
            if not _country_matches(
                _bowler_country(attributes.get(bowler_id)), against_country
            ):
                continue
        slot = aggregated.setdefault(
            bowler_id, {"runs": 0.0, "balls": 0.0, "dismissals": 0.0}
        )
        slot["runs"] += float(row["runs"])
        slot["balls"] += float(row["balls"])
        slot["dismissals"] += float(row["dismissals"])
        teams_seen.setdefault(bowler_id, set()).add(str(row["bowling_team"]))
        label = " / ".join(
            part
            for part in [_clean_place(row.get("venue")), _clean_place(row.get("city"))]
            if part
        )
        if label:
            venues_seen.add(label)

    return {
        "matchups": aggregated,
        "teams_by_bowler": {key: sorted(values) for key, values in teams_seen.items()},
        "venues_used": sorted(venues_seen) if venue_mode == "deliveries" else venues_for_pool,
        "venue_resolution": venue_resolution,
        "venue_mode": venue_mode,
        "eligible_bowlers_at_venue": (
            len(eligible_bowlers) if eligible_bowlers is not None else None
        ),
        "balls_total": int(sum(item["balls"] for item in aggregated.values())),
        "bowlers": len(aggregated),
    }


def rank_bowlers_vs_batter(
    *,
    batter_query: str,
    canonical_dir: Path,
    attributes_path: Path,
    effects_path: Path,
    against_country: str | None = None,
    bowling_team: str | None = None,
    venue: str | None = None,
    venue_mode: str = "bowlers",
    min_balls: int = 20,
    opportunity_balls: float = 12.0,
    limit: int = 15,
) -> dict[str, Any]:
    aliases = pq.read_table(canonical_dir / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(attributes_path)
    batter = resolve_player(batter_query, aliases, attributes=attributes)
    effects = _load_effects(effects_path)
    smoothing = json.loads(
        (effects_path.parent / "smoothing.json").read_text(encoding="utf-8")
    )
    names = (
        aliases.sort_values("match_count", ascending=False)
        .groupby("player_id")
        .first()["player_name"]
        .to_dict()
    )

    filtered = build_filtered_matchups(
        canonical_dir=canonical_dir,
        batter_id=batter["player_id"],
        against_country=against_country,
        bowling_team=bowling_team,
        venue_query=venue,
        venue_mode=venue_mode,
        attributes=attributes,
    )
    if filtered.get("warning"):
        return {
            "batter": {
                "player_id": batter["player_id"],
                "player_name": batter["player_name"],
                "query": batter_query,
            },
            "filters": {
                "against_country": against_country,
                "bowling_team": bowling_team,
                "venue": venue,
                "venue_mode": venue_mode,
                "min_balls": min_balls,
            },
            "warning": filtered["warning"],
            "venue_resolution": filtered.get("venue_resolution"),
            "strongest": [],
            "weakest": [],
        }

    matchups = {
        (batter["player_id"], bowler_id): stats
        for bowler_id, stats in filtered["matchups"].items()
    }

    ranked: list[dict[str, Any]] = []
    for bowler_id, stats in filtered["matchups"].items():
        if stats["balls"] < min_balls:
            continue
        ba = dict(attributes.get(bowler_id, {}))
        parsed = parse_bowling_style(ba.get("bowling_style_raw"))
        if not ba.get("bowling_arm"):
            ba["bowling_arm"] = parsed.bowling_arm
        if not ba.get("pace_group"):
            ba["pace_group"] = parsed.pace_group
        forecast = expected_runs_vs_bowler(
            batter_id=batter["player_id"],
            bowler_id=bowler_id,
            balls=opportunity_balls,
            effects=effects,
            matchups=matchups,
            bowler_attrs=ba,
            global_sr=float(smoothing["global_sr"]),
            matchup_strength=float(smoothing["matchup_strength"]),
            archetype_strength=float(smoothing["archetype_strength"]),
        )
        ranked.append(
            {
                "bowler_id": bowler_id,
                "bowler_name": names.get(bowler_id, bowler_id),
                "country": ba.get("country"),
                "bowling_style_raw": ba.get("bowling_style_raw"),
                "label": nation_arm_pace_label(ba.get("country"), parsed),
                "balls": int(stats["balls"]),
                "runs": int(stats["runs"]),
                "dismissals": int(stats["dismissals"]),
                "raw_sr": float(stats["runs"] / stats["balls"]) if stats["balls"] else None,
                "expected_sr": forecast["expected_sr"],
                "expected_runs": forecast["expected_runs"],
                "level": forecast["level"],
                "parent_sr": forecast["parent_sr"],
                "bowling_teams_seen": filtered["teams_by_bowler"].get(bowler_id, []),
            }
        )

    strongest = sorted(
        ranked, key=lambda row: (row["expected_sr"], -row["dismissals"], row["balls"])
    )
    weakest = sorted(
        ranked,
        key=lambda row: (-row["expected_sr"], -(row["raw_sr"] or 0.0), -row["balls"]),
    )
    for index, row in enumerate(strongest, start=1):
        row["rank_strongest"] = index
    for index, row in enumerate(weakest, start=1):
        row["rank_weakest"] = index

    return {
        "batter": {
            "player_id": batter["player_id"],
            "player_name": batter["player_name"],
            "query": batter_query,
        },
        "filters": {
            "against_country": against_country,
            "bowling_team": bowling_team,
            "venue": venue,
            "venue_mode": venue_mode,
            "min_balls": min_balls,
            "opportunity_balls": opportunity_balls,
        },
        "evidence": {
            "bowlers_before_min_balls": filtered["bowlers"],
            "bowlers_ranked": len(ranked),
            "balls_total": filtered["balls_total"],
            "venues_used": filtered.get("venues_used", []),
            "eligible_bowlers_at_venue": filtered.get("eligible_bowlers_at_venue"),
        },
        "venue_resolution": filtered.get("venue_resolution"),
        "strongest": strongest[:limit],
        "weakest": weakest[:limit],
        "top": {
            "strongest": strongest[0] if strongest else None,
            "weakest": weakest[0] if weakest else None,
        },
        "method": (
            "filtered train matchups → HB shrink toward batter arm/pace prior; "
            "venue_mode=bowlers keeps bowlers who appeared at venue; "
            "venue_mode=deliveries counts only balls at venue; "
            "strongest = lowest expected SR, weakest = highest"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batter", required=True)
    parser.add_argument(
        "--against",
        dest="against_country",
        default=None,
        help="Filter bowlers by attribute country, e.g. Pakistan / India",
    )
    parser.add_argument(
        "--bowling-team",
        default=None,
        help="Filter deliveries by bowling_team substring, e.g. Pakistan",
    )
    parser.add_argument(
        "--venue",
        default=None,
        help="Fuzzy venue/city match, e.g. Islamabad / Lahore / Rawalpindi",
    )
    parser.add_argument(
        "--venue-mode",
        choices=["bowlers", "deliveries"],
        default="bowlers",
        help=(
            "bowlers: restrict to bowlers who have bowled at venue (default); "
            "deliveries: only count batter balls faced at venue"
        ),
    )
    parser.add_argument("--min-balls", type=int, default=20)
    parser.add_argument("--balls", type=float, default=12.0, help="Opportunity size")
    parser.add_argument("--limit", type=int, default=12)
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
    args = parser.parse_args()
    result = rank_bowlers_vs_batter(
        batter_query=args.batter,
        canonical_dir=args.canonical,
        attributes_path=args.attributes,
        effects_path=args.effects,
        against_country=args.against_country,
        bowling_team=args.bowling_team,
        venue=args.venue,
        venue_mode=args.venue_mode,
        min_balls=args.min_balls,
        opportunity_balls=args.balls,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
