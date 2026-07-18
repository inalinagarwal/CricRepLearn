"""Player-centric lookup: profile + hierarchical matchup fallbacks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cric_rep_learn.baselines.historical import (
    HistoricalBaseline,
    MatchContext,
    N_RUN_CLASSES,
)
from cric_rep_learn.data.bowling_style import nation_arm_pace_label, parse_bowling_style
from cric_rep_learn.data.player_attributes import load_attributes_index


def resolve_player(
    query: str,
    aliases: pd.DataFrame,
    *,
    attributes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("empty player query")
    attributes = attributes or {}

    by_id = aliases[aliases["player_id"] == query]
    if len(by_id):
        top = by_id.sort_values("match_count", ascending=False).iloc[0]
        return {
            "player_id": top["player_id"],
            "player_name": top["player_name"],
            "match_count": int(top["match_count"]),
            "query": query,
            "matched_by": "player_id",
        }

    # Prefer cricketdata full_name hits mapped to our highest-evidence alias.
    # This resolves "Rohit Sharma" → RG Sharma (740742ef), not the rare alias.
    full_name_ids: set[str] = set()
    lowered = query.lower()
    tokens = [token for token in re.split(r"\s+", lowered) if token]
    for player_id, attrs in attributes.items():
        full_name = str(attrs.get("full_name") or "").lower()
        if not full_name:
            continue
        if (
            lowered == full_name
            or lowered in full_name
            or (tokens and all(token in full_name for token in tokens))
        ):
            full_name_ids.add(player_id)

    exact = aliases[
        aliases["player_name"].str.fullmatch(re.escape(query), case=False, na=False)
    ]
    contains = aliases[
        aliases["player_name"].str.contains(re.escape(query), case=False, na=False)
    ]
    if full_name_ids:
        candidates = aliases[aliases["player_id"].isin(full_name_ids)]
        matched_by = "full_name"
    elif len(exact):
        candidates = exact
        matched_by = "name"
    else:
        candidates = contains
        matched_by = "name"
    if candidates.empty:
        raise ValueError(f"no player matched query {query!r}")

    ranked = (
        candidates.sort_values("match_count", ascending=False)
        .groupby("player_id", as_index=False)
        .first()
        .sort_values("match_count", ascending=False)
    )
    top = ranked.iloc[0]
    return {
        "player_id": top["player_id"],
        "player_name": top["player_name"],
        "match_count": int(top["match_count"]),
        "query": query,
        "matched_by": matched_by,
        "alternates": [
            {
                "player_id": row.player_id,
                "player_name": row.player_name,
                "match_count": int(row.match_count),
            }
            for row in ranked.iloc[1:5].itertuples(index=False)
        ],
    }


def _embedding_rows(embeddings_path: Path | None, player_id: str) -> list[dict[str, Any]]:
    if embeddings_path is None or not embeddings_path.exists():
        return []
    frame = pq.read_table(embeddings_path).to_pandas()
    rows = frame[frame["player_id"] == player_id]
    out = []
    for row in rows.to_dict(orient="records"):
        vector = row.get("embedding")
        out.append(
            {
                "role": row["role"],
                "role_index": int(row["role_index"]),
                "deliveries": int(row["deliveries"]),
                "matches": int(row["matches"]),
                "first_date": str(row["first_date"]),
                "last_date": str(row["last_date"]),
                "embedding_norm": float(row["embedding_norm"]),
                "embedding_dim": len(vector) if vector is not None else None,
            }
        )
    return out


def _profile(
    player: dict[str, Any],
    attributes: dict[str, dict[str, Any]],
    embeddings_path: Path | None,
) -> dict[str, Any]:
    attrs = attributes.get(player["player_id"], {})
    parsed = parse_bowling_style(attrs.get("bowling_style_raw"))
    return {
        **player,
        "full_name": attrs.get("full_name") or player["player_name"],
        "country": attrs.get("country"),
        "batting_hand": attrs.get("batting_hand", "unknown"),
        "batting_style_raw": attrs.get("batting_style_raw"),
        "bowling_style_raw": attrs.get("bowling_style_raw"),
        "bowling_arm": attrs.get("bowling_arm", parsed.bowling_arm),
        "pace_group": attrs.get("pace_group", parsed.pace_group),
        "bowling_family": attrs.get("bowling_family", parsed.bowling_family),
        "arm_pace_key": attrs.get("arm_pace_key", parsed.arm_pace_key),
        "playing_role": attrs.get("playing_role"),
        "attributes_matched": bool(attrs.get("matched")),
        "embeddings": _embedding_rows(embeddings_path, player["player_id"]),
    }


def _delivery_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "deliveries": 0,
            "matches": 0,
            "mean_batter_runs": None,
            "dismissal_rate": None,
            "bowler_wicket_rate": None,
            "run_distribution": None,
        }
    runs = frame["runs_batter"].clip(upper=N_RUN_CLASSES - 1).astype(int)
    counts = np.bincount(runs, minlength=N_RUN_CLASSES).astype(np.float64)
    probs = counts / counts.sum()
    return {
        "deliveries": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "mean_batter_runs": float(frame["runs_batter"].mean()),
        "dismissal_rate": float(frame["batter_dismissed"].mean()),
        "bowler_wicket_rate": float((frame["bowler_wicket_count"] > 0).mean()),
        "run_distribution": {str(index): float(prob) for index, prob in enumerate(probs)},
    }


def hierarchical_matchup(
    deliveries: pd.DataFrame,
    attributes: dict[str, dict[str, Any]],
    batter_id: str,
    bowler_id: str,
) -> list[dict[str, Any]]:
    bowler_attrs = attributes.get(
        bowler_id,
        {"country": "unknown", "bowling_arm": "unknown", "pace_group": "unknown"},
    )
    country = bowler_attrs.get("country") or "unknown"
    arm = bowler_attrs.get("bowling_arm") or "unknown"
    pace = bowler_attrs.get("pace_group") or "unknown"
    parsed = parse_bowling_style(bowler_attrs.get("bowling_style_raw"))

    batter_balls = deliveries[deliveries["batter_id"] == batter_id].copy()
    if attributes:
        attr_frame = (
            pd.DataFrame.from_dict(attributes, orient="index")
            .reset_index(names="bowler_id")[["bowler_id", "country", "bowling_arm", "pace_group"]]
            .fillna("unknown")
        )
        batter_balls = batter_balls.merge(attr_frame, on="bowler_id", how="left")
    else:
        for column in ("country", "bowling_arm", "pace_group"):
            batter_balls[column] = "unknown"
    for column in ("country", "bowling_arm", "pace_group"):
        batter_balls[column] = batter_balls[column].fillna("unknown")

    levels = [
        {
            "level": "matchup",
            "label": "vs this bowler",
            "mask": batter_balls["bowler_id"] == bowler_id,
        },
        {
            "level": "vs_nation_arm_pace",
            "label": f"vs {nation_arm_pace_label(country, parsed)}",
            "mask": (
                (batter_balls["country"] == country)
                & (batter_balls["bowling_arm"] == arm)
                & (batter_balls["pace_group"] == pace)
            ),
        },
        {
            "level": "vs_arm_pace",
            "label": f"vs {parsed.label}",
            "mask": (batter_balls["bowling_arm"] == arm)
            & (batter_balls["pace_group"] == pace),
        },
        {
            "level": "vs_pace",
            "label": f"vs {pace}",
            "mask": batter_balls["pace_group"] == pace,
        },
        {
            "level": "batter",
            "label": "batter overall",
            "mask": pd.Series(True, index=batter_balls.index),
        },
    ]

    return [
        {"level": level["level"], "label": level["label"], **_delivery_summary(batter_balls.loc[level["mask"]])}
        for level in levels
    ]


def build_player_card(
    *,
    batter_query: str,
    bowler_query: str | None,
    canonical_dir: Path,
    attributes_path: Path,
    embeddings_path: Path | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    aliases = pq.read_table(canonical_dir / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(attributes_path)
    batter = resolve_player(batter_query, aliases, attributes=attributes)
    card: dict[str, Any] = {
        "batter": _profile(batter, attributes, embeddings_path),
    }
    if not bowler_query:
        return card

    bowler = resolve_player(bowler_query, aliases, attributes=attributes)
    card["bowler"] = _profile(bowler, attributes, embeddings_path)

    deliveries = pq.read_table(
        canonical_dir / "deliveries.parquet",
        columns=[
            "match_id",
            "batter_id",
            "bowler_id",
            "phase",
            "runs_batter",
            "batter_dismissed",
            "bowler_wicket_count",
        ],
    ).to_pandas()
    if phase:
        deliveries = deliveries[deliveries["phase"] == phase]

    card["phase_filter"] = phase
    card["fallback_chain"] = hierarchical_matchup(
        deliveries,
        attributes,
        batter["player_id"],
        bowler["player_id"],
    )

    baseline = HistoricalBaseline(player_attributes=attributes)
    row = {
        "phase": phase or "middle",
        "is_super_over": False,
        "innings": 1,
        "wickets_before": 1,
        "batter_id": batter["player_id"],
        "bowler_id": bowler["player_id"],
    }
    context = MatchContext(gender="male", team_type="international", venue="UNKNOWN")
    keys = baseline._keys(row, context)
    card["hierarchy_keys"] = {
        "vs_pace": list(keys[4]),
        "vs_arm_pace": list(keys[5]),
        "vs_nation_arm_pace": list(keys[6]),
        "matchup": list(keys[7]),
    }
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batter", required=True, help="Name or Cricsheet player ID")
    parser.add_argument("--bowler", help="Optional bowler name or ID for matchup chain")
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("artifacts/player-attributes/player_attributes.parquet"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/embeddings-residual-mps-user/player_embeddings.parquet"),
    )
    parser.add_argument("--phase", choices=("powerplay", "middle", "death"))
    args = parser.parse_args()
    embeddings = args.embeddings if args.embeddings.exists() else None
    card = build_player_card(
        batter_query=args.batter,
        bowler_query=args.bowler,
        canonical_dir=args.canonical,
        attributes_path=args.attributes,
        embeddings_path=embeddings,
        phase=args.phase,
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
