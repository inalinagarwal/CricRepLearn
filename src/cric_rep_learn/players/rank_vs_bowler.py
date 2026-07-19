"""Rank a batter and their co-batters against a specific bowler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from cric_rep_learn.data.bowling_style import nation_arm_pace_label, parse_bowling_style
from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.players.card import resolve_player
from cric_rep_learn.players.partnerships import top_partners
from cric_rep_learn.players.player_effects import expected_runs_vs_bowler


def _load_effects(path: Path) -> dict[str, dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    return {row["player_id"]: row for row in frame.to_dict(orient="records")}


def _load_matchups(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    frame = pq.read_table(path).to_pandas()
    return {
        (row["batter_id"], row["bowler_id"]): {
            "runs": float(row["runs"]),
            "balls": float(row["balls"]),
            "dismissals": float(row["dismissals"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _embedding_lookup(
    embeddings_path: Path | None, player_ids: list[str]
) -> dict[str, np.ndarray]:
    if embeddings_path is None or not embeddings_path.exists():
        return {}
    frame = pq.read_table(embeddings_path).to_pandas()
    batting = frame[frame["role"] == "batting"]
    wanted = set(player_ids)
    out: dict[str, np.ndarray] = {}
    for row in batting.to_dict(orient="records"):
        if row["player_id"] in wanted:
            out[row["player_id"]] = np.asarray(row["embedding"], dtype=np.float64)
    return out


def _combine_representation(
    effect_vector: list[float] | None,
    embedding: np.ndarray | None,
    *,
    hb_weight: float = 1.0,
    embedding_weight: float = 0.25,
) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    if effect_vector is not None:
        hb = np.asarray(effect_vector, dtype=np.float64)
        hb = hb / (np.linalg.norm(hb) + 1e-8)
        parts.append(hb_weight * hb)
    if embedding is not None:
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        parts.append(embedding_weight * emb)
    if not parts:
        return None
    return np.concatenate(parts)


def rank_vs_bowler(
    *,
    batter_query: str,
    bowler_query: str,
    canonical_dir: Path,
    attributes_path: Path,
    effects_path: Path,
    matchups_path: Path,
    co_batters_path: Path,
    embeddings_path: Path | None = None,
    balls: float = 12.0,
    peer_limit: int = 8,
) -> dict[str, Any]:
    aliases = pq.read_table(canonical_dir / "player_aliases.parquet").to_pandas()
    attributes = load_attributes_index(attributes_path)
    batter = resolve_player(batter_query, aliases, attributes=attributes)
    bowler = resolve_player(bowler_query, aliases, attributes=attributes)
    effects = _load_effects(effects_path)
    matchups = _load_matchups(matchups_path)
    smoothing = json.loads(
        (effects_path.parent / "smoothing.json").read_text(encoding="utf-8")
    )
    global_sr = float(smoothing["global_sr"])

    partners = top_partners(co_batters_path, batter["player_id"], limit=peer_limit)
    peer_ids = [batter["player_id"]] + [row["partner_id"] for row in partners]
    name_map = (
        aliases.sort_values("match_count", ascending=False)
        .groupby("player_id")
        .first()["player_name"]
        .to_dict()
    )
    partner_balls = {row["partner_id"]: int(row["balls_together"]) for row in partners}

    bowler_attrs = attributes.get(bowler["player_id"], {})
    parsed = parse_bowling_style(bowler_attrs.get("bowling_style_raw"))
    embeddings = _embedding_lookup(embeddings_path, peer_ids)
    anchor_repr = _combine_representation(
        effects.get(batter["player_id"], {}).get("effect_vector"),
        embeddings.get(batter["player_id"]),
    )

    ranked: list[dict[str, Any]] = []
    for player_id in peer_ids:
        forecast = expected_runs_vs_bowler(
            batter_id=player_id,
            bowler_id=bowler["player_id"],
            balls=balls,
            effects=effects,
            matchups=matchups,
            bowler_attrs=bowler_attrs,
            global_sr=global_sr,
            matchup_strength=float(smoothing["matchup_strength"]),
            archetype_strength=float(smoothing["archetype_strength"]),
        )
        peer_repr = _combine_representation(
            effects.get(player_id, {}).get("effect_vector"),
            embeddings.get(player_id),
        )
        similarity = None
        if anchor_repr is not None and peer_repr is not None:
            similarity = float(
                np.dot(anchor_repr, peer_repr)
                / (np.linalg.norm(anchor_repr) * np.linalg.norm(peer_repr) + 1e-8)
            )
        effect = effects.get(player_id, {})
        ranked.append(
            {
                "player_id": player_id,
                "player_name": effect.get("player_name")
                or name_map.get(player_id, player_id),
                "is_query_batter": player_id == batter["player_id"],
                "balls_batted_with_query": (
                    None
                    if player_id == batter["player_id"]
                    else partner_balls.get(player_id, 0)
                ),
                "hb_balls": int(effect.get("balls", 0)),
                "player_sr_effect": effect.get("player_sr_effect"),
                **forecast,
                "representation_similarity_to_query": similarity,
            }
        )

    ranked.sort(
        key=lambda row: (
            -float(row["expected_runs"]),
            -(row["representation_similarity_to_query"] or -1.0),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    return {
        "batter": {
            "player_id": batter["player_id"],
            "player_name": batter["player_name"],
            "query": batter_query,
        },
        "bowler": {
            "player_id": bowler["player_id"],
            "player_name": bowler["player_name"],
            "query": bowler_query,
            "country": bowler_attrs.get("country"),
            "bowling_style_raw": bowler_attrs.get("bowling_style_raw"),
            "label": nation_arm_pace_label(bowler_attrs.get("country"), parsed),
        },
        "opportunity_balls": balls,
        "peer_definition": "top non-striker partnership partners on train faced balls",
        "ranking": ranked,
        "method": (
            "hierarchical Bayes expected runs vs bowler "
            "(matchup→arm/pace→pace→player), "
            "peers = co-batters; representation = HB effects ⊕ batting embedding"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batter", required=True)
    parser.add_argument("--bowler", required=True)
    parser.add_argument("--balls", type=float, default=12.0)
    parser.add_argument("--peers", type=int, default=8)
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
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/embeddings-residual-mps-user/player_embeddings.parquet"),
    )
    args = parser.parse_args()
    embeddings = args.embeddings if args.embeddings.exists() else None
    result = rank_vs_bowler(
        batter_query=args.batter,
        bowler_query=args.bowler,
        canonical_dir=args.canonical,
        attributes_path=args.attributes,
        effects_path=args.effects,
        matchups_path=args.matchups,
        co_batters_path=args.co_batters,
        embeddings_path=embeddings,
        balls=args.balls,
        peer_limit=args.peers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
