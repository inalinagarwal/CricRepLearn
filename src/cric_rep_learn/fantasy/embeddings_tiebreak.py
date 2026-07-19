"""Optional HB ⊕ batting-embedding tie-break for near-tied fantasy points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def load_batting_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    frame = pq.read_table(path).to_pandas()
    if "role" in frame.columns:
        frame = frame[frame["role"] == "batting"]
    out: dict[str, np.ndarray] = {}
    emb_cols = [c for c in frame.columns if str(c).startswith("emb_")]
    for row in frame.to_dict(orient="records"):
        pid = str(row["player_id"])
        if emb_cols:
            out[pid] = np.asarray([row[c] for c in emb_cols], dtype=np.float64)
        elif "embedding" in row and row["embedding"] is not None:
            out[pid] = np.asarray(row["embedding"], dtype=np.float64)
    return out


def load_hb_effects(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    frame = pq.read_table(path).to_pandas()
    out: dict[str, np.ndarray] = {}
    for row in frame.to_dict(orient="records"):
        pid = str(row["player_id"])
        # Prefer explicit effect vector columns when present.
        if "effect_vector" in row and row["effect_vector"] is not None:
            out[pid] = np.asarray(row["effect_vector"], dtype=np.float64)
            continue
        out[pid] = np.asarray(
            [
                float(row.get("overall_sr") or row.get("sr") or row.get("batting_sr") or 1.2),
                float(row.get("dismissal_rate") or row.get("batting_dismissal_rate") or 0.05),
                float(np.log1p(float(row.get("balls") or row.get("batting_balls") or 0.0))),
            ],
            dtype=np.float64,
        )
    return out


def combined_representation(
    player_id: str,
    *,
    effects: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    emb_weight: float = 0.25,
) -> np.ndarray | None:
    hb = effects.get(player_id)
    emb = embeddings.get(player_id)
    if hb is None and emb is None:
        return None
    parts = []
    if hb is not None:
        parts.append(_unit(hb))
    if emb is not None:
        parts.append(_unit(emb) * emb_weight)
    return np.concatenate(parts)


def apply_embedding_tiebreak(
    pool: list[dict[str, Any]],
    *,
    effects_path: Path = Path("artifacts/player-effects/player_effects.parquet"),
    embeddings_path: Path = Path(
        "artifacts/embeddings-residual-mps-user/player_embeddings.parquet"
    ),
    epsilon: float = 1.5,
) -> list[dict[str, Any]]:
    """
    When two players' fantasy points are within epsilon, nudge ranking by
    combined HB⊕embedding norm (higher = slight preference). Does not change
    rates — garnish only.
    """
    effects = load_hb_effects(effects_path)
    embeddings = load_batting_embeddings(embeddings_path)
    if not effects and not embeddings:
        return pool

    scored = []
    for row in pool:
        rep = combined_representation(
            row["player_id"], effects=effects, embeddings=embeddings
        )
        garnish = float(np.linalg.norm(rep)) if rep is not None else 0.0
        scored.append({**row, "embedding_garnish": garnish})

    # Stable sort: fantasy_points primary, garnish secondary within epsilon bands.
    scored.sort(key=lambda r: (-r["fantasy_points"], -r["embedding_garnish"]))
    # Soft bump: if within epsilon of neighbor above, keep garnish order.
    return scored
