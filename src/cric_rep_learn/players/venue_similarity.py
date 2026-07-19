"""Venue resolution and similar-condition nearby expansion.

Weather ingestion is not wired yet, so similarity is a curated regional /
conditions proxy (subcontinent plains, Gulf, Caribbean, etc.). When a queried
ground is sparse, we expand evidence to the same cluster.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Alias cities with no / few Cricsheet rows → search seeds in-corpus.
CITY_ALIASES: dict[str, list[str]] = {
    "islamabad": ["rawalpindi"],
}

# Condition clusters: similar climate / pitch family proxies until weather joins.
# Membership is by city or distinctive venue token (normalized substring).
CONDITION_CLUSTERS: dict[str, list[str]] = {
    "pakistan_plains": [
        "islamabad",
        "rawalpindi",
        "lahore",
        "multan",
        "faisalabad",
        "peshawar",
        "gujranwala",
    ],
    "pakistan_coast": ["karachi"],
    "uae_gulf": ["dubai", "sharjah", "abu dhabi", "abu-dhabi"],
    "bangladesh": ["dhaka", "mirpur", "chittagong", "chattogram", "sylhet"],
    "india_north": ["delhi", "mohali", "chandigarh", "dharamsala", "lucknow", "kanpur"],
    "india_west": ["mumbai", "pune", "ahmedabad", "rajkot", "indore"],
    "india_south": ["chennai", "bengaluru", "bangalore", "hyderabad", "visakhapatnam"],
    "caribbean": [
        "kingston",
        "barbados",
        "bridgetown",
        "trinidad",
        "guyana",
        "st kitts",
        "antigua",
        "st lucia",
    ],
    "england": ["london", "southampton", "birmingham", "manchester", "leeds", "cardiff"],
    "australia_east": ["sydney", "melbourne", "brisbane"],
}


def normalize_place(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean_place(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def cluster_for_query(query: str) -> str | None:
    needle = normalize_place(query)
    for cluster, members in CONDITION_CLUSTERS.items():
        for member in members:
            if member in needle or needle in member:
                return cluster
    return None


def resolve_venues(
    canonical_dir: Path,
    query: str,
    *,
    limit: int = 12,
    include_similar: bool = False,
) -> dict[str, Any]:
    """Fuzzy-match venue/city; optionally expand to same condition cluster."""
    matches = pq.read_table(canonical_dir / "matches.parquet").to_pandas()
    needle = normalize_place(query)
    for alias, seeds in CITY_ALIASES.items():
        if alias in needle or needle == alias:
            # Prefer alias seed as primary search, keep original query noted.
            needle = normalize_place(seeds[0])
            break
    tokens = [token for token in needle.split() if token]

    scored: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    catalog: list[dict[str, Any]] = []
    for row in matches[["venue", "city"]].drop_duplicates().to_dict(orient="records"):
        venue = clean_place(row.get("venue"))
        city = clean_place(row.get("city"))
        key = (venue, city)
        if key in seen or (venue is None and city is None):
            continue
        seen.add(key)
        hay = normalize_place(f"{venue or ''} {city or ''}")
        if not hay:
            continue
        catalog.append({"venue": venue, "city": city, "hay": hay})
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
        scored.append({"venue": venue, "city": city, "score": score, "scope": "primary"})

    scored.sort(key=lambda item: (-item["score"], item["venue"] or "", item["city"] or ""))
    primary = [row for row in scored if row["score"] >= 60] or scored[:3]

    cluster = cluster_for_query(query)
    similar: list[dict[str, Any]] = []
    if include_similar and cluster:
        members = CONDITION_CLUSTERS[cluster]
        primary_keys = {(row["venue"], row["city"]) for row in primary}
        for place in catalog:
            if (place["venue"], place["city"]) in primary_keys:
                continue
            hay = place["hay"]
            if any(member in hay for member in members):
                similar.append(
                    {
                        "venue": place["venue"],
                        "city": place["city"],
                        "score": 40,
                        "scope": "similar_conditions",
                        "cluster": cluster,
                    }
                )
        similar.sort(key=lambda item: (item["venue"] or "", item["city"] or ""))

    suggestions: list[str] = []
    if not primary:
        if cluster:
            suggestions = list(CONDITION_CLUSTERS[cluster][:4])
        for alias, seeds in CITY_ALIASES.items():
            if alias in normalize_place(query):
                suggestions = seeds + suggestions

    accepted = list(primary)
    if include_similar:
        accepted = primary + similar

    return {
        "query": query,
        "cluster": cluster,
        "primary": primary[:limit],
        "similar": similar[:limit],
        "accepted": accepted[: max(limit * 3, limit)],
        "exact_or_substring": primary[:limit],
        "matches": scored[:limit],
        "suggestions": suggestions,
        "note": (
            "similar venues use regional/conditions proxies; "
            "historical weather join is not available yet"
        ),
    }


def venue_sql_clause(accepted: list[dict[str, Any]]) -> str:
    predicates: list[str] = []
    for row in accepted:
        parts: list[str] = []
        if row.get("venue"):
            esc = str(row["venue"]).replace("'", "''")
            parts.append(f"m.venue = '{esc}'")
        if row.get("city"):
            esc = str(row["city"]).replace("'", "''")
            parts.append(f"m.city = '{esc}'")
        if parts:
            predicates.append("(" + " AND ".join(parts) + ")")
    return "(" + " OR ".join(predicates) + ")" if predicates else "FALSE"
