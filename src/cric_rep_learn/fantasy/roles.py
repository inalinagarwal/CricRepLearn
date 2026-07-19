"""Map playing_role / attributes → fantasy WK|BAT|AR|BOWL and credit proxies."""

from __future__ import annotations

import re
from typing import Any


ROLE_ALIASES = {
    "wk": "WK",
    "wicketkeeper": "WK",
    "wicket-keeper": "WK",
    "wicket keeper": "WK",
    "wicketkeeper batter": "WK",
    "wicket-keeper batter": "WK",
    "keeper": "WK",
    "batter": "BAT",
    "batsman": "BAT",
    "bat": "BAT",
    "opening batter": "BAT",
    "middle order batter": "BAT",
    "top order batter": "BAT",
    "bowler": "BOWL",
    "bowl": "BOWL",
    "pace bowler": "BOWL",
    "spin bowler": "BOWL",
    "allrounder": "AR",
    "all-rounder": "AR",
    "all rounder": "AR",
    "batting allrounder": "AR",
    "bowling allrounder": "AR",
}


def map_playing_role(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = re.sub(r"\s+", " ", str(raw).strip().lower())
    if not text or text in {"nan", "none"}:
        return None
    if text in ROLE_ALIASES:
        return ROLE_ALIASES[text]
    # Longer keys first so "wicketkeeper batter" wins over "batter".
    for key, role in sorted(ROLE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if key in text:
            return role
    return None


def infer_role_from_attributes(
    attrs: dict[str, Any] | None,
    *,
    batting_order: int | None = None,
    bowls_in_attack: bool = False,
) -> str:
    attrs = attrs or {}
    mapped = map_playing_role(attrs.get("playing_role"))
    if mapped:
        return mapped

    bowling_raw = str(attrs.get("bowling_style_raw") or attrs.get("bowling_type") or "")
    has_bowl = bool(bowling_raw.strip()) and bowling_raw.lower() not in {"none", "nan"}
    order = batting_order if batting_order is not None else 99

    if bowls_in_attack and order <= 7:
        return "AR"
    if bowls_in_attack or (has_bowl and order >= 8):
        return "BOWL"
    if not has_bowl and order <= 7:
        return "BAT"
    if has_bowl and order <= 6:
        return "AR"
    return "BAT" if order <= 7 else "BOWL"


def credit_proxy(role: str, *, batting_sr: float | None = None) -> float:
    role = role.upper()
    if role == "WK":
        return 8.5
    if role == "AR":
        return 8.5
    if role == "BOWL":
        if batting_sr is not None and batting_sr >= 1.3:
            return 9.0
        return 8.5
    if batting_sr is None:
        return 8.5
    if batting_sr >= 1.45:
        return 10.0
    if batting_sr >= 1.25:
        return 9.0
    return 8.0


def resolve_squad_roles(
    players: list[dict[str, Any]],
    *,
    attributes: dict[str, dict[str, Any]],
    attack_ids: set[str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    attack_ids = attack_ids or set()
    overrides = {str(k): str(v).upper() for k, v in (overrides or {}).items()}
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(players):
        pid = row["player_id"]
        attrs = attributes.get(pid) or {}
        query = str(row.get("query") or row.get("player_name") or "")
        if pid in overrides:
            role = overrides[pid]
            source = "override"
        elif query in overrides:
            role = overrides[query]
            source = "override"
        else:
            order = row.get("batting_order")
            if order is None:
                order = idx + 1
            mapped = map_playing_role(attrs.get("playing_role"))
            bowls = pid in attack_ids
            if mapped and bowls and mapped in {"BAT", "WK"} and int(order) <= 8:
                # Meta often labels batting-first allrounders as batters.
                role = "AR"
                source = "playing_role+attack"
            elif mapped:
                role = mapped
                source = "playing_role"
            else:
                role = infer_role_from_attributes(
                    attrs,
                    batting_order=int(order),
                    bowls_in_attack=bowls,
                )
                source = "inferred"
        sr = float(row["batting_sr"]) if row.get("batting_sr") is not None else None
        out[pid] = {
            "role": role,
            "credits": credit_proxy(role, batting_sr=sr),
            "source": source,
            "playing_role_raw": attrs.get("playing_role"),
        }
    return out
