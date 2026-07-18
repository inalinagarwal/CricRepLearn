"""Normalize free-text bowling / batting styles into hierarchical archetypes."""

from __future__ import annotations

import re
from dataclasses import dataclass


_PACE_TOKENS = (
    "fast",
    "medium",
    "medium-fast",
    "fast-medium",
    "pace",
    "seam",
)
_SPIN_TOKENS = (
    "spin",
    "offbreak",
    "off-break",
    "off break",
    "legbreak",
    "leg-break",
    "leg break",
    "orthodox",
    "unorthodox",
    "chinaman",
    "wrist",
    "finger",
)


@dataclass(frozen=True, slots=True)
class ParsedBowlingStyle:
    raw: str | None
    bowling_arm: str
    pace_group: str
    bowling_family: str

    @property
    def arm_pace_key(self) -> str:
        return f"{self.bowling_arm}_{self.pace_group}"

    @property
    def label(self) -> str:
        if self.bowling_arm == "unknown" and self.pace_group == "unknown":
            return "unknown"
        arm = {"left": "left-arm", "right": "right-arm"}.get(
            self.bowling_arm, self.bowling_arm
        )
        pace = self.pace_group if self.pace_group != "unknown" else "unknown"
        return f"{arm} {pace}".strip()


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[\s_/]+", " ", value)
    value = value.replace("right hand", "right-hand").replace("left hand", "left-hand")
    value = value.replace("right arm", "right-arm").replace("left arm", "left-arm")
    return value


def parse_batting_hand(style: str | None) -> str:
    value = _normalize(style)
    if not value:
        return "unknown"
    if "left" in value:
        return "left"
    if "right" in value:
        return "right"
    return "unknown"


def parse_bowling_style(style: str | None) -> ParsedBowlingStyle:
    raw = None if style is None or (isinstance(style, float) and str(style) == "nan") else str(style)
    value = _normalize(raw)
    if not value or value in {"nan", "none", "null"}:
        return ParsedBowlingStyle(raw=None, bowling_arm="unknown", pace_group="unknown", bowling_family="unknown")

    if "left" in value:
        arm = "left"
    elif "right" in value:
        arm = "right"
    else:
        arm = "unknown"

    family = "unknown"
    if "chinaman" in value or "unorthodox" in value:
        family = "unorthodox_spin"
    elif "orthodox" in value:
        family = "orthodox_spin"
    elif "off" in value and "break" in value.replace("-", " "):
        family = "offbreak"
    elif "leg" in value and "break" in value.replace("-", " "):
        family = "legbreak"
    elif "wrist" in value:
        family = "wrist_spin"
    elif "finger" in value:
        family = "finger_spin"
    elif "fast" in value and "medium" in value:
        family = "fast_medium"
    elif "fast" in value:
        family = "fast"
    elif "medium" in value:
        family = "medium"
    elif "seam" in value or "pace" in value:
        family = "pace"

    if any(token in value for token in _SPIN_TOKENS) or family.endswith("spin") or family in {
        "offbreak",
        "legbreak",
        "orthodox_spin",
        "unorthodox_spin",
        "wrist_spin",
        "finger_spin",
    }:
        pace_group = "spin"
    elif any(token in value for token in _PACE_TOKENS) or family in {
        "fast",
        "medium",
        "fast_medium",
        "pace",
    }:
        pace_group = "pace"
    else:
        pace_group = "unknown"

    return ParsedBowlingStyle(
        raw=raw,
        bowling_arm=arm,
        pace_group=pace_group,
        bowling_family=family,
    )


def nation_arm_pace_label(country: str | None, parsed: ParsedBowlingStyle) -> str:
    nation = (country or "unknown").strip() or "unknown"
    return f"{nation} {parsed.label}"
