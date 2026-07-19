"""T20 phase helpers aligned with the canonical parser cutoffs."""

from __future__ import annotations


def t20_phase(legal_balls_before: int, *, scheduled_balls: int = 120) -> str:
    """Phase for the next delivery given legal balls already bowled."""
    if scheduled_balls == 120:
        if legal_balls_before < 36:
            return "powerplay"
        if legal_balls_before >= 96:
            return "death"
        return "middle"
    if scheduled_balls <= 0:
        return "unknown"
    progress = legal_balls_before / scheduled_balls
    if progress < 0.30:
        return "powerplay"
    if progress >= 0.80:
        return "death"
    return "middle"


def phase_over_range(phase: str) -> range:
    """0-based over indices for a T20 phase."""
    if phase == "powerplay":
        return range(0, 6)
    if phase == "middle":
        return range(6, 16)
    if phase == "death":
        return range(16, 20)
    return range(0, 20)
