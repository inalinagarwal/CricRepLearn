"""Leakage-safe statistical baselines for delivery outcomes."""

from .historical import (
    BaselinePrediction,
    HistoricalBaseline,
    MatchContext,
    SmoothingConfig,
)

__all__ = [
    "BaselinePrediction",
    "HistoricalBaseline",
    "MatchContext",
    "SmoothingConfig",
]
