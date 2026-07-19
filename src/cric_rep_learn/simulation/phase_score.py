"""Phase-weighted scoring helpers for innings summaries."""

from __future__ import annotations

from typing import Any

# Relative importance of runs scored in each phase (sums to 1).
# Death slightly higher: scarce balls, fantasy + match leverage.
DEFAULT_PHASE_WEIGHTS = {
    "powerplay": 0.28,
    "middle": 0.40,
    "death": 0.32,
}


def phase_for_over(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over >= 16:
        return "death"
    return "middle"


def summarize_phases(
    over_rows: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Aggregate over-level expected runs/wickets into PP / middle / death.

    Also returns a phase-weighted score =
      sum_phase (weight_phase * expected_runs_phase)
    so death contribution is emphasized relative to raw run share.
    """
    weights = dict(weights or DEFAULT_PHASE_WEIGHTS)
    totals = {
        phase: {"expected_runs": 0.0, "expected_wickets": 0.0, "overs": 0}
        for phase in ("powerplay", "middle", "death")
    }
    for row in over_rows:
        phase = row.get("phase") or phase_for_over(int(row["over"]))
        totals[phase]["expected_runs"] += float(row.get("expected_runs") or 0.0)
        totals[phase]["expected_wickets"] += float(row.get("expected_wickets") or 0.0)
        totals[phase]["overs"] += 1

    team_runs = sum(v["expected_runs"] for v in totals.values()) or 1.0
    weighted = 0.0
    out: dict[str, Any] = {}
    for phase, stats in totals.items():
        w = float(weights.get(phase, 0.0))
        share = stats["expected_runs"] / team_runs
        contribution = w * stats["expected_runs"]
        weighted += contribution
        out[phase] = {
            **stats,
            "run_share": float(share),
            "weight": w,
            "weighted_runs": float(contribution),
        }
    out["phase_weighted_score"] = float(weighted)
    out["weights"] = weights
    return out
