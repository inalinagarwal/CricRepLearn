"""Bowling attack scheduling with T20 cricket allocation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb

from cric_rep_learn.data.bowling_style import parse_bowling_style


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


SixBowlerPlan = Literal["4-4-4-4-2-2", "4-4-4-3-3-2"]


@dataclass
class BowlerSpell:
    player_id: str
    player_name: str
    max_overs: int = 4
    # "pace" | "spin" | "unknown" — unknown treated as pace for death/PP rules.
    pace_group: str = "unknown"
    # Higher score => prefer bowling this phase (from train wicket/economy).
    phase_scores: dict[str, float] = field(default_factory=dict)
    phase_evidence: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def is_spin(self) -> bool:
        return self.pace_group == "spin"

    @property
    def is_pace(self) -> bool:
        # Medium/fast/unknown bowl death and open vs top order.
        return self.pace_group != "spin"


def load_bowler_phase_profiles(
    canonical_dir: Path,
    bowler_ids: list[str],
    *,
    min_balls: int = 24,
    strength: float = 80.0,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Train bowling effectiveness by phase.

    score = shrunk_wicket_rate / (shrunk_sr_conceded + 0.35)
    Higher score => better to use in that phase (e.g. Bumrah at death).
    """
    if not bowler_ids:
        return {}
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    id_list = ", ".join(f"'{bid}'" for bid in bowler_ids)
    connection = duckdb.connect()
    try:
        globals_ = connection.execute(
            f"""
            SELECT
                d.phase,
                SUM(d.runs_batter)::DOUBLE / COUNT(*) AS sr,
                SUM(d.bowler_wicket_count)::DOUBLE / COUNT(*) AS wicket_rate
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND d.phase IN ('powerplay', 'middle', 'death')
            GROUP BY 1
            """
        ).fetchdf()
        global_map = {
            row["phase"]: {
                "sr": float(row["sr"]),
                "wicket_rate": float(row["wicket_rate"]),
            }
            for row in globals_.to_dict(orient="records")
        }
        frame = connection.execute(
            f"""
            SELECT
                d.bowler_id,
                d.phase,
                SUM(d.runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(d.bowler_wicket_count)::DOUBLE AS wickets
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND d.phase IN ('powerplay', 'middle', 'death')
              AND d.bowler_id IN ({id_list})
            GROUP BY 1, 2
            """
        ).fetchdf()
    finally:
        connection.close()

    out: dict[str, dict[str, dict[str, float]]] = {bid: {} for bid in bowler_ids}
    for row in frame.to_dict(orient="records"):
        phase = row["phase"]
        balls = float(row["balls"])
        g = global_map.get(phase, {"sr": 1.2, "wicket_rate": 0.05})
        sr = (float(row["runs"]) + strength * g["sr"]) / (balls + strength)
        wicket_rate = (float(row["wickets"]) + strength * g["wicket_rate"]) / (
            balls + strength
        )
        score = wicket_rate / (sr + 0.35)
        out[row["bowler_id"]][phase] = {
            "balls": balls,
            "raw_sr": float(row["runs"] / balls) if balls else g["sr"],
            "raw_wicket_rate": float(row["wickets"] / balls) if balls else g["wicket_rate"],
            "sr": sr,
            "wicket_rate": wicket_rate,
            "score": score,
            "enough_evidence": float(balls >= min_balls),
        }
    for bid in bowler_ids:
        for phase, g in global_map.items():
            if phase not in out[bid]:
                score = g["wicket_rate"] / (g["sr"] + 0.35)
                out[bid][phase] = {
                    "balls": 0.0,
                    "raw_sr": g["sr"],
                    "raw_wicket_rate": g["wicket_rate"],
                    "sr": g["sr"],
                    "wicket_rate": g["wicket_rate"],
                    "score": score,
                    "enough_evidence": 0.0,
                }
    return out


def attach_phase_profiles(
    attack: list[BowlerSpell],
    profiles: dict[str, dict[str, dict[str, float]]],
) -> list[BowlerSpell]:
    for bowler in attack:
        profile = profiles.get(bowler.player_id, {})
        bowler.phase_scores = {
            phase: float(stats["score"]) for phase, stats in profile.items()
        }
        bowler.phase_evidence = profile
    return attack


def attach_pace_groups(
    attack: list[BowlerSpell],
    attributes: dict[str, dict[str, Any]] | None,
) -> list[BowlerSpell]:
    """Set pace_group from player attributes / bowling_style_raw."""
    attributes = attributes or {}
    for bowler in attack:
        attrs = attributes.get(bowler.player_id) or {}
        pace = attrs.get("pace_group")
        if not pace or pace == "unknown":
            parsed = parse_bowling_style(
                attrs.get("bowling_style_raw") or attrs.get("bowling_type")
            )
            pace = parsed.pace_group
        bowler.pace_group = str(pace or "unknown")
    return attack


def assign_over_quotas(
    attack: list[BowlerSpell],
    *,
    n_overs: int = 20,
    six_bowler_plan: SixBowlerPlan = "4-4-4-4-2-2",
) -> list[BowlerSpell]:
    """
    Set max_overs from attack list order (input order = bowling priority).

    - 5 bowlers → 4 each
    - 6 bowlers → 4-4-4-4-2-2 (default) or 4-4-4-3-3-2
    - otherwise distribute up to 4/bowler until n_overs filled
    """
    n = len(attack)
    if n == 0:
        raise ValueError("attack must be non-empty")
    if n == 5:
        quotas = [4, 4, 4, 4, 4]
    elif n == 6:
        quotas = (
            [4, 4, 4, 3, 3, 2]
            if six_bowler_plan == "4-4-4-3-3-2"
            else [4, 4, 4, 4, 2, 2]
        )
    else:
        quotas = [0] * n
        left = n_overs
        for i in range(n):
            take = min(4, left)
            quotas[i] = take
            left -= take
            if left <= 0:
                break
        if left > 0:
            raise ValueError(
                f"cannot allocate {n_overs} overs across {n} bowlers (max 4 each)"
            )
    if n_overs == 20 and sum(quotas) != n_overs:
        raise ValueError(f"quota sum {sum(quotas)} != {n_overs}")
    for bowler, q in zip(attack, quotas, strict=True):
        bowler.max_overs = int(q)
    return attack


def assign_over_quotas_from_actual(
    attack: list[BowlerSpell],
    actual_overs: dict[str, float],
    *,
    n_overs: int = 20,
) -> list[BowlerSpell]:
    """
    Set max_overs from observed overs (capped at 4), then pad/trim to n_overs.

    Used for holdout MC so simulated bowling opportunity matches the match.
    List order should already be bowling priority (e.g. actual overs desc).
    """
    if not attack:
        raise ValueError("attack must be non-empty")
    for bowler in attack:
        raw = float(actual_overs.get(bowler.player_id, 0.0))
        quota = int(round(raw))
        if quota <= 0 and raw >= 0.5:
            quota = 1
        bowler.max_overs = max(0, min(4, quota))
    nonzero = [b for b in attack if b.max_overs > 0]
    if nonzero:
        attack[:] = nonzero
    total = sum(b.max_overs for b in attack)
    if total < n_overs:
        for bowler in attack:
            room = 4 - bowler.max_overs
            if room <= 0:
                continue
            add = min(room, n_overs - total)
            bowler.max_overs += add
            total += add
            if total >= n_overs:
                break
    elif total > n_overs:
        for bowler in reversed(attack):
            if total <= n_overs:
                break
            reducible = bowler.max_overs - (
                1 if float(actual_overs.get(bowler.player_id, 0.0)) > 0 else 0
            )
            cut = min(max(0, reducible), total - n_overs)
            bowler.max_overs -= cut
            total -= cut
        attack[:] = [b for b in attack if b.max_overs > 0] or attack[:1]
        total = sum(b.max_overs for b in attack)
        # If still over (all at minimum 1), drop lowest-priority extras.
        while total > n_overs and len(attack) > 1:
            victim = attack[-1]
            victim.max_overs -= 1
            total -= 1
            if victim.max_overs <= 0:
                attack.pop()
    if sum(b.max_overs for b in attack) < n_overs:
        # Abbreviated innings (e.g. short chase with ≤4 bowlers): keep actual
        # quotas; caller schedules min(20, sum) overs.
        return attack
    return attack


def configure_attack(
    attack: list[BowlerSpell],
    *,
    profiles: dict[str, dict[str, dict[str, float]]] | None = None,
    attributes: dict[str, dict[str, Any]] | None = None,
    six_bowler_plan: SixBowlerPlan = "4-4-4-4-2-2",
    assign_quotas: bool = True,
) -> list[BowlerSpell]:
    """Attach pace, phase scores, and over quotas."""
    if attributes is not None:
        attach_pace_groups(attack, attributes)
    if profiles is not None:
        attach_phase_profiles(attack, profiles)
    if assign_quotas:
        assign_over_quotas(attack, six_bowler_plan=six_bowler_plan)
    return attack


def _phase_for_over(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over >= 16:
        return "death"
    return "middle"


def build_over_schedule(
    attack: list[BowlerSpell],
    *,
    n_overs: int | None = None,
) -> list[dict[str, Any]]:
    """
    Assign each over under cricket-aware T20 rules:

    - List order = bowling priority; quotas already on ``max_overs``
    - Death (16–19): pace only, prefer best death phase scores
    - Spinners ranked in top 2–3: bowl out before death (PP/middle)
    - Powerplay: pace vs top order; if #1 bowler is spin, only over 0 is spin

    If ``n_overs`` is None, use ``min(20, sum(max_overs))`` so short-chase
    holdout attacks with <20 quota overs still schedule cleanly.
    """
    if not attack:
        raise ValueError("attack must be non-empty")
    quota_sum = sum(b.max_overs for b in attack)
    if n_overs is None:
        n_overs = min(20, quota_sum)
    if quota_sum < n_overs:
        raise ValueError(
            f"attack max overs sum to {quota_sum}; need ≥{n_overs}"
        )

    remaining = {b.player_id: b.max_overs for b in attack}
    by_id = {b.player_id: b for b in attack}
    rank = {b.player_id: i for i, b in enumerate(attack)}
    schedule: list[dict[str, Any] | None] = [None] * n_overs

    def available(pred) -> list[BowlerSpell]:
        return [b for b in attack if remaining[b.player_id] > 0 and pred(b)]

    def pick(
        over: int,
        *,
        candidates: list[BowlerSpell],
        phase: str,
        prefer_rank: bool = False,
    ) -> str:
        if not candidates:
            raise RuntimeError(f"no candidates for over {over}")
        prev = (
            schedule[over - 1]["bowler_id"]
            if over > 0 and schedule[over - 1] is not None
            else None
        )

        def sort_key(b: BowlerSpell) -> tuple:
            consecutive_penalty = (
                1 if prev is not None and b.player_id == prev else 0
            )
            if prefer_rank:
                return (
                    consecutive_penalty,
                    rank[b.player_id],
                    -b.phase_scores.get(phase, 0.0),
                    -remaining[b.player_id],
                )
            return (
                consecutive_penalty,
                -b.phase_scores.get(phase, 0.0),
                rank[b.player_id],
                -remaining[b.player_id],
            )

        ranked = sorted(candidates, key=sort_key)
        for cand in ranked:
            if cand.player_id == prev and any(c.player_id != prev for c in ranked):
                continue
            return cand.player_id
        return ranked[0].player_id

    def assign(over: int, bowler_id: str) -> None:
        phase = _phase_for_over(over)
        remaining[bowler_id] -= 1
        b = by_id[bowler_id]
        schedule[over] = {
            "over": over,
            "phase": phase,
            "bowler_id": bowler_id,
            "bowler_name": b.player_name,
            "pace_group": b.pace_group,
            "phase_score": b.phase_scores.get(phase),
        }

    # Death first: pace only.
    death_overs = [o for o in range(n_overs) if _phase_for_over(o) == "death"]
    for over in death_overs:
        pace_left = available(lambda b: b.is_pace)
        if not pace_left:
            pace_left = available(lambda _b: True)
        assign(over, pick(over, candidates=pace_left, phase="death"))

    # Powerplay: top-order vs pace; spinner opens only if #1 is spin.
    top = attack[0]
    pp_overs = [o for o in range(n_overs) if _phase_for_over(o) == "powerplay"]
    for over in pp_overs:
        if schedule[over] is not None:
            continue
        if over == 0 and top.is_spin and remaining[top.player_id] > 0:
            assign(over, top.player_id)
            continue
        pace_left = available(lambda b: b.is_pace)
        if pace_left:
            assign(
                over,
                pick(over, candidates=pace_left, phase="powerplay"),
            )
        else:
            assign(
                over,
                pick(
                    over,
                    candidates=available(lambda _b: True),
                    phase="powerplay",
                ),
            )

    # Middle: finish top-3 spinners before filling with pace.
    middle_overs = [o for o in range(n_overs) if _phase_for_over(o) == "middle"]
    for over in middle_overs:
        if schedule[over] is not None:
            continue
        top_spin = available(lambda b: b.is_spin and rank[b.player_id] <= 2)
        if top_spin:
            assign(
                over,
                pick(
                    over,
                    candidates=top_spin,
                    phase="middle",
                    prefer_rank=True,
                ),
            )
            continue
        other_spin = available(lambda b: b.is_spin)
        if other_spin:
            assign(
                over,
                pick(
                    over,
                    candidates=other_spin,
                    phase="middle",
                    prefer_rank=True,
                ),
            )
            continue
        pool = available(lambda _b: True)
        assign(
            over,
            pick(over, candidates=pool, phase="middle", prefer_rank=True),
        )

    for over in range(n_overs):
        if schedule[over] is not None:
            continue
        pool = available(lambda _b: True)
        if not pool:
            raise RuntimeError("overs left but no remaining bowler quota")
        assign(
            over,
            pick(over, candidates=pool, phase=_phase_for_over(over)),
        )

    _rebalance_spin_before_death(schedule, attack, remaining, by_id, rank)

    assert all(slot is not None for slot in schedule)
    return [slot for slot in schedule if slot is not None]


def _rebalance_spin_before_death(
    schedule: list[dict[str, Any] | None],
    attack: list[BowlerSpell],
    remaining: dict[str, int],
    by_id: dict[str, BowlerSpell],
    rank: dict[str, int],
) -> None:
    """Swap unused top-spinner quota into middle overs currently held by pace."""
    spin_left = [
        b
        for b in attack
        if b.is_spin and remaining[b.player_id] > 0 and rank[b.player_id] <= 2
    ]
    if not spin_left:
        return
    middle_idxs = [
        i
        for i, slot in enumerate(schedule)
        if slot is not None and slot["phase"] == "middle"
    ]
    for spin in sorted(spin_left, key=lambda b: rank[b.player_id]):
        while remaining[spin.player_id] > 0:
            donor = None
            for i in middle_idxs:
                slot = schedule[i]
                assert slot is not None
                if by_id[slot["bowler_id"]].is_pace:
                    donor = i
                    break
            if donor is None:
                return
            slot = schedule[donor]
            assert slot is not None
            old = slot["bowler_id"]
            remaining[old] += 1
            remaining[spin.player_id] -= 1
            schedule[donor] = {
                **slot,
                "bowler_id": spin.player_id,
                "bowler_name": spin.player_name,
                "pace_group": spin.pace_group,
                "phase_score": spin.phase_scores.get("middle"),
            }


def ball_bowler_schedule(over_schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand overs into 120 legal-ball slots."""
    balls: list[dict[str, Any]] = []
    for over in over_schedule:
        for ball_in_over in range(6):
            balls.append(
                {
                    **over,
                    "ball_in_over": ball_in_over,
                    "legal_balls_before": over["over"] * 6 + ball_in_over,
                }
            )
    return balls
