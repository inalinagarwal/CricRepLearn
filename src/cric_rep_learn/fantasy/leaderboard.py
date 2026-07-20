"""Evaluate fantasy XI baselines on holdout matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.fantasy.baselines import (
    BASELINE_STRATEGIES,
    pick_oracle_xi,
    score_xi_actual,
    top11_overlap,
)
from cric_rep_learn.fantasy.calibration import (
    align_actual_pred,
    build_match_box_scores,
    score_box_frame,
    score_prediction_frame,
)
from cric_rep_learn.fantasy.holdout_mc import reconstruct_holdout_matches
from cric_rep_learn.fantasy.roles import map_playing_role, resolve_squad_roles
from cric_rep_learn.fantasy.scoring import load_scoring_weights


def _build_match_pools(
    aligned: pd.DataFrame,
    setups: list[Any],
    *,
    attributes: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    setup_by_id = {s.match_id: s for s in setups}
    pools: dict[str, list[dict[str, Any]]] = {}
    scored = score_prediction_frame(aligned)
    for match_id, group in scored.groupby("match_id"):
        setup = setup_by_id.get(str(match_id))
        if setup is None:
            continue
        attack_ids = {
            b.player_id for b in setup.first_attack
        } | {b.player_id for b in setup.chase_attack}
        order_map: dict[str, int] = {}
        for idx, row in enumerate(setup.first_lineup + setup.chase_lineup):
            pid = str(row["player_id"])
            if pid not in order_map:
                order_map[pid] = idx + 1
        lineup = []
        for row in group.to_dict(orient="records"):
            pid = str(row["player_id"])
            lineup.append(
                {
                    "player_id": pid,
                    "player_name": str(row["player_name"]),
                    "team": str(row["team"]),
                    "batting_order": order_map.get(pid, 99),
                }
            )
        role_info = resolve_squad_roles(
            lineup,
            attributes=attributes,
            attack_ids=attack_ids,
        )
        roles = {pid: info["role"] for pid, info in role_info.items()}
        if "WK" not in roles.values():
            # Promote a batter with keeper metadata if the faced-batter pool omitted WK.
            keeper = None
            for row in lineup:
                attrs = attributes.get(row["player_id"]) or {}
                if map_playing_role(attrs.get("playing_role")) == "WK":
                    keeper = row["player_id"]
                    break
            if keeper is None and lineup:
                keeper = min(lineup, key=lambda r: r["batting_order"])["player_id"]
            if keeper:
                role_info[keeper]["role"] = "WK"
        pool: list[dict[str, Any]] = []
        for row in group.to_dict(orient="records"):
            pid = str(row["player_id"])
            info = role_info[pid]
            pool.append(
                {
                    "player_id": pid,
                    "player_name": str(row["player_name"]),
                    "team": str(row["team"]),
                    "role": info["role"],
                    "credits": float(info["credits"]),
                    "fantasy_points": float(row["fantasy_points"]),
                }
            )
        pools[str(match_id)] = pool
    return pools


def _pool_is_feasible(pool: list[dict[str, Any]]) -> bool:
    from cric_rep_learn.fantasy.optimize import DEFAULT_CONSTRAINTS

    counts = {"WK": 0, "BAT": 0, "AR": 0, "BOWL": 0}
    for p in pool:
        counts[str(p["role"]).upper()] = counts.get(str(p["role"]).upper(), 0) + 1
    c = DEFAULT_CONSTRAINTS
    return (
        len(pool) >= int(c["xi_size"])
        and counts.get("WK", 0) >= int(c["min_wk"])
        and counts.get("BAT", 0) >= int(c["min_bat"])
        and counts.get("BOWL", 0) >= int(c["min_bowl"])
        and counts.get("AR", 0) >= int(c["min_ar"])
    )


def evaluate_baselines(
    canonical_dir: Path,
    *,
    pred_frame: pd.DataFrame | None = None,
    pred_path: Path | None = None,
    splits: tuple[str, ...] = ("validation",),
    seed: int = 7,
    max_matches: int | None = None,
    attributes_path: Path = Path("artifacts/player-attributes/player_attributes.parquet"),
    strategies: tuple[str, ...] | None = None,
    include_oracle: bool = True,
    max_credits: float = 100.0,
) -> dict[str, Any]:
    load_scoring_weights()
    if pred_frame is None:
        if pred_path is None:
            pred_path = Path("artifacts/fantasy/mc_predictions.parquet")
        pred_frame = pd.read_parquet(pred_path)

    box_table = build_match_box_scores(canonical_dir, splits=splits)
    box_pdf = score_box_frame(box_table.to_pandas())
    aligned = align_actual_pred(box_pdf, pred_frame)
    match_ids = sorted(aligned["match_id"].unique().tolist())
    if max_matches is not None and len(match_ids) > max_matches:
        rng = np.random.default_rng(seed)
        match_ids = sorted(rng.choice(match_ids, size=max_matches, replace=False).tolist())
        aligned = aligned[aligned["match_id"].isin(match_ids)].copy()

    attributes = (
        load_attributes_index(attributes_path) if attributes_path.exists() else {}
    )
    setups = reconstruct_holdout_matches(
        canonical_dir,
        splits=splits,
        match_ids=match_ids,
        max_matches=len(match_ids),
        seed=seed,
        attributes=attributes,
    )
    pools = _build_match_pools(aligned, setups, attributes=attributes)

    chosen = strategies or tuple(BASELINE_STRATEGIES.keys())
    for name in chosen:
        if name not in BASELINE_STRATEGIES:
            raise ValueError(f"unknown strategy {name!r}")

    per_match_rows: list[dict[str, Any]] = []
    for match_id in match_ids:
        mid = str(match_id)
        pool = pools.get(mid)
        if not pool or len(pool) < 11 or not _pool_is_feasible(pool):
            continue
        actual_group = aligned[aligned["match_id"] == match_id]
        actual_points = {
            str(r["player_id"]): float(r["fantasy_points"])
            for r in actual_group.to_dict(orient="records")
        }
        actual_top = set(
            actual_group.nlargest(11, "fantasy_points")["player_id"].astype(str)
        )

        oracle_pts = None
        if include_oracle:
            oracle = pick_oracle_xi(pool, actual_points=actual_points)
            oracle_pts = score_xi_actual(
                oracle["players"],
                actual_points=actual_points,
                captain_id=oracle["captain_id"],
                vice_id=oracle["vice_id"],
            )

        for name in chosen:
            fn = BASELINE_STRATEGIES[name]
            if name == "random":
                pick = fn(pool, seed=seed + hash(mid) % 10_000)
            elif name == "credits_value":
                pick = fn(pool, max_credits=max_credits)
            else:
                pick = fn(pool)
            selected_ids = {p["player_id"] for p in pick["players"]}
            actual_pts = score_xi_actual(
                pick["players"],
                actual_points=actual_points,
                captain_id=pick["captain_id"],
                vice_id=pick["vice_id"],
            )
            per_match_rows.append(
                {
                    "match_id": mid,
                    "strategy": name,
                    "actual_xi_points": actual_pts,
                    "predicted_xi_points": float(pick["predicted_xi_points"]),
                    "legal": bool(pick.get("legal", True)),
                    "top11_overlap": top11_overlap(selected_ids, actual_top),
                    "oracle_actual_points": oracle_pts,
                    "regret_vs_oracle": (
                        float(oracle_pts - actual_pts) if oracle_pts is not None else None
                    ),
                }
            )

    detail = pd.DataFrame(per_match_rows)
    if detail.empty:
        return {
            "n_matches": 0,
            "strategies": list(chosen),
            "include_oracle": include_oracle,
            "summary": [],
            "per_match": [],
            "method": (
                "Pick XI from HB MC predicted fantasy points per match pool; "
                "score selected XI with realized box-score fantasy points and "
                "C/VC multipliers from predicted ranks."
            ),
            "note": "no feasible match pools (check WK/role inference)",
        }

    summary_rows = []
    random_means: dict[str, float] = {}
    for strategy, group in detail.groupby("strategy"):
        random_means[strategy] = float(group["actual_xi_points"].mean())
    for strategy, group in detail.groupby("strategy"):
        summary_rows.append(
            {
                "strategy": strategy,
                "n_matches": int(group["match_id"].nunique()),
                "mean_actual_xi_points": float(group["actual_xi_points"].mean()),
                "std_actual_xi_points": float(group["actual_xi_points"].std(ddof=0)),
                "mean_top11_overlap": float(group["top11_overlap"].mean()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean())
                if group["regret_vs_oracle"].notna().any()
                else None,
                "pct_legal": float(group["legal"].mean()),
                "lift_vs_random_pts": float(
                    group["actual_xi_points"].mean() - random_means.get("random", 0.0)
                )
                if strategy != "random" and "random" in random_means
                else None,
            }
        )
    summary = sorted(summary_rows, key=lambda r: -r["mean_actual_xi_points"])

    return {
        "n_matches": int(detail["match_id"].nunique()),
        "strategies": list(chosen),
        "include_oracle": include_oracle,
        "summary": summary,
        "per_match": detail.to_dict(orient="records"),
        "method": (
            "Pick XI from HB MC predicted fantasy points per match pool; "
            "score selected XI with realized box-score fantasy points and "
            "C/VC multipliers from predicted ranks."
        ),
    }


def run_leaderboard(
    canonical_dir: Path,
    output_dir: Path,
    *,
    pred_path: Path = Path("artifacts/fantasy/mc_predictions.parquet"),
    splits: tuple[str, ...] = ("validation",),
    seed: int = 7,
    max_matches: int | None = 50,
    include_oracle: bool = True,
    attributes_path: Path = Path("artifacts/player-attributes/player_attributes.parquet"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_baselines(
        canonical_dir,
        pred_path=pred_path,
        splits=splits,
        seed=seed,
        max_matches=max_matches,
        include_oracle=include_oracle,
        attributes_path=attributes_path,
    )
    report_path = output_dir / "baseline_leaderboard.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = output_dir / "baseline_leaderboard.md"
    md_lines = [
        "# Fantasy baseline leaderboard",
        "",
        f"Matches: **{report['n_matches']}** (holdout MC pools)",
        "",
        report["method"],
        "",
        "| Strategy | Mean actual XI pts | Top-11 overlap | Regret vs oracle | Lift vs random |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        regret_s = (
            f"{row['mean_regret_vs_oracle']:.1f}"
            if row["mean_regret_vs_oracle"] is not None
            else "—"
        )
        lift_s = (
            f"{row['lift_vs_random_pts']:+.1f}"
            if row["lift_vs_random_pts"] is not None
            else "—"
        )
        md_lines.append(
            f"| `{row['strategy']}` | {row['mean_actual_xi_points']:.1f} | "
            f"{row['mean_top11_overlap']:.1%} | {regret_s} | {lift_s} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    report["report_md"] = str(md_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fantasy"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/fantasy/mc_predictions.parquet"),
    )
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-matches",
        type=int,
        default=50,
        help="Subsample holdout matches for faster leaderboard (default 50)",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip per-match oracle upper bound (faster)",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("artifacts/player-attributes/player_attributes.parquet"),
    )
    args = parser.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    report = run_leaderboard(
        args.canonical,
        args.output,
        pred_path=args.predictions,
        splits=splits,
        seed=args.seed,
        max_matches=args.max_matches,
        include_oracle=not args.skip_oracle,
        attributes_path=args.attributes,
    )
    print(json.dumps({"summary": report["summary"], "paths": {
        "json": report["report_json"],
        "md": report["report_md"],
    }}, indent=2))


if __name__ == "__main__":
    main()
