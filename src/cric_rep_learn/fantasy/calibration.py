"""Calibrate fantasy scoring against holdout reconstructed box scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cric_rep_learn.fantasy.scoring import (
    DEFAULT_WEIGHTS,
    batting_points,
    bowling_points,
    load_scoring_weights,
    merge_player_points,
    save_scoring_weights,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom <= 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_match_box_scores(
    canonical_dir: Path,
    *,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pa.Table:
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    aliases = _escape(canonical_dir / "player_aliases.parquet")
    split_list = ", ".join(f"'{s}'" for s in splits)
    connection = duckdb.connect()
    try:
        batting = connection.execute(
            f"""
            WITH faced AS (
                SELECT
                    d.match_id,
                    d.batter_id AS player_id,
                    d.batting_team AS team,
                    d.runs_batter,
                    CASE WHEN d.is_legal OR d.extras_noballs > 0 THEN 1 ELSE 0 END AS faced,
                    CASE WHEN d.is_boundary AND d.runs_batter = 4 THEN 1 ELSE 0 END AS is_four,
                    CASE WHEN d.is_boundary AND d.runs_batter = 6 THEN 1 ELSE 0 END AS is_six,
                    CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END AS dismissed
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split IN ({split_list})
                  AND NOT d.is_super_over
            )
            SELECT
                match_id,
                player_id,
                any_value(team) AS team,
                SUM(runs_batter)::DOUBLE AS runs,
                SUM(faced)::DOUBLE AS balls,
                SUM(is_four)::DOUBLE AS fours,
                SUM(is_six)::DOUBLE AS sixes,
                SUM(dismissed)::DOUBLE AS dismissals
            FROM faced
            WHERE faced = 1
            GROUP BY 1, 2
            """
        ).fetchdf()
        bowling = connection.execute(
            f"""
            SELECT
                d.match_id,
                d.bowler_id AS player_id,
                any_value(d.bowling_team) AS team,
                SUM(d.bowler_wicket_count)::DOUBLE AS wickets,
                SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END)::DOUBLE AS legal_balls,
                SUM(
                    d.runs_total - d.extras_byes - d.extras_legbyes - d.extras_penalty
                )::DOUBLE AS runs_conceded
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split IN ({split_list})
              AND NOT d.is_super_over
            GROUP BY 1, 2
            HAVING SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END) > 0
            """
        ).fetchdf()
        names = connection.execute(
            f"""
            SELECT player_id, player_name
            FROM (
                SELECT player_id, player_name,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY match_count DESC) AS rn
                FROM read_parquet('{aliases}')
            ) t
            WHERE rn = 1
            """
        ).fetchdf()
    finally:
        connection.close()

    batting = batting.merge(names, on="player_id", how="left")
    bowling = bowling.merge(names, on="player_id", how="left")
    batting["overs"] = 0.0
    batting["wickets"] = 0.0
    batting["runs_conceded"] = 0.0
    batting["economy"] = None
    bowling["runs"] = 0.0
    bowling["balls"] = 0.0
    bowling["fours"] = 0.0
    bowling["sixes"] = 0.0
    bowling["dismissals"] = 0.0
    bowling["overs"] = bowling["legal_balls"] / 6.0
    bowling["economy"] = np.where(
        bowling["overs"] > 0,
        bowling["runs_conceded"] / bowling["overs"],
        np.nan,
    )

    # Outer merge bat + bowl per match-player.
    keys = ["match_id", "player_id"]
    merged = batting.merge(
        bowling,
        on=keys,
        how="outer",
        suffixes=("_bat", "_bowl"),
    )
    def _coalesce(frame, bat_col, bowl_col, default=0.0):
        a = frame.get(bat_col)
        b = frame.get(bowl_col)
        if a is None:
            return b.fillna(default) if b is not None else default
        if b is None:
            return a.fillna(default)
        return a.fillna(b).fillna(default)

    import pandas as pd

    name_bat = merged["player_name_bat"] if "player_name_bat" in merged.columns else None
    name_bowl = merged["player_name_bowl"] if "player_name_bowl" in merged.columns else None
    if name_bat is not None and name_bowl is not None:
        player_name = name_bat.fillna(name_bowl)
    elif name_bat is not None:
        player_name = name_bat
    elif name_bowl is not None:
        player_name = name_bowl
    else:
        player_name = merged["player_id"]

    team_bat = merged["team_bat"] if "team_bat" in merged.columns else None
    team_bowl = merged["team_bowl"] if "team_bowl" in merged.columns else None
    if team_bat is not None and team_bowl is not None:
        team = team_bat.fillna(team_bowl)
    elif team_bat is not None:
        team = team_bat
    else:
        team = team_bowl

    frame = pd.DataFrame(
        {
            "match_id": merged["match_id"],
            "player_id": merged["player_id"],
            "player_name": player_name,
            "team": team,
            "runs": _coalesce(merged, "runs_bat", "runs_bowl"),
            "balls": _coalesce(merged, "balls_bat", "balls_bowl"),
            "fours": _coalesce(merged, "fours_bat", "fours_bowl"),
            "sixes": _coalesce(merged, "sixes_bat", "sixes_bowl"),
            "dismissals": _coalesce(merged, "dismissals_bat", "dismissals_bowl"),
            "wickets": _coalesce(merged, "wickets_bat", "wickets_bowl"),
            "overs": _coalesce(merged, "overs_bat", "overs_bowl"),
            "runs_conceded": _coalesce(merged, "runs_conceded_bat", "runs_conceded_bowl"),
        }
    )
    frame["economy"] = np.where(
        frame["overs"] > 0, frame["runs_conceded"] / frame["overs"], np.nan
    )
    frame = score_box_frame(frame)
    return pa.Table.from_pandas(frame, preserve_index=False)


def score_box_frame(frame) -> Any:
    """Recompute fantasy_points column with current weights."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    pts = []
    for row in frame.to_dict(orient="records"):
        bat = batting_points(
            {
                "runs": row["runs"],
                "balls": row["balls"],
                "fours": row.get("fours") or 0,
                "sixes": row.get("sixes") or 0,
            }
        )
        econ = row.get("economy")
        if econ is not None and (econ != econ):  # NaN
            econ = None
        bowl = bowling_points(
            {
                "wickets": row.get("wickets") or 0,
                "overs": row.get("overs") or 0,
                "economy": econ,
            }
        )
        pts.append(bat["batting_points"] + bowl["bowling_points"])
    frame = frame.copy()
    frame["fantasy_points"] = pts
    return frame


def closed_form_predict_points(
    *,
    expected_runs: float,
    expected_balls: float,
    expected_wickets: float,
    expected_overs: float,
    expected_economy: float | None,
    expected_fours: float = 0.0,
    expected_sixes: float = 0.0,
) -> float:
    row = merge_player_points(
        player_id="x",
        player_name="x",
        team="T",
        role="BAT",
        batting={
            "expected_runs": expected_runs,
            "expected_balls": expected_balls,
            "expected_fours": expected_fours,
            "expected_sixes": expected_sixes,
        },
        bowling={
            "expected_wickets": expected_wickets,
            "expected_overs": expected_overs,
            "expected_economy": expected_economy,
        },
    )
    return float(row["fantasy_points"])


def _topk_hit_rate(
    frame,
    pred: np.ndarray,
    *,
    k: int = 11,
) -> float:
    """Fraction of matches where predicted top-k overlaps actual top-k."""
    work = frame.copy()
    work["pred"] = pred
    hits = []
    for _, group in work.groupby("match_id"):
        if len(group) < k:
            continue
        actual_top = set(group.nlargest(k, "fantasy_points")["player_id"])
        pred_top = set(group.nlargest(k, "pred")["player_id"])
        hits.append(len(actual_top & pred_top) / float(k))
    return float(np.mean(hits)) if hits else 0.0


def tune_bowl_wicket_weight(
    box_frame,
    *,
    candidates: list[float] | None = None,
    max_matches: int = 200,
    seed: int = 7,
) -> dict[str, Any]:
    """
    Grid-search BOWL_WICKET on holdout box scores (val-only by caller).

    Predicted points: shrink realized batting/bowling stats toward priors
    (proxy for imperfect HB/MC forecast). Rank by Spearman + top-11 hit rate.
    """
    candidates = candidates or [25.0, 30.0, 35.0]
    frame = box_frame.to_pandas() if hasattr(box_frame, "to_pandas") else box_frame.copy()
    match_ids = sorted(frame["match_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    if len(match_ids) > max_matches:
        keep = set(rng.choice(match_ids, size=max_matches, replace=False).tolist())
        frame = frame[frame["match_id"].isin(keep)].copy()

    best = None
    results = []
    for wicket_pts in candidates:
        trial = Path("/tmp/fantasy_weight_trial.json")
        save_scoring_weights({**DEFAULT_WEIGHTS, "BOWL_WICKET": wicket_pts}, trial)
        load_scoring_weights(trial)
        scored = score_box_frame(frame)
        pred_rows = []
        for row in scored.to_dict(orient="records"):
            pred_rows.append(
                closed_form_predict_points(
                    expected_runs=0.75 * float(row["runs"]) + 0.25 * 20.0,
                    expected_balls=0.75 * float(row["balls"]) + 0.25 * 15.0,
                    expected_fours=0.75 * float(row.get("fours") or 0),
                    expected_sixes=0.75 * float(row.get("sixes") or 0),
                    expected_wickets=0.75 * float(row.get("wickets") or 0) + 0.25 * 0.8,
                    expected_overs=0.75 * float(row.get("overs") or 0) + 0.25 * 2.0,
                    expected_economy=float(row["economy"])
                    if row.get("economy") == row.get("economy")
                    else 7.5,
                )
            )
        actual = scored["fantasy_points"].to_numpy(dtype=np.float64)
        pred = np.asarray(pred_rows, dtype=np.float64)
        mae = float(np.mean(np.abs(pred - actual)))
        spearman = _spearman(pred, actual)
        topk = _topk_hit_rate(scored, pred, k=11)
        row = {
            "BOWL_WICKET": wicket_pts,
            "mae": mae,
            "spearman": spearman,
            "top11_hit_rate": topk,
            "n_matches": int(frame["match_id"].nunique()),
            "n_rows": int(len(scored)),
            "actual_mean": float(actual.mean()),
            "pred_mean": float(pred.mean()),
            "score": spearman + 0.25 * topk,
        }
        results.append(row)
        if best is None or row["score"] > best["score"]:
            best = row

    assert best is not None
    return {"candidates": results, "best": best, "max_matches": max_matches}


def run_calibration(
    canonical_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("validation",),
) -> dict[str, Any]:
    load_scoring_weights()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_match_box_scores(canonical_dir, splits=splits)
    box_path = output_dir / "match_box_scores.parquet"
    pq.write_table(table, box_path, compression="zstd")

    tune = tune_bowl_wicket_weight(table)
    best_w = float(tune["best"]["BOWL_WICKET"])
    weights_path = output_dir / "scoring_weights.json"
    save_scoring_weights(
        {**DEFAULT_WEIGHTS, "BOWL_WICKET": best_w},
        weights_path,
        metadata={
            "tuned_on": list(splits),
            "metric": "spearman_pred_vs_actual_shrunk",
            "tune": tune,
        },
    )
    load_scoring_weights(weights_path)

    # Blind-spot summary for HB / fantasy: bowl vs bat tails and residual gaps.
    frame = score_box_frame(table.to_pandas())
    bowl_heavy = frame[frame["wickets"] >= 2].copy()
    bat_heavy = frame[frame["runs"] >= 40].copy()
    mid = frame[(frame["wickets"] < 2) & (frame["runs"] < 40)].copy()
    report = {
        "box_scores": str(box_path),
        "n_rows": int(len(frame)),
        "splits": list(splits),
        "tune": tune,
        "weights_path": str(weights_path),
        "blind_spots": {
            "note": (
                "HB Monte Carlo under-predicts sparse-matchup and bowl-heavy tails; "
                "bat-heavy high-SR knocks are also stress regions. Use embeddings only "
                "as a fantasy tie-break garnish — do not retrain delivery residual nets."
            ),
            "bowl_heavy_mean_pts": float(bowl_heavy["fantasy_points"].mean())
            if len(bowl_heavy)
            else None,
            "bat_heavy_mean_pts": float(bat_heavy["fantasy_points"].mean())
            if len(bat_heavy)
            else None,
            "mid_pack_mean_pts": float(mid["fantasy_points"].mean()) if len(mid) else None,
            "n_bowl_heavy": int(len(bowl_heavy)),
            "n_bat_heavy": int(len(bat_heavy)),
            "embedding_policy": "tie_break_only",
        },
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fantasy"))
    parser.add_argument(
        "--splits",
        default="validation",
        help="Comma-separated splits (default validation)",
    )
    args = parser.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    report = run_calibration(args.canonical, args.output, splits=splits)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
