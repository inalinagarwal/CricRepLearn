"""Build canonical player attributes from cricketdata-style metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .bowling_style import parse_batting_hand, parse_bowling_style
from .schema import PLAYER_ATTRIBUTES_SCHEMA


def load_player_meta_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {"cricsheet_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"player meta CSV missing columns: {sorted(missing)}")
    return frame


def build_player_attributes(
    canonical_dir: Path,
    player_meta_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    aliases = pq.read_table(canonical_dir / "player_aliases.parquet").to_pandas()
    primary = (
        aliases.sort_values("match_count", ascending=False)
        .groupby("player_id", as_index=False)
        .first()[["player_id", "player_name", "match_count"]]
    )
    meta = load_player_meta_csv(player_meta_csv)
    meta = meta.rename(
        columns={
            "cricsheet_id": "player_id",
            "batting_style": "batting_style_raw",
            "bowling_style": "bowling_style_raw",
        }
    )
    keep = [
        column
        for column in (
            "player_id",
            "cricinfo_id",
            "full_name",
            "name",
            "country",
            "batting_style_raw",
            "bowling_style_raw",
            "playing_role",
        )
        if column in meta.columns
    ]
    meta = meta[keep].drop_duplicates("player_id", keep="first")

    merged = primary.merge(meta, on="player_id", how="left", indicator=True)
    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        batting_raw = record.get("batting_style_raw")
        bowling_raw = record.get("bowling_style_raw")
        if isinstance(batting_raw, float) and pd.isna(batting_raw):
            batting_raw = None
        if isinstance(bowling_raw, float) and pd.isna(bowling_raw):
            bowling_raw = None
        parsed = parse_bowling_style(bowling_raw)
        country = record.get("country")
        if isinstance(country, float) and pd.isna(country):
            country = None
        full_name = record.get("full_name") or record.get("name") or record["player_name"]
        if isinstance(full_name, float) and pd.isna(full_name):
            full_name = record["player_name"]
        cricinfo_id = record.get("cricinfo_id")
        if isinstance(cricinfo_id, float) and pd.isna(cricinfo_id):
            cricinfo_id = None
        playing_role = record.get("playing_role")
        if isinstance(playing_role, float) and pd.isna(playing_role):
            playing_role = None
        matched = record["_merge"] == "both"
        rows.append(
            {
                "player_id": record["player_id"],
                "player_name": record["player_name"],
                "full_name": str(full_name),
                "cricinfo_id": None if cricinfo_id is None else str(cricinfo_id),
                "country": None if country is None else str(country),
                "batting_style_raw": batting_raw,
                "bowling_style_raw": bowling_raw,
                "batting_hand": parse_batting_hand(batting_raw),
                "bowling_arm": parsed.bowling_arm,
                "pace_group": parsed.pace_group,
                "bowling_family": parsed.bowling_family,
                "arm_pace_key": parsed.arm_pace_key,
                "playing_role": None if playing_role is None else str(playing_role),
                "source": "cricketdata_player_meta" if matched else None,
                "matched": bool(matched),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=PLAYER_ATTRIBUTES_SCHEMA)
    output_path = output_dir / "player_attributes.parquet"
    pq.write_table(table, output_path, compression="zstd")

    matched_n = sum(row["matched"] for row in rows)
    with_bowling = sum(
        row["matched"] and row["bowling_arm"] != "unknown" for row in rows
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir.resolve()),
        "player_meta_csv": str(player_meta_csv.resolve()),
        "output": str(output_path.resolve()),
        "players": len(rows),
        "matched": matched_n,
        "match_rate": matched_n / max(len(rows), 1),
        "known_bowling_arm": with_bowling,
        "source": "cricketdata player_meta joined on cricsheet_id",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_attributes_index(path: Path) -> dict[str, dict[str, Any]]:
    table = pq.read_table(path)
    return {
        row["player_id"]: row
        for row in table.to_pylist()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--player-meta",
        type=Path,
        default=Path("resources/player_meta_cricketdata.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/player-attributes"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_player_attributes(args.canonical, args.player_meta, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
