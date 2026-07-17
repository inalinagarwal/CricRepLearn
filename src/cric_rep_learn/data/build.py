"""Build canonical Parquet tables from a directory of Cricsheet JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .parser import CricsheetParser, ParseError
from .schema import (
    DELIVERY_SCHEMA,
    INNINGS_SCHEMA,
    MATCH_PLAYER_SCHEMA,
    MATCH_SCHEMA,
    PLAYER_ALIAS_SCHEMA,
    REPLACEMENT_SCHEMA,
    REVIEW_SCHEMA,
    SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA,
    WICKET_SCHEMA,
)


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    match_id: str
    path: Path
    source_file: str
    source_dataset: str
    sha256: str
    data_version: str
    revision: int
    created: str


class BufferedParquetWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        buffer_size: int = 50_000,
    ):
        self.path = path
        self.schema = schema
        self.buffer_size = buffer_size
        self.rows: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows.extend(rows)
        if len(self.rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _read_candidate(path: Path, input_root: Path) -> SourceCandidate:
    payload = path.read_bytes()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON in {path}: {exc}") from exc
    meta = raw.get("meta") or {}
    return SourceCandidate(
        match_id=path.stem,
        path=path,
        source_file=str(path.relative_to(input_root)),
        source_dataset=path.parent.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        data_version=str(meta.get("data_version", "")),
        revision=int(meta.get("revision", 0)),
        created=str(meta.get("created", "")),
    )


def discover_sources(input_root: Path) -> tuple[list[SourceCandidate], list[dict[str, Any]]]:
    """Select one source per Cricsheet match ID and record all duplicate decisions."""
    grouped: dict[str, list[SourceCandidate]] = defaultdict(list)
    for path in sorted(input_root.rglob("*.json")):
        candidate = _read_candidate(path, input_root)
        grouped[candidate.match_id].append(candidate)

    selected: list[SourceCandidate] = []
    manifest: list[dict[str, Any]] = []
    for match_id, candidates in sorted(grouped.items()):
        winner = max(
            candidates,
            key=lambda item: (item.revision, item.created, item.data_version, item.source_file),
        )
        selected.append(winner)
        for candidate in candidates:
            is_selected = candidate.path == winner.path
            if is_selected:
                reason = "only_source" if len(candidates) == 1 else "latest_revision"
            elif candidate.sha256 == winner.sha256:
                reason = "identical_duplicate"
            else:
                reason = "superseded_duplicate"
            manifest.append(
                {
                    "match_id": match_id,
                    "source_dataset": candidate.source_dataset,
                    "source_file": candidate.source_file,
                    "source_sha256": candidate.sha256,
                    "data_version": candidate.data_version,
                    "source_created_date": (
                        date.fromisoformat(candidate.created) if candidate.created else None
                    ),
                    "revision": candidate.revision,
                    "selected": is_selected,
                    "selection_reason": reason,
                }
            )
    return selected, manifest


def build_canonical_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    require_registry: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    candidates, manifest_rows = discover_sources(input_root)
    if not candidates:
        raise FileNotFoundError(f"No JSON files found below {input_root}")

    writers = {
        "matches": BufferedParquetWriter(
            output_dir / "matches.parquet", MATCH_SCHEMA, buffer_size=1_000
        ),
        "match_players": BufferedParquetWriter(
            output_dir / "match_players.parquet", MATCH_PLAYER_SCHEMA, buffer_size=20_000
        ),
        "innings": BufferedParquetWriter(
            output_dir / "innings.parquet", INNINGS_SCHEMA, buffer_size=5_000
        ),
        "deliveries": BufferedParquetWriter(output_dir / "deliveries.parquet", DELIVERY_SCHEMA),
        "wickets": BufferedParquetWriter(
            output_dir / "wickets.parquet", WICKET_SCHEMA, buffer_size=10_000
        ),
        "replacements": BufferedParquetWriter(
            output_dir / "replacements.parquet", REPLACEMENT_SCHEMA, buffer_size=2_000
        ),
        "reviews": BufferedParquetWriter(
            output_dir / "reviews.parquet", REVIEW_SCHEMA, buffer_size=5_000
        ),
        "source_manifest": BufferedParquetWriter(
            output_dir / "source_manifest.parquet",
            SOURCE_MANIFEST_SCHEMA,
            buffer_size=10_000,
        ),
    }
    writers["source_manifest"].append(manifest_rows)

    parser = CricsheetParser(require_registry=require_registry, t20_only=True)
    aliases: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    try:
        for index, candidate in enumerate(candidates, start=1):
            try:
                parsed = parser.parse_path(candidate.path, input_root=input_root)
            except ParseError as exc:
                skipped.append(
                    {
                        "match_id": candidate.match_id,
                        "source_file": candidate.source_file,
                        "error": str(exc),
                    }
                )
                continue

            writers["matches"].append([parsed.match])
            writers["match_players"].append(parsed.match_players)
            writers["innings"].append(parsed.innings)
            writers["deliveries"].append(parsed.deliveries)
            writers["wickets"].append(parsed.wickets)
            writers["replacements"].append(parsed.replacements)
            writers["reviews"].append(parsed.reviews)

            seen_aliases: set[tuple[str, str]] = set()
            for player in parsed.match_players:
                key = (player["player_id"], player["player_name"])
                if key in seen_aliases:
                    continue
                seen_aliases.add(key)
                current = aliases.get(key)
                if current is None:
                    aliases[key] = {
                        "player_id": player["player_id"],
                        "player_name": player["player_name"],
                        "first_seen": player["match_date"],
                        "last_seen": player["match_date"],
                        "match_count": 1,
                    }
                else:
                    current["first_seen"] = min(current["first_seen"], player["match_date"])
                    current["last_seen"] = max(current["last_seen"], player["match_date"])
                    current["match_count"] += 1

            if index % 500 == 0:
                print(f"Parsed {index:,}/{len(candidates):,} source matches")
    finally:
        for writer in writers.values():
            writer.close()

    alias_table = pa.Table.from_pylist(
        sorted(aliases.values(), key=lambda row: (row["player_id"], row["player_name"])),
        schema=PLAYER_ALIAS_SCHEMA,
    )
    pq.write_table(alias_table, output_dir / "player_aliases.parquet", compression="zstd")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "require_registry": require_registry,
        "source_json_files": len(manifest_rows),
        "unique_match_ids": len(candidates),
        "parsed_matches": writers["matches"].count,
        "skipped_matches": len(skipped),
        "match_players": writers["match_players"].count,
        "innings": writers["innings"].count,
        "deliveries": writers["deliveries"].count,
        "wickets": writers["wickets"].count,
        "replacements": writers["replacements"].count,
        "reviews": writers["reviews"].count,
        "player_aliases": len(aliases),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "skipped_matches.json").write_text(
        json.dumps(skipped, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="Use match-scoped fallback IDs instead of rejecting missing registry entries",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = build_canonical_dataset(
        args.input,
        args.output,
        require_registry=not args.allow_unregistered,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
