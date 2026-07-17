"""Create chronological match-level train, validation, and test assignments."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLIT_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("match_date", pa.date32()),
        ("split", pa.string()),
    ]
)


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")


def create_chronological_split(
    matches_path: Path,
    output_path: Path,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> dict[str, object]:
    """Assign complete match dates to splits, preventing date-level leakage."""
    _validate_fractions(train_fraction, validation_fraction)
    matches = pq.read_table(matches_path, columns=["match_id", "match_date"])
    if matches.num_rows < 3:
        raise ValueError("At least three matches are required for chronological splits")

    rows = matches.to_pylist()
    counts_by_date: dict[date, int] = {}
    for row in rows:
        counts_by_date[row["match_date"]] = counts_by_date.get(row["match_date"], 0) + 1

    ordered_dates = sorted(counts_by_date)
    if len(ordered_dates) < 3:
        raise ValueError("At least three distinct match dates are required")

    total = len(rows)
    train_target = total * train_fraction
    validation_target = total * (train_fraction + validation_fraction)
    cumulative = 0
    train_end = ordered_dates[0]
    validation_end = ordered_dates[1]
    for match_date in ordered_dates:
        cumulative += counts_by_date[match_date]
        if cumulative <= train_target:
            train_end = match_date
        if cumulative <= validation_target:
            validation_end = match_date

    # Guard tiny datasets while preserving strict date ordering.
    train_end_index = min(max(ordered_dates.index(train_end), 0), len(ordered_dates) - 3)
    train_end = ordered_dates[train_end_index]
    validation_end_index = min(
        max(ordered_dates.index(validation_end), train_end_index + 1),
        len(ordered_dates) - 2,
    )
    validation_end = ordered_dates[validation_end_index]

    split_rows = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        match_date = row["match_date"]
        if match_date <= train_end:
            split = "train"
        elif match_date <= validation_end:
            split = "validation"
        else:
            split = "test"
        counts[split] += 1
        split_rows.append({**row, "split": split})

    split_rows.sort(key=lambda row: (row["match_date"], row["match_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(split_rows, schema=SPLIT_SCHEMA)
    pq.write_table(table, output_path, compression="zstd")

    metadata: dict[str, object] = {
        "strategy": "chronological_by_complete_match_date",
        "train_fraction_requested": train_fraction,
        "validation_fraction_requested": validation_fraction,
        "test_fraction_requested": 1 - train_fraction - validation_fraction,
        "train_end_date": train_end.isoformat(),
        "validation_end_date": validation_end.isoformat(),
        "counts": counts,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_split_manifest(path: Path) -> None:
    table = pq.read_table(path)
    if table.num_rows != pc.count_distinct(table["match_id"]).as_py():
        raise ValueError("A match_id appears in more than one split")

    ranges: dict[str, tuple[date, date]] = {}
    for split in ("train", "validation", "test"):
        subset = table.filter(pc.equal(table["split"], split))
        if subset.num_rows == 0:
            raise ValueError(f"{split} split is empty")
        ranges[split] = (
            pc.min(subset["match_date"]).as_py(),
            pc.max(subset["match_date"]).as_py(),
        )

    if not ranges["train"][1] < ranges["validation"][0]:
        raise ValueError("Training dates overlap validation dates")
    if not ranges["validation"][1] < ranges["test"][0]:
        raise ValueError("Validation dates overlap test dates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("artifacts/canonical/matches.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/canonical/split_manifest.parquet"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()

    metadata = create_chronological_split(
        args.matches,
        args.output,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    validate_split_manifest(args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
