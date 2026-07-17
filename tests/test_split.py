from __future__ import annotations

from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from cric_rep_learn.data.split import (
    create_chronological_split,
    validate_split_manifest,
)


def test_chronological_split_keeps_dates_and_matches_isolated(tmp_path) -> None:
    start = date(2020, 1, 1)
    rows = []
    for index in range(20):
        # Two matches on each date ensure date groups cannot straddle splits.
        match_date = start + timedelta(days=index // 2)
        rows.append({"match_id": str(index), "match_date": match_date})

    matches_path = tmp_path / "matches.parquet"
    output_path = tmp_path / "split_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), matches_path)

    metadata = create_chronological_split(matches_path, output_path)
    validate_split_manifest(output_path)

    result = pq.read_table(output_path).to_pylist()
    date_splits: dict[date, set[str]] = {}
    for row in result:
        date_splits.setdefault(row["match_date"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in date_splits.values())
    assert sum(metadata["counts"].values()) == 20
    assert set(metadata["counts"]) == {"train", "validation", "test"}
