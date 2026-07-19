#!/usr/bin/env python3
"""Resume Open-Meteo daily weather for missing loc_keys (long-running)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cric_rep_learn.weather import (
    PROVIDER,
    _night_proxy,
    estimate_weather_impacts,
    fetch_daily_weather,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/weather"))
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()

    out = args.output
    daily = pq.read_table(out / "weather_daily.parquet").to_pandas()
    locs = pq.read_table(out / "venue_locations.parquet").to_pandas()
    matches = pq.read_table(
        args.canonical / "matches.parquet",
        columns=["match_id", "match_date", "venue", "city", "event_name"],
    ).to_pandas()
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date

    def coords(venue, city):
        city_s = None if pd.isna(city) else str(city).strip()
        venue_s = None if pd.isna(venue) else str(venue).strip()
        if city_s:
            hit = locs[locs["city"].fillna("").str.lower() == city_s.lower()]
            if len(hit):
                return hit.iloc[0]
        if venue_s:
            hit = locs[locs["venue"].fillna("") == venue_s]
            if len(hit):
                return hit.iloc[0]
        return None

    rows = []
    for r in matches.to_dict("records"):
        g = coords(r.get("venue"), r.get("city"))
        if g is None:
            continue
        rows.append(
            {
                **r,
                "latitude": float(g["latitude"]),
                "longitude": float(g["longitude"]),
                "timezone": g.get("timezone"),
            }
        )
    keyed = pd.DataFrame(rows)
    keyed["loc_key"] = (
        keyed["latitude"].round(2).astype(str)
        + ","
        + keyed["longitude"].round(2).astype(str)
    )
    need = (
        keyed.groupby("loc_key")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            timezone=("timezone", "first"),
            start=("match_date", "min"),
            end=("match_date", "max"),
        )
        .reset_index()
    )
    have = set(daily["loc_key"].astype(str).unique())
    missing = need[~need["loc_key"].isin(have)].to_dict("records")
    print(
        f"have={len(have)} missing={len(missing)} sleep={args.sleep}",
        flush=True,
    )

    extra = []
    errors = []
    for i, row in enumerate(missing):
        ok = False
        for attempt in range(args.max_retries):
            try:
                frame = fetch_daily_weather(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    start=row["start"],
                    end=row["end"],
                    timezone_name=str(row["timezone"] or "auto"),
                    sleep_s=args.sleep + attempt * 0.5,
                )
                if not frame.empty:
                    frame["loc_key"] = row["loc_key"]
                    extra.append(frame)
                    ok = True
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == args.max_retries - 1:
                    errors.append({"loc_key": row["loc_key"], "error": str(exc)})
                else:
                    time.sleep(2.0 * (attempt + 1))
        if (i + 1) % 10 == 0 or (i + 1) == len(missing):
            print(
                f"progress {i+1}/{len(missing)} ok={len(extra)} err={len(errors)} "
                f"last={'ok' if ok else 'fail'}",
                flush=True,
            )
        # Persist periodically so a kill doesn't lose progress.
        if extra and (i + 1) % 25 == 0:
            merged = pd.concat([daily] + extra, ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["loc_key", "weather_date"], keep="last"
            )
            pq.write_table(
                pa.Table.from_pandas(merged, preserve_index=False),
                out / "weather_daily.parquet",
                compression="zstd",
            )
            daily = merged
            extra = []
            print(f"checkpoint daily_locs={daily['loc_key'].nunique()}", flush=True)

    if extra:
        daily = pd.concat([daily] + extra, ignore_index=True)
        daily = daily.drop_duplicates(subset=["loc_key", "weather_date"], keep="last")
        pq.write_table(
            pa.Table.from_pandas(daily, preserve_index=False),
            out / "weather_daily.parquet",
            compression="zstd",
        )

    # Rebuild match_weather join.
    retrieved = datetime.now(timezone.utc)
    weather_rows = []
    joined = keyed.merge(
        daily,
        left_on=["loc_key", "match_date"],
        right_on=["loc_key", "weather_date"],
        how="left",
        suffixes=("", "_wx"),
    )
    for row in joined.to_dict("records"):
        if pd.isna(row.get("temperature_c")) and pd.isna(row.get("precipitation_mm")):
            continue
        night = _night_proxy(row.get("event_name"))
        precip = row.get("precipitation_mm")
        condition = None
        if precip is not None and not pd.isna(precip):
            condition = "rain" if float(precip) >= 1.0 else "dry"
        if night is True:
            condition = f"{condition or 'unknown'}_night"
        weather_rows.append(
            {
                "match_id": row["match_id"],
                "provider": PROVIDER,
                "observed_at_utc": datetime(
                    row["match_date"].year,
                    row["match_date"].month,
                    row["match_date"].day,
                    tzinfo=timezone.utc,
                ),
                "retrieved_at_utc": retrieved,
                "latitude": float(row.get("latitude_wx") or row["latitude"]),
                "longitude": float(row.get("longitude_wx") or row["longitude"]),
                "temporal_resolution": "daily",
                "is_forecast": False,
                "temperature_c": None
                if pd.isna(row.get("temperature_c"))
                else float(row["temperature_c"]),
                "feels_like_c": None,
                "relative_humidity_pct": None
                if pd.isna(row.get("relative_humidity_pct"))
                else float(row["relative_humidity_pct"]),
                "precipitation_mm": None
                if pd.isna(row.get("precipitation_mm"))
                else float(row["precipitation_mm"]),
                "wind_speed_kph": None
                if pd.isna(row.get("wind_speed_kph"))
                else float(row["wind_speed_kph"]),
                "wind_gust_kph": None,
                "wind_direction_deg": None,
                "cloud_cover_pct": None,
                "surface_pressure_hpa": None,
                "condition": condition,
                "is_night_proxy": night,
                "match_date": row["match_date"].isoformat(),
                "venue": row.get("venue"),
                "city": row.get("city"),
            }
        )
    weather = pd.DataFrame(weather_rows)
    pq.write_table(
        pa.Table.from_pandas(weather, preserve_index=False),
        out / "match_weather.parquet",
        compression="zstd",
    )
    impacts = estimate_weather_impacts(args.canonical, weather)
    (out / "weather_impacts.json").write_text(
        json.dumps(impacts, indent=2) + "\n", encoding="utf-8"
    )
    meta = {
        "generated_at_utc": retrieved.isoformat(),
        "matches_with_weather": int(len(weather)),
        "daily_locations": int(daily["loc_key"].nunique()),
        "retry_errors": errors[:30],
        "n_errors": len(errors),
        "impacts_rain": (impacts.get("buckets") or {}).get("rain"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2)[:900], flush=True)


if __name__ == "__main__":
    main()
