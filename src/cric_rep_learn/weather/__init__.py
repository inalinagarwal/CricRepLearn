"""Daily match weather from Open-Meteo (date-level, venue-local)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PROVIDER = "open-meteo-archive"

# Competitions that are predominantly night T20s (proxy without start time).
_NIGHT_EVENT_TOKENS = (
    "indian premier league",
    "big bash",
    "vitality blast",
    "natwest t20 blast",
    "caribbean premier",
    "pakistan super league",
    "the hundred",
    "international league t20",
    "sa20",
    "lanka premier",
    "bangladesh premier",
)


def _http_get_json(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """GET JSON via urllib+certifi, falling back to curl on SSL/HTTP failures."""
    import ssl
    import subprocess

    import certifi

    request = urllib.request.Request(url, headers={"User-Agent": "cric-rep-learn/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        result = subprocess.run(
            ["curl", "-sS", "--max-time", str(int(timeout)), url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)


def geocode_place(name: str, *, sleep_s: float = 0.15) -> dict[str, Any] | None:
    query = urllib.parse.urlencode({"name": name, "count": 1, "language": "en"})
    payload = _http_get_json(f"{GEOCODE_URL}?{query}")
    time.sleep(sleep_s)
    results = payload.get("results") or []
    if not results:
        return None
    hit = results[0]
    return {
        "latitude": float(hit["latitude"]),
        "longitude": float(hit["longitude"]),
        "timezone": str(hit.get("timezone") or "UTC"),
        "country": hit.get("country"),
        "geocode_name": hit.get("name"),
        "source": "open-meteo-geocoding",
        "verified": False,
    }


def fetch_daily_weather(
    *,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    timezone_name: str = "auto",
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_mean",
            ]
        ),
        "timezone": timezone_name,
    }
    # Archive for past dates; forecast endpoint for today/future.
    today = date.today()
    base = FORECAST_URL if end >= today else ARCHIVE_URL
    url = f"{base}?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url)
    time.sleep(sleep_s)
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "weather_date": pd.to_datetime(times).date,
            "temperature_c": daily.get("temperature_2m_mean"),
            "relative_humidity_pct": daily.get("relative_humidity_2m_mean"),
            "precipitation_mm": daily.get("precipitation_sum"),
            "wind_speed_kph": daily.get("wind_speed_10m_mean"),
            "latitude": float(payload.get("latitude", latitude)),
            "longitude": float(payload.get("longitude", longitude)),
            "timezone": str(payload.get("timezone") or timezone_name),
        }
    )


def _night_proxy(event_name: Any) -> bool | None:
    if event_name is None or (isinstance(event_name, float) and np.isnan(event_name)):
        return None
    text = str(event_name).lower()
    if any(token in text for token in _NIGHT_EVENT_TOKENS):
        return True
    return None


def build_venue_locations(
    matches: pd.DataFrame,
    *,
    cache_path: Path | None = None,
    sleep_s: float = 0.15,
    max_queries: int | None = None,
) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        cached = pq.read_table(cache_path).to_pandas()
        if len(cached) and "latitude" in cached.columns:
            return cached

    places: dict[str, dict[str, Any]] = {}
    # Geocode distinct cities first (better hit rate than long venue strings).
    city_values = sorted(
        {
            str(c).strip()
            for c in matches["city"].dropna().unique().tolist()
            if str(c).strip()
        }
    )
    venue_only = matches[matches["city"].isna() | (matches["city"].astype(str).str.strip() == "")]
    venue_values = sorted(
        {
            str(v).strip()
            for v in venue_only["venue"].dropna().unique().tolist()
            if str(v).strip()
        }
    )
    queries = city_values + venue_values
    if max_queries is not None:
        queries = queries[:max_queries]

    for query in queries:
        try:
            places[query] = geocode_place(query, sleep_s=sleep_s) or {}
        except Exception as exc:  # noqa: BLE001
            places[query] = {"error": str(exc)}

    rows: list[dict[str, Any]] = []
    unique = (
        matches[["venue", "city"]]
        .drop_duplicates()
        .sort_values(["city", "venue"], na_position="last")
    )
    for row in unique.to_dict(orient="records"):
        city = None if pd.isna(row.get("city")) else str(row["city"]).strip() or None
        venue = None if pd.isna(row.get("venue")) else str(row["venue"]).strip() or None
        query = city or venue
        if not query or query not in places:
            continue
        geo = places[query]
        if "latitude" not in geo:
            continue
        rows.append(
            {
                "venue": venue,
                "city": city,
                "country": geo.get("country"),
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "timezone": geo.get("timezone"),
                "source": geo.get("source"),
                "verified": bool(geo.get("verified")),
                "geocode_query": query,
                "geocode_name": geo.get("geocode_name"),
            }
        )
    frame = pd.DataFrame(rows)
    if cache_path and len(frame):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            cache_path,
            compression="zstd",
        )
    return frame


def build_match_weather(
    canonical_dir: Path,
    output_dir: Path,
    *,
    sleep_s: float = 0.2,
    max_locations: int | None = None,
) -> dict[str, Any]:
    """
    Pull day-average weather for every Cricsheet match date at geocoded venues.

    temporal_resolution = 'daily'. Day/night is a competition proxy only.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = pq.read_table(
        canonical_dir / "matches.parquet",
        columns=[
            "match_id",
            "match_date",
            "venue",
            "city",
            "event_name",
            "team_type",
        ],
    ).to_pandas()
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date

    locations = build_venue_locations(
        matches,
        cache_path=output_dir / "venue_locations.parquet",
        sleep_s=sleep_s,
        max_queries=max_locations,
    )
    if locations.empty or "latitude" not in locations.columns:
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "matches_with_weather": 0,
            "error": "no venues geocoded",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    def _coords_for_match(venue: Any, city: Any) -> dict[str, Any] | None:
        city_s = None if pd.isna(city) else str(city).strip()
        venue_s = None if pd.isna(venue) else str(venue).strip()
        if city_s:
            hit = locations[locations["city"].fillna("").str.lower() == city_s.lower()]
            if len(hit):
                return hit.iloc[0].to_dict()
        if venue_s:
            hit = locations[locations["venue"].fillna("") == venue_s]
            if len(hit):
                return hit.iloc[0].to_dict()
            hit = locations[
                locations["venue"].fillna("").str.contains(venue_s, case=False, na=False)
            ]
            if len(hit):
                return hit.iloc[0].to_dict()
        return None

    coord_rows = []
    for row in matches.to_dict(orient="records"):
        geo = _coords_for_match(row.get("venue"), row.get("city"))
        if not geo:
            continue
        coord_rows.append(
            {
                "match_id": row["match_id"],
                "match_date": row["match_date"],
                "venue": row.get("venue"),
                "city": row.get("city"),
                "event_name": row.get("event_name"),
                "latitude": float(geo["latitude"]),
                "longitude": float(geo["longitude"]),
                "timezone": geo.get("timezone"),
            }
        )
    keyed = pd.DataFrame(coord_rows)
    if keyed.empty:
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "matches_with_weather": 0,
            "error": "no venues geocoded",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    keyed["loc_key"] = (
        keyed["latitude"].round(2).astype(str)
        + ","
        + keyed["longitude"].round(2).astype(str)
    )

    loc_groups = (
        keyed.groupby("loc_key")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            timezone=("timezone", "first"),
            start=("match_date", "min"),
            end=("match_date", "max"),
            matches=("match_id", "count"),
        )
        .reset_index()
    )
    if max_locations is not None:
        loc_groups = loc_groups.head(max_locations)

    daily_frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    for row in loc_groups.to_dict(orient="records"):
        try:
            frame = fetch_daily_weather(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                start=row["start"],
                end=row["end"],
                timezone_name=str(row["timezone"] or "auto"),
                sleep_s=sleep_s,
            )
            if frame.empty:
                continue
            frame["loc_key"] = row["loc_key"]
            daily_frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            errors.append({"loc_key": row["loc_key"], "error": str(exc)})

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    if not daily.empty:
        pq.write_table(
            pa.Table.from_pandas(daily, preserve_index=False),
            output_dir / "weather_daily.parquet",
            compression="zstd",
        )

    retrieved = datetime.now(timezone.utc)
    weather_rows: list[dict[str, Any]] = []
    if not daily.empty:
        joined = keyed.merge(
            daily,
            left_on=["loc_key", "match_date"],
            right_on=["loc_key", "weather_date"],
            how="left",
            suffixes=("", "_wx"),
        )
        for row in joined.to_dict(orient="records"):
            if pd.isna(row.get("temperature_c")) and pd.isna(row.get("precipitation_mm")):
                continue
            night = _night_proxy(row.get("event_name"))
            precip = row.get("precipitation_mm")
            condition = None
            if precip is not None and not pd.isna(precip):
                condition = "rain" if float(precip) >= 1.0 else "dry"
            if night is True:
                condition = f"{condition or 'unknown'}_night"
            elif night is False:
                condition = f"{condition or 'unknown'}_day"
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
    if not weather.empty:
        # Write extended then also schema-aligned core columns.
        pq.write_table(
            pa.Table.from_pandas(weather, preserve_index=False),
            output_dir / "match_weather.parquet",
            compression="zstd",
        )

    impacts = estimate_weather_impacts(canonical_dir, weather)
    (output_dir / "weather_impacts.json").write_text(
        json.dumps(impacts, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "generated_at_utc": retrieved.isoformat(),
        "provider": PROVIDER,
        "temporal_resolution": "daily",
        "matches_total": int(len(matches)),
        "matches_with_weather": int(len(weather)),
        "locations_geocoded": int(len(locations)),
        "locations_fetched": int(len(loc_groups)),
        "fetch_errors": errors[:20],
        "note": (
            "Day averages at venue-local timezone; day/night is competition proxy only"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {**metadata, "impacts": impacts}


def estimate_weather_impacts(
    canonical_dir: Path, weather: pd.DataFrame
) -> dict[str, Any]:
    """Train-set SR / dismissal multipliers for weather buckets."""
    if weather.empty:
        return {"buckets": {}}
    deliveries = str((canonical_dir / "deliveries.parquet").resolve()).replace("'", "''")
    split = str((canonical_dir / "split_manifest.parquet").resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.register("weather", weather[["match_id", "precipitation_mm",
            "temperature_c", "relative_humidity_pct", "wind_speed_kph", "is_night_proxy"]])
        frame = connection.execute(
            f"""
            SELECT
                CASE WHEN w.precipitation_mm >= 1.0 THEN 'rain' ELSE 'dry' END AS rain_bucket,
                CASE WHEN w.temperature_c >= 30 THEN 'hot'
                     WHEN w.temperature_c < 18 THEN 'cool'
                     ELSE 'mild' END AS temp_bucket,
                CASE WHEN w.relative_humidity_pct >= 75 THEN 'humid' ELSE 'normal_humidity' END
                    AS humidity_bucket,
                CASE WHEN w.wind_speed_kph >= 20 THEN 'windy' ELSE 'calm' END AS wind_bucket,
                CASE WHEN w.is_night_proxy THEN 'night'
                     WHEN w.is_night_proxy IS NULL THEN 'unknown_session'
                     ELSE 'day' END AS session_bucket,
                SUM(d.runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            JOIN weather w USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
            GROUP BY 1, 2, 3, 4, 5
            """
        ).fetchdf()
        overall = connection.execute(
            f"""
            SELECT
                SUM(d.runs_batter)::DOUBLE / COUNT(*) AS sr,
                SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
                    AS dismiss
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            JOIN weather w USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
            """
        ).fetchone()
    finally:
        connection.close()

    g_sr = float(overall[0]) if overall and overall[0] else 1.2
    g_dismiss = float(overall[1]) if overall and overall[1] else 0.05

    def _bucket_impacts(column: str) -> dict[str, dict[str, float]]:
        grouped = (
            frame.groupby(column, as_index=False)[["runs", "balls", "dismissals"]]
            .sum()
        )
        out: dict[str, dict[str, float]] = {}
        for row in grouped.to_dict(orient="records"):
            balls = float(row["balls"])
            if balls < 5000:
                continue
            sr = float(row["runs"] / balls)
            dismiss = float(row["dismissals"] / balls)
            out[str(row[column])] = {
                "balls": balls,
                "sr": sr,
                "dismissal_rate": dismiss,
                "sr_mult": sr / g_sr if g_sr else 1.0,
                "dismiss_mult": dismiss / g_dismiss if g_dismiss else 1.0,
            }
        return out

    return {
        "baseline": {"sr": g_sr, "dismissal_rate": g_dismiss},
        "buckets": {
            "rain": _bucket_impacts("rain_bucket"),
            "temperature": _bucket_impacts("temp_bucket"),
            "humidity": _bucket_impacts("humidity_bucket"),
            "wind": _bucket_impacts("wind_bucket"),
            "session": _bucket_impacts("session_bucket"),
        },
        "application": (
            "Multiply batter expected_sr and dismissal_rate by the product of "
            "matching bucket sr_mult / dismiss_mult (vs train baseline)."
        ),
    }


def lookup_weather_features(
    weather_dir: Path,
    *,
    venue: str | None,
    city: str | None = None,
    match_date: date,
    fetch_if_missing: bool = True,
) -> dict[str, Any] | None:
    """Find day-average weather for a venue/city on a date."""
    daily_path = weather_dir / "weather_daily.parquet"
    loc_path = weather_dir / "venue_locations.parquet"
    locations = (
        pq.read_table(loc_path).to_pandas() if loc_path.exists() else pd.DataFrame()
    )
    daily = (
        pq.read_table(daily_path).to_pandas() if daily_path.exists() else pd.DataFrame()
    )

    row = None
    if len(locations):
        hit = pd.DataFrame()
        if city:
            hit = locations[locations["city"].fillna("").str.lower() == city.lower()]
        if hit.empty and venue:
            safe = venue.replace("(", "\\(").replace(")", "\\)")
            hit = locations[
                locations["venue"].fillna("").str.contains(safe, case=False, na=False, regex=True)
                | locations["city"].fillna("").str.contains(safe, case=False, na=False, regex=True)
            ]
        if len(hit):
            row = hit.iloc[0]

    lat = lon = None
    tz = "auto"
    if row is not None:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        tz = str(row.get("timezone") or "auto")
    elif fetch_if_missing and (city or venue):
        geo = geocode_place(city or venue or "", sleep_s=0.05)
        if not geo:
            return None
        lat, lon = geo["latitude"], geo["longitude"]
        tz = str(geo.get("timezone") or "auto")
    else:
        return None

    loc_key = f"{round(lat, 2)},{round(lon, 2)}"
    day = pd.DataFrame()
    if len(daily) and "loc_key" in daily.columns:
        day = daily[
            (daily["loc_key"] == loc_key)
            & (pd.to_datetime(daily["weather_date"]).dt.date == match_date)
        ]
    if day.empty and fetch_if_missing:
        try:
            fetched = fetch_daily_weather(
                latitude=lat,
                longitude=lon,
                start=match_date,
                end=match_date,
                timezone_name=tz,
                sleep_s=0.05,
            )
            if fetched.empty:
                return None
            fetched["loc_key"] = loc_key
            day = fetched
            # Append into cache for reuse.
            weather_dir.mkdir(parents=True, exist_ok=True)
            combined = (
                pd.concat([daily, fetched], ignore_index=True)
                if len(daily)
                else fetched
            )
            pq.write_table(
                pa.Table.from_pandas(combined, preserve_index=False),
                daily_path,
                compression="zstd",
            )
        except Exception:  # noqa: BLE001
            return None
    if day.empty:
        return None
    wx = day.iloc[0]
    precip = float(wx["precipitation_mm"]) if not pd.isna(wx["precipitation_mm"]) else 0.0
    temp = float(wx["temperature_c"]) if not pd.isna(wx["temperature_c"]) else None
    humidity = (
        float(wx["relative_humidity_pct"])
        if not pd.isna(wx["relative_humidity_pct"])
        else None
    )
    wind = float(wx["wind_speed_kph"]) if not pd.isna(wx["wind_speed_kph"]) else None
    return {
        "match_date": match_date.isoformat(),
        "temperature_c": temp,
        "relative_humidity_pct": humidity,
        "precipitation_mm": precip,
        "wind_speed_kph": wind,
        "rain_bucket": "rain" if precip >= 1.0 else "dry",
        "temp_bucket": (
            "hot" if temp is not None and temp >= 30 else
            "cool" if temp is not None and temp < 18 else
            "mild"
        ),
        "humidity_bucket": (
            "humid" if humidity is not None and humidity >= 75 else "normal_humidity"
        ),
        "wind_bucket": "windy" if wind is not None and wind >= 20 else "calm",
        "temporal_resolution": "daily",
        "loc_key": loc_key,
    }


def apply_weather_multipliers(
    *,
    sr: float,
    dismissal_rate: float,
    features: dict[str, Any] | None,
    impacts: dict[str, Any] | None,
) -> tuple[float, float, list[str]]:
    if not features or not impacts:
        return sr, dismissal_rate, []
    buckets = impacts.get("buckets") or {}
    notes: list[str] = []
    mapping = (
        ("rain", features.get("rain_bucket")),
        ("temperature", features.get("temp_bucket")),
        ("humidity", features.get("humidity_bucket")),
        ("wind", features.get("wind_bucket")),
    )
    for family, key in mapping:
        stats = (buckets.get(family) or {}).get(str(key) or "")
        if not stats:
            continue
        sr *= float(stats.get("sr_mult", 1.0))
        dismissal_rate *= float(stats.get("dismiss_mult", 1.0))
        notes.append(
            f"{family}:{key}(sr×{stats.get('sr_mult', 1):.3f},"
            f"out×{stats.get('dismiss_mult', 1):.3f})"
        )
    return sr, dismissal_rate, notes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/canonical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/weather"))
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Optional cap for smoke tests",
    )
    args = parser.parse_args()
    result = build_match_weather(
        args.canonical,
        args.output,
        sleep_s=args.sleep,
        max_locations=args.max_locations,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
