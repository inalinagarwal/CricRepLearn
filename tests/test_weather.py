"""Weather multiplier helpers (no network)."""

from __future__ import annotations

from cric_rep_learn.weather import apply_weather_multipliers


def test_rain_multiplier_changes_sr() -> None:
    impacts = {
        "buckets": {
            "rain": {
                "rain": {"sr_mult": 0.95, "dismiss_mult": 1.05},
                "dry": {"sr_mult": 1.0, "dismiss_mult": 1.0},
            },
            "temperature": {},
            "humidity": {},
            "wind": {},
        }
    }
    features = {
        "rain_bucket": "rain",
        "temp_bucket": "mild",
        "humidity_bucket": "normal_humidity",
        "wind_bucket": "calm",
    }
    sr, out, notes = apply_weather_multipliers(
        sr=1.2, dismissal_rate=0.05, features=features, impacts=impacts
    )
    assert abs(sr - 1.2 * 0.95) < 1e-9
    assert abs(out - 0.05 * 1.05) < 1e-9
    assert any(n.startswith("rain:rain") for n in notes)
